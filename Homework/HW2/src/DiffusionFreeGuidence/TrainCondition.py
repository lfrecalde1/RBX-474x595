

import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
import time
def val_step(trainer, net_model, val_loader, device):
    net_model.eval()
    epoch_loss_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in val_loader:
            b = images.shape[0]
            x_0 = images.to(device)
            labels = labels.to(device) + 1   # same label convention as train

            # IMPORTANT: no label-dropping / tricks in validation
            #loss = trainer(x_0, labels)      # your trainer already returns mean MSE
            loss = trainer(x_0, labels).sum() / b ** 2.
            epoch_loss_sum += loss.item()
            num_batches += 1

    avg_val_loss = epoch_loss_sum
    return avg_val_loss
    
def make_labels(num_samples: int, device: torch.device) -> torch.Tensor:
    num_classes = 10
    per_class = num_samples // num_classes
    remainder = num_samples % num_classes

    labels = []
    for c in range(num_classes):
        n_c = per_class + (1 if c < remainder else 0)
        if n_c > 0:
            labels.append(torch.full((n_c,), c, dtype=torch.long))
    labels = torch.cat(labels, dim=0).to(device)
    return labels
        
class SyntheticCIFARDiffusionDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, idx):
        x = self.imgs[idx]                
        y = int(self.labels[idx])
        img = self.to_pil(x)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    if modelConfig["augmented_data"] == "No":
        print("Using CIFAR ALONE")
        print(modelConfig["epoch"])
        dataset = CIFAR10(root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
        
        val_dataset = CIFAR10(root='./CIFAR10', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
        
    elif modelConfig["augmented_data"] == "Yes":
        print("Using CIFAR Augmented")
        print(modelConfig["epoch"])
        dataset = generation(modelConfig, None)

        val_dataset = CIFAR10(root='./CIFAR10', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    else:
        print("You should define with Yes or No to the parameter augmented_data")
        return None
        
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], modelConfig["schedule"]).to(device)
    
    # Define the empty list in order to the data of the loss and accuracy
    trainLossList = []
    valLossList = []
    # start training

    schedule_dir = os.path.join(modelConfig["save_dir"], modelConfig["schedule"])
    os.makedirs(schedule_dir, exist_ok=True)  # create save_dir/schedule if it doesn't exist

    for e in range(modelConfig["epoch"]):
        # Aux variable to save the loss of the training
        epoch_loss_sum = 0.0
        num_batches = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                
                # Accumulating loss
                epoch_loss_sum += loss.item()
                num_batches += 1
                
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        # Saving loss
        epoch_avg_loss = epoch_loss_sum
        trainLossList.append(epoch_avg_loss)

        # Saving validation
        val_loss = val_step(trainer, net_model, val_loader, device)
        valLossList.append(val_loss)
        
        warmUpScheduler.step()
        ckpt_name = f"ckpt_{e}_.pt"
        ckpt_path = os.path.join(schedule_dir, ckpt_name)

        torch.save(net_model.state_dict(), ckpt_path)

    # Save Data in case we need this
    np.save(os.path.join(schedule_dir, "train_loss.npy"),np.array(trainLossList))
    np.save(os.path.join(schedule_dir, "val_loss.npy"),np.array(valLossList))


def eval(modelConfig: Dict):
    print("Evaluation of the Network")
    device = torch.device(modelConfig["device"])
    torch.manual_seed(int(time.time()) % 2**31)
    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)
        print("size: ", labels.shape)
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

        schedule_weights = os.path.join(modelConfig["save_dir"], modelConfig["schedule"])
        ckpt = torch.load(os.path.join(
            schedule_weights, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"], schedule=modelConfig["schedule"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print(sampledImgs)
        
        # Path to save the data
        schedule_dir = os.path.join(modelConfig["save_dir"], modelConfig["schedule"], modelConfig["sampledImgName"])
        os.makedirs(schedule_dir, exist_ok=True)  # create save_dir/schedule if it doesn't exist
        save_image(sampledImgs, os.path.join(
            schedule_dir,  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])

def generation(modelConfig: Dict, number_of_loops):
    train_transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print("Generation of Augmented Images")
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=train_transform)

    
    device = torch.device(modelConfig["device"])
    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["data_generation"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["data_generation"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        schedule_weights = os.path.join(modelConfig["save_dir"], modelConfig["schedule"])
        ckpt = torch.load(os.path.join(
            schedule_weights, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"], schedule=modelConfig["schedule"]).to(device)
        # ----- loop to generate multiple synthetic batches -----
        N_per_batch = modelConfig["data_generation"]      
        if number_of_loops is None:
            num_loops = 300   
            print("We are increasing the number of loops to 300")
        else:
            num_loops = number_of_loops
            print("We are increasing the number of loops to:", num_loops)

        all_syn_imgs = []
        all_syn_labels = []

        with torch.no_grad():
            for loop_idx in range(num_loops):
                print(f"[Loop {loop_idx+1}/{num_loops}] generating {N_per_batch} synthetic samples")

                # balanced labels 0..9
                labels = make_labels(N_per_batch, device=device)

                noisyImage = torch.randn(
                    size=[N_per_batch, 3, modelConfig["img_size"], modelConfig["img_size"]],
                    device=device,)

                # diffusion sampling -> typically in [-1, 1]
                sampledImgs = sampler(noisyImage, labels)

                # map to [0, 1] for compatibility with ToPILImage / CIFAR pipeline
                sampledImgs = torch.clamp(sampledImgs * 0.5 + 0.5, 0.0, 1.0)

                all_syn_imgs.append(sampledImgs.cpu())
                all_syn_labels.append(labels.cpu())

        # stack everything into a single tensor
        synthetic_imgs = torch.cat(all_syn_imgs, dim=0)      # [N_total, 3, H, W]
        synthetic_labels = torch.cat(all_syn_labels, dim=0)  # [N_total]

        # ----- synthetic dataset -----
        synthetic_train = SyntheticCIFARDiffusionDataset(
            synthetic_imgs,
        synthetic_labels,
        transform=train_transform,
        )

        # ----- concatenated / augmented dataset -----
        augmented_dataset = ConcatDataset([dataset, synthetic_train])

        print("Real CIFAR-10 size:     ", len(dataset))
        print("Synthetic size:         ", len(synthetic_train))
        print("Augmented dataset size: ", len(augmented_dataset))

        # sanity-check pixel ranges after transform (both should be ~[-1, 1])
        img_real, y_real = dataset[0]
        print("Real min/max:     ", img_real.min().item(), img_real.max().item())

        img_syn, y_syn = synthetic_train[0]
        print("Synthetic min/max:", img_syn.min().item(), img_syn.max().item())

        return augmented_dataset
