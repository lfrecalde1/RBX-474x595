from torch import nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np


class GrayscaleWrapper(Dataset):
    """Wrap any dataset and convert images to 1-channel grayscale + normalize."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        # Normalize for grayscale images
        self.normalize = transforms.Normalize((0.5,), (0.5,))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]

        # img can be a tensor [C,H,W] or a PIL image
        if isinstance(img, torch.Tensor):
            # if RGB tensor: [3, H, W] -> grayscale [1, H, W]
            if img.ndim == 3 and img.size(0) == 3:
                img = TF.rgb_to_grayscale(img)
        else:
            # PIL image -> convert to tensor first
            img = transforms.ToTensor()(img)
            img = TF.rgb_to_grayscale(img)

        img = self.normalize(img)  # grayscale normalization
        return img, label


class Pipeline:
    def __init__(self, augmented_data, data, use_grayscale=True):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Base RGB transform (for non-grayscale case)
        rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        # Dummy load to check `data`, not really used
        aux = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=2)

        if augmented_data:
            print("Loading augmented data to the networks")
            trainset = data  # your augmented dataset
            if use_grayscale:
                # Wrap augmented data so it becomes grayscale everywhere
                trainset = GrayscaleWrapper(trainset)
        else:
            print("Without augmented data to the networks")
            if use_grayscale:
                # CIFAR10 -> ToTensor, then grayscale + normalize via wrapper
                base_train = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True,
                    transform=transforms.ToTensor()
                )
                trainset = GrayscaleWrapper(base_train)
            else:
                trainset = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True,
                    transform=rgb_transform
                )

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=32, shuffle=True, num_workers=2
        )
        
        # test data
        if use_grayscale:
            base_test = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True,
                transform=transforms.ToTensor()
            )
            testset = GrayscaleWrapper(base_test)
        else:
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True,
                transform=rgb_transform
            )

        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False, num_workers=2
        )

        self.classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        )

        self.lossFunc = nn.CrossEntropyLoss()

    def train_step(self, model, optimizer):
        model.train()
        epochloss = 0
        for batchcount, (images, labels) in enumerate(self.trainloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()

            y = model(images)

            loss = self.lossFunc(y, labels)     
            loss.backward()

            optimizer.step()
            
            epochloss += loss.item()

        return epochloss

    def val_step(self, model, name):
        correct = 0
        total = 0
        model.eval()
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batchcount, (images, labels) in enumerate(self.testloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                y = model(images)
                _, predicted = torch.max(y, 1)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        ConfusionMatrix(LabelsTrue=all_labels, LabelsPred=all_preds, class_names=self.classes, name=name)
        return correct*100/total
        

def ConfusionMatrix(LabelsTrue, LabelsPred, class_names, name):
    LabelsTrue = np.array(LabelsTrue)
    LabelsPred = np.array(LabelsPred)

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(LabelsTrue, LabelsPred):
        cm[t, p] += 1

    accuracy = 100.0 * np.sum(LabelsTrue == LabelsPred) / len(LabelsTrue)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(f"{name}_confusion.pdf", bbox_inches="tight")
    plt.close(fig)
    return None


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

LOSS_FN = nn.CrossEntropyLoss()

