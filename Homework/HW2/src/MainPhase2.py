
from DiffusionFreeGuidence.TrainCondition import train, eval, generation
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import scienceplots
import importlib
import torch
import torch.nn as nn
import networks as net
import time
importlib.reload(net)

# Section to save images
def plot_samples(data, label, name):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))

        
        ax.plot(data, label=label)

        ax.set_xlabel("Epochs")
        ax.set_ylabel(label)
        ax.autoscale(tight=True)
        ax.legend()

        
        fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")

        plt.close(fig)

    return None

def mlp_custom(pipeline, model_name, config):

    model = net.CustomMLP().to(pipeline.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    home_path = os.path.expanduser("./")
    JOB_FOLDER = os.path.join(home_path, "outputs/")
    TRAINED_MDL_PATH = os.path.join(JOB_FOLDER, f"cifar/{model_name}/")
    PATH_pdf = os.path.join(home_path, model_name)

    os.makedirs(JOB_FOLDER, exist_ok=True)
    os.makedirs(TRAINED_MDL_PATH, exist_ok=True)
    os.makedirs(PATH_pdf, exist_ok=True)

    epochs = 40
    trainLossList = []
    valAccList = []

    best_val_acc = -float("inf")
    best_epoch = -1

  
    for eIndex in range(epochs):
        print("Epoch count:", eIndex)

        train_epochloss = pipeline.train_step(model, optimizer)
        val_acc = pipeline.val_step(model, model_name)

        print(eIndex, train_epochloss, val_acc)

        valAccList.append(val_acc)
        trainLossList.append(train_epochloss)

        # checkpoint name includes model_name
        trainedMdlPath = os.path.join(TRAINED_MDL_PATH, f"{model_name}_epoch{eIndex}.pth")
        torch.save(model.state_dict(), trainedMdlPath)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = eIndex
            best_path = os.path.join(TRAINED_MDL_PATH, f"{model_name}_BEST.pth")
            torch.save(model.state_dict(), best_path)
            print(f"BEST saved {best_path} at epoch {eIndex} (val_acc={val_acc:.2f}%)")

    # Save logs with custom name also
    trainLosses = np.array(trainLossList)
    testAccuracies = np.array(valAccList)

    np.savetxt(os.path.join(TRAINED_MDL_PATH, f"{model_name}_train_log"), trainLosses)
    np.savetxt(os.path.join(TRAINED_MDL_PATH, f"{model_name}_val_log"), testAccuracies)

    print(f"Logs saved in {TRAINED_MDL_PATH}")

    plot_samples(trainLosses, "Loss",os.path.join(PATH_pdf,f"{model_name}_train.log"))
    plot_samples(testAccuracies, "Accuracy ", os.path.join(PATH_pdf, f"{model_name}_test.log"))

def ref_cnn_custom(pipeline, model_name, config):

    model = net.RefCNN().to(pipeline.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    home_path = os.path.expanduser("./")
    JOB_FOLDER = os.path.join(home_path, "outputs/")
    TRAINED_MDL_PATH = os.path.join(JOB_FOLDER, f"cifar/{model_name}/")
    PATH_pdf = os.path.join(home_path, model_name)
    
    os.makedirs(JOB_FOLDER, exist_ok=True)
    os.makedirs(TRAINED_MDL_PATH, exist_ok=True)
    os.makedirs(PATH_pdf, exist_ok=True)

    epochs = 40
    trainLossList = []
    valAccList = []

    best_val_acc = -float("inf")
    best_epoch = -1

  
    for eIndex in range(epochs):
        print("Epoch count:", eIndex)

        train_epochloss = pipeline.train_step(model, optimizer)
        val_acc = pipeline.val_step(model, model_name)

        print(eIndex, train_epochloss, val_acc)

        valAccList.append(val_acc)
        trainLossList.append(train_epochloss)

        # checkpoint name includes model_name
        trainedMdlPath = os.path.join(TRAINED_MDL_PATH, f"{model_name}_epoch{eIndex}.pth")
        torch.save(model.state_dict(), trainedMdlPath)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = eIndex
            best_path = os.path.join(TRAINED_MDL_PATH, f"{model_name}_BEST.pth")
            torch.save(model.state_dict(), best_path)
            print(f"BEST saved {best_path} at epoch {eIndex} (val_acc={val_acc:.2f}%)")

    # Save logs with custom name also
    trainLosses = np.array(trainLossList)
    testAccuracies = np.array(valAccList)

    np.savetxt(os.path.join(TRAINED_MDL_PATH, f"{model_name}_train_log"), trainLosses)
    np.savetxt(os.path.join(TRAINED_MDL_PATH, f"{model_name}_val_log"), testAccuracies)

    print(f"Logs saved in {TRAINED_MDL_PATH}")

    plot_samples(trainLosses, "Loss",os.path.join(PATH_pdf,f"{model_name}_train.log"))
    plot_samples(testAccuracies, "Accuracy ", os.path.join(PATH_pdf, f"{model_name}_test.log"))

def custom_cnn_custom(pipeline, model_name, config):

    model = net.RefCNN().to(pipeline.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    home_path = os.path.expanduser("./")
    JOB_FOLDER = os.path.join(home_path, "outputs/")
    TRAINED_MDL_PATH = os.path.join(JOB_FOLDER, f"cifar/{model_name}/")
    PATH_pdf = os.path.join(home_path, model_name)

    os.makedirs(JOB_FOLDER, exist_ok=True)
    os.makedirs(TRAINED_MDL_PATH, exist_ok=True)
    os.makedirs(PATH_pdf, exist_ok=True)

    epochs = 40
    trainLossList = []
    valAccList = []

    best_val_acc = -float("inf")
    best_epoch = -1

  
    for eIndex in range(epochs):
        print("Epoch count:", eIndex)

        train_epochloss = pipeline.train_step(model, optimizer)
        val_acc = pipeline.val_step(model, model_name)

        print(eIndex, train_epochloss, val_acc)

        valAccList.append(val_acc)
        trainLossList.append(train_epochloss)

        # checkpoint name includes model_name
        trainedMdlPath = os.path.join(TRAINED_MDL_PATH, f"{model_name}_epoch{eIndex}.pth")
        torch.save(model.state_dict(), trainedMdlPath)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = eIndex
            best_path = os.path.join(TRAINED_MDL_PATH, f"{model_name}_BEST.pth")
            torch.save(model.state_dict(), best_path)
            print(f"BEST saved {best_path} at epoch {eIndex} (val_acc={val_acc:.2f}%)")

    # Save logs with custom name also
    trainLosses = np.array(trainLossList)
    testAccuracies = np.array(valAccList)

    np.savetxt(os.path.join(TRAINED_MDL_PATH, f"{model_name}_train_log"), trainLosses)
    np.savetxt(os.path.join(TRAINED_MDL_PATH, f"{model_name}_val_log"), testAccuracies)

    print(f"Logs saved in {TRAINED_MDL_PATH}")

    plot_samples(trainLosses, "Loss",os.path.join(PATH_pdf,f"{model_name}_train.log"))
    plot_samples(testAccuracies, "Accuracy ", os.path.join(PATH_pdf, f"{model_name}_test.log"))

torch.manual_seed(1)
def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 40,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsWoAugmentation/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_149_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8,
        "data_generation": 10,
        "data_nrow": 1,
        "augmented_data": "No",
        "schedule": "linear",
    }

    # Create the neural networks
    dataset = generation(modelConfig, None)
    ## Augmented
    pipeline = net.Pipeline(True, dataset)
    mlp_custom(pipeline, model_name="mlp_augmented", config=model_config)

    ## No Augmented
    pipeline = net.Pipeline(False, dataset)
    mlp_custom(pipeline, model_name="mlp_no_augmented", config=model_config)
    
    # Ref CNN
    pipeline = net.Pipeline(True, dataset)
    ref_cnn_custom(pipeline, model_name="ref_cnn_augmented", config=model_config)

    # No Augmented
    pipeline = net.Pipeline(False, dataset)
    ref_cnn_custom(pipeline, model_name="ref_cnn_no_augmented",config=model_config)

    pipeline = net.Pipeline(True, dataset)
    custom_cnn_custom(pipeline, model_name="custom_cnn_augmented",config=model_config)

    # No Augmented
    pipeline = net.Pipeline(False, dataset)
    custom_cnn_custom(pipeline, model_name="custom_cnn_no_augmented",config=model_config)



if __name__ == '__main__':
    main()

