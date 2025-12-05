from DiffusionFreeGuidence.TrainCondition import train, eval, generation
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import scienceplots

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
        "schedule": "sine",
    }
    # Load Loss and accuracy over training and validation
    schedule_dir = os.path.join(modelConfig["save_dir"], modelConfig["schedule"])

    loss = np.load(os.path.join(schedule_dir, "train_loss.npy"))
    accuracy = np.load(os.path.join(schedule_dir, "val_loss.npy"))

    plot_samples(loss, "Loss", os.path.join(schedule_dir,"Diffusion_loss_training"))
    plot_samples(accuracy, "Accuracy ", os.path.join(schedule_dir,"Diffusion_accuracy_validation"))

    print(loss.shape)
    print(accuracy.shape)


if __name__ == '__main__':
    main()
