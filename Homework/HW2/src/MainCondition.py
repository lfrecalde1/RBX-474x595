from DiffusionFreeGuidence.TrainCondition import train, eval, generation
import torch

torch.manual_seed(1)
def main(model_config=None):
    modelConfig = {
        "state": "eval", # or eval
        "epoch": 150,
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
    if model_config is not None:
        modelConfig = model_config
    elif modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "eval":
        eval(modelConfig)
    else:
        generation(modelConfig)


if __name__ == '__main__':
    main()
