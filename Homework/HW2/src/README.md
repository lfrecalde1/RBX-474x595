# RBX-474x595

## Diffusion Model Weights (Linear Schedule)

Download the weights for the diffusion model with a linear schedule from:
https://wpi0-my.sharepoint.com/:u:/r/personal/lfrecalde_wpi_edu/Documents/PHD/Lectures/Learning_perception/linear/ckpt_149_.pt?csf=1&web=1&e=6fBWm8

Place the file in:
src/CheckpointsWoAugmentation/linear

## Phase 1

To run the first phase of the project:

python3 MainCondition.py

Set the parameter:
"schedule": "linear"

## Phase 2

To run Phase 2:

python3 MainPhase2.py

This step may take a while because many images are generated.

## Phase 3 (Sine Schedule)

Run the same script:

python3 MainPhase2.py

Set the parameter:
"schedule": "sine"

Download the sine-schedule weights from:
https://wpi0-my.sharepoint.com/:u:/r/personal/lfrecalde_wpi_edu/Documents/PHD/Lectures/Learning_perception/sine/ckpt_149_.pt?cs_

