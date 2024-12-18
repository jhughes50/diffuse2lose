## Occupancy Grid Completion with Diffusion and CNNs

 In this project, we aim to use deep learning methods to complete 2D occupancy grids. We first tried using Stable Diffusion for zero-shot completion of the unknown parts of occupancy grids. We then fine-tuned Stable Diffusion 2 on our own 2D occupancy grid dataset, and showed that fine-tuning improved color accuracy but did not improve the accuracy of the occupancy completion. Finally, we trained two models of our own. The first model was latent space diffusion model, we first pretrained a variational autoencoder, and then trained a 2D UNet model for the latent space diffusion. We then trained a seven layer UNet convolutional neural network (CNN) to do binary classification on the unknown parts of the 2D occupancy grid. This model achieved the best results with a mean-IoU score of 0.51.

## Repo Structure
 - `eval` evaluation scripts for generating images with our custom diffusion `unet` and our CNN model `onet`.
 - `loaders` scripts for generating the dataset. Does the masking and trasnaltion from semantic images to occupancy grid.
 - `logs` where to save loss logs, also contains plotting scripts.
 - `models` contains the pytorch code for each of our proposed models.
 - `sim` contains `c#` code for when we were trying to use a simulator instead of a dataset.
 - `train` scripts to train the models and fine-tune stable diffusion 2.
 - `utils` contains utility functions. 

## Docker 
There is a docker environment that you can build with `./build.bash` then run with `./run.bash`. In `run.bash` change `./data:/home/whoami/data` to the path to where the data is on your machine. If you want to run with more than one GPU make `--gpus all`. 


