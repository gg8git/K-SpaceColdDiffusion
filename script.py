import numpy as np
import matplotlib.pyplot as plt
import pathlib

from torch.utils.data import DataLoader

from utils.mri_data import SliceDataset
from utils.data_transform import DataTransform_Diffusion
from utils.sample_mask import RandomMaskGaussianDiffusion, RandomMaskDiffusion, RandomMaskDiffusion2D
from utils.misc import *
from help_func import print_var_detail

from diffusion.kspace_diffusion import KspaceDiffusion
from utils.diffusion_train import Trainer
from net.u_net_diffusion import Unet

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# ****** TRAINING SETTINGS ******
# dataset settings
acc = 8  # acceleration factor
frac_c = 0.04  # center fraction
path_dir_train = './data/singlecoil_train/'
path_dir_test = './data/singlecoil_test/'
img_mode = 'fastmri'  # 'fastmri' or 'B1000'
bhsz = 6
img_size = 320

# ====== Construct dataset ======
# initialize mask
mask_func = RandomMaskDiffusion(
    acceleration=acc,
    center_fraction=frac_c,
    size=(1, img_size, img_size),
)

# initialize dataset
data_transform = DataTransform_Diffusion(
    mask_func,
    img_size=img_size,
    combine_coil=True,
    flag_singlecoil=True,
)

# need to change this for multi-slice
# training set
dataset_train = SliceDataset(
    root=pathlib.Path(path_dir_train),
    transform=data_transform,
    challenge='singlecoil',
    num_skip_slice=5,
)

# need to change this for multi-slice
# test set
dataset_test = SliceDataset(
    root=pathlib.Path(path_dir_test),
    transform=data_transform,
    challenge='singlecoil',
    num_skip_slice=5,
)

dataloader_train = DataLoader(dataset_train, batch_size=bhsz, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=bhsz, shuffle=True)
print('len dataloader train:', len(dataloader_train))
print('len dataloader test:', len(dataloader_test))

# model settings
CH_MID = 64
# training settings
NUM_EPOCH = 50
learning_rate = 2e-5
time_steps = 1000
train_steps = NUM_EPOCH * len(dataloader_train) # can be customized to a fixed number, however, it should reflect the dataset size.
train_steps = max(train_steps, 700000)
print('train_steps:',train_steps)
# save settings
PATH_MODEL = './saved_models/fastmri_knee/diffusion_'+str(img_mode)+'_'+str(acc)+'x_T'+str(time_steps)+'_S'+str(train_steps)+'/'
create_path(PATH_MODEL)

# construct diffusion model
save_folder=PATH_MODEL
load_path=None
blur_routine='Constant'
train_routine='Final'
sampling_routine='x0_step_down'
discrete=False

# need to change this for multi-slice
model = Unet(
    dim=CH_MID,
    dim_mults=(1, 2, 4, 8),
    channels=2,
).cuda()
print('model size: %.3f MB' % (calc_model_size(model)))

diffusion = KspaceDiffusion(
    model,
    image_size=img_size,
    device_of_kernel='cuda',
    channels=2,
    timesteps=time_steps,  # number of steps
    loss_type='l1',  # L1 or L2
    blur_routine=blur_routine,
    train_routine=train_routine,
    sampling_routine=sampling_routine,
    discrete=discrete,
).cuda()

# construct trainer and train

trainer = Trainer(
    diffusion,
    image_size=img_size,
    train_batch_size=bhsz,
    train_lr=learning_rate,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    fp16=False,  # turn on mixed precision training with apex
    save_and_sample_every=50000,
    results_folder=save_folder,
    load_path=load_path,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
)
trainer.train()

# do something with the model