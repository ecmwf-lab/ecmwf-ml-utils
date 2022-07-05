# pip install climetlab-weatherbench --quiet
import pickle

# from weatherbench_score import *
from collections import OrderedDict

import climetlab as cml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras.layers import *
# import tensorflow.keras.backend as K
import seaborn as sns
import torch
import xarray as xr
from climetlab.profiling import print_counters
from torch import nn
from tqdm import tqdm
from weatherbench_common import MyDataset

# import climetlab.debug

# ds = cml.load_dataset( 'weatherbench', parameter = "geopotential_500", year = list(range(2015,2019)),).to_xarray()
ds = MyDataset(
    "directory",
    "/perm/mafp/weather-bench-links/data-from-mat-chantry-symlinks-to-files",
    # "/perm/mafp/weather-bench-links/data-from-mat-chantry-symlinks-to-files-2015-2016-2017-2018/grib",
    variable="z",
    levelist="500",
)
# ds = xr.open_mfdataset('/perm/mafp/weather-bench-links/data-from-mihai-alexe/netcdf/pl_*.nc')
ds_train = ds.sel(time=slice("2015", "2016"))
ds_valid = ds.sel(time=slice("2017", "2017"))
# For speed of testing just look at the first few months of 2018
# ds_test = ds.sel(time=slice("2018", "2018"))  # .isel(time=range(0,2000))

# What size of batch do we want to use?
batch_size = 32
# How many hours do we want to predict forward (multiple of 6)
# lead_time = 6
# assert lead_time % 6 == 0, "Lead time must be a multiple of 6"


def stats():
    # print(stats)
    ds_train.source.statistics()
    ds_train.mean = ds_train.source.statistics()["average"]
    ds_train.std = ds_train.source.statistics()["stdev"]
    ds_train.count = ds_train.source.statistics()["count"]

    # dg_train = ds_train.to_tfdataset(normalise=lambda x: (x - ds_train.mean) / ds_train.std)
    # dg_test = ds_test.to_tfdataset(normalise=lambda x:(x-ds_train.mean)/ds_train.std)
    # dg_valid = ds_valid.to_tfdataset(normalise=lambda x:(x-ds_train.mean)/ds_train.std)


dl_train = ds_train.to_pytorch(offset=3)
# dl_test = ds_test.to_pytorch(offset=3)
dl_valid = ds_valid.to_pytorch(offset=3)

# train and validation dataloaders
# dl_train = torch.utils.data.DataLoader(
#    ds_train,
#    batch_size=128,
#    # multi-process data loading
#    # use as many workers as you have cores on your machine
#    num_workers=8,
#    # default: no shuffle, so need to explicitly set it here
#    shuffle=True,
#    # uses pinned memory to speed up CPU-to-GPU data transfers
#    # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
#    pin_memory=True,
#    # function used to collate samples into batches
#    # if None then Pytorch uses the default collate_fn (see below)
#    collate_fn=None,
# )
#
# dl_valid = torch.utils.data.DataLoader(
#    ds_valid,
#    batch_size=128,
#    num_workers=8,
#    shuffle=False,
#    pin_memory=True,
#    collate_fn=None,
# )


class ResidualBlock(nn.Module):
    """Residual conv-block. See https://arxiv.org/pdf/2203.12297.pdf"""

    _KERNEL = 5
    _PADDING = _KERNEL // 2
    _LRU_ALPHA = 0.2

    def __init__(
        self, nf_in: int, nf_out: int, stride: int = 1, batch_norm: bool = False
    ) -> None:
        """Initializes the residual block.
        Args:
            nf_in: number of input channels
            nf_out: number of output channels
            stride: stride of 2D convolution
            batch_norm: use batch normalization [T/F]
        """
        super().__init__()
        self.batch_norm = batch_norm

        self.activation1 = nn.LeakyReLU(self._LRU_ALPHA)
        self.conv1 = nn.Conv2d(
            nf_in,
            nf_out,
            kernel_size=self._KERNEL,
            padding=self._PADDING,
            stride=stride,
        )
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(nf_out)

        self.activation2 = nn.LeakyReLU(self._LRU_ALPHA)
        self.conv2 = nn.Conv2d(
            nf_out, nf_out, kernel_size=self._KERNEL, padding=self._PADDING
        )
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(nf_out)

        # define the skip connection
        if nf_in != nf_out:
            self.skip_path = nn.Conv2d(nf_in, nf_out, kernel_size=1)
            if stride > 1:
                self.skip_path = nn.Sequential(nn.AvgPool2d(stride), self.skip_path)
        elif stride > 1:
            self.skip_path = nn.AvgPool2d(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given tensor input x, returns ResidualBlock(x)."""
        out = self.activation1(x)
        out = self.conv1(out)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        if hasattr(self, "skip_path"):
            x = self.skip_path(x)
        return out + x


res_block = ResidualBlock(32, 32)

# for p in res_block.parameters():
#    print(p.shape, p.requires_grad)


class WBModel(nn.Module):
    def __init__(self):
        super().__init__()  # don't forget this

        self._resnet = nn.Sequential(
            *[
                ResidualBlock(
                    1, 2
                ),  # input tensor: [batch_size, 1, H, W]  -> [batch_size, 32, H, W]
                # ResidualBlock(32, 32),
                # ResidualBlock(32, 32),
                ResidualBlock(2, 1),
            ]
        )

    # special method that defines the forward action of the neural network on the input tensor(s)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._resnet(x)


import gc
import os
import sys

import psutil

# def memReport():
#    for obj in gc.get_objects():
#        if torch.is_tensor(obj):
#            print(type(obj), obj.size())


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0**30  # memory use in GB...I think
    print("memory GB:", memoryUse)


import datetime


def training():
    print("training")
    test_model = WBModel()
    for X, y in dl_train:
        print(datetime.datetime.now())

        #        cpuStats()
        ##        memReport()
        y_pred = test_model(X)  # calls forward()
        print(datetime.datetime.now())
        print("--")


training()
training()
training()
print_counters()
