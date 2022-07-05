from tkinter import W

import climetlab as cml
import numpy as np
import pandas as pd

# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras.layers import *
# import tensorflow.keras.backend as K
import torch
from climetlab.profiling import call_counter

FREQ = "30d"
# FREQ = "1d"


class MyDataset:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    #   self.kwargs["time"] = "0000"

    @property
    def source(self):
        show = {}
        for k, v in self.kwargs.items():
            if k == "date":
                v = str(v)[:10] + " " + str(len(v)) + " dates"
            show[k] = v
        print(f"Getting selection: {self.args}, {show}")
        return cml.load_source(*self.args, **self.kwargs)

    def to_xarray(self, *args, **kwargs):
        return self.source.to_xarray(*args, **kwargs)

    def to_numpy(self, *args, **kwargs):
        return self.source.to_numpy(*args, **kwargs)

        # def sel(self, *args, **kwargs):
        return self.source.sel(*args, **kwargs)

    def sel(self, time=None, **kwargs):
        if time is not None:
            if not isinstance(time, slice):
                time = slice(time, time)
            start = str(int(time.start)) + "-01-01"
            end = str(int(time.stop)) + "-12-31"
            dates = list(
                pd.date_range(start=start, end=end, freq=FREQ).strftime("%Y%m%d")
            )
            kwargs["date"] = dates
        return self.source.sel(**kwargs)

    def __getitem__(self, param):
        return self.sel(param=param)

    def to_tfdataset(self, *args, offset, **kwargs):
        source = self.source
        # σ = self.statistics()["stdev"]

        def normalise(a):
            return a
            # return (a - μ) / σ

        def generate():
            fields = []
            for s in source:
                fields.append(normalise(s.to_numpy()))
                if len(fields) >= offset:
                    yield fields[0], fields[-1]
                    fields.pop(0)

        import tensorflow as tf

        shape = source.first.shape

        dtype = kwargs.get("dtype", tf.float32)
        return tf.data.Dataset.from_generator(
            generate,
            output_signature=(
                tf.TensorSpec(shape, dtype=dtype, name="input"),
                tf.TensorSpec(shape, dtype=dtype, name="output"),
            ),
        )

    def to_pytorch(self, *args, **kwargs):
        return self.source.to_pytorch(*args, **kwargs)
