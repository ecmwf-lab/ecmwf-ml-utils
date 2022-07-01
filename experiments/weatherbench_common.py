import climetlab as cml
import numpy as np
import pandas as pd

# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras.layers import *
# import tensorflow.keras.backend as K
import torch

FREQ = "30d"
# FREQ = "1d"

class MyDataset:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.kwargs["time"] = "0000"

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

    def sel(self, time=None, **kwargs):
        new_kwargs = {k: v for k, v in self.kwargs.items()}

        if time is not None:
            if not isinstance(time, slice):
                time = slice(time, time)
            start = str(int(time.start)) + "-01-01"
            end = str(int(time.stop)) + "-12-31"
            dates = list(
                pd.date_range(start=start, end=end, freq=FREQ).strftime("%Y%m%d")
            )
            new_kwargs["date"] = dates

        new_kwargs.update(kwargs)
        return MyDataset(*self.args, **new_kwargs)

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

    def to_pytorch(self, offset):
        return WrapperWeatherBenchDataset(self, offset)

class WrapperWeatherBenchDataset(torch.utils.data.Dataset):
    def __init__(self, ds, offset) -> None:
        super().__init__()

        self.ds = ds.source

        self.stats = self.ds.statistics()
        self.offset = offset

    def __len__(self):
        """Returns the length of the dataset. This is important! Pytorch must know this."""
        return self.stats["count"] - self.offset

    def __getitem__(self, i):  # -> Tuple[np.ndarray, ...]:
        """Returns the i-th sample (x, y). Pytorch will take care of the shuffling after each epoch."""
        return self.ds[i].to_numpy(), self.ds[i + self.offset].to_numpy()