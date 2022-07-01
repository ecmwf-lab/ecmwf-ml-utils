# pip install climetlab-weatherbench --quiet
import pickle

# from weatherbench_score import *
from collections import OrderedDict

import climetlab as cml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import xarray as xr
from tensorflow.keras.layers import *
from tqdm import tqdm


from weatherbench_common import MyDataset


# ds = cml.load_dataset( 'weatherbench', parameter = "geopotential_500", year = list(range(2015,2019)),).to_xarray()
ds = MyDataset(
    "local",
    "/perm/mafp/weather-bench-links/data-from-mat-chantry-symlinks-to-files-2015-2016-2017-2018/grib",
    param="z",
    level="500",
)
# ds = xr.open_mfdataset('/perm/mafp/weather-bench-links/data-from-mihai-alexe/netcdf/pl_*.nc')
ds_train = ds.sel(time=slice("2015", "2016"))
ds_valid = ds.sel(time=slice("2017", "2017"))
# For speed of testing just look at the first few months of 2018
ds_test = ds.sel(time=slice("2018", "2018"))  # .isel(time=range(0,2000))

# What size of batch do we want to use?
batch_size = 32
# How many hours do we want to predict forward (multiple of 6)
lead_time = 6
assert lead_time % 6 == 0, "Lead time must be a multiple of 6"


def fixit(x):
    x = x.to_tfdataset(offset=3)
    x = x.shuffle(25)
    x = x.batch(32)
    return x


dg_train = fixit(ds_train)
dg_test = fixit(ds_test)
dg_valid = fixit(ds_valid)


class PeriodicPadding2D(tf.keras.layers.Layer):
    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def call(self, inputs, **kwargs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = tf.concat(
            [
                inputs[:, :, -self.pad_width :, :],
                inputs,
                inputs[:, :, : self.pad_width, :],
            ],
            axis=2,
        )
        # Zero padding in the lat direction
        inputs_padded = tf.pad(
            inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0]]
        )
        return inputs_padded

    def get_config(self):
        config = super().get_config()
        config.update({"pad_width": self.pad_width})
        return config


# Once we have created a periodic padding layer, the rest is easy
# A field comes in, we pad it, then pass to a normal 2D convolutional layer.
class PeriodicConv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        conv_kwargs={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        if type(kernel_size) is not int:
            assert (
                kernel_size[0] == kernel_size[1]
            ), "PeriodicConv2D only works for square kernels"
            kernel_size = kernel_size[0]
        pad_width = (kernel_size - 1) // 2
        self.padding = PeriodicPadding2D(pad_width)
        self.conv = Conv2D(filters, kernel_size, padding="valid", **conv_kwargs)

    def call(self, inputs):
        return self.conv(self.padding(inputs))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "conv_kwargs": self.conv_kwargs,
            }
        )
        return config


# With the layer defined, our model looks really small!
# we use the functional API (see Wednesday's NN notebook)
def build_cnn(filters, kernels, input_shape, dr=0):
    """
    Fully convolutional network
    Filters & kernels, lists of same length (this length is the model depth)
    input_shape, the shape of our input tensor
    dr the dropout rate
    """
    # Create an input layer of the appropriate shape
    x = input = Input(shape=input_shape)
    # Loop over the depth
    for f, k in zip(filters[:-1], kernels[:-1]):
        # First a Periodic Conv2D
        x = PeriodicConv2D(f, k)(x)
        # Now a nonlinearity
        x = LeakyReLU()(x)
        # If we are worried about overfitting we can use Dropout
        # during training
        if dr > 0:
            x = Dropout(dr)(x)
    # One final linear layer to get the right number of outputs
    output = PeriodicConv2D(filters[-1], kernels[-1])(x)
    # Construct the model from the graph
    return keras.models.Model(input, output)


# ### Build a model
#
# We'll use 4 hidden layers with 32 filters in each.

cnn = build_cnn(
    filters=[32, 1],
    # filters = [32, 32, 32, 32, 1],  # orig
    kernels=[3, 3],
    # kernels = [5, 5, 5, 5, 5], # orig
    input_shape=(181, 360, 1)
    # input_shape = (32, 64, 1) # orig
)

# Use a very standard loss & optimiser
cnn.compile(keras.optimizers.Adam(1e-4), "mse")
cnn.summary()

# Since we didn't load the full data this is only for demonstration.
# without a GPU training is slow. You could try this on Colab or similar
cnn.fit(
    dg_train,
    epochs=3,
    validation_data=dg_valid,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=2, verbose=1, mode="auto"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.25,
            patience=3,
        ),
    ],
)


print("Training is finished.")
exit()

## 
## # We can save the weights to load later
## # cnn.save_weights('test.h5')
## 
## # Here's one I trained earlier
## # cnn_long_training.load_weights('cnn_6h.h5')
## 
## # ## Create predictions
## #
## # Now that we have our model we need to create a prediction xarray object. This function does this.
## #
## # Depending on the model we trained we might have a model that directly predicts 3/5 days, or one that only predicts in 6hr steps.
## # With a direct approach, we would use the method below to construct our prediction.
## #
## # With the iterative approach we would use `create_iterative_predictions` (see futher below) to iterate foreward to our desired lead times.
## 
## 
## def create_predictions(model, dg):
##     """Create predictions for non-iterative model"""
##     preds = model.predict_generator(dg)
##     # Unnormalize
##     preds = preds * dg.std.values + dg.mean.values
##     fcs = []
##     lev_idx = 0
##     for var, levels in dg.var_dict.items():
##         if levels is None:
##             fcs.append(
##                 xr.DataArray(
##                     preds[:, :, :, lev_idx],
##                     dims=["time", "lat", "lon"],
##                     coords={"time": dg.valid_time, "lat": dg.ds.lat, "lon": dg.ds.lon},
##                     name=var,
##                 )
##             )
##             lev_idx += 1
##         else:
##             nlevs = len(levels)
##             fcs.append(
##                 xr.DataArray(
##                     preds[:, :, :, lev_idx : lev_idx + nlevs],
##                     dims=["time", "lat", "lon", "level"],
##                     coords={
##                         "time": dg.valid_time,
##                         "lat": dg.ds.lat,
##                         "lon": dg.ds.lon,
##                         "level": levels,
##                     },
##                     name=var,
##                 )
##             )
##             lev_idx += nlevs
##     return xr.merge(fcs)
## 
## 
## # We won't use this here as we've made a model designed to predict only 6hr forward, so we will need to chain predictions together to reach 3 & 5 days.
## 
## # fc = create_predictions(cnn, dg_test)
## # compute_weighted_rmse(fc, valid).compute()
## 
## # Here create a function for creating predictions for longer lead times by chaining increments together.
## def create_iterative_predictions(model, dg, max_lead_time=5 * 24):
##     state = dg.data[: dg.n_samples]
##     preds = []
##     # Do the prediction
##     for _ in tqdm(range(max_lead_time // dg.lead_time)):
##         state = model.predict(state)
##         p = state * dg.std.values + dg.mean.values
##         preds.append(p)
##     preds = np.array(preds)
## 
##     # Create the xarray object
##     lead_time = np.arange(dg.lead_time, max_lead_time + dg.lead_time, dg.lead_time)
##     das = []
##     lev_idx = 0
##     for var, levels in dg.var_dict.items():
##         if levels is None:
##             das.append(
##                 xr.DataArray(
##                     preds[:, :, :, :, lev_idx],
##                     dims=["lead_time", "time", "lat", "lon"],
##                     coords={
##                         "lead_time": lead_time,
##                         "time": dg.init_time,
##                         "lat": dg.ds.lat,
##                         "lon": dg.ds.lon,
##                     },
##                     name=var,
##                 )
##             )
##             lev_idx += 1
##         else:
##             nlevs = len(levels)
##             das.append(
##                 xr.DataArray(
##                     preds[:, :, :, :, lev_idx : lev_idx + nlevs],
##                     dims=["lead_time", "time", "lat", "lon", "level"],
##                     coords={
##                         "lead_time": lead_time,
##                         "time": dg.init_time,
##                         "lat": dg.ds.lat,
##                         "lon": dg.ds.lon,
##                         "level": levels,
##                     },
##                     name=var,
##                 )
##             )
##             lev_idx += nlevs
##     return xr.merge(das)
## 
## 
## # %% [markdown]
## # ### Let's evalute our model on the test set.
## # We'll roll the model out for 5 days and evaluate it.
## 
## # %%
## fc_iter = create_iterative_predictions(cnn, dg_test)
## 
## # %%
## rmse = evaluate_iterative_forecast(fc_iter, ds_test, func=compute_weighted_rmse)
## rmse.load()
## 
## # %% [markdown]
## # ### Let's look at the error at 3 days
## 
## # %%
## display(rmse.sel(lead_time=3 * 24))
## 
## # %% [markdown]
## # ### Plot the headline scores against a few benchmarks
## 
## # %%
## rmse.z.plot(label="CNN")
## plt.plot([3 * 24, 5 * 24], [154, 334], "x", label="Operational IFS", markersize=8)
## plt.plot([3 * 24, 5 * 24], [489, 743], ".", label="T42 IFS", markersize=8)
## plt.plot([3 * 24, 5 * 24], [175, 350], "o", label="Keisler GNN (2022)", markersize=8)
## plt.legend()
## 
## # %% [markdown]
## # ### Our model lags a long way behind the Operational IFS.
## #
## # However as you can see, [Keisler's GraphNN](https://arxiv.org/pdf/2202.07575) produced very comparable values. However there are some caveats with that work, as you integrate forward in time the images become blurred, scoring very well on RMSE but not capturing extreme values. More work will be required to establish if ML can produce sharp yet well scoring predictions.
## #
## # Note that the test sets are not identical between our experiments and the others (due to time constraints), but we would expect to see only small differences with a different test set.
## #
## # ### Let's visualise our prediction
## #
## # This plotting is taken from [here](https://github.com/pangeo-data/WeatherBench/blob/master/notebooks/4-evaluation.ipynb), where Rasp et al. show how to do a complete evaluation of a model.
## 
## import warnings
## 
## # %%
## import cartopy.crs as ccrs
## import matplotlib.cbook
## 
## warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
## 
## cmap_z = "cividis"
## cmap_t = "RdYlBu_r"
## cmap_diff = "bwr"
## cmap_error = "BrBG"
## 
## 
## def imcol(ax, data, title="", **kwargs):
##     if not "vmin" in kwargs.keys():
##         mx = np.abs(data.max().values)
##         kwargs["vmin"] = -mx
##         kwargs["vmax"] = mx
##     #     I = ax.imshow(data, origin='lower',  **kwargs)
##     I = data.plot(
##         ax=ax,
##         transform=ccrs.PlateCarree(),
##         add_colorbar=False,
##         add_labels=False,
##         rasterized=True,
##         **kwargs,
##     )
##     cb = fig.colorbar(I, ax=ax, orientation="horizontal", pad=0.01, shrink=0.90)
##     ax.set_title(title)
##     ax.coastlines(alpha=0.5)
##     return
## 
## 
## fig, axs = plt.subplots(
##     2, 5, figsize=(18, 8), subplot_kw={"projection": ccrs.PlateCarree()}
## )
## # True
## for iax, var, cmap, r, t in zip(
##     [0], ["z"], [cmap_z], [[47000, 58000]], [r"Z500 [m$^2$ s$^{-2}$]"]
## ):
##     imcol(
##         axs[iax, 0],
##         ds_test[var].isel(time=0),
##         cmap=cmap,
##         vmin=r[0],
##         vmax=r[1],
##         title=f"ERA5 {t} t=0h",
##     )
##     imcol(
##         axs[iax, 1],
##         ds_test[var].isel(time=6),
##         cmap=cmap,
##         vmin=r[0],
##         vmax=r[1],
##         title=f"ERA5 {t} t=6h",
##     )
##     imcol(
##         axs[iax, 2],
##         ds_test[var].isel(time=6) - ds_test[var].isel(time=0),
##         cmap=cmap_diff,
##         title=f"ERA5 {t} diff (6h-0h)",
##     )
##     imcol(
##         axs[iax, 3],
##         ds_test[var].isel(time=5 * 24),
##         cmap=cmap,
##         vmin=r[0],
##         vmax=r[1],
##         title=f"ERA5 {t} t=5d",
##     )
##     imcol(
##         axs[iax, 4],
##         ds_test[var].isel(time=5 * 24) - ds_test[var].isel(time=0),
##         cmap=cmap_diff,
##         title=f"ERA5 {t} diff (5d-0h)",
##     )
## 
## # CNN
## for iax, var, cmap, r, t in zip(
##     [1], ["z"], [cmap_z], [[47000, 58000]], [r"Z500 [m$^2$ s$^{-2}$]"]
## ):
##     imcol(
##         axs[iax, 0],
##         ds_test[var].isel(time=0),
##         cmap=cmap,
##         vmin=r[0],
##         vmax=r[1],
##         title=f"ERA5 {t} t=0h",
##     )
##     imcol(
##         axs[iax, 1],
##         fc_iter[var].isel(time=0).sel(lead_time=6),
##         cmap=cmap,
##         vmin=r[0],
##         vmax=r[1],
##         title=f"CNNi {t} t=6h",
##     )
##     imcol(
##         axs[iax, 2],
##         fc_iter[var].isel(time=0).sel(lead_time=6) - ds_test[var].isel(time=6),
##         cmap=cmap_error,
##         title=f"Error CNNi - ERA5 {t} t=6h",
##     )
##     imcol(
##         axs[iax, 3],
##         fc_iter[var].isel(time=0).sel(lead_time=5 * 24),
##         cmap=cmap,
##         vmin=r[0],
##         vmax=r[1],
##         title=f"CNNd {t} t=5d",
##     )
##     imcol(
##         axs[iax, 4],
##         fc_iter[var].isel(time=0).sel(lead_time=5 * 24)
##         - ds_test[var].isel(time=5 * 24),
##         cmap=cmap_error,
##         title=f"Error CNNd - ERA5 {t} t=5d",
##     )
## 
## for ax in axs.flat:
##     ax.set_xticks([])
##     ax.set_yticks([])
## plt.tight_layout(pad=0)
## # plt.savefig('../figures/examples.pdf', bbox_inches='tight')
## # plt.savefig('../figures/examples.jpeg', bbox_inches='tight', dpi=300)
## 
## # %% [markdown]
## # # What happens when we iterate for very long times?
## #
## # Is the model stable, do the results look sensible?
## #
## # PS Don't do this for the whole dataset as it will take a very long time.
## #
## # # What could we do to improve this?
## #
## # 1. Rewrite the model with residual style blocks, i.e. seek to learn the increment to the current state.
## #
## # 2. Use a different type of network?
## #
## # 3. Change the hyper-parameters.
## #
## # 4. Use a local normalisation (i.e. the local mean & std).
## #
## # 5. Use more data?
## #
## # 6. Dilated convolutions (need to match with periodic padding)
## #
## # 7. Training the model with rollout
## # [See figure 2](https://arxiv.org/pdf/2202.11214.pdf)
## #
## # 8. Train a probabilistic model?
## #
## # # Is WeatherBench a useful benchmark problem?
## #
## # Can you think of applications for a fairly accurate model, perhaps one with slightly larger error than the IFS.
## 
## # One possible configuration of a residual network
## def build_residual_cnn(filters, kernels, input_shape, dr=0):
##     """Fully convolutional network"""
##     x = input = Input(shape=input_shape)
##     for i, (f, k) in enumerate(zip(filters[:-1], kernels[:-1])):
##         x_in = x
##         x = PeriodicConv2D(f, k)(x)
##         x = LeakyReLU()(x)
##         if dr > 0:
##             x = Dropout(dr)(x)
##         if (i > 0) and (k == kernels[i - 1]):
##             x = Add()([x_in, x])
##     output = PeriodicConv2D(filters[-1], kernels[-1])(x)
##     return keras.models.Model(input, output)
## 