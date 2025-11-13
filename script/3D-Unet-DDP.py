# Suppress all warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Import datetime utilities
from datetime import timedelta

# Core libraries
import numpy as np
import xarray as xr

# PyEarthTools modules for data access and pipelines
import pyearthtools.data as petdata
from pyearthtools.data.time import TimeResolution

import pyearthtools.pipeline as petpipe
from pyearthtools.pipeline.operation import Operation

# Internal index handling and warning control
import pyearthtools.data.indexes._indexes as idx
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=idx.IndexWarning, message="Data requested at a higher resolution")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="NetCDF: HDF error")

# PyTorch and Lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from functools import lru_cache

# Set manual seed and precision
torch.manual_seed(42)                                      
torch.set_float32_matmul_precision('medium')             

# Select device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Local import for JASMIN path configuration
import site_archive_jasmin as saj          #type: ignore         
print(saj.ROOT_DIRECTORIES)

# Select variables to load from Himawari archive
vars_to_load = ["surface_global_irradiance", "cloud_optical_depth"]
himawari = petdata.archive.Himawari(vars_to_load)

# Merge function to align multiple timesteps in a single sample
def proper_temporal_merge(samples):
    valid = []
    for i, s in enumerate(samples):
        if s is not None:
            s = s.copy(deep=True)
            s = s.assign_coords(time=[s.time.values[0] + np.timedelta64(i*10, "m")])
            valid.append(s)
    return xr.concat(valid, dim="time", combine_attrs="override")

# Scaling operation applied per variable
class PerChannelScale(Operation):
    def __init__(self, scale_dict):
        # scale_dict: variable name → normalisation factor
        super().__init__(operation="apply")
        self.scale_dict = scale_dict

    def apply_func(self, sample):
        # If TemporalRetrieval returns a tuple, handle recursively
        if isinstance(sample, tuple):
            return tuple(self.apply_func(s) for s in sample)
        # Apply scaling to each variable in dataset
        ds = sample.copy()
        for var, factor in self.scale_dict.items():
            if var in ds:
                ds[var] = ds[var] / float(factor)
        return ds

# Scaling constants for Himawari variables
scales = {
    "surface_global_irradiance": 1200,   # W/m²
    "cloud_optical_depth": 60            # unitless
}

# Function to build pipeline for a specific date range

@lru_cache(maxsize=32)
def make_pipeline(start_time, end_time, region=(-15.5, -8, 130, 135.5)):

    # Set temporal resolution to 10 minutes
    himawari.data_resolution = TimeResolution("minute")
    himawari.data_resolution.value = 10

    return petpipe.Pipeline(
        himawari,
        petpipe.operations.xarray.Sort(order=["time", "latitude", "longitude"]),
        petpipe.operations.xarray.AlignDataVariableDimensionsToDatasetCoords(),

        # Temporal window: 7 past frames (t−6 to t₀, every 10 min) and 3 future frames (+1h, +2h, +3h)
        petpipe.modifications.TemporalWindow(
            prior_indexes=[-6, -5, -4, -3, -2, -1, 0],
            posterior_indexes=[6, 12, 18],
            timedelta=timedelta(minutes=10),
            merge_method=proper_temporal_merge
        ),

        # Spatial domain
        petdata.transform.region.Bounding(*region),

        # Per-channel scaling
        PerChannelScale(scales),

        # Convert to NumPy and rearrange
        petpipe.operations.xarray.conversion.ToNumpy(),
        petpipe.operations.numpy.reshape.Rearrange("t c h w -> c t h w"),

        # Define iteration range
        iterator=petpipe.iterators.DateRange(
            start_time,
            end_time,
            interval="10 minutes"
        ),

        # Ignore missing files or invalid datasets
        exceptions_to_ignore=(petdata.exceptions.DataNotFoundError, ValueError),
    )

# 3D U-Net model definition for spatio-temporal nowcasting
class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_filters=32, use_lag=False):
        super().__init__()
        self.use_lag = use_lag

        # Encoder
        self.enc1 = self._block(in_channels + (1 if use_lag else 0), base_filters)
        self.enc2 = self._block(base_filters, base_filters * 2)
        self.enc3 = self._block(base_filters * 2, base_filters * 4)
        self.pool = nn.MaxPool3d((1, 2, 2))  # halve spatial dimensions only

        # Bottleneck
        self.bottleneck = self._block(base_filters * 4, base_filters * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self._block(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self._block(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self._block(base_filters * 2, base_filters)

        # Temporal output head
        self.temporal_head = nn.Sequential(
            nn.Conv3d(base_filters, base_filters, (3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_filters, out_channels, (3, 1, 1), padding=(1, 0, 0))
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, lag=None):
        # Input: (B, 2, T, 276, 276)
        if self.use_lag and lag is not None:
            x = torch.cat([x, lag], dim=1)

        # Downscale to 256×256 for better performance
        x256 = F.interpolate(x, size=(x.shape[2], 256, 256), mode="trilinear", align_corners=False)

        # Encoder path
        e1 = self.enc1(x256)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        # Decoder path
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Temporal projection and resize to 276×276
        out256 = self.temporal_head(d1)
        out256 = out256[:, :, -3:, :, :]
        out = F.interpolate(out256, size=(out256.shape[2], 276, 276), mode="trilinear", align_corners=False)
        return out

# Pooled mean squared error loss for spatial structure similarity
class PooledMSELoss(nn.Module):
    def __init__(self, pool=9):
        super().__init__()
        self.pool = pool

    def forward(self, pred, target):
        B, C, T, H, W = pred.shape
        kernel = torch.ones((C, 1, 1, self.pool, self.pool), device=pred.device) / (self.pool ** 2)
        pred_smooth = F.conv3d(pred, kernel, padding=(0, self.pool // 2, self.pool // 2), groups=C)
        target_smooth = F.conv3d(target, kernel, padding=(0, self.pool // 2, self.pool // 2), groups=C)
        return F.mse_loss(pred_smooth, target_smooth)

# Lightning wrapper for training and validation
class NowcastLitModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = UNet3D()
        self.lr = lr
        self.criterion = PooledMSELoss(pool=9)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        x, y_true = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y_true)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        x, y_true = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Iterable dataset directly reading from PET pipeline
import os
from torch.utils.data import get_worker_info

class PetNowcastIterableDataset(IterableDataset):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def _get_worker_and_rank(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers
        return global_worker_id, global_num_workers

    def __iter__(self):
        worker_id, n_workers_global = self._get_worker_and_rank()

        it = iter(self.pipeline.iterator)
        for idx, t in enumerate(it):
            if idx % n_workers_global != worker_id:
                continue

            try:
                inputs, outputs = self.pipeline[t]
            except Exception:
                continue

            x = torch.from_numpy(inputs).permute(1, 0, 2, 3).float()
            y = torch.from_numpy(outputs).permute(1, 0, 2, 3).float()

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            yield x, y



# Training pipeline (1 year), Validation pipeline (1 week)
trainpipe = make_pipeline("20190101T0000", "20220101T0000")
valpipe   = make_pipeline("20220101T0000", "20220201T0000")

# Prepare datasets and loaders
train_ds = PetNowcastIterableDataset(trainpipe)
val_ds   = PetNowcastIterableDataset(valpipe)

def worker_init_fn(worker_id):
    import random
    import numpy as np
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

train_loader = DataLoader(
    train_ds,
    batch_size=8,               # start small, you can increase later
    num_workers=2,              # per rank
    pin_memory=True,
    persistent_workers=False,   # important for stability
    prefetch_factor=2,
    worker_init_fn=worker_init_fn
)

val_loader = DataLoader(
    val_ds,
    batch_size=8,
    num_workers=1,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2,
    worker_init_fn=worker_init_fn
)


# Initialise Lightning model
lit_model = NowcastLitModule(lr=2e-4)

wandb_logger = WandbLogger(
    project="pyearthtools-nowcasting",
    name="unet3d_himawari_ddp",
    save_dir="/gws/ssde/j25a/mmh_storage/train110/wandb_logs"
)

early_stop_cb = EarlyStopping(
    monitor="val_loss",     # same metric as checkpoint
    patience=5,             # stop after 3 epochs with no improvement
    mode="min",
    verbose=True
)

# Save checkpoints after each epoch
checkpoint_cb = ModelCheckpoint(
    dirpath="/gws/ssde/j25a/mmh_storage/train110/chk",
    filename="unet3d-ddp-nowcast-final",
    monitor="val_loss",
    mode="min",
    save_top_k=1
)
# Lightning trainer configuration
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    precision="16-mixed",
    sync_batchnorm=True,
    log_every_n_steps=10,
    num_sanity_val_steps=0,
    logger=wandb_logger,
    callbacks=[checkpoint_cb, early_stop_cb],
    limit_train_batches=200,
    limit_val_batches=20,
)

# Run training
trainer.fit(lit_model, train_loader, val_loader)
