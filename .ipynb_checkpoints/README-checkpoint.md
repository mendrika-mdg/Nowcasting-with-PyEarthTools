# 3D U-Net Nowcasting of Cloud Optical Depth and Irradiance over Darwin

This repository performs satellite-based nowcasting of **Cloud Optical Depth (COD)** and **Surface Global Irradiance (SGI)** over Darwin using **Himawari** data, **PyEarthTools**, **PyTorch**, and **PyTorch Lightning**.

## Data Pipeline
- Loads COD and SGI from the Himawari archive using PyEarthTools.
- Builds a **temporal window** of:
  - 7 past frames: t−60, t−50, t−40, t−30, t−20, t−10, t₀  
  - 3 future targets: t+60, t+120, t+180 minutes
- Applies:
  - dataset sorting and coordinate alignment  
  - cropping to the Darwin region  
  - per-channel scaling (SGI ÷ 1200, COD ÷ 60)  
  - conversion to NumPy  
  - reshaping to `(2, 7, H, W)` for inputs and `(2, 3, H, W)` for targets  
- Uses an **IterableDataset** that streams samples on the fly, handles missing data, cleans NaNs, and automatically shards work across multiple GPUs and workers.

## Model
- A custom **3D U-Net** for spatio-temporal prediction.
- Preserves the time dimension (no temporal pooling).
- Downsamples only the spatial dimensions using 3D max pooling.
- Uses skip connections, 3D transposed convolutions, and a temporal head to produce the 3 future frames.
- Processes inputs at 256×256 internally and outputs predictions at 276×276.

## Loss and Training
- Uses a **pooled MSE loss** computed over 9×9 windows for structural consistency.
- Trains with **PyTorch Lightning** using:
  - 4× A100 GPUs (DDP)
  - mixed precision (`16-mixed`)
  - batch size 8
  - 2 workers per rank
  - early stopping and model checkpointing
  - Weights & Biases logging
- Training data: **2019–2021**
- Validation: **January 2022**
- Test: **January–July 2024**
- Trainer limits: 200 training batches and 20 validation batches per epoch for up to 30 epochs.

## Summary
In short, the repository provides a fully automated on-the-fly data pipeline and a 3D U-Net model to generate 1 h, 2 h, and 3 h nowcasts of COD and SGI using Himawari satellite imagery.
