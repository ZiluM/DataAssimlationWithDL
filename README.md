# Code for "Reconstructing the Tropical Pacific Upper Ocean using Online Data Assimilation with a Deep Learning model"

## Authors

Zilu Meng (1),  Gregory J. Hakim (1)

1. Department of Atmospheric Sciences, University of Washington, Seattle, WA, USA

## Abstract

A deep learning (DL) model, based on a transformer architecture, is  trained on a climate-model dataset and compared with a standard linear inverse model (LIM) in the tropical Pacific. We show that the DL model produces more accurate forecasts compared to the LIM when tested on a reanalysis dataset. We then assess the ability of an ensemble Kalman filter to reconstruct the monthly-averaged upper ocean from a noisy set of 24  sea-surface temperature observations designed to mimic existing coral proxy measurements, and compare results for the DL model and LIM. Due to signal damping in the DL model, we implement a novel inflation technique by adding noise from hindcast experiments. Results show that assimilating observations with the DL model yields better reconstructions than the LIM for observation averaging times ranging from one month to one year. The improved reconstruction is due to the enhanced predictive capabilities of the DL model, which map the memory of past observations to future assimilation times.

## Plain Language Summary

We use a deep learning (DL) model to better predict climate patterns in the tropical Pacific upper ocean, and to reconstruct past conditions from a sparse network of noisy observations. The DL model forecasts are more accurate than a reference Linear Inverse Model (LIM), which has approximately comparable computational demand. After we adjust DL model forecasts to better approximate errors, we show that this model can more accurately reconstruct climate fields than the LIM. This success highlights the significant potential of deep learning to improve our understanding and prediction of climate change through reconstructing climate variables from sparse information such as from coral proxies. 

## Code description

da.py: Data assimilation code

myconfig1.py: Configuration file for DL model

utils.py: Utility functions

da.sh: Shell script to run the code

config.yml : config file for DA

./Code/ : Directory containing the code for DL model; from Zhou&Zhang 2023 SciAdv (DOI:10.5281/zenodo.7445610)

./model: model weights



