# TAMLP Bearing Fault Diagnosis Code Description
Paper model: CNNTransformer and AlphaEvolution algorithm synergistically optimized MLP fault diagnosis model

## Hardware & Software Environment
Hardware: Intel i714700 CPU, 64 GB RAM, NVIDIA RTX 3090 / 4090 / 5090 GPU with 24 GB video memory
OS: Ubuntu 22.04 / Windows 10 / Windows 11
Python: 3.9
CUDA: 11.7

## Install Dependencies
pip install -r requirements.txt

## Important Notes
ycl.py: Preprocessing script for converting raw vibration signals into images.
Note: Preprocessing takes a very long runtime when processing the fullsize dataset.

## Running Order
First run sy.py, then run ycl.py to generate all images, and finally run TAMLP.py.
Note: Working Condition 2 only performs forward inference. No training or parameter tuning is performed.

## Dataset Availability
The Southeast University bearing dataset is a public dataset.
Owing to the large volume of the laboratory dataset, the raw dataset is not uploaded alongside the code. Readers may contact the authors through the corresponding laboratory data application channel to obtain it.