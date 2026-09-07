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
preprocess.py: Preprocessing script for converting raw vibration signals into images.
Note: Preprocessing takes a very long runtime when processing the fullsize dataset.

## Running Order
First run sy.py, then run ycl.py to generate all images, and finally run TAMLP.py.

## Output Description
weights: Stores the model weight files obtained from each fold of five‑fold cross‑validation.
Two figures are output:
lab_cond2_raw_confusion.png (raw confusion matrix with sample counts)
lab_cond2_row_norm_confusion.png (rownormalized confusion matrix; diagonal elements represent perclass recall)
3. Metrics including fivefold accuracy and WorkingCondition2 test results are printed on the console.

Note: Working Condition 2 only performs forward inference. No training or parameter tuning is performed.

## Dataset Availability
The Southeast University bearing dataset is a public dataset.
The laboratory selfbuilt dataset has a large file size. Please contact the corresponding author via email to request the fullversion data.
