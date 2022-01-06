# Deep Learning-based Soundfield Synthesis using Irregular Loudspeaker arrays
|--data_lib                               # Data generation module
|  ├── soundfield_generation_wideband.py  # Contains class for data generation
├─ dataset                                # Dataset
|  ├── masks                              # Contains masks used to create irregular array
├─ train_lib                              # Module for network training
|  ├── network_utils.py                   # Network architecture
|  ├── params_wideband.py                 # Simulation parameters
|  ├── train_utils.py                     # Utilities for training/tensorboard
├── generate_data_numpy.py                # Generates data in numpy arrays
├── generate_data_tfrecords.py            # Converts numpy arrays to TFRecords
├── test_pm.py                            # Computes SSIM and NMSE using pressure matching
├──test_pwdr_pwdr_holes_pwdr_cnn.py       # Computes SSIM and NMSE using PWDR, PWDR_holes and PWDR-CNN
├── train.py                              # Trains the network
└── README.md

