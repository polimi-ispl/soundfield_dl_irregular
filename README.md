# Deep Learning-based Soundfield Synthesis using Irregular Loudspeaker arrays
├─ data_lib                               # Data generation module<br />
|  ├── soundfield_generation_wideband.py  # Contains class for data generation<br />
├─ dataset                                # Dataset<br />
|  ├── masks                              # Contains masks used to create irregular array<br />
├─ train_lib                              # Module for network training<br />
|  ├── network_utils.py                   # Network architecture<br />
|  ├── params_wideband.py                 # Simulation parameters<br />
|  ├── train_utils.py                     # Utilities for training/tensorboard<br />
├── generate_data_numpy.py                # Generates data in numpy arrays<br />
├── generate_data_tfrecords.py            # Converts numpy arrays to TFRecords<br />
├── test_pm.py                            # Computes SSIM and NMSE using pressure matching<br />
├──test_pwdr_pwdr_holes_pwdr_cnn.py       # Computes SSIM and NMSE using PWDR, PWDR_holes and PWDR-CNN<br />
├── train.py                              # Trains the network<br />
└── README.md<br />

