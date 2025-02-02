# Training

### Standard

| Max epochs | LR       | Stopped Early | Best R@5 Val | Best R@5 Test | Link                          |
| ---------- | -------- | ------------- | ------------ | ------------- | ----------------------------- |
| 15         | 0.000001 | N (15)        | **81.2**     | **79.9**      | [example](/Runs/std_10_10e-6) |
| 10         | 0.00001  | Y (8)         | 81.0         | 80.6          | [example](/Runs/std_10_10e-5) |
| 10         | 0.0001   | Y (6)         | 79.6         | 78.3          | [example](/Runs/std_10_10e-4) |
| 10         | 0.001    | Y (8)         | 76.9         | 77.8          | [example](/Runs/std_10_10e-3) |
| 10         | 0.01     | Y (7)         | 71.6         | 72.3          | [example](/Runs/std_10_10e-2) |
| 10         | 0.1      | Y (10)        | 71.7         | 70.2          | [example](/Runs/std_10_10e-1) |

### NetVLAD

| Max epochs | Num Clusters | LR       | Stopped Early | Optimizer  | Best R@5 Val | Best R@5 Test | Link                                          |
| ---------- | ------------ | -------- | ------------- | ---------- | ------------ | ------------- | --------------------------------------------- |
| 20         | 64           | 0.000001 | N (20)        | Adam       | 95.8         | 92.8          | [example](/Runs/netvlad_20_64_10e-6_adam)     |
| 20         | 64           | 0.00001  | Y (13)        | Adam       | **96.3**     | **93.2**      | [example](/Runs/netvlad_20_64_10e-5_adam)     |
| 20         | 64           | 0.0001   | Y (7)         | Adam       | 95.9         | 91.9          | [example](/Runs/netvlad_20_64_10e-4_adam)     |
| 20         | 64           | 0.001    | Y (6)         | Adam       | 92.9         | 90.3          | [example](/Runs/netvlad_20_64_10e-3_adam)     |
| 20         | 64           | 0.00001  | N (20)        | SGD m=0.9  | 90.6         | 90.1          | [example](/Runs/netvlad_20_64_10e-5_sgd_0.9)  |
| 20         | 64           | 0.00001  | N (20)        | SGD m=0.99 | 95.6         | 92.8          | [example](/Runs/netvlad_20_64_10e-5_sgd_0.99) |
| 20         | 64           | 0.00001  | N (20)        | Adagrad    | 92.3         | 90.7          | [example](/Runs/netvlad_20_64_10e-5_adagrad)  |

### GeM

| Max epochs | LR       | Stopped Early | Optimizer  | Best R@5 Val | Best R@5 Test | Link                                           |
| ---------- | -------- | ------------- | ---------- | ------------ | ------------- | ---------------------------------------------- |
| 20         | 0.000001 | N (20)        | Adam       | **90.1**     | **89.0**      | [example](/Runs/gem_20_3_10e-6_10e-6_adam)     |
| 20         | 0.00001  | Y (8)         | Adam       | 89.9         | 88.6          | [example](/Runs/gem_20_3_10e-6_10e-5_adam)     |
| 20         | 0.0001   | Y (8)         | Adam       | 89.6         | 87.9          | [example](/Runs/gem_20_3_10e-6_10e-4_adam)     |
| 20         | 0.001    | Y (11)        | Adam       | 86.9         | 86.0          | [example](/Runs/gem_20_3_10e-6_10e-3_adam)     |
| 20         | 0.000001 | N (20)        | SGD m=0.9  | 48.9         | 57.4          | [example](/Runs/gem_20_3_10e-6_10e-6_sgd_0.9)  |
| 20         | 0.000001 | N (20)        | SGD m=0.99 | 59.9         | 67.3          | [example](/Runs/gem_20_3_10e-6_10e-6_sgd_0.99) |
| 20         | 0.000001 | N (20)        | Adagrad    | 52.0         | 60.1          | [example](/Runs/gem_20_3_10e-6_10e-6_adagrad)  |

# Ablation

## StLucia test

### NetVLAD and GeM Ablation Test on StLucia

| Aggr.   | Max epochs | Params              | LR       | Optimizer  | Best R@5 Test | Link                                                        |
| ------- | ---------- | ------------------- | -------- | ---------- | ------------- | ----------------------------------------------------------- |
| NetVLAD | 20         | netvlad_clusters=64 | 0.00001  | Adam       | 72.4          | [example](/Runs/netvlad_20_64_10e-5_adam_test_st_lucia)     |
| NetVLAD | 20         | netvlad_clusters=64 | 0.00001  | Adagrad    | 70.6          | [example](/Runs/netvlad_20_64_10e-5_adagrad_test_st_lucia)  |
| NetVLAD | 20         | netvlad_clusters=64 | 0.00001  | SGD m=0.99 | 72.3          | [example](/Runs/netvlad_20_64_10e-5_sgd_0.99_test_st_lucia) |
| GeM     | 20         | p=3, eps=10^-6      | 0.000001 | Adam       | 64.5          | [example](/Runs/gem_20_3_10e-6_10e-6_adam_test_st_lucia)    |
| GeM     | 20         | p=3, eps=10^-6      | 0.000001 | Adagrad    | 53.2          |                                                             |
| GeM     | 20         | p=3, eps=10^-6      | 0.000001 | SGD m=0.99 | 58.4          |                                                             |

## train_positives_dist_threshold and val_positive_dist_threshold tests

### NetVLAD train_positives_dist_threshold on Pitts30k

| Max epochs | Num Clusters | train_positives_threshold_val | LR      | Stopped Early | Optimizer | Best R@5 Val | Best R@5 Test | Link                                                                        |
| ---------- | ------------ | ----------------------------- | ------- | ------------- | --------- | ------------ | ------------- | --------------------------------------------------------------------------- |
| 20         | 64           | 5                             | 0.00001 | Y (6)         | Adam      | 96.4         | 93.2          | [example](/Runs/netvlad_20_64_10e-5_adam_train_positives_dist_threshold_5)  |
| 20         | 64           | 15                            | 0.00001 | Y (9)         | Adam      | 96.0         | 93.1          | [example](/Runs/netvlad_20_64_10e-5_adam_train_positives_dist_threshold_15) |

### NetVLAD train_positives_dist_threshold on StLucia

| Max epochs | Num Clusters | train_positives_threshold_val | LR      | Optimizer | Best R@5 Test | Link                                                                                      |
| ---------- | ------------ | ----------------------------- | ------- | --------- | ------------- | ----------------------------------------------------------------------------------------- |
| 20         | 64           | 5                             | 0.00001 | Adam      | 70.9          | [example](/Runs/netvlad_20_64_10e-5_adam_train_positives_dist_threshold_5_test_st_lucia)  |
| 20         | 64           | 15                            | 0.00001 | Adam      | 70.6          | [example](/Runs/netvlad_20_64_10e-5_adam_train_positives_dist_threshold_15_test_st_lucia) |

### NetVLAD val_positive_dist_threshold

| Max epochs | Num Clusters | val_positive_threshold_val | LR      | Optimizer | Best R@5 Test | Link                                                                                   |
| ---------- | ------------ | -------------------------- | ------- | --------- | ------------- | -------------------------------------------------------------------------------------- |
| 20         | 64           | 20                         | 0.00001 | Adam      | 91.9          | [example](/Runs/netvlad_20_64_10e-5_adam_val_positive_dist_threshold_20_test_pitts30k) |
| 20         | 64           | 30                         | 0.00001 | Adam      | 93.8          | [example](/Runs/netvlad_20_64_10e-5_adam_val_positive_dist_threshold_30_test_pitts30k) |

### NetVLAD val_positive_dist_threshold on StLucia

| Max epochs | Num Clusters | val_positive_threshold_val | LR      | Optimizer | Best R@5 Test | Link                                                                                   |
| ---------- | ------------ | -------------------------- | ------- | --------- | ------------- | -------------------------------------------------------------------------------------- |
| 20         | 64           | 20                         | 0.00001 | Adam      | 71.9          | [example](/Runs/netvlad_20_64_10e-5_adam_val_positive_dist_threshold_20_test_st_lucia) |
| 20         | 64           | 30                         | 0.00001 | Adam      | 73.0          | [example](/Runs/netvlad_20_64_10e-5_adam_val_positive_dist_threshold_30_test_st_lucia) |

## Data augmentation

### NetVLAD data augmentation on Pitts30k

| Max epochs | Num Clusters | Augmentation     | LR      | Stopped Early | Optimizer | Best R@5 Val | Best R@5 Test | Link                                                       |
| ---------- | ------------ | ---------------- | ------- | ------------- | --------- | ------------ | ------------- | ---------------------------------------------------------- |
| 20         | 64           | sharpness_adjust | 0.00001 | Y (8)         | Adam      | 95.9         | 93.1          | [example](/Runs/netvlad_20_64_10e-5_adam_sharpness_adjust) |
| 20         | 64           | greyscale        | 0.00001 | Y (11)        | Adam      | 96.0         | 92.5          | [example](/Runs/netvlad_20_64_10e-5_adam_grayscale)        |
| 20         | 64           | color_jitter     | 0.00001 | Y (11)        | Adam      | 96.2         | 92.9          | [example](/Runs/netvlad_20_64_10e-5_adam_color_jitter)     |
| 20         | 64           | downscale        | 0.00001 | Y (10)        | Adam      | 96.4         | 93.0          | [example](/Runs/netvlad_20_64_10e-5_adam_downscale)        |
| 20         | 64           | upscale          | 0.00001 | Y (12)        | Adam      | 96.4         | 93.3          | [example](/Runs/netvlad_20_64_10e-5_adam_upscale)          |

### NetVLAD data augmentation on StLucia

| Max epochs | Num Clusters | Augmentation     | LR      | Optimizer | Best R@5 Test | Link                                                                     |
| ---------- | ------------ | ---------------- | ------- | --------- | ------------- | ------------------------------------------------------------------------ |
| 20         | 64           | sharpness_adjust | 0.00001 | Adam      | 72.3          | [example](/Runs/netvlad_20_64_10e-5_adam_sharpness_adjust_test_st_lucia) |
| 20         | 64           | greyscale        | 0.00001 | Adam      | 71.6          | [example](/Runs/netvlad_20_64_10e-5_adam_grayscale_test_st_lucia)        |
| 20         | 64           | color_jitter     | 0.00001 | Adam      | 70.2          | [example](/Runs/netvlad_20_64_10e-5_adam_color_jitter_test_st_lucia)     |
| 20         | 64           | downscale        | 0.00001 | Adam      | **77.5**      | [example](/Runs/netvlad_20_64_10e-5_adam_downscale_test_st_lucia)        |
| 20         | 64           | upscale          | 0.00001 | Adam      | 68.6          | [example](/Runs/netvlad_20_64_10e-5_adam_upscale_test_st_lucia)          |

# Personal Contribution

## Backbone

### NetVLAD PC AlexNet train

| Train from layer | Max epochs | LR      | Stopped Early | Optimizer | Best R@5 Val | Best R@5 Test | Link                                              |
| ---------------- | ---------- | ------- | ------------- | --------- | ------------ | ------------- | ------------------------------------------------- |
| 5                | 20         | 0.00001 | Y (20)        | Adam      | 93.5         | 90.0          | [example](/Runs/netvlad_20_64_10e-5_adam_alexnet) |

### NetVLAD PC VGG16 train

| Train from layer | Max epochs | LR      | Stopped Early | Optimizer | Best R@5 Val | Best R@5 Test | Link                                            |
| ---------------- | ---------- | ------- | ------------- | --------- | ------------ | ------------- | ----------------------------------------------- |
| 5                | 20         | 0.00001 | Y (9)         | Adam      | 96.2         | 92.4          | [example](/Runs/netvlad_20_64_10e-5_adam_vgg16) |

### NetVLAD PC ResNet50

| Train from layer | Max epochs | LR      | Stopped Early | Optimizer | Best R@5 Val | Best R@5 Test | Link                                                     |
| ---------------- | ---------- | ------- | ------------- | --------- | ------------ | ------------- | -------------------------------------------------------- |
| 4                | 20         | 0.00001 | Y (5)         | Adam      | 96.8         | 92.9          | [example](/Runs/netvlad_20_64_10e-5_adam_resnet50-conv4) |

### NetVLAD PC ResNet50MoCo

| Train from layer | Max epochs | LR      | Stopped Early | Optimizer | Best R@5 Val | Best R@5 Test | Link                                                         |
| ---------------- | ---------- | ------- | ------------- | --------- | ------------ | ------------- | ------------------------------------------------------------ |
| 4                | 20         | 0.00001 | Y (5)         | Adam      | 95.8         | 92.2          | [example](/Runs/netvlad_20_64_10e-5_adam_resnet50moco-conv4) |

## Attention modules

### NetVLAD CRN

| Max epochs | LR      | Stopped Early | Augmentation | Optimizer | CRN LR                      | Best R@5 Val | Best R@5 Test | Link                                                           |
| ---------- | ------- | ------------- | ------------ | --------- | --------------------------- | ------------ | ------------- | -------------------------------------------------------------- |
| 20         | 0.00001 | Y (9)         | x            | Adam      | 0.001                       | 96.4         | 93.6          | [example](/Runs/netvlad_20_64_10e-5_adam_crn_0.001)            |
| 20         | 0.00001 | Y (9)         | downscale    | Adam      | 0.0001 (failure with 0.001) | 96.7         | 93.6          | [example](/Runs/netvlad_20_64_10e-5_adam_downscale_crn_0.0001) |

### NetVLAD CBAM

| Max epochs | LR      | Stopped Early | Modified | Optimizer | CBAM LR | Best R@5 Val | Best R@5 Test | Link                                                       |
| ---------- | ------- | ------------- | -------- | --------- | ------- | ------------ | ------------- | ---------------------------------------------------------- |
| 20         | 0.00001 | Y (12)        | n        | Adam      | 0.01    | 96.3         | 93.1          | [example](/Runs/netvlad_20_64_10e-5_adam_cbam_0.01)        |
| 20         | 0.00001 | Y (13)        | y        | Adam      | 0.01    | 96.2         | 92.8          | [example](/Runs/netvlad_20_64_10e-5_adam_cbam_0.01_modded) |

### NetVLAD CRN on StLucia

| Max epochs | LR      | Augmentation train | Augmentation test | Optimizer | CRN LR | Best R@5 Test | Link                                                                                        |
| ---------- | ------- | ------------------ | ----------------- | --------- | ------ | ------------- | ------------------------------------------------------------------------------------------- |
| 20         | 0.00001 | x                  | x                 | Adam      | 0.001  | 77.0          | [example](/Runs/netvlad_20_64_10e-5_adam_crn_0.001_test_st_lucia)                           |
| 20         | 0.00001 | x                  | downscale         | Adam      | 0.001  | 78.8          | [example](/Runs/netvlad_20_64_10e-5_adam_crn_0.001_train_default_test_downscale_st_lucia)   |
| 20         | 0.00001 | downscale          | downscale         | Adam      | 0.001  | 81.1          | [example](/Runs/netvlad_20_64_10e-5_adam_crn_0.001_train_downscale_test_downscale_st_lucia) |
