### Standard

| Max epochs | LR       | Stopped Early | Best R@5 Val | Best R@5 Test | Link                            |
| ---------- | -------- | ------------- | ------------ | ------------- | ------------------------------- |
| 10         | 0.000001 | N (10)        | 79.6         | 77.9          | [example](Runs/std_10_0.000001) |
| 10         | 0.00001  | Y (8)         | **81.0**     | **80.6**      | [example](Runs/std_10_0.00001)  |
| 10         | 0.0001   | Y (6)         | 79.6         | 78.3          | [example](Runs/std_10_0.0001)   |
| 10         | 0.001    | Y (8)         | 76.9         | 77.8          | [example](Runs/std_10_0.001)    |
| 10         | 0.01     | Y (7)         | 71.6         | 72.3          | [example](Runs/std_10_0.01)     |
| 10         | 0.1      | Y (10)        | 71.7         | 70.2          | [example](Runs/std_10_0.1)      |

### NetVLAD

| Max epochs | Num Clusters | LR       | Stopped Early | Optimizer  | Best R@5 Val | Best R@5 Test | Link                                      |
| ---------- | ------------ | -------- | ------------- | ---------- | ------------ | ------------- | ----------------------------------------- |
| 10         | 64           | 0.000001 | N (10)        | Adam       | **78.8**     | 77.5          | [example](Runs\netvlad_10_0.000001_64)    |
| 10         | 64           | 0.00001  | Y (8)         | Adam       | 78.1         | **77.6**      | [example](Runs\netvlad_10_0.00001_64)     |
| 10         | 64           | 0.0001   | N (10)        | Adam       | 76.2         | 75.2          | [example](Runs\netvlad_10_0.0001_64)      |
| 10         | 64           | 0.00001  | Y (4)         | SGD m=0.9  | 38.4         | 46.4          | [example](Runs\netvlad_sgd_m_0.9_epoc_10) |
| 10         | 64           | 0.00001  | Y (4)         | SGD m=0.99 | 43.7         | 52.8          | [example](Runs\nevlad_sgd_m_0.99_epoc_10) |
| 10         | 64           | 0.00001  | N (10)        | Adagrad    | 60.7         | 67.0          | [example](Runs\netvlad_adagrad_std_10)    |

### GeM

| Max epochs | LR       | Stopped Early | Optimizer | Best R@5 Val | Best R@5 Test | Link                                            |
| ---------- | -------- | ------------- | --------- | ------------ | ------------- | ----------------------------------------------- |
| 10         | 0.000001 | N (10)        | Adam      | 79.6         | 77.9          | [example](Runs\GeM_p_3_lr_10e-6_10)             |
| 10         | 0.00001  | Y (8)         | Adam      | **81.0**     | **80.6**      | [example](Runs\GeM_p_3_lr_10e-5_10)             |
| 10         | 0.00001  | N (10)        | SGD m=0.9 | 43.0         | 52.3          | [example](Runs\GeM_p_3_sgd_m_0.9_lr_10e-5_10)   |
| 10         | 0.00001  | N (10)        | Adagrad   | 61.3         | 67.3          | [example](Runs\GeM_p_3_adagrad_std_lr_10e-5_10) |
