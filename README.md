# PU-Distillation-Noisy-Label

This repository is the official implementation of Training Classifiers that are Universally Robust to All Label Noise Levels.

To train classifiers that are universally robust to all noise levels, and that are not sensitive to any variation in the noise model, we propose a distillation-based framework that incorporates a new subcategory of Positive-Unlabeled learning. In particular, we shall assume that a small subset of any given noisy dataset is known to have correct labels, which we treat as “positive”, while the remaining noisy subset is treated as “unlabeled”. Our framework consists of the following 3 steps: (1) We shall generate, via iterative updates, an augmented clean subset with additional reliable “positive” samples filtered from “unlabeled” samples; (2) We shall train a teacher model on this larger augmented clean set; (3) With the guidance of the teacher model, we then train a student model on the whole dataset.

![](flow.png)

## Requirements

- Python 3.6
- Pytorch 1.4.0

## Training

#### Hyper-parameters:

- ##### CIFAR-10

  + **N_bagging** (number of binary classifiers for each class): 20
  + **K_iteration** (number of iterations to augment clean set): 10
  + **threshold** (decision threshold of binary classifiers): 0.9
  + **add_criterion** (criterion of moving an unsure sample to clean set): 19

  For symmetric noise:

| Parameter\Noise Level | 30%  | 40%  | 50%  | 60%  | 70%  | 80%  | 90%  |
| --------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| **student_lambda**    |      |      |      |      |      |      |      |
| **eta**               |      |      |      |      |      |      |      |

​		For asymmetric noise:

| Parameter\Noise Level | 30%  | 40%  | 50%  | 60%  | 70%  | 80%  | 90%  |
| --------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| **student_lambda**    |      |      |      |      |      |      |      |
| **eta**               |      |      |      |      |      |      |      |



- ##### Clothing1M

  + **N_bagging**: 10
  + **K_iteration**: 6
  + **threshold**: 0.95
  + **add_criterion**: 10
  + **student_lambda**: 0.5
  + **eta**: 0.8



#### Examples:

- ##### CIFAR-10 (given 10% clean set, symmetric noise level=70%):

```
python generate_clean_set_cifar.py --clean_data_ratio 0.1 --threshold 0.9 --add_criterion 19 --N_bagging 20 --K_iteration 10
python teacher_cifar.py --n 5 --mixup --entropy_reg
python student_cifar.py  --noise_type syn --noise_level 0.7 --label_type soft_bootstrap --student_lambda 0.8
```

- ##### Clothing1M:

```
python generate_clean_set_clothing.py --threshold 0.95 --add_criterion 9 --N_bagging 10 --K_iteration 5
python teacher_clothing.py --n 5 --mixup
python student_clothing.py --label_type soft_bootstrap --student_lambda 0.8
```



## Results

#### Accuracies on CIFAR10 with symmetric synthetic noise (average of 5 trials)

| Algorithm\Noise Level | 30%            | 40%            | 50%            | 60%            | 70%            | 80%            | 90%             |
| --------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | --------------- |
| SOTA                  | 95.95$\pm$0.05 | 94.66$\pm$0.08 | 94.85$\pm$0.28 | 94.89$\pm$0.10 | 94.14$\pm$0.14 | 93.21$\pm$0.14 | 61.62$\pm$10.29 |
| Our Method            |                |                |                |                |                |                |                 |



#### Accuracies on CIFAR10 with asymmetric synthetic noise (average of 5 trials)

| Algorithm\Noise Level | 30%            | 40%            | 50%            | 60%            | 70%            | 80%            | 90%            |
| --------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| SOTA                  | 93.95$\pm$0.06 | 89.56$\pm$0.78 | 84.56$\pm$0.94 | 78.21$\pm$0.32 | 76.70$\pm$0.11 | 76.44$\pm$0.21 | 76.00$\pm$0.14 |
| Our Method            |                |                |                |                |                |                |                |



#### Accuracies on Clothing1M

| Algorithm  | Accuracy |
| ---------- | -------- |
| SOTA       | 74.76    |
| Our Method | 77.70    |

