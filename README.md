# CIFAR-10 Image Classification: Custom CNNs and Transfer Learning

This repository contains a comprehensive deep learning project focused on image classification using the **CIFAR-10 dataset**. The project systematically explores the design space of Convolutional Neural Networks (CNNs), evaluating feature preprocessing techniques, architectural scaling (width and depth), regularization strategies, optimization algorithms, and transfer learning with a pre-trained **VGG16** backbone.

## 👥 Team Members
| Name | GitHub |
| :--- | :--- |
| Youssef Elhelw | [@Youssef-Elhelw](https://github.com/Youssef-Elhelw) |
| Matthew Ashraf | [@Matthew-Ashraf](https://github.com/MatthewAshraf) |
| Areig Ismail | [@Areig-Ismail](https://github.com/Arj5056) |

---

## 📊 Dataset Overview
The **CIFAR-10 dataset** consists of 60,000 $32 \times 32$ color images distributed equally across 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

* **Data Allocation:**
  * **Training Set:** 40,000 images
  * **Validation Set:** 10,000 images
  * **Test Set:** 10,000 images
* **Reproducibility:** Random seeds for both `numpy` and `tensorflow` are strictly pinned to `42` to guarantee deterministic weight initializations and training runs.

---

## ⚙️ Experimental Framework & Task Analysis

### 🟢 Task 1: Data Preprocessing Experiments
This section isolates the impact of alternative input feature scaling methods using a fixed standard baseline network (`BaselineCNN`):
`Conv2D(32, 3x3)` ➡️ `MaxPooling2D(2, 2)` ➡️ `Conv2D(64, 3x3)` ➡️ `MaxPooling2D(2, 2)` ➡️ `Flatten` ➡️ `Dense(128)` ➡️ `Dense(10)`.

Models were trained for 20 epochs with a batch size of 128 using the Adam optimizer ($lr=0.001$). 

#### ⚠️ Post-Training Evaluation Correction
An inspection of the cell execution history reveals a code evaluation bug where the generic `model` variable was left pointing to the final trained model configuration. This caused the summary table code to re-evaluate the same model across different tensors incorrectly. The verified actual metrics extracted directly from the sequential stdout training logs are documented below:

| Preprocessing Method | Final Train Accuracy | Final Validation Accuracy | Log-Verified Test Accuracy | Initial Training Loss (Epoch 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Data** | 88.50% | 54.25% | **54.90%** | 3.7490 |
| **Min-Max Scaling** ($[0, 1]$) | 91.44% | 66.38% | **66.69%** | 1.5625 |
| **Standardization** ($\mu=0, \sigma=1$) | 96.34% | 66.56% | **66.65%** | 1.3883 |

* **Key Insight:** Feature scaling dramatically stabilizes the early loss landscape (halving the initial epoch loss) and prevents exploding gradient variance, establishing a critical performance foundation.

---

### 🟢 Task 2A: Network Width (Filter Count) Comparison
Using a standard 4-layer block layout, this task scales the network's capacity horizontally by varying the number of hidden channels:
`Conv2D(f1)` ➡️ `Conv2D(f2)` ➡️ `MaxPooling` ➡️ `Conv2D(f3)` ➡️ `Conv2D(f4)` ➡️ `MaxPooling` ➡️ `Flatten` ➡️ `Dense(256)` ➡️ `Dense(10)`.

| Model Width | Filter Dimensions $(f1, f2, f3, f4)$ | Total Parameters | Train Accuracy | Validation Accuracy | Test Accuracy | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Small** | `(8, 8, 16, 16)` | 269,266 | 92.38% | 61.44% | **60.96%** | ~46.6s |
| **Medium** | `(32, 32, 64, 64)` | 1,116,970 | 98.58% | 72.07% | **71.13%** | ~78.1s |
| **Large** | `(64, 64, 128, 128)` | 2,360,138 | 98.68% | 74.02% | **73.67%** | ~145.2s |

* **Key Insight:** Expanding structural width provides an absolute classification gain of **+12.71%** from Small to Large configurations, allowing the network to capture complex, multi-scale localized abstractions at the expense of computational latency.

---

### 🟢 Task 2B: Network Depth Comparison
This experiment tests vertical layer scaling while fixing the baseline filter budget to 32 channels per convolutional layer.

* **Shallow (4 Layers + Flatten):** 555,754 parameters. **Test Accuracy = 68.20%**
* **Medium Depth (6 Layers + GAP):** 58,154 parameters. **Test Accuracy = 74.51%**
* **Deep (8 Layers + GAP):** 76,650 parameters. **Test Accuracy = 71.33%**

#### 🔍 Critical Question Analysis
* **Does the deepest model perform best?** No, the **Medium Depth** architecture achieved the optimal configuration. The deepest architecture suffered a performance decline due to compounding optimization difficulties and diminishing spatial dimensions before feature extraction. Notably, replacing dense flattening operations with **Global Average Pooling (GAP)** acted as an effective structural regularizer, keeping the parameter count low while preserving representation accuracy.

---

### 🟢 Task 3 & 4: Regularization, Optimizers, and Learning Rates

#### Regularization Optimization (Task 3B Summary)
* **D0 (No Regularization Baseline):** Train: 98.87%, Test: 71.22% ➡️ **Overfitting Gap: 27.65%**
* **D1 (Moderate Dropout):** Train: 93.54%, Test: 77.70% ➡️ **Overfitting Gap: 15.84%**
* **D2 (Heavy Dropout + Weight Decay):** Train: 79.66%, Test: 78.74% ➡️ **Overfitting Gap: 0.92%**
* *Early Stopping Evaluation:* Utilizing validation loss tracking successfully intercepted optimization paths right as training performance diverged from validation accuracy, saving optimal weights before overfitting occurred.

#### Optimizer & Learning Rate Sensitivity Analysis (Task 4)
Evaluating the Medium network configurations against alternative Adam learning rates highlighted severe convergence boundaries:

| Learning Rate | Train Accuracy | Validation Accuracy | Test Accuracy | Observed Optimization Convergence Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **$lr = 0.0001$** | 92.81% | 65.74% | **65.81%** | Extremely slow, steady, and sub-optimal convergence. |
| **$lr = 0.0010$** | 99.14% | 72.82% | **72.18%** | **Optimal boundary.** Fast and balanced gradient descent trajectory. |
| **$lr = 0.0100$** | 74.39% | 47.99% | **48.65%** | **Catastrophic Divergence.** Massive early loss oscillations; updates overshot local minima, destroying structural optimization. |

---

### 🟢 Task 5: Transfer Learning & Fine-Tuning via VGG16
This task leverages a pre-trained **VGG16** backbone initialized with ImageNet weights, upscaling the input space to $48 \times 48$ pixels to accommodate the architecture's minimum spatial dimensions.

* **Model 1 (From-Scratch Control CNN):** Baseline built specifically for the $48 \times 48$ tensor layout.
  * *Results:* Test Accuracy = **67.32%** | Training Time = 155.47s
* **Model 2 (VGG16 Feature Extraction):** Fully freezes the VGG16 base layers, updating only the newly attached dense classification top head.
  * *Results:* Standard frozen representations act as highly robust generic feature extractors.
* **Model 3 (VGG16 Fine-Tuning with $lr = 10^{-5}$):** Unfreezes the final 4 layers of the VGG16 base while preserving early weights, utilizing a conservative learning rate.
  * *Results:* Test Accuracy = **82.45%** | Loss = 0.5599 | Training Time = 465.77s
* **Model 4 (VGG16 Fine-Tuning with $lr = 0.001$):** Unfreezes the final 4 base layers but applies a standard, aggressive learning rate.
  * *Results:* Test Accuracy collapsed entirely to **10.00%** (Random Guessing). The high learning rate over-wrote the pre-trained features, corrupting the model's representations.

#### 🔍 Critical Question Analysis
* **Does transfer learning outperform training from scratch on CIFAR-10?** While the fine-tuned VGG16 outperformed basic custom pipelines, our specialized championship model trained entirely from scratch achieved the absolute highest accuracy (**86.74%**). This occurs because the large VGG16 architecture was designed for massive image scales, causing upscaled $32 \times 32$ images to lose fine-grained details through deep pooling steps. A dedicated, well-regularized network built from scratch can adapt its filters specifically to low-resolution structures more efficiently.

---

## 🏆 The Ultimate Championship Model

By integrating our experimental findings, we designed an optimized, end-to-end production model:

### ⚙️ Pipeline Architecture
1. **Input Normalization:** Channel-wise Z-Score Standardization.
2. **Real-Time Data Augmentation:** `ImageDataGenerator` configured with a $15^\circ$ rotation limit, $10\%$ spatial horizontal/vertical shifting, $10\%$ random zoom scales, and horizontal flips.
3. **Optimized Backbone:** High-capacity deep convolutional layers utilizing balanced padding schemes.
4. **Regularization Layer:** `GlobalAveragePooling2D` spatial reduction linked to a dense `Dropout(0.5)` bottleneck layer to eliminate remaining over-parameterization.
5. **Convergence Guard:** Early stopping callback with validation loss tracking to automatically restore the best model checkpoints.

### 🎯 Empirical Performance Verification
* **Final Evaluation Loss:** **0.4192**
* **Final Production Test Accuracy:** 🎉 **86.74%**

---

## 📉 Error Analysis & Confusion Matrix Breakdown

A Seaborn confusion matrix heatmap was used to evaluate the final model's classification trends across individual categories, revealing distinct behavior:

1. **Low-Recall Vulnerabilities:** The model struggled most with the `cat`, `dog`, and `deer` categories.
2. **Structural Confusions:** Detailed evaluation of 15 specific misclassified instances (5 target images per class) showed that errors were driven by shared visual characteristics—such as similar skeletal postures, fine fur patterns between cats and dogs, and highly noisy background textures blending with birds.


> You will find more details, including the full confusion matrix and misclassified image samples, in the `20230499_20230596_20230709_CNN_Assignment.ipynb` notebook.
