# 👁️ Cataract Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project for automated cataract detection from retinal images using Convolutional Neural Networks (CNN) and Transfer Learning with EfficientNet.

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)

## 🔍 Overview

This project implements a binary classification system to detect cataracts from retinal fundus images. Two different approaches are compared:
1. **Custom CNN Model** - A neural network built from scratch
2. **Transfer Learning Model** - Using pre-trained EfficientNetB0

## 📊 Dataset

The project uses two datasets for training:

### 1. Cataract Dataset
- **Source**: [Kaggle Cataract Dataset](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)
- **Categories**: Normal, Cataract, Glaucoma, Retina Disease
- **Used**: Only Normal and Cataract images

### 2. Ocular Disease Recognition Dataset (ODIR-5K)
- **Source**: ODIR-5K Training Images
- **Processing**: Extracted images with cataract mentions in diagnostic keywords
- **Balancing**: Downsampled to address class imbalance

### 📈 Dataset Statistics
- Combined dataset with balanced classes (Normal vs Cataract)
- Train/Validation/Test split: 68%/12%/20%
- Image preprocessing: Resized to 256x192 pixels, normalized to [0,1]

## 🔬 Methodology

```
┌─────────────┐  ┌─────────────┐
│  Cataract   │  │   Ocular    │
│   Dataset   │  │   Disease   │
└──────┬──────┘  └──────┬──────┘
       └────────┬────────┘
                │
         ┌──────▼──────┐
         │   Combined  │
         │   Dataset   │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │    Data     │
         │Preprocessing│
         └──────┬──────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼────┐          ┌───────▼──────┐
│ Custom │          │  EfficientNet│
│  CNN   │          │     B0       │
└───┬────┘          └───────┬──────┘
    │                       │
    └───────────┬───────────┘
                │
         ┌──────▼──────┐
         │   Training  │
         │ & Evaluation│
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │Best Model   │
         │  Selection  │
         └─────────────┘
```

### 🔄 Data Preprocessing Steps
1. ✅ Load images from both datasets
2. ✅ Extract normal and cataract cases
3. ✅ Balance dataset by downsampling
4. ✅ Resize images to 256x192 pixels
5. ✅ Normalize pixel values (0-1 range)
6. ✅ Split into train, validation, and test sets
7. ✅ Apply data augmentation (horizontal flip, height shift)

## 🧠 Models

### Model 1: Custom CNN Architecture
```
📐 Architecture:
├── Conv2D (16 filters, 3x3) + ReLU + BatchNorm + MaxPool
├── Conv2D (32 filters, 3x3) + ReLU + BatchNorm + MaxPool
├── Conv2D (64 filters, 3x3) + ReLU + BatchNorm + MaxPool
├── Flatten
├── Dense (1024) + Dropout (0.5)
├── Dense (512) + Dropout (0.7)
├── Dense (128) + Dropout (0.5)
└── Dense (2, softmax)
```

**Specifications:**
- ⚙️ Optimizer: Adam
- 📉 Loss: Categorical Crossentropy
- 📊 Metrics: Accuracy
- 🔢 Epochs: 100
- 📦 Batch Size: 32

### Model 2: Transfer Learning with EfficientNetB0
```
📐 Architecture:
├── EfficientNetB0 (pretrained on ImageNet)
├── GlobalAveragePooling2D
└── Dense (2, softmax)
```

**Specifications:**
- ⚙️ Optimizer: Adam (lr=0.000003)
- 📉 Loss: Categorical Crossentropy with Label Smoothing (0.01)
- 📊 Metrics: Accuracy
- 🔢 Epochs: 100 (with Early Stopping)
- 📦 Batch Size: 32
- 🛑 Callbacks: EarlyStopping (patience=20), ReduceLROnPlateau (patience=10)

### 🎨 Custom Activation Function
- Implemented **Mish activation** as a custom Keras layer

## 🚀 Installation

### Prerequisites
```bash
Python 3.x
TensorFlow 2.x
```

### Clone Repository
```bash
git clone https://github.com/yourusername/cataract-detection.git
cd cataract-detection
```

### Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow opencv-python efficientnet scikit-image tqdm openpyxl
```

## 💻 Usage

### 1️⃣ Download Datasets
Download the datasets from:
- [Cataract Dataset](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)
- [ODIR-5K Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

### 2️⃣ Update Paths
Update the dataset paths in the notebook:
```python
IMG_ROOT = '../input/cataractdataset/dataset/'
OCU_IMG_ROOT = '../input/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images/'
```

### 3️⃣ Run Notebook
```bash
jupyter notebook Cataract_Detection.ipynb
```

### 4️⃣ Training Process
The notebook will:
- 📂 Load and preprocess both datasets
- ⚖️ Balance the classes
- 📊 Split data into train/validation/test sets
- 🎯 Train both models
- 📈 Compare model performances
- 💾 Save the best model

## 📊 Results

The models are evaluated on:
- ✅ **Accuracy**: Classification accuracy on test set
- 📉 **Loss**: Training and validation loss curves
- 📈 **Training History**: Accuracy improvement over epochs

Both models use:
- 🔄 **Data Augmentation**: Horizontal flips and height shifts
- 🎯 **Early Stopping**: Prevents overfitting
- 📉 **Learning Rate Reduction**: Adaptive learning rate scheduling

## 📁 Project Structure

```
cataract-detection/
│
├── 📓 Cataract_Detection.ipynb      # Main Jupyter notebook
├── 📄 Cataract_Dataset.txt          # Dataset link information
├── 🖼️  Cataract_Detection_Methodology.png  # Methodology diagram
├── 📋 README.md                      # Project documentation
│
└── 📂 datasets/                      # (Download separately)
    ├── cataractdataset/
    └── ocular-disease-recognition-odir5k/
```

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Programming Language |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep Learning Framework |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | Neural Network API |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical Computing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data Manipulation |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) | Image Processing |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine Learning Tools |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white) | Data Visualization |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive Development |

### 📚 Key Libraries
- **TensorFlow/Keras** - Deep learning framework
- **EfficientNet** - Pre-trained model for transfer learning
- **OpenCV** - Image processing
- **scikit-learn** - Train/test split, metrics
- **scikit-image** - Image I/O operations
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization

## 🎯 Key Features

✨ **Dual Model Approach**: Compare custom CNN vs. transfer learning
✨ **Data Augmentation**: Improves model generalization
✨ **Class Balancing**: Handles imbalanced datasets
✨ **Multiple Datasets**: Combines data from two sources
✨ **Custom Activation**: Implements Mish activation function
✨ **Callbacks**: Early stopping and learning rate reduction
✨ **Visualization**: Training curves and sample images

## 🔮 Future Enhancements

- 🎯 Multi-class classification (cataract severity levels)
- 📱 Deploy as web/mobile application
- 🔍 Add explainability (Grad-CAM visualization)
- 📊 Implement additional metrics (Precision, Recall, F1-Score)
- 🚀 Try other architectures (ResNet, DenseNet, Vision Transformers)
- 💾 Model optimization and quantization
- 🌐 Real-time inference pipeline
