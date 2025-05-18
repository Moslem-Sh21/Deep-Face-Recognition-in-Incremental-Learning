# Deep Face Recognition Framework

## Overview
This implementation was developed as part of the MRI segmentation course project for ELEC 872 AI and Interactive Systems.
## Dataset Requirements
- The VGGFace2 dataset should be located at: `D:/vgg_face2/train_test/`
- If your dataset is stored in a different location, please update the paths in the `train.lst` and `val.lst` files accordingly.

## Repository Structure
All implementation code is located in the `\scail\codes` directory.

## 1. Deep Face Recognition Models

### Available Models
The framework includes three main CNN architectures:
- **ResNet** (`ResNet.py`): Implements ResNet18 or ResNet34 architectures
- **VGG16** (`VGG16.py`): Implements the VGG16 architecture
- **Inception** (`Inception.py`): Implements the Inception architecture

### Using ResNet Models
The `ResNet.py` script provides flexibility in model configuration:

#### Loss Function Selection
You can select between three different loss functions by modifying lines 52-54:
```python
# ------------ selecting loss ------------
loss_fun = 'softmax'
# loss_fun = 'hing'
# loss_fun = 'kl_div'
```

#### Path Configuration
Modify the following paths if needed (lines 75-83):
```python
# Training image list path
train_file_path = 'D:/FR_codes/data/images_list_files/vgg_faces/S~10/batch1/train.lst'

# Validation image list path
val_file_path = 'D:/FR_codes/data/images_list_files/vgg_faces/S~10/batch1/val.lst'

# Dataset mean and standard deviation path
datasets_mean_std_file_path = 'D:/FR_codes/data/datasets_mean_std.txt'
```

#### Data Augmentation Options
Various data augmentation techniques can be enabled by uncommenting lines 109-112:
```python
train_dataset = ImagesListFileFolder(
    train_file_path,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # ---------- Additional data augmentation options ----------
        # transforms.RandomRotation(degrees=(-45, 45)),
        # transforms.RandomAdjustSharpness(2, p=0.5),
        # transforms.GaussianBlur((5, 9), sigma=(0.1, 2.0)),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.2, 0.5)),
        transforms.ToTensor(),
        normalize,
    ])
)
```

#### Model Architecture Selection
You can choose between ResNet18 and ResNet34 by modifying lines 156-157:
```python
model = models.resnet18(pretrained=False, num_classes=num_classes)
# model = models.resnet34(pretrained=False, num_classes=num_classes)
```

#### Model Saving Location
The trained model will be saved to:
```
D:/FR_codes/scail/model_FR
```

#### Default Configuration
The default configuration uses:
- RandomResizedCrop(224)
- RandomHorizontalFlip()
- ResNet18 architecture
- Softmax loss function

### Using VGG16 and Inception Models
The `VGG16.py` and `Inception.py` scripts follow the same usage pattern as `ResNet.py`, with the only difference being the underlying CNN architecture.

## 2. Incremental Learning for Face Recognition

### Training Pipeline

#### Step 1: Training the First Batch of Classes from Scratch
```bash
python codes/scratch.py
```

Important path configurations:
```python
# Training image list path
train_file_path = 'D:/FR_codes/data/images_list_files/vgg_faces/S~10/batch1/train.lst'

# Validation image list path
val_file_path = 'D:/FR_codes/data/images_list_files/vgg_faces/S~10/batch1/val.lst'

# Dataset mean and standard deviation path
datasets_mean_std_file_path = 'D:/FR_codes/data/datasets_mean_std.txt'
```

#### Step 2: Incremental Learning with Fine-tuning
```bash
python codes/ft.py configs/ft.cf
```

Important configurations in `ft.cf`:
```
train_files_dir = D:/FR_codes/data/images_list_files/vgg_faces/S~10/unbalanced/train/
dataset_files_dir = D:/FR_codes/data/images_list_files/vgg_faces/S~10
first_model_load_path = D:/FR_codes/scail/model_s/vgg_faces_s10_batch1.pt
```

#### Step 3: Validation Features Extraction
```bash
python codes/features_extraction.py configs/features_extraction.cf
```

Important configurations in `features_extraction.cf`:
```
datasets_mean_std_file_path = D:/FR_codes/data/datasets_mean_std.txt
first_model_load_path = D:/FR_codes/scail/model_s/vgg_faces_s10_batch1.pt
models_load_path_prefix = D:/FR_codes/data/images_list_files/vgg_faces/S~10/unbalanced/train/model_ft/ift_vgg_faces_s10_5k_b
val_images_list_dir = D:/FR_codes/data/images_list_files/vgg_faces/S~10/accumulated/val
```

#### Step 4: Last Layer Parameters Extraction
For the first batch:
```bash
python codes/extract_last_layer_weights_for_first_batch.py
```

Important configuration:
```python
model_load_path = "D:/FR_codes/scail/model_s/vgg_faces_s10_batch1.pt"
```

For fine-tuned models:
```bash
python codes/extract_last_layer_weights_for_ft.py
```

Important configuration:
```python
models_load_path_prefix = 'D:/FR_codes/data/images_list_files/vgg_faces/S~10/unbalanced/train/model_ft/ift_vgg_faces_s10_5k_b'
```

#### Step 5: ScaIL Implementation
```bash
python codes/scail.py
```

Important configuration:
```python
list_root_dir = 'D:/FR_codes/data/images_list_files'
local_root_dir = 'D:/FR_codes/data/images_list_files'
```

