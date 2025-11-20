# Image Classification With Transfer Learning

A comprehensive PyTorch implementation demonstrating fine-tuning of pre-trained CNNs on custom datasets, with detailed comparisons between transfer learning and training from scratch.

## 📋 Overview

This project showcases transfer learning for image classification using PyTorch. It includes:
- Fine-tuning pre-trained ResNet50 on custom datasets
- Comprehensive data augmentation pipeline
- Detailed evaluation with confusion matrices and misclassification analysis
- Side-by-side comparison of transfer learning vs. training from scratch
- Production-ready Jupyter notebooks with extensive documentation

## 🚀 Features

- **Multiple Pre-trained Models**: ResNet50 (easily adaptable to EfficientNet, ViT)
- **Data Augmentation**: Random flips, rotations, color jitter, and affine transformations
- **Small Dataset Friendly**: Optimized for datasets with ≤10k images
- **Comprehensive Metrics**: Accuracy, confusion matrix, per-class performance, misclassifications
- **Training Visualizations**: Loss/accuracy curves, learning progress tracking
- **Comparative Analysis**: Transfer learning vs. from-scratch training
- **GPU Support**: Automatic CUDA detection and optimization

## 📁 Repository Structure

```
.
├── transfer_learning.ipynb          # Main notebook: Transfer learning implementation
├── comparison_notebook.ipynb        # Comparison: Transfer learning vs. from scratch
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DeLyXHardiee/Image-Classification-With-Transfer-Learning.git
cd Image-Classification-With-Transfer-Learning
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Option 1: Use Demo Dataset
The notebooks automatically download the Flowers102 dataset for demonstration if no custom dataset is provided.

### Option 2: Custom Dataset
Organize your images in the following structure:

```
data/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class2/
│   ├── img1.jpg
│   └── ...
└── classN/
    └── ...
```

Place your dataset in the `./data` directory, and the notebooks will automatically load it.

## 📓 Usage

### 1. Transfer Learning Notebook

Open `transfer_learning.ipynb` to:
- Load and explore your dataset
- Fine-tune ResNet50 on your custom classes
- Visualize training progress
- Analyze model performance
- Examine misclassifications

**Quick Start:**
```bash
jupyter notebook transfer_learning.ipynb
```

**Key Steps:**
1. Run all cells sequentially
2. Adjust hyperparameters in the Configuration section if needed
3. Model trains for 10 epochs by default (adjustable)
4. Results saved automatically (model weights, plots)

### 2. Comparison Notebook

Open `comparison_notebook.ipynb` to:
- Compare transfer learning vs. training from scratch
- Understand convergence speed differences
- Analyze computational efficiency
- Get recommendations for your use case

**Quick Start:**
```bash
jupyter notebook comparison_notebook.ipynb
```

## ⚙️ Configuration

Key hyperparameters (found in both notebooks):

```python
CONFIG = {
    'batch_size': 32,           # Batch size for training
    'num_epochs': 10,           # Number of training epochs
    'learning_rate': 0.001,     # Learning rate for optimizer
    'num_classes': 5,           # Number of classes (auto-detected)
    'train_split': 0.8,         # Train/validation split ratio
    'image_size': 224,          # Input image size (224x224)
    'num_workers': 2,           # Data loader workers
}
```

## 📈 Outputs

After training, the notebooks generate:

### Saved Files
- `best_model.pth` - Best model weights based on validation accuracy
- `transfer_learning_model.pth` - Complete training checkpoint
- `training_history.png` - Training and validation curves
- `confusion_matrix.png` - Confusion matrix heatmap
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `misclassifications.png` - Visualization of misclassified samples
- `per_class_accuracy.png` - Per-class performance breakdown

### Performance Metrics
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Misclassification analysis
- Training/validation loss curves

## 🎯 Model Architecture

### Transfer Learning Approach
```
ResNet50 (Pre-trained on ImageNet)
    ├── Convolutional Layers [FROZEN]
    └── Custom Classifier [TRAINABLE]
        ├── Linear(2048 → 512)
        ├── ReLU
        ├── Dropout(0.3)
        └── Linear(512 → num_classes)
```

**Advantages:**
- Trains only ~1-2% of total parameters
- Faster convergence (5-10 epochs)
- Better performance on small datasets
- Less prone to overfitting

## 📊 Expected Results

### Transfer Learning (10 epochs, small dataset)
- **Validation Accuracy**: 70-90% (dataset dependent)
- **Training Time**: 5-15 minutes (with GPU)
- **Convergence**: Rapid improvement in first few epochs

### Training From Scratch (same dataset)
- **Validation Accuracy**: 50-70% (dataset dependent)
- **Training Time**: 20-60 minutes (with GPU)
- **Convergence**: Slower, requires more epochs

## 🔬 Experiments

### Try Different Models

**EfficientNet:**
```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
```

**Vision Transformer (ViT):**
```python
from torchvision.models import vit_b_16, ViT_B_16_Weights
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
```

### Fine-tune More Layers

Unfreeze last few layers for better performance:
```python
# Unfreeze last ResNet block
for param in model.layer4.parameters():
    param.requires_grad = True
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{image-classification-transfer-learning,
  author = {DeLyXHardiee},
  title = {Image Classification With Transfer Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/DeLyXHardiee/Image-Classification-With-Transfer-Learning}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share your results

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [ImageNet Dataset](https://www.image-net.org/)

## 💡 Tips for Best Results

1. **Data Quality**: Clean, well-labeled data is crucial
2. **Augmentation**: Essential for small datasets
3. **Learning Rate**: Start with 0.001, adjust if needed
4. **Batch Size**: Larger batches (32-64) generally work better
5. **Early Stopping**: Monitor validation loss to prevent overfitting
6. **Class Balance**: Ensure relatively balanced class distribution

## 🐛 Troubleshooting

**CUDA Out of Memory:**
- Reduce batch size
- Use smaller image size (e.g., 128x128)
- Reduce model capacity

**Poor Performance:**
- Increase training epochs
- Adjust learning rate
- Check data quality and labels
- Try different augmentation strategies

**Slow Training:**
- Ensure GPU is being used (`device = 'cuda'`)
- Increase num_workers in DataLoader
- Use mixed precision training

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Training! 🚀**