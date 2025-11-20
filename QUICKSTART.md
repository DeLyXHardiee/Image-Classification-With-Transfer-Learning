# Quick Start Guide

Get up and running with transfer learning in just a few steps!

## 🚀 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
jupyter notebook
```

### 3. Open a Notebook
- **For basic transfer learning**: Open `transfer_learning.ipynb`
- **For comparison study**: Open `comparison_notebook.ipynb`

### 4. Run All Cells
Click "Cell" → "Run All" or press Shift+Enter to run cells sequentially

## 📊 Using Your Own Dataset

### Organize Your Images
```
data/
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
├── cats/
│   ├── cat1.jpg
│   └── ...
└── birds/
    └── ...
```

### Update Config
In the notebook, update the configuration:
```python
CONFIG = {
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'num_classes': 3,  # Change to your number of classes
    # ... other settings
}

data_dir = './data'  # Path to your dataset
```

### Run Training
Execute all cells - the notebook will:
1. ✅ Load your dataset
2. ✅ Apply data augmentation
3. ✅ Fine-tune ResNet50
4. ✅ Generate evaluation metrics
5. ✅ Save trained model

## 📈 What You'll Get

After training completes, you'll have:

### Saved Files
- `best_model.pth` - Your trained model
- `training_history.png` - Training curves
- `confusion_matrix.png` - Performance visualization
- `misclassifications.png` - Error analysis
- `per_class_accuracy.png` - Per-class metrics

### Performance Metrics
- Overall accuracy
- Per-class precision/recall/F1
- Confusion matrix
- Misclassification analysis

## 🎯 Expected Training Time

| Dataset Size | GPU | CPU  |
|-------------|-----|------|
| 1,000 images | 2-5 min | 10-20 min |
| 5,000 images | 5-10 min | 30-60 min |
| 10,000 images | 10-20 min | 1-2 hours |

*Times are approximate for 10 epochs with batch size 32*

## ⚙️ Quick Tweaks

### Speed Up Training
```python
CONFIG['batch_size'] = 64  # Increase batch size
CONFIG['num_workers'] = 4   # More data loading threads
```

### Improve Accuracy
```python
CONFIG['num_epochs'] = 20    # Train longer
CONFIG['learning_rate'] = 0.0001  # Lower learning rate
```

### Reduce Memory Usage
```python
CONFIG['batch_size'] = 16    # Smaller batches
CONFIG['image_size'] = 128   # Smaller images
```

## 🐛 Quick Fixes

**Problem**: "CUDA out of memory"
```python
# Solution: Reduce batch size
CONFIG['batch_size'] = 16
```

**Problem**: "Dataset not found"
```python
# Solution: Check path
import os
print(os.path.exists('./data'))  # Should be True
```

**Problem**: Poor accuracy
```python
# Solutions:
# 1. Train more epochs
CONFIG['num_epochs'] = 20

# 2. Check data quality
# - Are images labeled correctly?
# - Are classes balanced?
# - Is there enough variety?

# 3. Adjust learning rate
CONFIG['learning_rate'] = 0.0001
```

## 📚 Next Steps

1. ✅ Complete `transfer_learning.ipynb` first
2. ✅ Try `comparison_notebook.ipynb` to understand the benefits
3. ✅ Experiment with different hyperparameters
4. ✅ Try other architectures (EfficientNet, ViT)
5. ✅ Apply to your own dataset

## 💡 Tips for Best Results

1. **Data Quality First**: Clean, well-labeled data is crucial
2. **Start Simple**: Use default settings first, then optimize
3. **Monitor Training**: Watch for overfitting (train/val gap)
4. **Save Checkpoints**: Best model is automatically saved
5. **Visualize Results**: Check confusion matrix for insights

## 🔗 Resources

- [Main README](README.md) - Complete documentation
- [PyTorch Docs](https://pytorch.org/docs/) - Framework reference
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) - Official guide

---

**Ready to start?** Open `transfer_learning.ipynb` and run all cells! 🚀
