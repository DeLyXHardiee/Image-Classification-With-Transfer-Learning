# run-20251125-150335

# Changes
Went down from 102 classes in Flower102 set to only using 20.

- Timestamp: 2025-11-25T15:03:35.996107
- Device: cuda
- Training samples: 160
- Validation samples: 40
- Best validation accuracy: 90.00%
- Final validation accuracy: 100.00%
- Final train accuracy: 88.54%
- Final val accuracy (from history): 87.50%
- Final train loss: 1.0904
- Final val loss: 0.9756
- Misclassifications: 4 / 40

## Configuration
| Key | Value |
| --- | --- |
| batch_size | 8 |
| fine_tune_lr | 0.0001 |
| image_size | 224 |
| label_smoothing | 0.1 |
| learning_rate | 0.001 |
| max_classes | 20 |
| mixup_alpha | 0.4 |
| num_classes | 20 |
| num_epochs | 20 |
| num_workers | 6 |
| train_split | 0.8 |
| weight_decay | 0.0001 |

## Saved Artifacts
- **Best Model**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-150335\best_model.pth
- **Final Model**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-150335\transfer_learning_model.pth
- **Training History**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-150335\training_history.png
- **Confusion Matrix**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-150335\confusion_matrix.png
- **Confusion Matrix Normalized**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-150335\confusion_matrix_normalized.png
- **Misclassifications**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-150335\misclassifications.png
- **Per Class Accuracy**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-150335\per_class_accuracy.png