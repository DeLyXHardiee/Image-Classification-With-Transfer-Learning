# run-20251125-151512

# Changes
Went from unfreezing layer 3 and 4 to only unfreezing layer 4

- Timestamp: 2025-11-25T15:15:12.144011
- Device: cuda
- Training samples: 160
- Validation samples: 40
- Best validation accuracy: 92.50%
- Final train accuracy: 88.09%
- Final val accuracy (from history): 87.50%
- Final train loss: 1.1045
- Final val loss: 1.0508
- Misclassifications: 3 / 40

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
- **Best Model**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-151512\best_model.pth
- **Final Model**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-151512\transfer_learning_model.pth
- **Training History**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-151512\training_history.png
- **Confusion Matrix**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-151512\confusion_matrix.png
- **Confusion Matrix Normalized**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-151512\confusion_matrix_normalized.png
- **Misclassifications**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-151512\misclassifications.png
- **Per Class Accuracy**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-151512\per_class_accuracy.png