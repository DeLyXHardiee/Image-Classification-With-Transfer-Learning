# run-20251125-121143

# Changes

- Enabled label smoothing and mixup to regularize the classifier and training loop.
- Unfroze ResNet50 `layer3` and `layer4`, fine-tuning them alongside the classifier while introducing parameter groups with a smaller learning rate and `1e-4` weight decay.
- Added mixup helper utilities plus configuration entries (`fine_tune_lr`, `weight_decay`, `label_smoothing`, `mixup_alpha`) so experiments can be tuned from a single dict.


- Timestamp: 2025-11-25T12:11:43.405851
- Device: cuda
- Training samples: 816
- Validation samples: 204
- Best validation accuracy: 89.22%
- Final validation accuracy: 100.00%
- Final train accuracy: 89.73%
- Final val accuracy (from history): 87.25%
- Final train loss: 1.3602
- Final val loss: 1.5166
- Misclassifications: 22 / 204

## Configuration
| Key | Value |
| --- | --- |
| batch_size | 32 |
| fine_tune_lr | 0.0001 |
| image_size | 224 |
| label_smoothing | 0.1 |
| learning_rate | 0.001 |
| mixup_alpha | 0.4 |
| num_classes | 102 |
| num_epochs | 20 |
| num_workers | 6 |
| train_split | 0.8 |
| weight_decay | 0.0001 |

## Saved Artifacts
- **Best Model**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-121143\best_model.pth
- **Final Model**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-121143\transfer_learning_model.pth
- **Training History**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-121143\training_history.png
- **Confusion Matrix**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-121143\confusion_matrix.png
- **Confusion Matrix Normalized**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-121143\confusion_matrix_normalized.png
- **Misclassifications**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-121143\misclassifications.png
- **Per Class Accuracy**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251125-121143\per_class_accuracy.png