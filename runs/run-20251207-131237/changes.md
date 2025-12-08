# run-20251207-131237
Added a test split.


- Timestamp: 2025-12-07T13:12:37.903239
- Device: cuda
- Training samples: 160
- Validation samples: 20
- Test samples: 20
- Best validation accuracy: 95.00%
- Final validation accuracy: 95.00%
- Test accuracy: 85.00%
- Final train accuracy: 88.09%
- Final val accuracy (from history): 90.00%
- Final train loss: 1.1043
- Final val loss: 1.0561
- Misclassifications (Val): 1 / 20

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
| test_split | 0.1 |
| train_split | 0.8 |
| val_split | 0.1 |
| weight_decay | 0.0001 |

## Saved Artifacts
- **Best Model**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251207-131237\best_model.pth
- **Final Model**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251207-131237\transfer_learning_model.pth
- **Training History**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251207-131237\training_history.png
- **Confusion Matrix**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251207-131237\confusion_matrix.png
- **Confusion Matrix Normalized**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251207-131237\confusion_matrix_normalized.png
- **Misclassifications**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251207-131237\misclassifications.png
- **Per Class Accuracy**: C:\Users\mar20\Desktop\Computer Vision\Image-Classification-With-Transfer-Learning\runs\run-20251207-131237\per_class_accuracy.png