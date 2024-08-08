# Cat vs Dog Image Classification with SVM

## Overview
This project focuses on building an image classification model to distinguish between cats and dogs using Support Vector Machine (SVM). The dataset contains 25,000 images, split equally between cats and dogs.

## Project Structure
- `data/`: Directory containing the dataset of images.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model building.
- `scripts/`: Python scripts for data preprocessing, training, and evaluation.
- `models/`: Saved models and evaluation metrics.

## Data Preprocessing
- **Resizing:** All images are resized to 64x64 pixels.
- **Flattening:** Resized images are flattened to create a feature vector.
- **Standard Scaling:** Feature vectors are scaled using `StandardScaler`.

## Model Training
- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **Hyperparameters:** Default settings with `gamma='scale'`

## Evaluation Metrics
- **Accuracy:** 67.3%
- **Precision, Recall, F1-Score:** Approximately 67% for both cats and dogs.
- **Silhouette Score:** 0.0103
- **Davies-Bouldin Index:** 10.5236

## Results
| Metric                | Value      |
|-----------------------|------------|
| Accuracy              | 67.3%      |
| Precision (Cat)       | 68%        |
| Precision (Dog)       | 67%        |
| Recall (Cat)          | 67%        |
| Recall (Dog)          | 68%        |
| F1-Score (Cat)        | 67%        |
| F1-Score (Dog)        | 67%        |
| Silhouette Score      | 0.0103     |
| Davies-Bouldin Index  | 10.5236    |

## Visualizations
### Sample Predictions
![Sample Predictions](images/sample_predictions.png)

## Next Steps
1. **Data Augmentation:** Improve model performance by augmenting the dataset with transformations like rotation, flipping, and scaling.
2. **Hyperparameter Tuning:** Use techniques like GridSearchCV to find the optimal hyperparameters.
3. **Advanced Models:** Experiment with more complex models like Convolutional Neural Networks (CNNs).

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
