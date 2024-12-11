### Model Performance Comparison

| Classifier              | Accuracy | Precision (0) | Recall (0) | F1-score (0) | Precision (1) | Recall (1) | F1-score (1) | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-score |
|-------------------------|----------|----------------|-------------|---------------|----------------|-------------|---------------|---------------------|-------------------|---------------------|
| **Naive Bayes on original data**         | 0.6875   | 0.74           | 0.48        | 0.58          | 0.67           | 0.86        | 0.75          | 0.70                | 0.67              | 0.67                |
| **Naive Bayes on preprocessed data**         | 0.6808   | 0.73           | 0.47        | 0.57         | 0.66           | 0.85        | 0.75          | 0.69                | 0.66              | 0.66                |
| **K-Nearest Neighbors** | 0.9075   | 0.90           | 0.89        | 0.90          | 0.91           | 0.92        | 0.92          | 0.91                | 0.91              | 0.91                |
| **Random Forest**       | 0.9233   | 0.91           | 0.92        | 0.92          | 0.93        | **0.93**        | 0.93          | 0.92                | 0.92              | 0.92                |
| **Logistic Regression** | 0.91     | 0.87           | 0.94        | 0.90          | **0.95**           | 0.89        | 0.92          | 0.91                | 0.91              | 0.91                |
| **Stacking model** | 0.91     | 0.91           | 0.90        | 0.90          | 0.92           | **0.93**        | 0.92          | 0.91                | 0.91              | 0.91                |
| **BERT**       | **0.9363**   | **0.92**           | **0.94**        | **0.93**          | **0.95**           | **0.93**        | **0.94**          | **0.94**                | **0.94**              | **0.94**                |
| **Neural Network**        | 0.8446   |            |         |           |                |             |               | 0.87                    | 0.85                  | 0.86                    |



---

### Explanation of the Columns:
- **Accuracy**: Overall accuracy of the model.
- **Precision (0)**, **Recall (0)**, **F1-score (0)**: Metrics for class **0**.
- **Precision (1)**, **Recall (1)**, **F1-score (1)**: Metrics for class **1**.
- **Macro Avg Precision**, **Macro Avg Recall**, **Macro Avg F1-score**: Averages of precision, recall, and F1-score for both classes