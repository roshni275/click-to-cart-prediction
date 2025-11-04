# Click-to-Cart: Predicting Purchase Intent from User Behavior
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Library](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange)


This project analyzes session-level customer behavior on an e-commerce platform and predicts whether a user will make a purchase.  
The goal was not only to build a classifier, but to understand behavioral patterns and compare how different machine learning models handle this type of data.

## Datasets & Notebook
Training Dataset: [Click to download](https://drive.google.com/drive/u/0/folders/1bDGWF8N644N-aG0zAoK6D7PuxMD1SQRK)  
Contains user session data for training the model.

Test Dataset: [Click to download](https://drive.google.com/drive/u/0/folders/1bDGWF8N644N-aG0zAoK6D7PuxMD1SQRK)  
Used to evaluate model performance.

Colab Notebook: [Open in Colab](https://colab.research.google.com/drive/1fX5E8ZhXEjn2NA7S7_v__-yRdyJQW0hk?usp=sharing)  
Full analysis, modeling, and visualization.

---

## 1. Exploratory Data Analysis (EDA)

Before modeling, detailed EDA was conducted to understand data distributions, correlations, and noise.

### Dataset Overview:
- 4 float columns
- 2 integer columns (including target `Purchase`)
- 2 object/categorical columns
- No missing values, no duplicate rows
- Slight class imbalance: ~70% `Purchase=0`, 30% `Purchase=1`

### Key Observations:
- Users who purchased generally had **higher time on site** and **more pages viewed**.
- Browser refresh rate was **irrelevant** for the target and removed.
- `Referral` and `Last_Ad_Seen` were one-hot encoded.
- Google is the largest referral source (~41.17%).
- Nearly 30% of users in each referral or ad category made a purchase.

---

## 2. Model 1: AdaBoost

### Overview:
- Stagewise additive model using **Decision Stumps** (weak learners).
- Iteratively focuses on misclassified instances by adjusting their weights.
- Final prediction is a weighted sum of all predictors.


### Assumptions:
- Dataset has no missing values.
- Less outliers (AdaBoost is sensitive to noisy data).

### Hyperparameters:
- `n_estimators = 10`
- `learning_rate = 0.5`

### Performance:
| Metric | Score |
|--------|-------|
| Accuracy | 79.56% |
| Precision | 0.72 |
| Recall | 0.54 |
| F1 Score | 0.62 |
| AUC | 0.86 |

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/d8cc7aca-44d1-41f4-9aea-3a74dc344c3f" />



**From Scratch Version:** Accuracy = 76.89%

---

## 3. Model 2: XGBoost

### Overview:
- Gradient boosting on decision trees, optimizing a differentiable loss function.
- Handles class imbalance, supports parallel processing, and includes regularization.



### Assumptions:
- Differentiable loss function.
- Less outliers to avoid overfitting.
- Dataset is not extremely small.

### Hyperparameters:
- `n_estimators = 20`
- `learning_rate = 0.1`
- `max_depth = 3`
- `colsample_bytree = 0.8`
- `subsample = 0.8`
- `scale_pos_weight = 1`

### Performance:
| Metric | Score |
|--------|-------|
| Accuracy | 79.11% |
| Precision | 0.72 |
| Recall | 0.51 |
| F1 Score | 0.59 |
| AUC | 0.84 |

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/e121d7de-9d60-4896-9964-d5f40f2a47a8" />

**From Scratch Version:** Accuracy = 78.67%

## 4. Comparative Analysis

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|---------|-----------|--------|----|-----|
| AdaBoost | 79.56% | 0.72 | 0.54 | 0.62 | 0.86 |
| XGBoost  | 79.11% | 0.72 | 0.51 | 0.59 | 0.84 |

**Insights:**
- Both models perform similarly overall.
- AdaBoost slightly outperforms XGBoost on Recall, F1, and AUC.
- High precision is important as **false positives are costly** in marketing campaigns.
- Slightly better AdaBoost performance may be due to small dataset size and fewer hyperparameters compared to XGBoost.

---
## 5. Conclusion

AdaBoost slightly outperforms XGBoost on recall, F1, and AUC, while both models achieve high precision.  
High precision is important to reduce false positives and marketing costs.

