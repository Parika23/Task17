# Task17 — Student Habits & Performance

## Overview

This project applies a complete machine learning workflow to a Student Performance dataset. The analysis starts with exploratory data analysis (EDA), prepares the data for modeling, builds a regression model to predict exam scores, and then builds a classification model to predict whether a student passes or fails.

## Objectives

- Understand the structure and quality of the dataset.
- Identify numerical and categorical variables.
- Examine distributions, relationships, correlations, missing values, and potential outliers.
- Identify factors associated with `exam_score`.
- Predict `exam_score` using Linear Regression.
- Create a Pass/Fail target using a 50 mark threshold.
- Predict Pass/Fail using Logistic Regression.
- Evaluate both models on training and testing data.
- Assess possible overfitting or underfitting.
- Summarize five meaningful student performance insights.

## Dataset

The dataset contains 1,000 student records and 16 columns, including:

- `age`
- `gender`
- `study_hours_per_day`
- `social_media_hours`
- `netflix_hours`
- `part_time_job`
- `attendance_percentage`
- `sleep_hours`
- `diet_quality`
- `exercise_frequency`
- `parental_education_level`
- `internet_quality`
- `mental_health_rating`
- `extracurricular_participation`
- `exam_score`

`student_id` is treated as an identifier and excluded from modeling.

## Machine Learning Workflow

### Regression

**Target:** `exam_score`

**Model:** Linear Regression

**Evaluation metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification

**Target:** Pass/Fail derived from `exam_score`

- Pass: `exam_score >= 50`
- Fail: `exam_score < 50`

**Model:** Logistic Regression

**Evaluation metrics:**
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score

## Preprocessing

The notebook uses a `scikit-learn` preprocessing pipeline:

- Numeric variables → median imputation + standardization
- Categorical variables → most frequent imputation + one-hot encoding
- `student_id` → removed
- Train/test split → 80/20
- Classification split → stratified 80/20

The preprocessing is fitted only on the training data through the pipeline to reduce data leakage.

## Key EDA Findings

The dataset shows that:

- `study_hours_per_day` has the strongest positive relationship with `exam_score`.
- `mental_health_rating` has a moderate positive relationship.
- `exercise_frequency` and `sleep_hours` have weaker positive relationships.
- `social_media_hours` and `netflix_hours` are negatively associated with exam score.
- `parental_education_level` contains missing values that require preprocessing.
- Some numeric observations are flagged as potential IQR outliers, but they are retained because they remain plausible observations.
- The Pass/Fail target is imbalanced toward Pass.

## Files

```text
Task17/
├── Student_habits_performance.ipynb
├── Day18_19_student_habits_performance.csv
└── README.md
```

## Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## How to Run

1. Open `Student_habits_performance.ipynb` in Jupyter Notebook or Google Colab.
2. Keep `Day18_19_student_habits_performance.csv` in the same folder, or upload the CSV when prompted in Colab.
3. Run the notebook from top to bottom.

## Conclusion

The project demonstrates an end-to-end data analytics and machine learning workflow, from EDA and preprocessing to regression, classification, evaluation, and interpretation. The analysis indicates that study habits—especially daily study hours—provide the strongest predictive signal for exam performance in this dataset, while lifestyle and well-being variables contribute additional information.
