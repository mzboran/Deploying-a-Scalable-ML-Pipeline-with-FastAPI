# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a binary classifier model, using a `RandomForestClassifier`, trained to predict if an individual's income
exceeds $50,000.00 based on features from the UCI Census Income dataset.

## Intended Use
The model is intended for educational purposes.  This exercise is part of a machine learning pipeline deployment.

## Training Data
The training data comes from the UCI Census Income dataset. It includes features such as:

    workclass
    education
    marital-status
    occupation
    relationship
    race
    sex
    native-country

The target variable is income bracket: `<=50K` or `>50K`.

## Evaluation Data
The dataset is split into 80% training and 20% testing data.  The fixed random seed is `random_state=1337`.
Evaluation was based on the hold-out test set and performance on slices of features.

## Metrics
Overall Performance:

    Precision: 0.7372
    Recall: 0.6096
    F1 Score: 0.6674

Example Slices:

| Workclass       | Count | Precision | Recall | F1 Score |
|----------------|-------|-----------|--------|----------|
| ?              | 377   | 0.6111    | 0.4681 | 0.5301   |
| Federal-gov    | 176   | 0.7042    | 0.7692 | 0.7353   |
| Local-gov      | 415   | 0.7742    | 0.6000 | 0.6761   |
| Never-worked   | 1     | 1.0000    | 1.0000 | 1.0000   |


## Ethical Considerations
Predictions should not be used in decision-making scenarios with high possibility to cause harm.

## Caveats and Recommendations
Users of this model should not interpret predictions as definitive indicators of income.  
Further work is recommended to:

- Tune hyperparameters  
- Explore fairness mitigation strategies  
- Benchmark alternative models
