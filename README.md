# Mushroom Classification with Decision Tree

A machine-learning project that uses a **Decision Tree Classifier** to
classify mushrooms as edible (`e`) or poisonous (`p`). The project
compares different values of `min_samples_split` and selects the best
value using **Macro F1-score** and 5-fold stratified cross-validation.

## Dataset

-   File: `mushrooms.csv`
-   Rows: **8,124**
-   Columns: **23** (22 input features + 1 target)
-   Target column: `class`
-   Classes:
    -   `e`: edible
    -   `p`: poisonous
-   All 22 input features are categorical.

## Project structure

``` text
mushroom_decision_tree/
├── data/
│   └── mushrooms.csv
├── Notebook.ipynb
├── train_decision_tree.py
├── f1_results.csv
└── README.md
```

The notebook code expects the dataset at `data/mushrooms.csv`. If the
CSV is in the same directory as the notebook, change the `DATA_PATH`
line as explained in the code comments.

## Requirements

Install the required packages:

``` bash
pip install pandas numpy matplotlib scikit-learn
```

## Method

1.  Load the dataset and separate the target (`class`) from the input
    features.
2.  Split the data into training (80%) and test (20%) sets using
    stratification.
3.  Encode categorical features with `OneHotEncoder`.
4.  Build a pipeline containing the encoder and
    `DecisionTreeClassifier`.
5.  Test `min_samples_split` values from **2 through 100**.
6.  Evaluate each value using **5-fold Stratified Cross-Validation** and
    `f1_macro`.
7.  Select the parameter with the highest mean cross-validation
    F1-score.
8.  Fit the selected model on the full training set and evaluate it once
    on the held-out test set.

The encoder is inside the pipeline, so it is fitted separately within
each cross-validation training fold. This helps prevent preprocessing
leakage.

## Best results

The experiment produced a tie: every tested value from **2 to 32**
achieved the same highest mean cross-validation score. The code sorts
ties by the smaller parameter value, so it selects
`min_samples_split = 2`.

     `min_samples_split`   Mean CV Macro F1   Std. CV Macro F1
  ---------------------- ------------------ ------------------
    **2--32 (all tied)**       **0.999692**       **0.000616**
                      33           0.999230           0.000974
                      34           0.998768           0.001045
                      35           0.998152           0.001045
                  36--38           0.997998           0.000785
                  39--40           0.997844           0.000754

### Final test-set result

  Metric                                         Score
  ------------------------------ ---------------------
  Selected `min_samples_split`                   **2**
  Test Macro F1-score              **1.000000 (100%)**

The test score is calculated on the held-out 20% test set and is not
used to choose the hyperparameter.

## Output files

-   `f1_results.csv`: mean and standard deviation of the 5-fold Macro
    F1-score for each tested `min_samples_split` value.
-   The notebook displays the F1-score comparison plot and the final
    classification report and confusion matrix.

## Run

Open `Notebook.ipynb` in Jupyter Notebook or JupyterLab, make sure the
dataset path is correct, and run the cells from top to bottom.

## Notes

-   `min_samples_split` is the minimum number of samples required for a
    node to be considered for splitting. It is a tree-growth
    hyperparameter; it is **not** the same as cost-complexity pruning.
-   Macro F1 computes F1 separately for each class and takes their
    unweighted average.
-   The reported results correspond to the experiment settings above
    (`random_state=42`, 80/20 split, 5 folds, `criterion="gini"`, and
    values 2--100). Results may vary if these settings or the dataset
    change.
