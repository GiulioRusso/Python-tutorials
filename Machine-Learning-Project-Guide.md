[Back to Index 🗂️](./README.md)

<center><h1>🧪 Machine Learning Project Guide</h1></center>

<br>

Welcome to your practical guide for setting up a machine‑learning project in Python with tabular data—walking you through every essential step from raw CSVs to a production‑ready model. This checklist maps each notebook cell to a concise narrative so you can switch effortlessly between code and guidance.

## 1️⃣ Imports & Environment Setup

Bundles every critical import and seed‑setting command into one upfront cell—eliminating the hunting for scattered imports later and ensuring each run is perfectly reproducible.

```python
# Add all your necessary imports here
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np, random, os

# Set the random seeds 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

## 2️⃣ File Path

Centralises fundamental project constants—dataset path and target column—into one location. Keeping these variables explicit and in a single cell minimises hard‑coded surprises, streamlines parameter tweaks, and ensures every downstream script references the exact same configuration.

```python
FILE_PATH   = "./data/features.csv"      # e.g.: CSV directory
TARGET_COL  = "target"     # Replace with your column name
```

## 3️⃣ Load Dataset

Loads the raw CSV into memory and immediately separates features from the target column for downstream processing.

```python
# Load the raw data set from disk and handle a missing-file error gracefully
try:
    # Attempt to read the CSV file located at FILE_PATH
    data = pd.read_csv(FILE_PATH)
    print(f"Data successfully loaded from {FILE_PATH}.")
    
# Raised if the file does not exist at the specified path
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {FILE_PATH}")
```

```python
# Attempt to separate features and target in the DataFrame
try:
    # Drop the target column to obtain the feature matrix X
    X = data.drop(columns=[TARGET_COL])
    
    # Extract the target vector y
    y = data[TARGET_COL]
    
    # Basic sanity-check prints
    print(f"Features (X) shape: {X.shape}")   # rows × feature-columns
    print(f"Target (y) shape: {y.shape}")     # rows × 1 (or Series length)

# Handle the case where the provided TARGET_COL is not present
except KeyError:
    raise KeyError(f"Target column '{TARGET_COL}' not found in the dataset.")

```

## 4️⃣ Data Exploration and Visualization

Generates two high‑impact plots to ground your intuition before modelling.

- **Class distribution chart**: A seaborn.countplot shows how many samples belong to each target class (in case you are facing a classification task) or target values (in case you are facing a regression task). This quickly surfaces any class imbalance that may require resampling, re‑weighting, or alternative evaluationmetrics.

    ```python
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y, palette='viridis')

    # Add value counts on top of the bars
    for i, v in enumerate(pd.Series(y).value_counts()):
        plt.text(i, v, str(v), ha='center', va='bottom', fontsize=12)

    plt.title('Distribution of Classes', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()
    ```

- **Correlation heat‑map**: A colour‑coded matrix of Pearson correlations between numerical features. Strong off‑diagonal values highlight redundant predictors (multicollinearity) or reveal domain relationships worth engineering into new features.

    ```python
    correlation_matrix = X.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f', 
                square=True, 
                linewidths=0.5)

    plt.title('Correlation Heatmap of Features', fontsize=16)
    plt.tight_layout()
    plt.show()
    ```

Use these visuals to decide whether to balance classes, drop or combine variables, or apply dimensionality reduction before continuing.

## 5️⃣ Data preprocessing

When raw data arrives, it rarely lines up with the mathematical assumptions baked into our machine-learning algorithms. Two of the most common hurdles are categorical features and missing values, and both can introduce subtle but serious bias if left untreated.

- **One-hot encoding**: converts each category into a separate binary column so that algorithms can measure “distance” between examples without imposing an arbitrary numerical order (e.g., “red = 1, blue = 2” would falsely imply blue > red). By turning categories into orthogonal indicators, we preserve all the information in the original labels while letting linear models, trees, and neural networks process them just like any other numeric input.

    ```python
    # Identify columns whose dtype is object or category
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns

    # Instantiate a OneHotEncoder that:
    # - ignores unseen categories at transform time
    # - returns a dense array (sparse_output=False)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit the encoder on the categorical columns and transform them
    X_encoded = encoder.fit_transform(X[categorical_columns])

    # Get the generated one-hot column names, e.g. "color_red", "color_blue"
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    # Wrap the encoded array in a DataFrame aligned to the original index
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns, index=X.index)

    # Replace the original categorical columns with their one-hot expansion
    X = pd.concat([X.drop(columns=categorical_columns), X_encoded_df], axis=1)

    # Encode the target vector y if it is categorical
    if y.dtype in ['object', 'category']:
        # Multiclass target: more than two unique labels
        if len(y.unique()) > 2:
            # One-hot encode y, producing one column per class
            encoder_y = OneHotEncoder(sparse_output=False)
            y_encoded = encoder_y.fit_transform(y.values.reshape(-1, 1))
            encoded_columns_y = encoder_y.get_feature_names_out([TARGET_COL])
            # Convert back to DataFrame for consistent indexing
            y = pd.DataFrame(y_encoded, columns=encoded_columns_y, index=y.index)
        else:
            # Binary target: encode as 0/1 with LabelEncoder
            encoder_y = LabelEncoder()
            y = pd.Series(encoder_y.fit_transform(y), index=y.index)
    ```

- **Missing-value** strategies: such as mean/median imputation, learned models, or flagging with a “missing” indicator—ensure that the absence of a value doesn’t silently skew results or shrink the training set. Proper handling keeps useful rows in play, reduces variance, and helps the model learn any predictive signal hidden in the pattern of missingness itself. The logic implemented in our example is:

1. Percentage of Missing Values: <br>
→ 0%: No action needed <br>
→ \>70%: Drop the column entirely <br>
→ 5-70%: Create missing indicator + imputation <br>
→ <5%: Simple imputation <br>

2. Data Type:
- Numerical Columns: <br>
    → Imputation using median values <br>
    → Missing indicators for columns with >5% missing <br>

- Categorical Columns: <br>
    → Imputation using mode (most frequent value) for low missingness <br>
    → Imputation with 'Missing' category for higher missingness <br>
    → Missing indicators for columns with >5% missing <br>

    ```python
    # Summarize the amount of missing data in every feature
    missing_stats = pd.DataFrame({
        # Total count of NaNs per column
        'Total Missing': X.isnull().sum(),
        # Percentage of NaNs relative to the full data set
        'Percent Missing': (X.isnull().sum() / len(X) * 100).round(2)
    })

    # Display only the columns that actually contain missing values, ordered by % missing
    missing_stats[missing_stats['Total Missing'] > 0].sort_values(
        'Percent Missing',
        ascending=False
    )

    # Decide how to handle each column’s missing data
    if len(missing_stats) > 0:
        print("Missing Value Statistics:")
        print(missing_stats)
        print("\nShape before handling missing values:", X.shape)
        
        # Iterate over every column in the feature matrix
        for column in X.columns:
            # Percentage of missing values in this column
            missing_pct = (X[column].isnull().sum() / len(X)) * 100
            
            # Skip columns with no missing data
            if missing_pct == 0:
                continue
            
            # Drop columns with > 70 % missing
            elif missing_pct > 70:
                X = X.drop(columns=[column])
                print(f"\nDropped column '{column}' with {missing_pct:.1f}% missing values")
            
            # Numeric columns
            elif pd.api.types.is_numeric_dtype(X[column]):
                if missing_pct < 5:
                    # Low missingness → simple median imputation
                    X[column] = X[column].fillna(X[column].median())
                    print(f"\nImputed '{column}' with median")
                else:
                    # Higher missingness → add flag then median-impute
                    X[f'{column}_missing'] = X[column].isnull().astype(int)
                    X[column] = X[column].fillna(X[column].median())
                    print(f"\nCreated missing indicator and imputed '{column}' with median")
            
            # Categorical / object columns
            else:
                if missing_pct < 5:
                    # Low missingness → fill with the mode (most frequent category)
                    X[column] = X[column].fillna(X[column].mode()[0])
                    print(f"\nImputed '{column}' with mode")
                else:
                    # Higher missingness → add flag then replace NaNs with explicit 'Missing'
                    X[f'{column}_missing'] = X[column].isnull().astype(int)
                    X[column] = X[column].fillna('Missing')
                    print(f"\nCreated missing indicator and imputed '{column}' with 'Missing' category")
        
        print("\nShape after handling missing values:", X.shape)
        
    else:
        print("No missing values found in the dataset.")

    # Final sanity check: there should be zero missing values left
    assert X.isnull().sum().sum() == 0, "Missing values still present in the dataset"
    ```

## 6️⃣ Model and Parameters definition

With cleansed, fully numeric data in hand, the next milestone is choosing a learning algorithm and specifying how its behaviour should be searched and optimised. Whether you end up with a tree ensemble, a gradient-boosted model, a linear classifier, or a neural network, the workflow—and the reasons behind it—follow the same pattern.

- **Instantiate an estimator**: In scikit-learn (and most modern ML libraries) every algorithm is wrapped in a class whose constructor exposes a set of keyword arguments. Instantiating that class—e.g. SomeEstimator(**kwargs)—creates a model object with behaviour but no knowledge of your data yet.
- **Specify hyper-parameters**: Hyper-parameters are settings the training procedure cannot learn on its own; they steer how the algorithm searches for patterns.

**Grid search** is a systematic way to tune a model’s hyper-parameters: you define a “grid” of possible values for each knob, then the procedure trains and validates the model on every combination in that grid—usually with cross-validation—to measure performance. By exhaustively scoring these configurations, grid search pinpoints the hyper-parameter set that yields the best average metric, giving you an evidence-based choice instead of guesswork.

Here some examples of model and hyper-parameter grids:

```python
model = RandomForestClassifier(random_state=RANDOM_SEED)

grid_params = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, 20],
}
```

```python
model = SVC(random_state=RANDOM_SEED)

grid_params = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf', 'poly'],
    'model__gamma': ['scale', 'auto'],
}
```

```python
model = MLPClassifier(random_state=RANDOM_SEED)

grid_params = {
    'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam', 'sgd'],
    'model__alpha': [0.0001, 0.001, 0.01],
}
```

```python
model = LinearRegression()

grid_params = {
    'model__fit_intercept': [True, False],
    'model__normalize': [True, False]
}
```

```python
model = SVR()

grid_params = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf', 'poly'],
    'model__gamma': ['scale', 'auto'],
}
```

## 7️⃣ Building Pipelines

A pipeline is a conveyor belt that feeds your raw features through a fixed sequence of transformations and finally into an estimator.
Wrapping the whole journey in a single scikit-learn Pipeline object delivers cleaner code and deplyment portability

Inside the pipeline, every step is an independent transformer and you can build them as you prefere. A good structure is:

- **Feature scaling**: Puts numerical variables on a common scale so that algorithms that rely on distance or gradient magnitude don’t get dominated by large-unit features.
- **Dimensionality reduction**: Shrinks the feature space, improving interpretability and often boosting generalisation by cutting noise. Two broad flavours are Feature Selection, based on different statistical tests, or Principal Component Analysis.
- **Model**: the predefined model to investigate.

Here two pipeline examples:

```python
# Pipeline 1:  Standardise → Select top-K features → Train model
NUM_SELECTED_FEATURES = 2   # keep only the two most informative predictors

pipeline = Pipeline([
    # Feature scaling: centre each numeric column at μ=0 and scale to σ=1
    ('scaler', StandardScaler()),
    
    # Univariate feature selection:
    # - f_classif = ANOVA F-test (categorical target vs continuous feature)
    # - k         = how many best-scoring features to retain
    ('feature_selection', SelectKBest(f_classif, k=NUM_SELECTED_FEATURES)),
    
    # Estimator placeholder (any classifier/regressor assigned to `model`)
    ('model', model)
])
```

```python
# Pipeline 2:  Standardise → PCA dimensionality reduction → Model
VARIANCE = 0.95   # retain 95 % of the original variance in the principal components

pipeline = Pipeline([
    # Feature scaling (required before PCA to give all dimensions equal weight)
    ('scaler', StandardScaler()),
    
    # Principal Component Analysis:
    # - n_components = 0.95 → keep enough components to explain 95 % variance
    ('pca', PCA(n_components=VARIANCE)),
    
    # Estimator (supplied via the variable `model`)
    ('model', model)
])
```

## 8️⃣ Training & Cross-Validation

Before we let the model see any data, we first **ring-fence a test set** that will stay completely untouched until final evaluation.  The call to `train_test_split` does this in a single line:

- `train_size=TRAIN_SIZE` – controls the proportion of rows that go into training (e.g. 80 %).  
- `stratify=y` – keeps the class distribution identical in both splits, a must-have when classes are imbalanced.  
- `random_state=RANDOM_SEED` – guarantees the same split every run for reproducibility.

With the dataset split, we launch a **grid search with cross-validation** (`GridSearchCV`) around the whole preprocessing-plus-model pipeline:

| Parameter | Description |
|---------|---------|
| `estimator=pipeline` | The complete chain (scaling → dimension reduction → estimator) is evaluated as a single unit, preventing data leakage. |
| `param_grid=grid_params` | The hyper-parameter combinations we defined earlier. |
| `cv=NUM_FOLD` | Number of folds; each fold acts once as validation while the others form the training mini-set. |
| `scoring='accuracy'` or `'neg_mean_squared_error'` | Metric used to rank the parameter sets for classification or regression problems respectively. |
| `n_jobs=-1` | Uses every CPU core to run folds in parallel. |


Then, the `grid_search.fit(X_train, y_train)` call:
1. Splits the training portion into *k* folds.  
2. Fits the pipeline on *k – 1* folds and scores on the hold-out fold for each hyper-parameter set.  
3. Repeats until every fold has served as validation, then averages the scores.  
4. Returns `best_estimator_`, the pipeline trained with the parameter combo that produced the highest mean accuracy.

We capture the full CV table with `grid_search.cv_results_` to inspect:

* `mean_test_score` – average accuracy across folds.  
* `std_test_score` – variability; a high standard deviation warns that performance is unstable across splits.

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TRAIN_SIZE = 0.8   # 80 % of data for training, 20 % held out for testing
NUM_FOLD   = 5     # k-fold cross-validation

# Stratified train/test split so each class keeps the same proportion
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size   = TRAIN_SIZE,
    stratify     = y,
    random_state = RANDOM_SEED
)

# Grid-search object that will tune the hyper-parameters in `grid_params`
grid_search = GridSearchCV(
    estimator = pipeline,       # full preprocessing+model chain
    param_grid = grid_params,   # dictionary of parameter ranges
    scoring = 'accuracy',       # or 'neg_mean_squared_error' for regression tasks 
    cv = NUM_FOLD,              # number of CV folds
    n_jobs = -1                 # run folds in parallel on all CPU cores
)

# Run the exhaustive search across folds
grid_search.fit(X_train, y_train)

# Pipeline retrained on the full CV training split with the best parameters
best_pipeline = grid_search.best_estimator_

# Collect mean and std of test scores for every parameter set
cv_results      = pd.DataFrame(grid_search.cv_results_)
mean_cv_scores  = cv_results['mean_test_score']
std_cv_scores   = cv_results['std_test_score']
```

We can inspect the results for each fold based on the task:
- Classification:
```python
print("Cross-Validation Results:")
for i, (mean, std) in enumerate(zip(mean_cv_scores, std_cv_scores)):
    print(f"Parameter set {i+1}: Mean Accuracy = {mean:.4f}, Std = {std:.4f}")
```
- Regression:
```python
print("Cross-Validation Results:")
for i, (mean, std) in enumerate(zip(mean_cv_scores, std_cv_scores)):
    print(f"Parameter set {i+1}: Mean MSE = {mean:.4f}, Std = {std:.4f}")
```

## 9️⃣ Evaluation

After hyper-parameter tuning we lock the winning pipeline and run it once on the held-out test set—data that played no part in training or cross-validation. This single pass simulates real-world deployment: features go through the exact same preprocessing, the estimator produces predictions, and we compare them to the true labels.

- For classification we inspect overall accuracy, a per-class precision/recall/F1 report, and a confusion matrix that spotlights where the model confuses classes.

    ```python
    # Generate predictions for the unseen test features
    y_pred = best_pipeline.predict(X_test)

    # Compute overall accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Detailed precision / recall / F1 per class
    report = classification_report(y_test, y_pred)

    # Confusion matrix counts true vs. predicted labels
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Visualise the confusion matrix for easier interrogation
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                xticklabels=[f"Class {i}" for i in range(len(conf_matrix))],
                yticklabels=[f"Class {i}" for i in range(len(conf_matrix))])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    ```

- For regression we look at error magnitudes (MSE & MAE) and the proportion of variance explained (R²).
These metrics provide an honest snapshot of generalisation performance and guide any final tweaks or business decisions.

    ```python
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Predict numeric targets for the test set
    y_pred = best_pipeline.predict(X_test)

    # Mean squared error penalises large errors more heavily
    mse = mean_squared_error(y_test, y_pred)

    # Mean absolute error gives a more interpretable “average mistake”
    mae = mean_absolute_error(y_test, y_pred)

    # R² indicates the fraction of variance the model explains
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest Mean Squared Error (MSE): {mse:.4f}")
    print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Test R² Score: {r2:.4f}")
    ```

Go check [this repository](https://github.com/GiulioRusso/Machine-Learning-boilerplate) for a Machine Learning boilerplate project.
