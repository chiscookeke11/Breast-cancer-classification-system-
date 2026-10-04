# Breast Cancer Classification System

A Machine Learning classification project that uses **Logistic Regression** to classify breast-cancer observations as **Malignant (M)** or **Benign (B)** based on numerical measurements of cell nuclei.

The project demonstrates a complete beginner-friendly Machine Learning workflow:

- Loading a dataset
- Exploring the data
- Checking dataset dimensions and information
- Detecting missing values
- Removing unnecessary columns
- Encoding categorical target values
- Separating features and target
- Splitting data into training and testing sets
- Standardizing numerical features
- Training a Logistic Regression model
- Evaluating model performance
- Making predictions on new observations

> **Important:** This project is an educational Machine Learning project. It is **not a medical diagnostic system** and should not be used to make real-world medical decisions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Machine Learning Problem](#machine-learning-problem)
3. [Project Structure](#project-structure)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [Features in the Dataset](#features-in-the-dataset)
7. [How the System Works](#how-the-system-works)
8. [Installation](#installation)
9. [Running the Project](#running-the-project)
10. [Step-by-Step Code Explanation](#step-by-step-code-explanation)
11. [1. Importing Dependencies](#1-importing-dependencies)
12. [2. Loading the Dataset](#2-loading-the-dataset)
13. [3. Exploring the Dataset](#3-exploring-the-dataset)
14. [4. Checking Dataset Shape](#4-checking-dataset-shape)
15. [5. Checking Dataset Information](#5-checking-dataset-information)
16. [6. Checking for Missing Values](#6-checking-for-missing-values)
17. [7. Removing Unnecessary Columns](#7-removing-unnecessary-columns)
18. [8. Encoding the Diagnosis Column](#8-encoding-the-diagnosis-column)
19. [9. Checking Class Distribution](#9-checking-class-distribution)
20. [10. Statistical Description](#10-statistical-description)
21. [11. Separating Features and Target](#11-separating-features-and-target)
22. [12. Splitting the Dataset](#12-splitting-the-dataset)
23. [13. Scaling the Dataset](#13-scaling-the-dataset)
24. [14. Creating the Logistic Regression Model](#14-creating-the-logistic-regression-model)
25. [15. Training the Model](#15-training-the-model)
26. [16. Evaluating the Training Data](#16-evaluating-the-training-data)
27. [17. Evaluating the Testing Data](#17-evaluating-the-testing-data)
28. [18. Building the Prediction System](#18-building-the-prediction-system)
29. [Understanding the Prediction Output](#understanding-the-prediction-output)
30. [Model Performance](#model-performance)
31. [Complete Machine Learning Pipeline](#complete-machine-learning-pipeline)
32. [Important Implementation Note](#important-implementation-note)
33. [Possible Improvements](#possible-improvements)
34. [Learning Objectives](#learning-objectives)
35. [Disclaimer](#disclaimer)
36. [Author](#author)

---

# Project Overview

The goal of this project is to build a binary classification model capable of predicting whether a breast-cancer observation is **malignant** or **benign**.

The project uses the following general workflow:

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Encoding
   ↓
Feature / Target Separation
   ↓
Train / Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Training
   ↓
Predictions
   ↓
Accuracy Evaluation
```

The model receives numerical measurements describing characteristics of cell nuclei and learns patterns associated with the two diagnosis classes.

---

# Machine Learning Problem

This is a **supervised binary classification problem**.

The model has:

### Input

30 numerical features describing characteristics of cell nuclei.

### Output

One of two classes:

```text
0 → Malignant
1 → Benign
```

The model therefore learns:

```text
30 numerical features
        ↓
Logistic Regression
        ↓
0 or 1
```

---

# Project Structure

The current repository has the following structure:

```text
Breast-cancer-classification-system-/
│
├── README.md
│
├── breast_cancer_classification.ipynb
│
└── sample_data/
    └── data.csv
```

### `README.md`

This documentation file.

### `breast_cancer_classification.ipynb`

The main Jupyter Notebook containing:

- Data loading
- Data preprocessing
- Exploratory analysis
- Feature engineering/preparation
- Model training
- Model evaluation
- Prediction

### `sample_data/data.csv`

The dataset used by the notebook.

---

# Technologies Used

The project is implemented in Python.

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Jupyter Notebook | Interactive development environment |
| Logistic Regression | Classification algorithm |
| StandardScaler | Feature scaling |
| Accuracy Score | Model evaluation |

---

# Dataset

The dataset loaded by the project contains:

```text
569 rows
33 columns
```

Initially, the dataset contains:

- `id`
- `diagnosis`
- 30 numerical predictive features
- `Unnamed: 32`

After preprocessing:

```text
569 observations
31 columns
```

After separating the target from the features:

```text
X → 569 × 30
Y → 569 × 1
```

---

# Features in the Dataset

The dataset contains measurements grouped into three categories:

### Mean measurements

Examples:

```text
radius_mean
texture_mean
perimeter_mean
area_mean
smoothness_mean
compactness_mean
concavity_mean
concave points_mean
symmetry_mean
fractal_dimension_mean
```

### Standard error measurements

Examples:

```text
radius_se
texture_se
perimeter_se
area_se
smoothness_se
compactness_se
concavity_se
concave points_se
symmetry_se
fractal_dimension_se
```

### Worst measurements

Examples:

```text
radius_worst
texture_worst
perimeter_worst
area_worst
smoothness_worst
compactness_worst
concavity_worst
concave points_worst
symmetry_worst
fractal_dimension_worst
```

These numerical measurements become the model's input features.

---

# How the System Works

The complete system can be represented as:

```text
data.csv
   │
   ▼
Pandas DataFrame
   │
   ▼
Inspect Dataset
   │
   ├── head()
   ├── tail()
   ├── shape
   ├── info()
   ├── isnull()
   └── describe()
   │
   ▼
Remove Unnecessary Columns
   │
   ▼
Encode Diagnosis
   │
   ├── M → 0
   └── B → 1
   │
   ▼
Separate Features and Target
   │
   ├── X = features
   └── Y = diagnosis
   │
   ▼
Train/Test Split
   │
   ├── Training → 455 samples
   └── Testing → 114 samples
   │
   ▼
StandardScaler
   │
   ▼
Logistic Regression
   │
   ▼
Predictions
   │
   ▼
Accuracy
```

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
```

Move into the project:

```bash
cd Breast-cancer-classification-system-
```

---

## 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn jupyter
```

---

# Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
breast_cancer_classification.ipynb
```

Run the notebook cells from top to bottom.

The dataset is expected at:

```text
./sample_data/data.csv
```

Therefore, make sure the notebook is executed from the project root.

---

# Step-by-Step Code Explanation

This section explains the actual code used in the notebook.

---

# 1. Importing Dependencies

The notebook starts with:

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
```

Let's break this down line by line.

---

## `import pandas as pd`

```python
import pandas as pd
```

This imports the Pandas library.

`pandas` is primarily used for:

- Reading datasets
- Working with tables
- Cleaning data
- Selecting columns
- Performing statistical analysis

The alias:

```python
pd
```

allows us to write:

```python
pd.read_csv()
```

instead of:

```python
pandas.read_csv()
```

---

## `import numpy as np`

```python
import numpy as np
```

NumPy is used for numerical operations.

The project later uses NumPy to convert the manually supplied input data into an array:

```python
np.asarray(input_data)
```

The alias:

```python
np
```

is the conventional NumPy abbreviation.

---

## Importing `train_test_split`

```python
from sklearn.model_selection import train_test_split
```

This imports Scikit-learn's dataset splitting function.

It is used to divide the dataset into:

```text
Training data
Testing data
```

The model learns from the training data and is evaluated using unseen testing data.

---

## Importing Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
```

This imports the Logistic Regression classification algorithm.

Despite its name, Logistic Regression is commonly used for classification problems.

In this project it performs:

```text
Features → Malignant or Benign
```

---

## Importing `accuracy_score`

```python
from sklearn.metrics import accuracy_score
```

This imports a metric used to calculate classification accuracy.

The formula is:

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

---

## Importing `StandardScaler`

```python
from sklearn.preprocessing import StandardScaler
```

This imports Scikit-learn's standardization tool.

It transforms features so that they have approximately:

```text
mean = 0
standard deviation = 1
```

This is especially useful when features have very different numerical scales.

For example:

```text
area_mean       → hundreds/thousands
smoothness_mean → decimals
```

Without scaling, large numerical ranges can disproportionately affect some algorithms.

---

# 2. Loading the Dataset

The notebook uses:

```python
cancer_data = pd.read_csv("./sample_data/data.csv")
```

### `pd.read_csv()`

Reads the CSV file.

### `"./sample_data/data.csv"`

Specifies the location of the dataset.

### `cancer_data`

Stores the resulting Pandas DataFrame.

Conceptually:

```text
CSV file
   ↓
pd.read_csv()
   ↓
Pandas DataFrame
   ↓
cancer_data
```

---

# 3. Exploring the Dataset

The notebook first examines the first five records:

```python
cancer_data.head()
```

`head()` returns the first five rows by default.

This is useful for checking:

- Column names
- Data types visually
- Dataset values
- Target format
- Potential problems

The first rows show values such as:

```text
diagnosis = M
radius_mean = 17.99
texture_mean = 10.38
...
```

---

## Checking the Last Five Rows

The notebook also uses:

```python
cancer_data.tail()
```

`tail()` displays the final five rows.

This helps confirm that:

- The dataset loaded completely.
- The final records look reasonable.
- No obvious formatting issue exists at the end of the file.

---

# 4. Checking Dataset Shape

The notebook executes:

```python
cancer_data.shape
```

The result is:

```text
(569, 33)
```

This means:

```text
569 rows
33 columns
```

The first number represents observations.

The second number represents variables/columns.

---

# 5. Checking Dataset Information

The notebook uses:

```python
cancer_data.info()
```

This provides information about:

- Number of rows
- Number of columns
- Column names
- Non-null values
- Data types
- Memory usage

The dataset contains:

```text
569 entries
33 columns
```

Most feature columns are:

```text
float64
```

The `id` column is:

```text
int64
```

The `diagnosis` column is:

```text
object
```

The `Unnamed: 32` column contains no useful values.

---

# 6. Checking for Missing Values

The notebook checks missing values using:

```python
cancer_data.isnull().sum()
```

This calculates the number of missing values in every column.

The result shows:

```text
Unnamed: 32    569
```

while the other columns contain:

```text
0
```

missing values.

Therefore:

```text
Unnamed: 32
```

is completely empty.

This makes it unnecessary for Machine Learning.

---

# 7. Removing Unnecessary Columns

The notebook removes two columns:

```python
cancer_data.drop(
    columns=['Unnamed: 32', 'id'],
    axis=1,
    inplace=True
)
```

Let's break this down.

### `cancer_data.drop()`

Removes rows or columns from the DataFrame.

### `columns=`

Tells Pandas that we want to remove columns.

### `['Unnamed: 32', 'id']`

Specifies the columns to remove.

The project removes:

```text
Unnamed: 32
id
```

### Why remove `Unnamed: 32`?

Because all 569 values are missing.

It provides no useful information.

### Why remove `id`?

The ID identifies a record but does not represent a biological feature that the model needs for classification.

Using it as a predictive feature could introduce meaningless patterns.

### `axis=1`

Means we are operating on columns.

In Pandas:

```text
axis=0 → rows
axis=1 → columns
```

### `inplace=True`

Means the original DataFrame is modified directly.

Without `inplace=True`, we would normally assign the result back:

```python
cancer_data = cancer_data.drop(...)
```

After this operation, the dataset contains:

```text
31 columns
```

---

# 8. Encoding the Diagnosis Column

Initially, diagnosis contains:

```text
M
B
```

where:

```text
M = Malignant
B = Benign
```

Machine Learning models generally require numerical representations of target classes.

The notebook therefore performs:

```python
cancer_data["diagnosis"] = cancer_data["diagnosis"].replace({
    "M": 0,
    "B": 1
})
```

---

## Line-by-line

### Select the diagnosis column

```python
cancer_data["diagnosis"]
```

This selects the target column.

### `.replace()`

```python
.replace(...)
```

replaces existing values with new values.

### Mapping

```python
{
    "M": 0,
    "B": 1
}
```

means:

```text
Malignant → 0
Benign    → 1
```

After encoding, the model works with:

```text
0 = Malignant
1 = Benign
```

---

# 9. Checking Class Distribution

The notebook uses:

```python
cancer_data["diagnosis"].value_counts()
```

The result is:

```text
1    357
0    212
```

Therefore:

```text
Benign    → 357
Malignant → 212
```

Total:

```text
357 + 212 = 569
```

This also confirms that every observation belongs to one of the two classes.

The classes are not perfectly balanced, but both classes have substantial representation.

---

# 10. Statistical Description

The notebook uses:

```python
cancer_data.describe()
```

This generates descriptive statistics such as:

```text
count
mean
std
min
25%
50%
75%
max
```

For example, the `radius_mean` feature has:

```text
mean ≈ 14.13
std  ≈ 3.52
min  ≈ 6.98
max  ≈ 28.11
```

These statistics help us understand:

- Feature ranges
- Central tendencies
- Variation
- Potential outliers
- Differences in scale

The feature ranges vary significantly.

For example:

```text
smoothness_mean → small decimal values
area_mean       → hundreds/thousands
```

This becomes important when training Logistic Regression, which is why the project later applies feature scaling.

---

# 11. Separating Features and Target

The notebook creates:

```python
X = cancer_data.drop(columns="diagnosis", axis=1)
Y = cancer_data["diagnosis"]
```

This is one of the most important Machine Learning steps.

---

## Creating `X`

```python
X = cancer_data.drop(columns="diagnosis", axis=1)
```

We remove the target column from the dataset.

Everything remaining becomes a feature.

Therefore:

```text
X = input features
```

There are:

```text
30 features
```

---

## Creating `Y`

```python
Y = cancer_data["diagnosis"]
```

The diagnosis column becomes the target.

Therefore:

```text
Y = target
```

The relationship is:

```text
X → 30 numerical features

Y → 0 or 1
```

---

# 12. Splitting the Dataset

The notebook uses:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)
```

This divides the dataset into training and testing sets.

---

## `X`

The feature dataset.

---

## `Y`

The target labels.

---

## `test_size=0.2`

This means:

```text
20% → testing
80% → training
```

Since the dataset contains 569 records:

```text
80% ≈ 455 records
20% ≈ 114 records
```

The notebook confirms:

```text
X.shape       → (569, 30)
X_train.shape → (455, 30)
X_test.shape  → (114, 30)
```

---

## `random_state=2`

```python
random_state=2
```

ensures reproducibility.

Without a fixed random state, running the notebook multiple times could produce different train/test splits.

With:

```python
random_state=2
```

the same split is generated each time.

---

# 13. Scaling the Dataset

The project uses:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

This is a very important part of the project.

---

## Creating the scaler

```python
scaler = StandardScaler()
```

This creates a standardization object.

---

## Fitting and transforming training data

```python
X_train_scaled = scaler.fit_transform(X_train)
```

This performs two operations.

### `fit()`

The scaler calculates statistics from the training data.

Primarily:

```text
mean
standard deviation
```

### `transform()`

The scaler uses those statistics to standardize the training data.

The resulting data is stored in:

```text
X_train_scaled
```

---

## Transforming testing data

```python
X_test_scaled = scaler.transform(X_test)
```

Notice that this uses:

```python
transform()
```

rather than:

```python
fit_transform()
```

This is intentional.

The scaler must learn its scaling parameters only from the training data.

The test data should then be transformed using the same parameters.

This prevents information from the test set from leaking into the training process.

The flow is:

```text
X_train
   ↓
fit + transform
   ↓
X_train_scaled


X_test
   ↓
transform using training statistics
   ↓
X_test_scaled
```

---

# 14. Creating the Logistic Regression Model

The notebook creates the model:

```python
model = LogisticRegression(max_iter=1000)
```

This creates a Logistic Regression classifier.

---

## Why Logistic Regression?

The problem has two possible classes:

```text
Malignant
Benign
```

This makes Logistic Regression suitable as a baseline binary classification algorithm.

Logistic Regression estimates the probability of belonging to a class and then assigns the observation to a class based on the learned decision boundary.

---

## `max_iter=1000`

```python
max_iter=1000
```

controls the maximum number of optimization iterations.

The model needs to find appropriate coefficients during training.

Increasing `max_iter` gives the optimizer more opportunities to converge.

This can be important when working with datasets where the default iteration limit is insufficient.

---

# 15. Training the Model

The actual training operation is:

```python
model.fit(X_train_scaled, Y_train)
```

This is where the model learns from the training data.

### `X_train_scaled`

The standardized input features.

### `Y_train`

The corresponding diagnosis labels.

Conceptually:

```text
X_train_scaled + Y_train
          ↓
    Logistic Regression
          ↓
     Learned model
```

The model learns coefficients associated with the different features.

Once training is complete, `model` contains the learned parameters.

---

# 16. Evaluating the Training Data

The notebook predicts the training labels:

```python
training_data_prediction = model.predict(X_train_scaled)
```

The model receives the training features and generates predictions.

The predictions are stored in:

```text
training_data_prediction
```

---

## Calculating training accuracy

```python
accuracy_score_of_training_data = accuracy_score(
    Y_train,
    training_data_prediction
)
```

The first argument:

```python
Y_train
```

contains the actual labels.

The second:

```python
training_data_prediction
```

contains the model's predicted labels.

The accuracy is then printed:

```python
print(
    "Prediction on training data",
    accuracy_score_of_training_data
)
```

The notebook produced:

```text
Prediction on training data 0.989010989010989
```

Approximately:

```text
98.90%
```

training accuracy.

---

# 17. Evaluating the Testing Data

The notebook then evaluates the model using data it did not train on.

First:

```python
test_data_prediction = model.predict(X_test_scaled)
```

The model predicts the labels of the test set.

Then:

```python
accuracy_score_of_test_data = accuracy_score(
    Y_test,
    test_data_prediction
)
```

compares the predictions against the actual test labels.

Finally:

```python
print(
    "Prediction on testing data:",
    accuracy_score_of_test_data
)
```

The notebook produced:

```text
Prediction on testing data: 0.9736842105263158
```

Approximately:

```text
97.37%
```

test accuracy.

---

# 18. Building the Prediction System

The notebook then demonstrates how to use the trained model with a new observation.

The sample input is:

```python
input_data = (
    20.29,
    14.34,
    135.1,
    1297,
    0.1003,
    0.1328,
    0.198,
    0.1043,
    0.1809,
    0.05883,
    0.7572,
    0.7813,
    5.438,
    94.44,
    0.01149,
    0.02461,
    0.05688,
    0.01885,
    0.01756,
    0.005115,
    22.54,
    16.67,
    152.2,
    1575,
    0.1374,
    0.205,
    0.4,
    0.1625,
    0.2364,
    0.07678
)
```

There are 30 values because the model expects the same 30 features used during training.

---

## Converting the tuple to NumPy

```python
input_data_as_numpy_array = np.asarray(input_data)
```

The tuple is converted into a NumPy array.

For example:

```text
tuple
  ↓
NumPy array
```

This makes it easier to manipulate the input in the format expected by Scikit-learn.

---

## Reshaping the input

The notebook uses:

```python
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
```

Why?

The model expects input shaped like:

```text
number of samples × number of features
```

For one observation with 30 features:

```text
1 × 30
```

The `1` means:

```text
one patient/observation
```

The `-1` tells NumPy to automatically calculate the required number of columns.

Therefore:

```text
(30,)
```

becomes:

```text
(1, 30)
```

---

## Making the prediction

The notebook then performs:

```python
prediction = model.predict(input_data_reshaped)
```

This sends the observation to the trained model.

The output is:

```text
[0]
```

Based on the encoding used earlier:

```text
0 → Malignant
1 → Benign
```

the output:

```text
[0]
```

corresponds to:

```text
Malignant
```

---

# Understanding the Prediction Output

The project encoded the target as:

```text
M → 0
B → 1
```

Therefore:

| Model Output | Meaning |
|---:|---|
| `0` | Malignant |
| `1` | Benign |

For example:

```python
prediction = [0]
```

means:

```text
Malignant
```

while:

```python
prediction = [1]
```

means:

```text
Benign
```

---

# Model Performance

The notebook reports the following results:

| Dataset | Accuracy |
|---|---:|
| Training | 98.90% |
| Testing | 97.37% |

The relatively small difference between training and testing accuracy is encouraging for this particular experiment.

```text
Training Accuracy ≈ 98.90%

Testing Accuracy  ≈ 97.37%
```

The difference is approximately:

```text
1.53 percentage points
```

This suggests that the model is not showing a large gap between its performance on the training and test sets.

However, **accuracy alone is not sufficient to establish that a medical classification model is clinically reliable**.

---

# Complete Machine Learning Pipeline

The entire implementation can be summarized as follows:

## Step 1 — Import libraries

```python
import pandas as pd
import numpy as np
```

Load data and perform numerical operations.

---

## Step 2 — Import Machine Learning tools

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
```

Load the tools required for:

- Dataset splitting
- Model creation
- Evaluation
- Feature scaling

---

## Step 3 — Load the dataset

```python
cancer_data = pd.read_csv("./sample_data/data.csv")
```

Read the CSV into a DataFrame.

---

## Step 4 — Inspect the data

```python
cancer_data.head()
cancer_data.tail()
cancer_data.shape
cancer_data.info()
```

Understand the dataset before modeling.

---

## Step 5 — Check missing values

```python
cancer_data.isnull().sum()
```

Identify columns containing missing values.

---

## Step 6 — Remove unnecessary columns

```python
cancer_data.drop(
    columns=["Unnamed: 32", "id"],
    axis=1,
    inplace=True
)
```

Remove columns that should not be used as predictive features.

---

## Step 7 — Encode the target

```python
cancer_data["diagnosis"] = cancer_data["diagnosis"].replace({
    "M": 0,
    "B": 1
})
```

Convert categorical labels into numbers.

---

## Step 8 — Separate features and target

```python
X = cancer_data.drop(columns="diagnosis", axis=1)
Y = cancer_data["diagnosis"]
```

Create:

```text
X → inputs
Y → output
```

---

## Step 9 — Split the data

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)
```

Create training and testing datasets.

---

## Step 10 — Scale the features

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Standardize the numerical features.

---

## Step 11 — Create the model

```python
model = LogisticRegression(max_iter=1000)
```

Create the Logistic Regression classifier.

---

## Step 12 — Train the model

```python
model.fit(X_train_scaled, Y_train)
```

Learn the relationship between features and diagnosis.

---

## Step 13 — Make training predictions

```python
training_data_prediction = model.predict(X_train_scaled)
```

Predict labels for training data.

---

## Step 14 — Calculate training accuracy

```python
accuracy_score(
    Y_train,
    training_data_prediction
)
```

Measure performance on training data.

---

## Step 15 — Make test predictions

```python
test_data_prediction = model.predict(X_test_scaled)
```

Predict labels for unseen test data.

---

## Step 16 — Calculate test accuracy

```python
accuracy_score(
    Y_test,
    test_data_prediction
)
```

Measure generalization performance.

---

# Important Implementation Note

There is an important issue in the final prediction section of the current notebook.

The model is trained using:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

and:

```python
X_test_scaled = scaler.transform(X_test)
```

Therefore, the model was trained on **scaled features**.

However, the final prediction currently uses:

```python
prediction = model.predict(input_data_reshaped)
```

The manually supplied input has not been scaled.

This means the production prediction pipeline is inconsistent with the training pipeline.

The correct process should be:

```python
input_data_scaled = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_scaled)
```

The complete corrected prediction section should therefore look like:

```python
input_data = (
    20.29, 14.34, 135.1, 1297, 0.1003,
    0.1328, 0.198, 0.1043, 0.1809,
    0.05883, 0.7572, 0.7813, 5.438,
    94.44, 0.01149, 0.02461, 0.05688,
    0.01885, 0.01756, 0.005115,
    22.54, 16.67, 152.2, 1575,
    0.1374, 0.205, 0.4, 0.1625,
    0.2364, 0.07678
)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

input_data_scaled = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_scaled)

print(prediction)
```

The key addition is:

```python
input_data_scaled = scaler.transform(input_data_reshaped)
```

This ensures that new observations go through the same preprocessing procedure as the training data.

The ideal production pipeline is:

```text
Raw Input
   ↓
Reshape
   ↓
Scale using existing scaler
   ↓
Logistic Regression
   ↓
Prediction
```

not:

```text
Raw Input
   ↓
Logistic Regression
```

---

# Why Scaling Matters

Consider two features:

```text
area_mean = 1001
smoothness_mean = 0.1184
```

They have completely different numerical ranges.

Without scaling:

```text
area_mean
1001
```

is numerically much larger than:

```text
smoothness_mean
0.1184
```

Standardization converts the features to comparable scales.

The standardization formula is:

```text
z = (x - μ) / σ
```

where:

- `x` = original value
- `μ` = mean of the feature
- `σ` = standard deviation
- `z` = standardized value

This produces features with approximately:

```text
mean = 0
standard deviation = 1
```

---

# Possible Improvements

Although the project achieves high accuracy, there are several ways it can be improved.

## 1. Use a Pipeline

Instead of manually managing scaling and prediction, Scikit-learn's `Pipeline` can combine preprocessing and modeling.

Example:

```python
from sklearn.pipeline import Pipeline

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

Then:

```python
model_pipeline.fit(X_train, Y_train)
```

and:

```python
prediction = model_pipeline.predict(input_data_reshaped)
```

This makes it much harder to accidentally forget preprocessing during inference.

---

## 2. Add More Evaluation Metrics

The project currently evaluates mainly using accuracy.

For medical classification, additional metrics are important:

```text
Precision
Recall
F1-score
Confusion Matrix
ROC-AUC
Specificity
Sensitivity
```

For example:

```python
from sklearn.metrics import classification_report

print(classification_report(Y_test, test_data_prediction))
```

---

## 3. Add a Confusion Matrix

A confusion matrix can show:

```text
True Positives
True Negatives
False Positives
False Negatives
```

This is especially important because different types of classification errors can have very different consequences.

---

## 4. Use Stratified Splitting

Because the target classes are not perfectly balanced, a more robust split could use:

```python
train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2,
    stratify=Y
)
```

This helps maintain approximately the same class distribution in both training and testing sets.

---

## 5. Cross-Validation

Instead of relying on one train/test split, cross-validation could be used to obtain a more reliable estimate of model performance.

For example:

```python
from sklearn.model_selection import cross_val_score
```

This would allow the model to be evaluated across multiple data splits.

---

## 6. Compare Multiple Models

Logistic Regression is a good baseline, but other algorithms could be evaluated:

```text
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine
K-Nearest Neighbors
Gradient Boosting
```

The models could then be compared using the same evaluation metrics.

---

## 7. Hyperparameter Tuning

The project currently uses:

```python
LogisticRegression(max_iter=1000)
```

Hyperparameters such as:

```text
C
solver
penalty
max_iter
```

could be tuned using:

```text
GridSearchCV
RandomizedSearchCV
```

---

## 8. Improve the Prediction Interface

The current notebook requires users to manually provide 30 numerical values.

A future version could provide:

```text
Web interface
       ↓
Form inputs
       ↓
Validation
       ↓
Scaler
       ↓
Model
       ↓
Prediction
```

Possible technologies include:

```text
Streamlit
FastAPI
Flask
Django
Next.js + FastAPI
```

---

## 9. Save the Trained Model

The current notebook trains the model every time it is executed.

A production version could save:

```text
trained model
scaler
feature order
```

using tools such as:

```python
import joblib
```

For example:

```python
joblib.dump(model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

The saved model could then be loaded without retraining.

---

# Learning Objectives

This project demonstrates several important Machine Learning concepts.

### Data Analysis

You learn how to:

```text
Load data
Inspect data
Check shape
Check data types
Find missing values
Generate statistics
```

### Data Preprocessing

You learn:

```text
Removing unnecessary columns
Encoding categorical variables
Feature/target separation
Feature scaling
```

### Supervised Learning

You learn the difference between:

```text
Features
Target
Training data
Testing data
```

### Classification

You implement a binary classification model using:

```text
Logistic Regression
```

### Model Evaluation

You learn how to measure:

```text
Training accuracy
Testing accuracy
```

### Prediction

You learn how to take a new observation and pass it through a trained model.

---

# Key Machine Learning Concepts Demonstrated

```text
Dataset
   ↓
Features
   ↓
Target
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Scaling
   ↓
Model
   ↓
Training
   ↓
Prediction
   ↓
Evaluation
```

This workflow is fundamental to many Machine Learning projects.

---

# Example End-to-End Code

The core workflow of the project can be represented by:

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Load dataset
cancer_data = pd.read_csv("./sample_data/data.csv")


# Remove unnecessary columns
cancer_data.drop(
    columns=["Unnamed: 32", "id"],
    axis=1,
    inplace=True
)


# Encode target
cancer_data["diagnosis"] = cancer_data["diagnosis"].replace({
    "M": 0,
    "B": 1
})


# Separate features and target
X = cancer_data.drop(
    columns="diagnosis",
    axis=1
)

Y = cancer_data["diagnosis"]


# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create model
model = LogisticRegression(max_iter=1000)


# Train model
model.fit(
    X_train_scaled,
    Y_train
)


# Training prediction
training_data_prediction = model.predict(
    X_train_scaled
)


# Training accuracy
training_accuracy = accuracy_score(
    Y_train,
    training_data_prediction
)


# Testing prediction
test_data_prediction = model.predict(
    X_test_scaled
)


# Testing accuracy
testing_accuracy = accuracy_score(
    Y_test,
    test_data_prediction
)


print(
    "Training Accuracy:",
    training_accuracy
)

print(
    "Testing Accuracy:",
    testing_accuracy
)
```

---

# Results

The current notebook produced:

```text
Training Accuracy: 0.989010989010989
Testing Accuracy: 0.9736842105263158
```

Which corresponds approximately to:

```text
Training Accuracy: 98.90%
Testing Accuracy: 97.37%
```

These results demonstrate that the Logistic Regression model performs strongly on the dataset used in this experiment.

However, these values should be interpreted as **experimental model performance on this dataset**, not as evidence of clinical diagnostic accuracy.

---

# Disclaimer

This project is intended for:

- Educational purposes
- Machine Learning practice
- Demonstrating binary classification
- Understanding data preprocessing
- Learning model evaluation

It is **not intended for medical diagnosis, treatment decisions, screening, or clinical use**.

A real-world medical Machine Learning system would require extensive validation, appropriate clinical datasets, careful handling of false negatives and false positives, external validation, calibration, bias assessment, clinical oversight, regulatory review, and deployment controls.

---

# Author

**Chinedu Okeke**

Software Engineer | AI/ML Engineer | Open Source Contributor

GitHub: `chiscookeke11`

---

# Conclusion

This project demonstrates a complete introductory Machine Learning classification workflow.

Starting from a raw CSV dataset, the project:

```text
1. Loads the data
2. Inspects the data
3. Checks its structure
4. Checks for missing values
5. Removes unnecessary columns
6. Encodes the target variable
7. Separates features and target
8. Splits the data
9. Scales the features
10. Creates a Logistic Regression model
11. Trains the model
12. Generates predictions
13. Calculates training accuracy
14. Calculates testing accuracy
15. Demonstrates prediction on a new observation
```

The project achieved approximately:

```text
98.90% training accuracy
97.37% testing accuracy
```

The most important lesson from the project is that a Machine Learning model is not just the algorithm itself. A reliable workflow also requires appropriate:

```text
Data
↓
Preprocessing
↓
Feature preparation
↓
Training
↓
Evaluation
↓
Inference
```

The same preprocessing used during training must also be applied when making predictions on new data.