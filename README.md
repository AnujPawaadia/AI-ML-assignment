# README

## How to Run the Code

This guide explains how to set up your environment and execute the provided Python code for analyzing event data, defining churn, engineering features, training a machine-learning model, and generating actionable insights.

---

### Prerequisites

#### Python Libraries
Ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `google.colab` (for Google Colab environments)
- `docx` (if creating reports in Word format)

You can install missing libraries using pip:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

### Step-by-Step Instructions

#### 1. Mount Google Drive

The dataset should be stored in your Google Drive. Update the file path in the code to match the location of your dataset.

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 2. Load the Dataset

Replace `data_path` with the correct path to your dataset in Google Drive:

```python
data_path = '/content/drive/MyDrive/Colab Notebooks/events.csv'
df = pd.read_csv(data_path)
```

#### 3. Data Cleaning

The `load_and_clean_data` function performs the following tasks:
- Removes missing values and duplicates.
- Ensures correct data types for analysis.

Call this function with your dataset:
```python
data = load_and_clean_data('events.csv')
```

#### 4. Perform Exploratory Data Analysis (EDA)

Visualize the dataset using the `exploratory_data_analysis` function:
```python
exploratory_data_analysis(data)
```

This generates plots for:
- Event type distribution.
- Daily trends.
- Popular brands and categories.

#### 5. Define Churn

Churn is defined as users who have not made a purchase in the last 30 days. Use the `define_churn` function to identify churned users:
```python
data = define_churn(data)
```

#### 6. Feature Engineering

Generate features for modeling using the `feature_engineering` function:
```python
features = feature_engineering(data)
```

Features include:
- Total events per user.
- Total spend per user.
- Average time between events.
- Churn labels.

#### 7. Train the Model

Train a Random Forest Classifier using the `train_model` function:
```python
model = train_model(features)
```

This function:
- Splits the dataset into training and testing sets.
- Encodes categorical variables if necessary.
- Evaluates the model using a confusion matrix and classification report.

#### 8. Interpret the Model

Visualize feature importance using the `model_interpretation` function:
```python
model_interpretation(model, features)
```

#### 9. Recommendations

Generate actionable insights with the `recommendations` function:
```python
recommendations()
```

#### 10. Exporting Results (Optional)

If you want to export a Word document report, ensure you have the `docx` library installed. Save your outputs using:
```python
from docx import Document
```

---

### Notes

1. **Dataset**:
   - Ensure your dataset has the required columns, such as `event_time`, `event_type`, `brand`, `category_code`, `price`, and `user_id`.

2. **Google Colab**:
   - If running in Google Colab, remember to mount your Drive and set the correct file paths.

3. **Custom Adjustments**:
   - Modify file paths, column names, or thresholds as needed to suit your dataset.

---

### Output

The outputs include:
- Data visualizations.
- A trained machine-learning model.
- Insights and recommendations for user engagement and retention.

For any issues, ensure your environment is correctly set up with the necessary libraries and the dataset is formatted properly.
