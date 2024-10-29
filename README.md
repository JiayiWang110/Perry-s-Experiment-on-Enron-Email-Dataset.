# Perry-s-Experiment-on-Enron-Email-Dataset.

## Project Overview
This readme file demonstrates an **experiment using the Enron Email Dataset**. The main focus of the analysis is to uncover insights from the dataset, utilizing various data processing and machine learning techniques. The project is designed to explore communication patterns, detect anomalies, and potentially identify key entities within the email network.

## Dataset
The Enron Email Dataset contains a large collection of email communications from Enron employees. It is widely used for research in **natural language processing (NLP)**, **network analysis**, and **machine learning**.

**Dataset Source**: The dataset is publicly available and can be accessed [here](https://www.cs.cmu.edu/~enron/). Please download the dataset and place it in the appropriate directory to run the notebook.

## Key Features
- **Data Preprocessing**: Cleaning, tokenization, and preparing the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Visualization of communication patterns, email traffic over time, and key employee interactions.
- **Machine Learning Models**: Training and evaluation of models to classify or predict certain patterns (e.g., spam detection, anomaly detection).
- **Network Analysis**: Constructing a communication graph and identifying central individuals or clusters in the network.

## Dependencies
To run this notebook, the following libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `networkx`
- `nltk` 
- `jupyter`

You can install these dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn networkx nltk jupyter
```

## 1. Data Preprocessing According to Zhou's Method
The following steps are taken to extract the sender and receiver information from the email data.

### Step 1: Extract Email Entities
We use regular expressions to extract the `From`, `To`, `Cc`, and `Bcc` fields from the email messages.

```python

# import dataset
emails_df = pd.read_csv('/...csv')
emails_df.head(5)
import re

# Extract email entities like From, To, Cc, Bcc from the message text
def extract_email_entities(message):
    # Extract From, To, Cc, Bcc fields using regular expressions
    from_match = re.search(r'From:\s*(.*)', message)
    to_match = re.search(r'To:\s*(.*)', message)
    cc_match = re.search(r'Cc:\s*(.*)', message)
    bcc_match = re.search(r'Bcc:\s*(.*)', message)

    return {
        'From': from_match.group(1) if from_match else None,
        'To': to_match.group(1) if to_match else None,
        'Cc': cc_match.group(1) if cc_match else None,
        'Bcc': bcc_match.group(1) if bcc_match else None
    }

# Apply the function to the dataset
emails_df['entities'] = emails_df['message'].apply(extract_email_entities)
```
### Step 2: Standardize Email Format
After extracting the email addresses, we standardize them by converting them to lowercase and removing any extra spaces.

```python
# Standardize the extracted email addresses
def standardize_format(header_value):
    # 提取邮箱地址并转换为标准格式
    if header_value:
        email_matches = re.findall(r'[\w\.-]+@[\w\.-]+', header_value)
        return [email.lower().strip() for email in email_matches] if email_matches else None
    return None

# Apply the standardization function to the 'From' and 'To' fields
emails_df['From_standardized'] = emails_df['entities'].apply(lambda x: standardize_format(x['From']))
emails_df['To_standardized'] = emails_df['entities'].apply(lambda x: standardize_format(x['To']))
```
### Step 3: Generalize Email Formats
To further process the email addresses, we remove duplicates and create a generalized list of unique emails.

```python
# Generalize email formats to remove duplicates
def generalize_formats(employee_email_list):
    if employee_email_list is None:
        return None  # 如果为空，直接返回 None
    generalized = set()
    for email in employee_email_list:
        generalized.add(email)
    return list(generalized)

# Apply the generalization function to the standardized 'From' field
emails_df['generalized'] = emails_df['From_standardized'].apply(generalize_formats)
```

### Step 4: Remove Duplicate Emails
In this step, we generate a unique identifier for each email based on the recipient, sender, and message content. This allows us to remove any duplicate emails from the dataset.

```python
import hashlib

# Generate a unique email ID based on recipient, sender, and message content
def generate_unique_email_id(row):
    # 将接收者、发件人和邮件内容组合生成唯一标识符
    combined = f"{row['To_standardized']}{row['From_standardized']}{row['message']}"
    return hashlib.md5(combined.encode()).hexdigest()

# Apply the function to generate unique email IDs
emails_df['email_id'] = emails_df.apply(generate_unique_email_id, axis=1)

# Remove duplicate emails based on the unique email ID
emails_df_unique = emails_df.drop_duplicates(subset='email_id')

# Check the number of unique emails after processing
email_count = emails_df_unique.shape[0]
print(f"处理后的邮件数量: {email_count}")

# Display the first few rows to confirm the result
print(emails_df_unique.head())

# Remove columns with all missing values
emails_df_unique = emails_df_unique.dropna(axis=1, how='all')

# Display the first few rows to confirm the result
print(emails_df_unique.head())
```

### Step 5: Filter Emails by Date
In this step, we extract the dates from the email content, ensure they are in a consistent format, and filter the dataset based on a specific date range.

```python
from dateutil import parser

# Extract date from email message
def extract_date(message):
    date_match = re.search(r'Date:\s*(.*)', message)
    if date_match:
        date_str = date_match.group(1).strip()
        try:
            return parser.parse(date_str)
        except (ValueError, TypeError):
            return None
    return None

# Apply the date extraction function
emails_df_unique['date'] = emails_df_unique['message'].apply(extract_date)

# Check for missing dates and remove them
emails_df_unique['date'].dropna().head()

# Check the date range
min_date = emails_df_unique['date'].min()
max_date = emails_df_unique['date'].max()
print(f"日期范围: {min_date} 到 {max_date}")

# Sort the dataset by date
emails_df_unique = emails_df_unique.sort_values(by='date')

import pytz
from datetime import datetime

# Convert the date to UTC format (or set another timezone like 'PST')
emails_df_unique['date'] = pd.to_datetime(emails_df_unique['date'], errors='coerce')

# Set the start and end dates in UTC
start_date = datetime(1998, 1, 1, tzinfo=pytz.UTC)
end_date = datetime(2002, 6, 23, tzinfo=pytz.UTC)

# Filter the emails within the date range
filtered_emails_df = emails_df_unique[(emails_df_unique['date'] >= start_date) & (emails_df_unique['date'] <= end_date)]

# Check the number of filtered emails
filtered_email_count = filtered_emails_df.shape[0]
print(f"筛选后的邮件数量: {filtered_email_count}")

# Display the first few rows of the filtered dataset
filtered_emails_df.head()

import pandas as pd

# Extract emails from 'From_standardized' and 'To_standardized' columns, and remove None values
from_emails = pd.Series([email for sublist in filtered_emails_df['From_standardized'] if sublist is not None for email in sublist])
to_emails = pd.Series([email for sublist in filtered_emails_df['To_standardized'] if sublist is not None for email in sublist])

# Combine the email addresses and remove duplicates
all_employees = pd.concat([from_emails, to_emails]).unique()

# Calculate the number of unique employees
unique_employee_count = len(all_employees)
print(f"在这些邮件中涉及的独特员工数量为: {unique_employee_count}")
```

### Step 6: Random Sampling and Ensure Total Email Count is 21635
In this step, we randomly sample emails from the filtered dataset to ensure that the total number of emails is 21635.

```python
# Randomly sample 21635 emails from the filtered dataset
sampled_emails = filtered_emails_df.sample(n=21635, random_state=42)

# Display the first few rows of the sampled emails
print(sampled_emails.head())

# Check the size of the sampled dataset
print(f"抽取后的数据集大小: {len(sampled_emails)}")

# Extract emails from the 'From_standardized' and 'To_standardized' fields
from_emails = pd.Series([email for sublist in sampled_emails['From_standardized'] if sublist is not None for email in sublist])
to_emails = pd.Series([email for sublist in sampled_emails['To_standardized'] if sublist is not None for email in sublist])

# Combine the email addresses and remove duplicates
all_employees = pd.concat([from_emails, to_emails]).unique()

# Calculate the number of unique employees
unique_employee_count = len(all_employees)
print(f"在这些邮件中涉及到的独特员工数量为: {unique_employee_count}")

# Save the sampled emails to a CSV file
sampled_emails.to_csv('sampled_emails.csv', index=False)
```

## 2. Reconstructing Perry's Experiment
### Step 1: Creating Dynamic Covariates

```python
sampled_emails['From_standardized'] = sampled_emails['From_standardized'].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)
sampled_emails['To_standardized'] = sampled_emails['To_standardized'].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)
sampled_emails['From_standardized'] = sampled_emails['From_standardized'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
sampled_emails['To_standardized'] = sampled_emails['To_standardized'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
# 计算'send'和'receive'协变量
sampled_emails['send'] = sampled_emails.groupby(['From_standardized', 'To_standardized']).cumcount()
sampled_emails['receive'] = sampled_emails.groupby(['To_standardized', 'From_standardized']).cumcount()

# 计算'2-send'和'2-receive'协变量
# 创建一个辅助表格，获取所有行交换后的邮件对 (From -> To)
sampled_emails_swap = sampled_emails[['From_standardized', 'To_standardized']].copy()
sampled_emails_swap.columns = ['To_standardized', 'From_standardized']  # 交换列名以便匹配

# 将原始数据框与交换过的表格合并，找到2_send和2_receive的匹配对
merged_df = pd.merge(sampled_emails, sampled_emails_swap, on=['From_standardized', 'To_standardized'], how='inner')

# 计算2_send和2_receive的计数
send_count = merged_df.groupby(['From_standardized', 'To_standardized']).size().reset_index(name='2_send')
receive_count = merged_df.groupby(['To_standardized', 'From_standardized']).size().reset_index(name='2_receive')

# 将计数结果合并回原始数据框中
sampled_emails = pd.merge(sampled_emails, send_count, on=['From_standardized', 'To_standardized'], how='left')
sampled_emails = pd.merge(sampled_emails, receive_count, left_on=['From_standardized', 'To_standardized'], right_on=['To_standardized', 'From_standardized'], how='left')

# 填补缺失值（如果有）
sampled_emails['2_send'] = sampled_emails['2_send'].fillna(0)
sampled_emails['2_receive'] = sampled_emails['2_receive'].fillna(0)

print(sampled_emails.columns)

# 计算'sibling'和'cosibling'协变量
sampled_emails['sibling'] = sampled_emails.groupby(['From_standardized_x', 'To_standardized_x']).cumcount()
sampled_emails['cosibling'] = sampled_emails.groupby(['To_standardized_x', 'From_standardized_x']).cumcount()

sampled_emails['sibling'] = sampled_emails.apply(lambda row: sum((sampled_emails['From_standardized_x'] == row['From_standardized_x']) &
                                          (sampled_emails['To_standardized_x'] != row['To_standardized_x'])), axis=1)
sampled_emails['cosibling'] = sampled_emails.apply(lambda row: sum((sampled_emails['To_standardized_x'] == row['To_standardized_x']) &
                                           (sampled_emails['From_standardized_x'] != row['From_standardized_x'])), axis=1)
```

### Step 2: Building the Cox Proportional Hazards Model
In this step, we prepare the data and fit a Cox Proportional Hazards model to study the time-to-event data.

```python
from lifelines import CoxPHFitter

# Convert the 'date' column to datetime objects
sampled_emails['date'] = pd.to_datetime(sampled_emails['date'])

# Find the start date (minimum date) to compute time duration
start_date = sampled_emails['date'].min()

# Compute the 'time' column as the number of days from the start date
sampled_emails['time'] = (sampled_emails['date'] - start_date).dt.days

# Create an 'event' column where all rows are marked as event occurred (1)
sampled_emails['event'] = 1

# Check for missing values in the dataset
print(sampled_emails.isnull().sum())

# Drop rows with missing values
sampled_emails.dropna(inplace=True)

# Fill missing values with 0 (if necessary)
sampled_emails.fillna(0, inplace=True)

# Select covariates to include in the Cox model
covariates = ['send', 'receive', '2_send', '2_receive', 'sibling', 'cosibling']

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix for the covariates
corr_matrix = sampled_emails[covariates].corr().abs()

# Plot a heatmap to visualize the correlations between covariates
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

from lifelines import CoxPHFitter

# Create the Cox Proportional Hazards model with regularization
cph = CoxPHFitter(penalizer=0.1)

# Fit the model on the selected covariates and time/event data
cph.fit(sampled_emails[covariates + ['time', 'event']], duration_col='time', event_col='event')

# Print the model summary
cph.print_summary()
```

### Step 3: PCA for Dimensionality Reduction
We use Principal Component Analysis (PCA) to reduce the dimensionality of the covariates and simplify the data.
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data before PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sampled_emails[['send', 'receive', '2_send', '2_receive', 'sibling', 'cosibling']])
# Apply PCA to reduce the dimensionality of the covariates
pca = PCA(n_components=4)  # Keep 4 principal components
principal_components = pca.fit_transform(scaled_data)

# Add the principal components to the dataset
sampled_emails['PC1'] = principal_components[:, 0]
sampled_emails['PC2'] = principal_components[:, 1]
sampled_emails['PC3'] = principal_components[:, 2]
sampled_emails['PC4'] = principal_components[:, 3]
# Use the principal components as covariates in the model
covariates = ['PC1', 'PC2', 'PC3', 'PC4']
from lifelines import CoxPHFitter

# Fit the Cox Proportional Hazards model with the principal components
cph = CoxPHFitter()
cph.fit(sampled_emails[covariates + ['time', 'event']], duration_col='time', event_col='event')

# Print the model summary
cph.print_summary()
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix for the principal components
corr_matrix = sampled_emails[covariates].corr().abs()

# Plot a heatmap to visualize the correlations between the principal components
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
```

### Step 4: Bootstrap Bias Correction
In this step, we perform Bootstrap Bias Correction using resampling to estimate the variability of the Cox model parameters.
```python

# Convert the 'time' column to datetime and calculate time since a reference date
sampled_emails['time'] = pd.to_datetime(sampled_emails['time'])
sampled_emails['time'] = (sampled_emails['time'] - pd.Timestamp("1998-11-13")).dt.total_seconds()

# Remove non-numeric columns
sampled_emails = sampled_emails.select_dtypes(include=[np.number])

# Standardize the covariates using StandardScaler
scaler = StandardScaler()
scaled_sample = sampled_emails.copy()
scaled_sample[covariates] = scaler.fit_transform(scaled_sample[covariates])

from sklearn.utils import resample

# Initialize list to store bootstrap coefficients
bootstrap_coefs = []

# Perform 500 rounds of bootstrap resampling
for i in range(500):
    sample = resample(sampled_emails)
    cph = CoxPHFitter()
    cph.fit(sample[covariates + ['time', 'event']], duration_col='time', event_col='event')
    bootstrap_coefs.append(cph.params_)

# Convert bootstrap coefficients to a DataFrame
bootstrap_coefs = pd.DataFrame(bootstrap_coefs)

# Calculate the mean and standard error of the coefficients
mean_coefs = bootstrap_coefs.mean()
std_error = bootstrap_coefs.std()

# Calculate the residuals
terms = mean_coefs.index
residuals = mean_coefs / std_error

# Plot the bootstrap residuals
plt.errorbar(terms, residuals, yerr=std_error, fmt='o', capsiz=5)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Term")
plt.ylabel("Normalized Residual")
plt.title("Bootstrap Residuals")
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

# Add a constant to the dataset
X = add_constant(sampled_emails[covariates])

# Calculate the VIF to check for multicollinearity among the covariates.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

# Display the VIF values
print(vif)

import numpy as np

# Calculate the Pearson residuals
observed_events = sampled_emails['event']
expected_events = cph.predict_partial_hazard(sampled_emails)
pearson_residuals = (observed_events - expected_events) / np.sqrt(expected_events)

# Plot the Pearson residuals
plt.figure(figsize=(10, 6))
plt.scatter(expected_events, pearson_residuals, alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Expected Count")
plt.ylabel("Pearson Residual")
plt.title("Pearson Residuals vs Expected Count")
plt.show()

# Calculate the sum of squares of Pearson residuals
pearson_residuals_squared_sum = np.sum(pearson_residuals ** 2)
print(f"Sum of Squares of Pearson Residuals: {pearson_residuals_squared_sum}")

# Initialize list to store bootstrap residuals
bootstrap_residuals = []

# Perform 500 rounds of bootstrap resampling for Pearson residuals
for i in range(500):
    bootstrap_sample = resample(sampled_emails).reset_index(drop=True)
    cph.fit(bootstrap_sample[covariates + ['time', 'event']], duration_col='time', event_col='event')
    expected_bootstrap = cph.predict_partial_hazard(bootstrap_sample)
    pearson_residuals_bootstrap = (bootstrap_sample['event'] - expected_bootstrap) / np.sqrt(expected_bootstrap)
    bootstrap_residuals.append(pearson_residuals_bootstrap)

# Calculate the mean residuals across all bootstrap samples
mean_bootstrap_residuals = np.mean(bootstrap_residuals, axis=0)
print(f"Mean Bootstrap Residuals: {mean_bootstrap_residuals}")
```

### Step 5: Evaluating the Quality of the Fit
We evaluate the quality of the fit by examining the coefficients of different covariates.
```python
# Calculate the coefficients for different covariates
coefficients = cph.summary['coef']
print(coefficients)
# Display the PCA component matrix
pca_components = pd.DataFrame(pca.components_, columns=['send', 'receive', '2_send', '2_receive', 'sibling', 'cosibling'])
print(pca_components)
# Extract coefficients and p-values
coefficients = cph.summary['coef']
p_values = cph.summary['p']
print(coefficients)
print(p_values)
import matplotlib.pyplot as plt

# Plot the coefficients of the network effects
plt.figure(figsize=(10, 6))
coefficients.plot(kind='barh')
plt.title('Coefficients of Network Effects')
plt.xlabel('Coefficient Value')
plt.ylabel('Network Effect')
plt.show()
# Filter and display significant network effects (p < 0.05)
significant_effects = coefficients[p_values < 0.05]
print("Significant Network Effects:")
print(significant_effects)
