# Perry-s-Experiment-on-Enron-Email-Dataset.

## Project Overview
This readme file demonstrates an **experiment using the Enron Email Dataset**. The main focus of the analysis is to uncover insights from the dataset, utilizing various data processing and machine learning techniques. The project is designed to explore communication patterns, detect anomalies, and potentially identify key entities within the email network.

## Dataset
The Enron Email Dataset contains a large collection of email communications from Enron employees. It is widely used for research in **natural language processing (NLP)**, **network analysis**, and **machine learning**.

**Dataset Source**: The dataset is publicly available and can be accessed [here](https://www.cs.cmu.edu/~enron/). Please download the dataset and place it in the appropriate directory to run the notebook.

We should note that in [Perry's experiment](https://ptrckprry.com/course/ssd/reading/Perr13.pdf), they analyze the dataset compiled by [Zhou et al. (2007)](https://www.cs.rpi.edu/~goldberg/publications/cleaning.pdf), comprising 21,635 messages sent between 156 employees from November 13th, 1998, to June 21st, 2002, along with the genders, seniorities, and departments of these employees. In our reconstruction, since I was unable to find the processed Enron dataset by Zhou online and received no response after emailing the original authors for the dataset, I handled the raw Enron Email Dataset according to Zhou's method in the first part. And then reconstructed Perry's experiment in the second part.
Since the dataset used is not entirely the same, the results of the experiment reconstruction may not be completely identical. However, the experimental methods employed are the same.


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
-  `hashlib`
-  `pytz`
-  `lifelines `
-  `re`


You can install these dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn networkx nltk jupyter hashlib pytz lifelines re
```

## 1. Data Preprocessing According to Zhou's Method
The following steps are taken to extract the sender and receiver information from the email data.

### Step 1: Extract Email Entities
We use regular expressions to extract the `From`, `To`, `Cc`, and `Bcc` fields from the email messages.

```python

# import dataset
emails_df = pd.read_csv('/...csv')
emails_df.head(5)


# Extract email entities like From, To, Cc, Bcc from the message text
def extract_email_entities(message):
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

emails_df['entities'] = emails_df['message'].apply(extract_email_entities)
```
### Step 2: Standardize Email Format
After extracting the email addresses, we standardize them by converting them to lowercase and removing any extra spaces.

```python
# Standardize the extracted email addresses
def standardize_format(header_value):
    if header_value:
        email_matches = re.findall(r'[\w\.-]+@[\w\.-]+', header_value)
        return [email.lower().strip() for email in email_matches] if email_matches else None
    return None

emails_df['From_standardized'] = emails_df['entities'].apply(lambda x: standardize_format(x['From']))
emails_df['To_standardized'] = emails_df['entities'].apply(lambda x: standardize_format(x['To']))
```
### Step 3: Generalize Email Formats
To further process the email addresses, we remove duplicates and create a generalized list of unique emails.

```python
def generalize_formats(employee_email_list):
    if employee_email_list is None:
        return None 
    generalized = set()
    for email in employee_email_list:
        generalized.add(email)
    return list(generalized)

emails_df['generalized'] = emails_df['From_standardized'].apply(generalize_formats)
```

### Step 4: Remove Duplicate Emails
In this step, we generate a unique identifier for each email based on the recipient, sender, and message content. This allows us to remove any duplicate emails from the dataset.

```python

# Generate a unique email ID based on recipient, sender, and message content
def generate_unique_email_id(row):
    combined = f"{row['To_standardized']}{row['From_standardized']}{row['message']}"
    return hashlib.md5(combined.encode()).hexdigest()

emails_df['email_id'] = emails_df.apply(generate_unique_email_id, axis=1)

emails_df_unique = emails_df.drop_duplicates(subset='email_id')

email_count = emails_df_unique.shape[0]
print(f"number of emails: {email_count}")

print(emails_df_unique.head())

emails_df_unique = emails_df_unique.dropna(axis=1, how='all')

print(emails_df_unique.head())
```

### Step 5: Filter Emails by Date
In this step, we extract the dates from the email content, ensure they are in a consistent format, and filter the dataset based on a specific date range.

```python
from dateutil import parser
from datetime import datetime

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

emails_df_unique['date'] = emails_df_unique['message'].apply(extract_date)

emails_df_unique['date'].dropna().head()

min_date = emails_df_unique['date'].min()
max_date = emails_df_unique['date'].max()
print(f"date range: {min_date} to {max_date}")

emails_df_unique = emails_df_unique.sort_values(by='date')

# Convert the date to UTC format (or set another timezone like 'PST')
emails_df_unique['date'] = pd.to_datetime(emails_df_unique['date'], errors='coerce')

start_date = datetime(1998, 1, 1, tzinfo=pytz.UTC)
end_date = datetime(2002, 6, 23, tzinfo=pytz.UTC)

filtered_emails_df = emails_df_unique[(emails_df_unique['date'] >= start_date) & (emails_df_unique['date'] <= end_date)]

filtered_email_count = filtered_emails_df.shape[0]
print(f"number of filtered emails: {filtered_email_count}")

filtered_emails_df.head()


from_emails = pd.Series([email for sublist in filtered_emails_df['From_standardized'] if sublist is not None for email in sublist])
to_emails = pd.Series([email for sublist in filtered_emails_df['To_standardized'] if sublist is not None for email in sublist])

all_employees = pd.concat([from_emails, to_emails]).unique()

unique_employee_count = len(all_employees)
print(f"the number of unique employees: {unique_employee_count}")
```

### Step 6: Random Sampling and Ensure Total Email Count is 21635
To keep the dataset as consistent as possible with the one used by Perry, I randomly sampled 21,635 messages from the cleaned dataset, specifically between November 13th, 1998, and June 21st, 2002, for the subsequent analysis.
In this step, we randomly sample emails from the filtered dataset to ensure that the total number of emails is 21635.

```python
# Randomly sample 21635 emails from the filtered dataset
sampled_emails = filtered_emails_df.sample(n=21635, random_state=42)

print(sampled_emails.head())

print(f"the size of the sampled dataset: {len(sampled_emails)}")

from_emails = pd.Series([email for sublist in sampled_emails['From_standardized'] if sublist is not None for email in sublist])
to_emails = pd.Series([email for sublist in sampled_emails['To_standardized'] if sublist is not None for email in sublist])

all_employees = pd.concat([from_emails, to_emails]).unique()

unique_employee_count = len(all_employees)
print(f"the numbers of unique employees: {unique_employee_count}")

# Save the sampled emails to a CSV file
sampled_emails.to_csv('sampled_emails.csv', index=False)
```

## 2. Reconstructing Perry's Experiment
### Step 1: Creating Dynamic Covariates
Since we don't have Employee's Information file which contains the information of the actors’ genders, departments and seniorities, we can not creat the static covariates.

```python
sampled_emails['From_standardized'] = sampled_emails['From_standardized'].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)
sampled_emails['To_standardized'] = sampled_emails['To_standardized'].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)
sampled_emails['From_standardized'] = sampled_emails['From_standardized'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
sampled_emails['To_standardized'] = sampled_emails['To_standardized'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)

# 'send'&'receive'
sampled_emails['send'] = sampled_emails.groupby(['From_standardized', 'To_standardized']).cumcount()
sampled_emails['receive'] = sampled_emails.groupby(['To_standardized', 'From_standardized']).cumcount()

# '2-send'&'2-receive'
sampled_emails_swap = sampled_emails[['From_standardized', 'To_standardized']].copy()
sampled_emails_swap.columns = ['To_standardized', 'From_standardized'] 
merged_df = pd.merge(sampled_emails, sampled_emails_swap, on=['From_standardized', 'To_standardized'], how='inner')
send_count = merged_df.groupby(['From_standardized', 'To_standardized']).size().reset_index(name='2_send')
receive_count = merged_df.groupby(['To_standardized', 'From_standardized']).size().reset_index(name='2_receive')

sampled_emails = pd.merge(sampled_emails, send_count, on=['From_standardized', 'To_standardized'], how='left')
sampled_emails = pd.merge(sampled_emails, receive_count, left_on=['From_standardized', 'To_standardized'], right_on=['To_standardized', 'From_standardized'], how='left')
sampled_emails['2_send'] = sampled_emails['2_send'].fillna(0)
sampled_emails['2_receive'] = sampled_emails['2_receive'].fillna(0)

print(sampled_emails.columns)

# 'sibling'&'cosibling'
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
import seaborn as sns
import matplotlib.pyplot as plt

sampled_emails['date'] = pd.to_datetime(sampled_emails['date'])

start_date = sampled_emails['date'].min()

sampled_emails['time'] = (sampled_emails['date'] - start_date).dt.days

sampled_emails['event'] = 1

print(sampled_emails.isnull().sum())

sampled_emails.dropna(inplace=True)

sampled_emails.fillna(0, inplace=True)

covariates = ['send', 'receive', '2_send', '2_receive', 'sibling', 'cosibling']

corr_matrix = sampled_emails[covariates].corr().abs()

# Plot a heatmap to visualize the correlations between covariates
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

cph = CoxPHFitter(penalizer=0.1)

# Fit the model on the selected covariates and time/event data
cph.fit(sampled_emails[covariates + ['time', 'event']], duration_col='time', event_col='event')

cph.print_summary()
```

### Step 3: PCA for Dimensionality Reduction
We use Principal Component Analysis (PCA) to reduce the dimensionality of the covariates and simplify the data.
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Standardize the data before PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sampled_emails[['send', 'receive', '2_send', '2_receive', 'sibling', 'cosibling']])

pca = PCA(n_components=4)  # Keep 4 principal components
principal_components = pca.fit_transform(scaled_data)

sampled_emails['PC1'] = principal_components[:, 0]
sampled_emails['PC2'] = principal_components[:, 1]
sampled_emails['PC3'] = principal_components[:, 2]
sampled_emails['PC4'] = principal_components[:, 3]
covariates = ['PC1', 'PC2', 'PC3', 'PC4']

# Fit the Cox Proportional Hazards model with the principal components
cph = CoxPHFitter()
cph.fit(sampled_emails[covariates + ['time', 'event']], duration_col='time', event_col='event')

cph.print_summary()

corr_matrix = sampled_emails[covariates].corr().abs()

# Plot a heatmap to visualize the correlations between the principal components
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
```

### Step 4: Bootstrap Bias Correction
In this step, we perform Bootstrap Bias Correction using resampling to estimate the variability of the Cox model parameters.
```python
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

sampled_emails['time'] = pd.to_datetime(sampled_emails['time'])
sampled_emails['time'] = (sampled_emails['time'] - pd.Timestamp("1998-11-13")).dt.total_seconds()

sampled_emails = sampled_emails.select_dtypes(include=[np.number])

scaler = StandardScaler()
scaled_sample = sampled_emails.copy()
scaled_sample[covariates] = scaler.fit_transform(scaled_sample[covariates])

bootstrap_coefs = []

for i in range(500):
    sample = resample(sampled_emails)
    cph = CoxPHFitter()
    cph.fit(sample[covariates + ['time', 'event']], duration_col='time', event_col='event')
    bootstrap_coefs.append(cph.params_)

bootstrap_coefs = pd.DataFrame(bootstrap_coefs)

mean_coefs = bootstrap_coefs.mean()
std_error = bootstrap_coefs.std()

terms = mean_coefs.index
residuals = mean_coefs / std_error

# Plot the bootstrap residuals
plt.errorbar(terms, residuals, yerr=std_error, fmt='o', capsiz=5)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Term")
plt.ylabel("Normalized Residual")
plt.title("Bootstrap Residuals")
plt.show()


X = add_constant(sampled_emails[covariates])

# Calculate the VIF to check for multicollinearity among the covariates.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)

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

pearson_residuals_squared_sum = np.sum(pearson_residuals ** 2)
print(f"Sum of Squares of Pearson Residuals: {pearson_residuals_squared_sum}")

bootstrap_residuals = []

# Perform 500 rounds of bootstrap resampling for Pearson residuals
for i in range(500):
    bootstrap_sample = resample(sampled_emails).reset_index(drop=True)
    cph.fit(bootstrap_sample[covariates + ['time', 'event']], duration_col='time', event_col='event')
    expected_bootstrap = cph.predict_partial_hazard(bootstrap_sample)
    pearson_residuals_bootstrap = (bootstrap_sample['event'] - expected_bootstrap) / np.sqrt(expected_bootstrap)
    bootstrap_residuals.append(pearson_residuals_bootstrap)

mean_bootstrap_residuals = np.mean(bootstrap_residuals, axis=0)
print(f"Mean Bootstrap Residuals: {mean_bootstrap_residuals}")
```

### Step 5: Evaluating the Quality of the Fit
We evaluate the quality of the fit by examining the coefficients of different covariates.
```python
# Calculate the coefficients for different covariates
coefficients = cph.summary['coef']
print(coefficients)

pca_components = pd.DataFrame(pca.components_, columns=['send', 'receive', '2_send', '2_receive', 'sibling', 'cosibling'])
print(pca_components)
coefficients = cph.summary['coef']
p_values = cph.summary['p']
print(coefficients)
print(p_values)


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
