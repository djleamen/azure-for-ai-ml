"""
Practice activity: Implementing preprocessing techniques
From Microsoft Azure for AI and Machine Learning by Microsoft on Coursera
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# Generate synthetic dataset for testing
np.random.seed(42)
N_SAMPLES = 100
data = {
    'income': np.random.normal(50000, 15000, N_SAMPLES),
    'credit_score': np.random.normal(650, 50, N_SAMPLES),
    'job_title': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist'], N_SAMPLES),
    'target': np.random.choice([0, 1], N_SAMPLES)
}

# Introduce missing values and outliers for testing
data['income'][np.random.randint(0, N_SAMPLES, 5)] = np.nan
data['credit_score'][np.random.randint(0, N_SAMPLES, 3)] = np.nan
data['income'][np.random.randint(0, N_SAMPLES, 2)] = 150000  # Outliers
df = pd.DataFrame(data)

# Handle missing values by filling with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median()) # type: ignore

# Remove duplicates
df = df.drop_duplicates()

df['income_log'] = np.log1p(df['income'])

scaler = StandardScaler()
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Detect and remove outliers using Z-score
z_scores = np.abs(stats.zscore(df.select_dtypes(include=['float64', 'int64']))) # type: ignore
df = df[(z_scores < 3).all(axis=1)]

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
