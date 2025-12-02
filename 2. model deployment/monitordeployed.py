"""
Practice activity: Monitoring deployed models
From Microsoft Azure for AI and Machine Learning by Microsoft on Coursera
"""

import logging
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

# Sample incoming data
incoming_data = pd.DataFrame(
    {'feature1': [1.4, 1.6, 1.8], 'feature2': [3.3, 3.9, 4.2]})

# Training data metrics
training_mean = {'feature1': 1.5, 'feature2': 3.7}
training_std = {'feature1': 0.2, 'feature2': 0.3}

# Calculate statistics for incoming data
incoming_mean = incoming_data.mean()

# Compare data to check for drift
for feature in incoming_data.columns:
    if abs(incoming_mean[feature] - training_mean[feature]) > training_std[feature] * 3:
        print(f"Alert: Significant data deviation detected in {feature}")

# Alert example (can be integrated with Azure Monitor)
alert_condition = {
    "threshold": 85,
    "metric": "accuracy",
    "operator": "LessThan",
    "alert_action": "EmailNotification"
}

print("Alert condition set for accuracy below 85%.")


# Training data and incoming data samples
training_data_sample = np.random.normal(1.5, 0.2, 100)
incoming_data_sample = incoming_data['feature1'].values

# Perform KS test
statistic, p_value = ks_2samp(training_data_sample, incoming_data_sample)
if p_value < 0.05:
    print("Model drift detected for feature1")


# Configure logging settings
logging.basicConfig(filename='model_performance_logs.log', level=logging.INFO)

# Log prediction request and response
input_data = {'feature1': 1.6, 'feature2': 3.8}
output_prediction = {'prediction': 0.9}

logging.info("Input: %s, Output: %s", input_data, output_prediction)
