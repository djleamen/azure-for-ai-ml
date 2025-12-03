"""
Practice activity: Implementing automated alerts and remediation
From Microsoft Azure for AI and Machine Learning by Microsoft on Coursera
"""

import time
import random
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient

# Connect to Azure Logs Query Client (for monitoring)
credential = DefaultAzureCredential()
client = LogsQueryClient(credential)

# Define alert conditions
alert_conditions = {
    "metric_name": "response_time",
    "threshold": 200,
    "operator": "GreaterThan",
    "alert_action": "EmailNotification"
}
print("Alert set up for response time exceeding 200 ms.")


# Simulate response time metric
RESPONSE_TIME = 200  # Normal response time in milliseconds
THRESHOLD = 300  # Alert threshold in milliseconds

# Simulate an increase in response time
# Add random delay to exceed the threshold
response_time = RESPONSE_TIME + random.randint(100, 200)

# Check if the response time exceeds the threshold
if response_time > THRESHOLD:
    print(
        f"Alert: Response time exceeded! Current response time: {RESPONSE_TIME} ms")
    # Trigger notification
    print("Notification sent: Response time alert.")
    # Placeholder for initiating remediation (e.g., scaling up resources)
    print("Initiating remediation: Scaling up resources.")

# Simulate model accuracy metric
MODEL_ACCURACY = 0.85  # Normal accuracy
THRESHOLD_ACCURACY = 0.80  # Minimum acceptable accuracy

# Simulate a drop in accuracy
# Decrease accuracy below the threshold
MODEL_ACCURACY -= random.uniform(0.1, 0.15)
# Check if the model accuracy drops below the threshold
if MODEL_ACCURACY < THRESHOLD_ACCURACY:
    print(
        f"Alert: Model accuracy dropped! Current accuracy: {MODEL_ACCURACY:.2f}")
    # Trigger notification
    print("Notification sent: Model accuracy alert.")
    # Placeholder for initiating remediation (e.g., retraining the model)
    print("Initiating remediation: Retraining the model.")
