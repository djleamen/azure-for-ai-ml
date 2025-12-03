"""
Practice activity: Troubleshooting a sample pipeline
From Microsoft Azure for AI and Machine Learning by Microsoft on Coursera
"""

import logging

import pandas as pd
from azureml.core import Experiment, Model, Workspace
from azureml.core.compute import AksCompute
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksWebservice
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Connect to your workspace
ws = Workspace.from_config()

# Create example experiment
EXPERIMENT_NAME = 'sample_pipeline'  # replace with your pipeline name
experiment = Experiment(ws, EXPERIMENT_NAME)

# Access the run details
for run in experiment.get_runs():
    print(f"Run ID: {run.id}, Status: {run.status}")
    print(run.get_details())


# Create and save a sample dataset
data = {
    "customer_id": [1, 2, 3, 4, 5],
    "membership_level": ["Bronze", "Silver", "Gold", "Silver", "Bronze"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
CSV_FILENAME = "customer_data.csv"
df.to_csv(CSV_FILENAME, index=False)

# Load the dataset
incoming_data = pd.read_csv(CSV_FILENAME)
# Validate the data


def validate_data(df):
    """Validate incoming data for expected schema and values."""
    # Check for null values in 'customer_id'
    if df['customer_id'].isnull().any():
        print("Validation Error: 'customer_id' column contains null values.")
    else:
        print("No null values in 'customer_id' column.")

    # Check that 'membership_level' contains only allowed values
    allowed_values = {"Bronze", "Silver", "Gold"}
    invalid_values = set(df['membership_level']) - allowed_values
    if invalid_values:
        print(
            f"Validation Error: 'membership_level' contains invalid values: {invalid_values}")
    else:
        print("All values in 'membership_level' are valid.")


# Run validation
validate_data(incoming_data)


# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a LinearRegression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Example of deploying a model to Azure Kubernetes Service (AKS)
# Note: This section is commented out as it requires a registered model in the workspace

# # Connect to your workspace
# ws = Workspace.from_config()

# # Load the model (replace 'my_model' with an actual registered model name)
# model = Model(ws, 'my_model')

# # Define inference configuration
# inference_config = InferenceConfig(
#     entry_script='score.py', environment='myenv')

# # Define deployment configuration
# aks_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# # Deploy the model
# service = Model.deploy(workspace=ws, name='my-aks-service', models=[model],
#                        inference_config=inference_config, deployment_config=aks_config,
#                        deployment_target=AksCompute(ws, 'aks-compute'))
# service.wait_for_deployment(show_output=True)


# Configure logging settings
logging.basicConfig(filename='pipeline_logs.log', level=logging.INFO)

# Log pipeline events
logging.info("Pipeline step completed successfully.")
