"""
Practice activity: Using diagnostic and monitoring tools
From Microsoft Azure for AI and Machine Learning by Microsoft on Coursera
"""
from azureml.core import Workspace
import pandas as pd

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
        print(f"Validation Error: 'membership_level' contains invalid values: {invalid_values}")
    else:
        print("All values in 'membership_level' are valid.")

# Run validation
validate_data(incoming_data)

# Connect to Azure ML Workspace
workspace = Workspace.from_config()

# Access the deployed service
service_name = 'my-model-service'
if service_name in workspace.webservices:
    service = workspace.webservices[service_name]
    
    # Enable Application Insights if not already enabled
    if not service.app_insights_enabled:
        service.update(enable_app_insights=True)
        print(f"Application Insights enabled for service: {service_name}")
    else:
        print(f"Application Insights is already enabled for service: {service_name}")
    
    # Check the Application Insights link
    print(f"Application Insights URL: {service.scoring_uri}")
    
    # Apply canary deployment to a limited number of users
    # This is pseudocode; it shows the business logic, but isn't a full implementation
    # Note: canary_deploy() is not a real method, this is illustrative
    # canary_deployment_successful = service.canary_deploy()
    # if canary_deployment_successful:
    #     print("Canary deployment successful. Proceeding to full deployment.")
    # else:
    #     print("Canary deployment failed. Investigate issues before full deployment.")
else:
    print(f"Service '{service_name}' not found in workspace.")
    print("Available services:", list(workspace.webservices.keys()))
