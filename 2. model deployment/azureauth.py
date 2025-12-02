"""
Practice activity: Authenticating to Azure Machine Learning
From Microsoft Azure for AI and Machine Learning by Microsoft on Coursera
"""

from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication

# Use InteractiveLoginAuthentication - will open browser for login
interactive_auth = InteractiveLoginAuthentication()
ws = Workspace.from_config(auth=interactive_auth)

print(f"Connected to workspace: {ws.name}")
print(f"Subscription ID: {ws.subscription_id}")
print(f"Resource Group: {ws.resource_group}")
print(f"Location: {ws.location}")
