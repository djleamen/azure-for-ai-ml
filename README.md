# Azure for AI and Machine Learning

Practice activities from the **Microsoft Azure for AI and Machine Learning** course by Microsoft on Coursera. This repository contains hands-on scripts demonstrating end-to-end ML workflows on Azure Machine Learning, from data preparation through deployment and troubleshooting.

## Repository structure

```
azure-for-ai-ml/
├── 1. data prep and model training/
│   └── preprocessing.py             # Data cleaning, scaling, outlier handling
├── 2. model deployment/
│   ├── azureauth.py                 # Authenticate to an Azure ML workspace
│   └── monitordeployed.py           # Monitor deployed models, detect drift
├── 3. troubleshooting/
│   ├── automatedalerts.py           # Alerting and automated remediation
│   ├── diagnostictools.py           # Diagnostic and monitoring tooling
│   └── troubleshootsample.py        # Troubleshoot a sample ML pipeline
```

## Modules

### 1. Data prep and model training
Generates a synthetic dataset and demonstrates common preprocessing techniques: handling missing values, removing duplicates, log-transforming skewed features, scaling with `StandardScaler`, and detecting outliers.

### 2. Model deployment
- **`azureauth.py`** — Connects to an Azure ML workspace using `InteractiveLoginAuthentication`.
- **`monitordeployed.py`** — Compares training vs. incoming data statistics, runs a Kolmogorov–Smirnov drift test, and sketches an Azure Monitor-style alert configuration.

### 3. Troubleshooting
- **`automatedalerts.py`** — Defines alert conditions and simulates response-time and accuracy threshold breaches.
- **`diagnostictools.py`** — Validates incoming data schema/values and inspects a deployed Azure ML web service.
- **`troubleshootsample.py`** — Walks through inspecting experiment runs and diagnosing a sample pipeline.

## Prerequisites

- Python 3.9+
- An Azure subscription with an Azure Machine Learning workspace
- Azure CLI (for `DefaultAzureCredential` flows)

## Setup

```bash
# Clone the repo
git clone https://github.com/djleamen/azure-for-ai-ml.git
cd azure-for-ai-ml

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install azureml-core azure-identity azure-monitor-query \
            pandas numpy scipy scikit-learn
```

## Configuration

Update `.azureml/config.json` with your own Azure ML workspace details:

```json
{
    "subscription_id": "<your-subscription-id>",
    "resource_group": "<your-resource-group>",
    "workspace_name": "<your-workspace-name>"
}
```

## Running the scripts

```bash
python "1. data prep and model training/preprocessing.py"
python "2. model deployment/azureauth.py"
python "2. model deployment/monitordeployed.py"
python "3. troubleshooting/automatedalerts.py"
python "3. troubleshooting/diagnostictools.py"
python "3. troubleshooting/troubleshootsample.py"
```

## License

Educational use. Course content © Microsoft / Coursera.
