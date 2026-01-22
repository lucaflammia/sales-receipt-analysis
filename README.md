# sales-receipt-analysis

## Project Overview

This repository provides a framework for performing retail analytics on Italian GDO (Sales Receipt) data. It is designed to support a dual-workflow, allowing for efficient local prototyping on a work laptop and scalable cloud production on AWS instances (with potential for GPU acceleration for LLM/Deep Learning tasks).

**Core Technologies:**
- **Language:** Python 3.12 (standardized via DevContainers)
- **Data Engine:** Polars (primarily LazyFrame API for memory efficiency)
- **Stats/ML:** `statsmodels` (VIF), `scikit-learn` (IsolationForest), `mlxtend` (FP-Growth)
- **AI:** Integration with Ollama (Llama 3) for receipt summarization

## Getting Started with DevContainers

This project uses VS Code DevContainers for a consistent and reproducible development environment.

### Prerequisites

1.  **Docker Desktop:** Ensure Docker is installed and running on your system.
2.  **VS Code:** Install Visual Studio Code.
3.  **VS Code Dev Containers Extension:** Install the "Dev Containers" extension in VS Code.

### Launching the DevContainer

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/extendi/IBC-analisi-scontrini.git
    cd sales-receipt-analysis
    ```
2.  **Open in VS Code:**
    Open the `sales-receipt-analysis` directory in VS Code.
3.  **Reopen in Container:**
    VS Code should automatically detect the `.devcontainer` configuration and prompt you to "Reopen in Container". If it doesn't, click the green remote indicator in the bottom-left corner of the VS Code window and select "Reopen in Container".

    The first time you do this, Docker will build the image (which might take a few minutes as it installs system dependencies, Rust, and Python packages). Subsequent openings will be much faster.

4.  **Verify Setup:**
    Once the DevContainer is running, open a terminal within VS Code (Terminal > New Terminal) and run:
    ```bash
    python --version
    pip list | grep polars
    ```
    You should see Python 3.12 and Polars listed.

## Hardware Profiles and Configuration (`config.yaml`)

The `config.yaml` file is central to managing environment-specific settings, particularly for switching between local prototyping and cloud production.

```yaml
environment: local # Options: local, cloud

local:
  data_path: data/raw/  # Path for local data prototyping
  sampling_rate: 0.1    # 10% sampling for work laptop
  processed_data_path: data/processed/
  models_path: models/

cloud:
  data_path: s3://sales-receipt-analysis/raw/ # Example S3 path for AWS
  sampling_rate: 1.0    # 100% data for AWS instances
  processed_data_path: s3://sales-receipt-analysis/processed/
  models_path: s3://sales-receipt-analysis/models/

# General settings
random_state: 42
```

**How to Switch Profiles:**

Simply change the `environment` key at the top of `config.yaml` to either `local` or `cloud`. The `main.py` script and other modules will automatically adjust their behavior (e.g., data paths, sampling rates) based on this setting.

-   **`local`:** Uses local file paths and a `sampling_rate` of `0.1` (10% of data) to optimize performance and memory usage on less powerful machines.
-   **`cloud`:** Assumes a cloud environment (e.g., AWS S3 for data paths) and uses a `sampling_rate` of `1.0` (100% of data) for full-scale processing.

## Running the First Anomaly Detection Pass

This project includes a starter implementation for anomaly detection using `IsolationForest` from `scikit-learn`.

1.  **Ensure `config.yaml` is set to `local` (recommended for initial runs).**
2.  **Run the Main Script:**
    Open a terminal in your DevContainer and execute:
    ```bash
    python main.py
    ```

    The script will perform the following steps:
    -   Load configuration.
    -   Ingest a dummy `receipts.csv` (created automatically if not present in `data/raw/`).
    -   Extract shopping mission features (Basket Size, Basket Value, Timestamp features).
    -   Collect a sample of the data (based on `sampling_rate` in `config.yaml`).
    -   Calculate Variance Inflation Factor (VIF) for numerical features to check for multicollinearity.
    -   Train an `IsolationForest` model on selected features (`basket_size`, `basket_value`, `hour_of_day`, `day_of_week`).
    -   Add an `anomaly_score` column to the processed data.
    -   Save the processed data to `data/processed/processed_receipts.parquet` (if in `local` environment).

    You will see console output detailing the steps, resource usage (RAM/CPU), VIF results, and a preview of the data with anomaly scores.

## Project Structure

```
.devcontainer/
├── Dockerfile
└── devcontainer.json
.github/
└── workflows/ # CI/CD workflows (e.g., GitHub Actions)
data/
├── processed/ # Cleaned, transformed, and feature-engineered data
└── raw/       # Original, immutable raw data
models/          # Trained machine learning models
notebooks/       # Jupyter notebooks for exploration and prototyping
src/
├── engineering.py # Feature engineering logic (VIF, shopping mission)
└── ingestion.py   # Data ingestion and loading
tests/           # Unit and integration tests
config.yaml      # Environment-specific configuration
main.py          # Main entry point for the data pipeline
README.md        # Project documentation
requirements.txt # Python dependencies
setup_project.sh # Initial project setup script
```
