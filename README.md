# 🛒 sales-receipt-analysis

## Project Overview

This repository provides a high-performance framework for performing retail analytics on Italian GDO (Sales Receipt) data. It is designed to support a dual-workflow, allowing for efficient local prototyping on a work laptop and scalable cloud production on AWS instances.

The project focuses on replicating a "Canonical Baseline" while enhancing it with **Consensus-based Anomaly Detection** to identify irregular shopping behaviors and pricing discrepancies.

**Core Technologies:**
- **Language:** Python 3.12 (standardized via DevContainers)
- **Data Engine:** [Polars](https://pola.rs/) (primarily LazyFrame API for memory efficiency)
- **Stats/ML:** `scikit-learn` (IsolationForest, RandomForest, LassoCV), `statsmodels` (VIF)
- **Cloud:** AWS S3 integration via `fsspec` and `s3fs`

---

## 🚀 Getting Started with DevContainers

This project uses VS Code DevContainers for a consistent and reproducible development environment.

### Prerequisites

1.  **Docker Desktop:** Ensure Docker is installed and running on your system.
2.  **VS Code:** Install Visual Studio Code.
3.  **VS Code Dev Containers Extension:** Install the "Dev Containers" extension.

### Launching the DevContainer

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/extendi/IBC-analisi-scontrini.git](https://github.com/extendi/IBC-analisi-scontrini.git)
    cd sales-receipt-analysis
    ```
2.  **Open in VS Code:** Open the `sales-receipt-analysis` directory.
3.  **Reopen in Container:** VS Code should prompt you to "Reopen in Container". If not, click the green remote indicator in the bottom-left and select it.

4.  **Verify Setup:**
    ```bash
    python --version
    python -m pytest tests/
    ```

---

## ⚙️ Hardware Profiles and Configuration (`config.yaml`)

The `config.yaml` file manages environment-specific settings, switching between local prototyping and cloud production.

```yaml
environment: local # Options: local, cloud

local:
  data_path: data/raw/
  sampling_rate: 0.1     # 10% sampling for local work
  processed_data_path: data/processed/
  models_path: models/

cloud:
  data_path: s3://sales-receipt-analysis/raw/ 
  sampling_rate: 1.0     # 100% data for production
  processed_data_path: s3://sales-receipt-analysis/processed/
  models_path: s3://sales-receipt-analysis/models/

# General settings
random_state: 42
```

**How to Switch Profiles:** Simply change the `environment` key at the top of `config.yaml`. The `main.py` script automatically adjusts data paths and sampling rates based on this setting.

## 🧠 The Analytics Pipeline
The pipeline implements a **Consensus-based Anomaly Detection** strategy to ensure high-precision results:



1. **Ingestion & Denormalization:** Scans partitioned Parquet files and enforces strict schema (Dates, Strings, Decimals).
2. **Feature Engineering:**
    * **Base:** `hour_of_day`, `is_weekend`, `day_of_week`.
    * **Research:** `freshness_weight_ratio`, `peak_hour_pressure`, `price_delta` (vs. median).
3. **Consensus Feature Selection:** Merges **Random Forest** and **Lasso Regression** importance scores to identify the most robust predictors for `basket_value`.
4. **Anomaly Scoring:** Trains an `IsolationForest` on consensus features and maps results to `critical`, `warning`, and `info` severities based on quantiles.

## 🏃 Running the Pipeline
The `main.py` entry point supports flexible, partitioned execution.

**Specific Month:**
```bash
python main.py --year 2024 --month 5
```

### Outputs
* **Parquet:** `canonical_baseline.parquet` is exported for cross-validation with reference models.
* **Plots:** 3D and 2D anomaly distribution charts are generated in `/plots`.
* **Diagnostics:** Feature null counts and VIF scores are logged to ensure data quality.

## 🛠️ Troubleshooting

### AWS Credential Conflicts
If you receive "Access Key Rejected" even after updating your `~/.aws/credentials` file:

**Clear Env Vars:** AWS prioritizes shell variables. Run:
```bash
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
```
**Clock Sync:** Ensure your container's system time matches the host, or AWS will reject the request signature.

### Memory Issues
If the pipeline crashes during `df.collect()`, reduce the `sampling_rate` in `config.yaml` or increase Docker RAM limits in **Settings > Resources**.

## Project Structure

```
├── .devcontainer/    # Environment definition
├── data/              # Raw (immutable) and Processed data
├── src/
│   ├── ingestion.py   # Config loader and S3 abstraction
│   └── engineering.py # Feature logic & Consensus ML classes
├── main.py            # Pipeline orchestration
├── config.yaml        # Environment toggle (local vs cloud)
├── plots/             # Visualizations (Anomaly distributions)
└── requirements.txt   # Dependencies
```