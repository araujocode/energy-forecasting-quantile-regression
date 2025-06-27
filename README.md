# Quantile Regression for Energy Forecasting

## 1. Project Overview

This project implements a complete, end-to-end machine learning pipeline to forecast appliance energy consumption in a residential building. The primary challenge of this dataset is the volatile and spiky nature of the target variable, where standard regression models often fail to capture critical peak usage events.

The dataset used in this project is the "Appliances energy prediction dataset" provided by Luis M. Candanedo, Véronique Feldheim, and Dominique Deramaix. It is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).

Original Paper Citation: Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. Energy and Buildings, 140, 81-97.

Through an iterative process of experimentation and debugging, this project demonstrates that a simple regression approach is insufficient. The final, successful implementation utilizes a sophisticated **Quantile Regression** strategy with two specialized LightGBM models: one to reliably predict the **median (baseline) energy usage** and another to specifically forecast the **90th percentile (peak) usage**.

The entire pipeline is architected using professional Object-Oriented Programming (OOP) principles and includes advanced techniques such as robust feature engineering, grouped feature scaling, and state-of-the-art model interpretability with SHAP. The project culminates in an interactive Streamlit web application for analyzing the dual forecasts.

### Key Findings

* Standard regression models (optimizing for RMSE/R²) fail to generalize and produce poor results (R² ≈ 0.3) due to the skewed nature of the target variable and the dominance of autoregressive features.
* A **log-transformation** of the target variable is a critical first step to stabilize the data distribution.
* **Quantile Regression** is a superior strategy for this problem. It allows for the creation of specialized models that can separately forecast typical usage (median) and high-consumption events (peaks), providing a much more useful and nuanced result than a single-point forecast.
* Model interpretability through **SHAP** reveals that the two quantile models learn different strategies: the median model relies on the recent average consumption, while the peak model focuses more on recent volatility (standard deviation) to anticipate spikes.

---

## 2. Directory Structure

The project is organized into a modular structure to ensure clarity, reusability, and maintainability.

```
energy_forecasting_project/
│
├── data/
│   └── energydata_complete.csv     # Raw dataset
│
├── models/
│   └── (Contains the trained .pkl files for models and scalers)
│
├── report/
│   └── figures/
│       └── (Contains all generated plots from a training run)
│
├── src/
│   ├── __init__.py                 # Makes src a Python package
│   ├── data_handler.py             # Class for loading and initial cleaning
│   ├── feature_engineer.py         # Class for all feature engineering logic
│   └── model_trainer.py            # Class for training, evaluation, and visualization
│
├── .gitignore                      # Specifies files for Git to ignore
├── app.py                          # The interactive Streamlit web application
├── requirements.txt                # Project dependencies
├── train.py                        # Main script to execute the training pipeline
└── README.md                       # This file
```

---

## 3. Methodology

The project follows a rigorous pipeline, encapsulated within the classes in the `src/` directory.

### 3.1. Data Handling (`DataHandler`)

* Loads the dataset from `data/energydata_complete.csv`.
* Parses the `date` column into datetime objects and sets it as the index.
* Removes the irrelevant `rv1` and `rv2` columns as specified by the dataset authors.

### 3.2. Feature Engineering (`FeatureEngineer`)

A curated set of features is created to provide the model with temporal context without causing overfitting.

1. **Time-Based Features:** `hour`, `day_of_week`, `month`.
2. **Cyclical Features:** The time-based features are transformed using sine and cosine functions (e.g., `hour_sin`, `hour_cos`) to allow the model to understand their cyclical nature (e.g., hour 23 is close to hour 0).
3. **Autoregressive Features:** Lag and rolling window features are created to give the model memory of the recent past.
    * `Appliances_lag_24hr`: The energy usage from the same time yesterday, to capture daily seasonality.
    * `Appliances_rolling_mean_1hr` & `Appliances_rolling_std_1hr`: The average usage and volatility over the last hour, to capture recent trends.
4. **Target Transformation:** A **log1p transformation** (`numpy.log1p`) is applied to the `Appliances` target variable. This is a critical step to handle the high skewness of the data and stabilize the variance, making it easier for the models to learn.

### 3.3. Modeling Strategy (`ModelTrainer`)

This project's core complexity lies in its advanced modeling and evaluation strategy.

1. **Data Splitting:** A robust chronological split is used: 85% for the training set and 15% for the final, unseen test set. The training set is further subdivided (80/20) into a training part and a validation part for early stopping.
2. **Grouped Feature Scaling:** To prevent powerful autoregressive features from "drowning out" the more subtle environmental signals, features are split into two groups and scaled independently using `StandardScaler`, which is robust to outliers.
3. **Quantile Regression:** Instead of standard regression, this pipeline uses **Quantile Regression** with a LightGBM model. This state-of-the-art technique allows us to train specialized models:
    * **Median Forecast (`alpha=0.5`):** Predicts the 50th percentile, representing the most likely, typical energy usage.
    * **Peak Forecast (`alpha=0.9`):** Predicts the 90th percentile, specifically trained to identify conditions that lead to high-energy spikes.
4. **Training & Evaluation:**
    * **Early Stopping:** Models are trained with early stopping using the validation set to find the optimal number of boosting rounds and prevent overfitting.
    * **Appropriate Metrics:** The models are evaluated on the unseen test set using metrics suitable for this problem. The misleading R² score is discarded in favor of:
        * **Pinball Loss:** The industry-standard metric for evaluating quantile regression models.
        * **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** for interpretability.
    * All predictions and metrics are inverse-transformed (`numpy.expm1`) back to the original Watt-hour scale for clear interpretation.

### 3.4. Model Interpretability (`SHAP`)

The project uses the **SHAP (SHapley Additive exPlanations)** library to explain the predictions of both the Median and Peak forecast models, providing deep insight into how each feature contributes to the final output.

---

## 4. How to Run the Project

Ensure you have Python 3.10+ installed.

### Step 1: Set Up the Environment

First, navigate to the project's root directory in your terminal. It is highly recommended to use a virtual environment.

```powershell
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### Step 2: Run the Training Pipeline

Execute the `train.py` script. This will run the entire pipeline: loading data, engineering features, training the two quantile models, and saving the final model artifacts (`.pkl` files) to the `models/` directory. It will also generate all evaluation plots and save them to a new, timestamped folder inside `report/figures/`.

```bash
python train.py
```

### Step 3: Launch the Interactive Application

After the training script has run successfully, launch the Streamlit web application.

```bash
streamlit run app.py
```

Your default web browser will open with the interactive forecasting tool.

-----