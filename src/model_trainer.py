# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_pinball_loss,
)  # <-- ADDED Pinball Loss
import lightgbm as lgb
import shap


class ModelTrainer:
    """
    Final version: Implements Quantile Regression, generates correct training
    curves, and evaluates using the correct metric (Pinball Loss).
    """

    def __init__(self, df: pd.DataFrame, features: list, target: str):
        self.df = df
        self.features = features
        self.target = target

        # Using pre-determined robust hyperparameters for quantile regression
        self.models = {
            "Median_Forecast (50th)": lgb.LGBMRegressor(
                objective="quantile",
                alpha=0.5,
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            ),
            "Peak_Forecast (90th)": lgb.LGBMRegressor(
                objective="quantile",
                alpha=0.9,
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            ),
        }

        self.trained_models = {}
        self.training_histories = {}
        self.scaler_ar = StandardScaler()
        self.scaler_env = StandardScaler()

    def prepare_data(self):
        """Splits data and scales feature groups separately."""
        if "lights" in self.features:
            self.features.remove("lights")
            print("Removed 'lights' feature.")

        self.autoregressive_features = [
            col for col in self.features if "Appliances_" in col
        ]
        self.environmental_features = [
            col for col in self.features if col not in self.autoregressive_features
        ]
        X = self.df[self.features]
        y = self.df[self.target]

        train_end_idx = int(len(self.df) * 0.85)
        self.X_train, self.X_test = X.iloc[:train_end_idx], X.iloc[train_end_idx:]
        self.y_train, self.y_test = y.iloc[:train_end_idx], y.iloc[train_end_idx:]

        # Create a validation set from the end of the training set
        train_val_split_idx = int(len(self.X_train) * 0.8)
        self.X_train_part = self.X_train.iloc[:train_val_split_idx]
        self.y_train_part = self.y_train.iloc[:train_val_split_idx]
        self.X_val_part = self.X_train.iloc[train_val_split_idx:]
        self.y_val_part = self.y_train.iloc[train_val_split_idx:]

        self.X_test_df = self.X_test.copy()

        self.scaler_ar.fit(self.X_train[self.autoregressive_features])
        self.scaler_env.fit(self.X_train[self.environmental_features])

        def scale_and_combine(df):
            df_ar_scaled = pd.DataFrame(
                self.scaler_ar.transform(df[self.autoregressive_features]),
                columns=self.autoregressive_features,
                index=df.index,
            )
            df_env_scaled = pd.DataFrame(
                self.scaler_env.transform(df[self.environmental_features]),
                columns=self.environmental_features,
                index=df.index,
            )
            return pd.concat([df_ar_scaled, df_env_scaled], axis=1)[self.features]

        self.X_train_scaled = scale_and_combine(self.X_train)
        self.X_train_part_scaled = scale_and_combine(self.X_train_part)
        self.X_val_part_scaled = scale_and_combine(self.X_val_part)
        self.X_test_scaled = scale_and_combine(self.X_test)
        print("Data split and scaled in separate groups using StandardScaler.")

    def train(self):
        """Trains all defined quantile models and captures training history."""
        print("--- Model Training Started ---")
        for name, model in self.models.items():
            print(f"  Training {name}...")

            model.fit(
                self.X_train_part_scaled,
                self.y_train_part,
                eval_set=[(self.X_val_part_scaled, self.y_val_part)],
                eval_metric="quantile",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            self.training_histories[name] = model.evals_result_

            best_iteration = model.best_iteration_ if model.best_iteration_ > 0 else 1
            print(f"    Best iteration found: {best_iteration}")
            final_model = lgb.LGBMRegressor(**model.get_params())
            final_model.set_params(n_estimators=best_iteration)
            final_model.fit(self.X_train_scaled, self.y_train)

            self.trained_models[name] = final_model
        print("--- All models trained. ---")

    def evaluate(self):
        """
        Evaluates models using appropriate metrics for quantile regression:
        Pinball Loss, MAE, and RMSE.
        """
        from sklearn.metrics import mean_pinball_loss

        results = pd.DataFrame(columns=["Pinball Loss", "MAE", "RMSE"])
        y_test_original_scale = np.expm1(self.y_test)

        for name, model in self.trained_models.items():
            if "50th" in name:
                alpha = 0.50
            elif "90th" in name:
                alpha = 0.90
            else:
                alpha = 0.50

            preds_log = model.predict(self.X_test_scaled)
            preds_original_scale = np.expm1(preds_log)
            preds_original_scale[preds_original_scale < 0] = 0

            pinball = mean_pinball_loss(
                y_test_original_scale, preds_original_scale, alpha=alpha
            )
            mae = mean_absolute_error(y_test_original_scale, preds_original_scale)
            rmse = np.sqrt(
                mean_squared_error(y_test_original_scale, preds_original_scale)
            )

            results.loc[name] = [pinball, mae, rmse]

        print("\n--- Final Model Performance on Unseen Test Set (Original Scale) ---")
        self.results = results.sort_values(by="Pinball Loss")
        print(self.results)

    def save_artifacts(self, path="models"):
        """Saves all models and scalers."""
        os.makedirs(path, exist_ok=True)
        for name, model in self.trained_models.items():
            joblib.dump(model, os.path.join(path, f"model_{name}.pkl"))
        joblib.dump(self.scaler_ar, os.path.join(path, "scaler_ar.pkl"))
        joblib.dump(self.scaler_env, os.path.join(path, "scaler_env.pkl"))
        print(f"All models and scalers saved to '{path}' directory.")

    def generate_visualizations(self, figures_path: str):
        """Generates and saves all evaluation plots for each model."""
        print("\n--- Generating Visualizations ---")
        y_test_original = np.expm1(self.y_test)

        for name, model in self.trained_models.items():
            preds_log = model.predict(self.X_test_scaled)
            preds_original = np.expm1(preds_log)
            plt.figure(figsize=(15, 7))
            plt.plot(self.y_test.index, y_test_original, label="Actuals", alpha=0.8)
            plt.plot(
                self.y_test.index,
                preds_original,
                label=f"Predictions ({name})",
                linestyle="--",
                alpha=0.8,
            )
            plt.title(f"{name} - Predictions vs. Actuals on Test Set")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_path, f"predictions_{name}.png"))
            plt.close()
            print(f"  - Saved predictions_{name}.png")

            history = self.training_histories[name]
            plt.figure(figsize=(10, 6))
            plt.plot(history["valid_0"]["quantile"], label="Validation Score")
            plt.title(f"Training Curve ({name})")
            plt.xlabel("Boosting Round")
            plt.ylabel("Quantile Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_path, f"training_curve_{name}.png"))
            plt.close()
            print(f"  - Saved training_curve_{name}.png")

            print(f"--- Generating SHAP visualizations for {name} ---")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test_scaled)
            plt.figure()
            shap.summary_plot(
                shap_values,
                self.X_test_scaled,
                plot_type="bar",
                show=False,
                max_display=20,
            )
            plt.title(f"SHAP Feature Importance ({name}) - Log Scale")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_path, f"shap_summary_{name}.png"))
            plt.close()
            print(f"  - Saved shap_summary_{name}.png")
