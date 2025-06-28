import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_pinball_loss
from sklearn.model_selection import (
    TimeSeriesSplit,
)  # <-- Re-introducing TimeSeriesSplit
import lightgbm as lgb
import shap
import optuna


class ModelTrainer:
    """
    Final version: Implements and tunes Quantile Regression models using Optuna
    with a robust TimeSeries Cross-Validation strategy.
    """

    def __init__(self, df: pd.DataFrame, features: list, target: str):
        self.df = df
        self.features = features
        self.target = target

        self.models = {
            "Median_Forecast (50th)": lgb.LGBMRegressor(
                objective="quantile", alpha=0.5, random_state=42, verbose=-1
            ),
            "Peak_Forecast (90th)": lgb.LGBMRegressor(
                objective="quantile", alpha=0.9, random_state=42, verbose=-1
            ),
        }

        self.trained_models = {}
        self.training_histories = {}
        self.scaler_ar = StandardScaler()
        self.scaler_env = StandardScaler()

    def prepare_data(self):
        """Prepares data for cross-validation and final testing."""
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

        # Fit scalers on the entire training set. They will be applied within each CV fold.
        self.scaler_ar.fit(self.X_train[self.autoregressive_features])
        self.scaler_env.fit(self.X_train[self.environmental_features])

        print("Data prepared. Scalers fitted on the full training set.")

    def optimize_hyperparameters(self, n_trials=50):
        """Uses Optuna with TimeSeries Cross-Validation for robust tuning."""
        print(
            f"--- Starting Cross-Validated Hyperparameter Optimization ({n_trials} trials) ---"
        )

        def objective(trial):
            params = {
                "objective": "quantile",
                "alpha": 0.5,
                "metric": "quantile",
                "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

            tscv = TimeSeriesSplit(n_splits=10)
            scores = []

            def scale_and_combine(df, features, ar_features, env_features):
                df_ar_s = pd.DataFrame(
                    self.scaler_ar.transform(df[ar_features]),
                    columns=ar_features,
                    index=df.index,
                )
                df_env_s = pd.DataFrame(
                    self.scaler_env.transform(df[env_features]),
                    columns=env_features,
                    index=df.index,
                )
                # Return a DataFrame with the correct feature order
                return pd.concat([df_ar_s, df_env_s], axis=1)[features]

            for train_idx, val_idx in tscv.split(self.X_train):
                X_train_fold, X_val_fold = (
                    self.X_train.iloc[train_idx],
                    self.X_train.iloc[val_idx],
                )
                y_train_fold, y_val_fold = (
                    self.y_train.iloc[train_idx],
                    self.y_train.iloc[val_idx],
                )

                X_train_fold_scaled = scale_and_combine(
                    X_train_fold,
                    self.features,
                    self.autoregressive_features,
                    self.environmental_features,
                )
                X_val_fold_scaled = scale_and_combine(
                    X_val_fold,
                    self.features,
                    self.autoregressive_features,
                    self.environmental_features,
                )

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train_fold_scaled,
                    y_train_fold,
                    eval_set=[(X_val_fold_scaled, y_val_fold)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )

                preds_log = model.predict(X_val_fold_scaled)
                loss = mean_pinball_loss(y_val_fold, preds_log, alpha=0.5)
                scores.append(loss)

            return np.mean(scores)  # Return the average score across all folds

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("\nBest trial found by Optuna (Cross-Validated):")
        print(f"  Value (Average Pinball Loss): {study.best_value:.4f}")
        self.best_params = study.best_params
        for key, value in self.best_params.items():
            print(f"    {key}: {value}")

        base_params = self.best_params
        base_params["random_state"] = 42
        base_params["n_jobs"] = -1
        base_params["verbose"] = -1

        self.models["Median_Forecast (50th)"] = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, **base_params
        )
        self.models["Peak_Forecast (90th)"] = lgb.LGBMRegressor(
            objective="quantile", alpha=0.9, **base_params
        )
        print("\n--- Quantile models updated with robustly tuned hyperparameters. ---")

    def train(self):
        """Trains the final models and captures training history."""
        print("--- Model Training Started ---")

        # We need a validation set for early stopping to find the best n_estimators
        train_val_split_idx = int(len(self.X_train) * 0.8)
        X_train_part = self.X_train.iloc[:train_val_split_idx]
        y_train_part = self.y_train.iloc[:train_val_split_idx]
        X_val_part = self.X_train.iloc[train_val_split_idx:]
        y_val_part = self.y_train.iloc[train_val_split_idx:]

        # Helper to scale data for this stage
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

        X_train_part_scaled = scale_and_combine(X_train_part)
        X_val_part_scaled = scale_and_combine(X_val_part)

        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(
                X_train_part_scaled,
                y_train_part,
                eval_set=[(X_val_part_scaled, y_val_part)],
                eval_metric="quantile",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            self.training_histories[name] = model.evals_result_
            best_iteration = (
                model.best_iteration_
                if model.best_iteration_ is not None and model.best_iteration_ > 0
                else model.get_params()["n_estimators"]
            )
            print(f"    Best iteration found: {best_iteration}")

            final_model = lgb.LGBMRegressor(**model.get_params())
            final_model.set_params(n_estimators=best_iteration)

            X_train_scaled = scale_and_combine(self.X_train)
            final_model.fit(X_train_scaled, self.y_train)

            self.trained_models[name] = final_model
        print("--- All models trained. ---")

    def evaluate(self):
        """Evaluates models using Pinball Loss, MAE, and RMSE."""
        results = pd.DataFrame(columns=["Pinball Loss", "MAE", "RMSE"])
        y_test_original_scale = np.expm1(self.y_test)

        X_test_scaled = pd.concat(
            [
                pd.DataFrame(
                    self.scaler_ar.transform(self.X_test[self.autoregressive_features]),
                    columns=self.autoregressive_features,
                    index=self.X_test.index,
                ),
                pd.DataFrame(
                    self.scaler_env.transform(self.X_test[self.environmental_features]),
                    columns=self.environmental_features,
                    index=self.X_test.index,
                ),
            ],
            axis=1,
        )[self.features]

        for name, model in self.trained_models.items():
            if "50th" in name:
                alpha = 0.50
            elif "90th" in name:
                alpha = 0.90
            else:
                alpha = 0.50

            preds_log = model.predict(X_test_scaled)
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
        os.makedirs(path, exist_ok=True)
        for name, model in self.trained_models.items():
            joblib.dump(model, os.path.join(path, f"model_{name}.pkl"))
        joblib.dump(self.scaler_ar, os.path.join(path, "scaler_ar.pkl"))
        joblib.dump(self.scaler_env, os.path.join(path, "scaler_env.pkl"))
        print(f"All models and scalers saved to '{path}' directory.")

    def generate_visualizations(self, figures_path: str):
        print("\n--- Generating Visualizations ---")
        y_test_original = np.expm1(self.y_test)
        X_test_scaled = pd.concat(
            [
                pd.DataFrame(
                    self.scaler_ar.transform(self.X_test[self.autoregressive_features]),
                    columns=self.autoregressive_features,
                    index=self.X_test.index,
                ),
                pd.DataFrame(
                    self.scaler_env.transform(self.X_test[self.environmental_features]),
                    columns=self.environmental_features,
                    index=self.X_test.index,
                ),
            ],
            axis=1,
        )[self.features]

        for name, model in self.trained_models.items():
            preds_log = model.predict(X_test_scaled)
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
            shap_values = explainer.shap_values(X_test_scaled)
            plt.figure()
            shap.summary_plot(
                shap_values, X_test_scaled, plot_type="bar", show=False, max_display=20
            )
            plt.title(f"SHAP Feature Importance ({name}) - Log Scale")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_path, f"shap_summary_{name}.png"))
            plt.close()
            print(f"  - Saved shap_summary_{name}.png")
