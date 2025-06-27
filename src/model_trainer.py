# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import optuna
import shap  # <-- ADD THIS IMPORT

class ModelTrainer:
    """
    A class to handle model training, evaluation, and visualization.
    """
    def __init__(self, df: pd.DataFrame, features: list, target: str):
        self.df = df
        self.features = features
        self.target = target
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10),
            "LightGBM": lgb.LGBMRegressor(random_state=42) # Start with a basic model before optimization
        }
        self.trained_models = {}
        self.scaler = MinMaxScaler()

    def prepare_data(self):
        """Splits and scales the data for training and testing."""
        X = self.df[self.features]
        y = self.df[self.target]
        split_index = int(len(self.df) * 0.8)
        self.X_train, self.X_test = X.iloc[:split_index], X.iloc[split_index:]
        self.y_train, self.y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        # We need to keep a DataFrame version of X_test for SHAP plots
        self.X_test_df = self.X_test.copy() 

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("Data split and scaled.")
        
    def optimize_hyperparameters(self, n_trials=50):
        """Uses Optuna to find the best hyperparameters for LightGBM."""
        print(f"--- Starting Hyperparameter Optimization with Optuna ({n_trials} trials) ---")
        
        def objective(trial):
            params = {
                'objective': 'regression_l1', 'metric': 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 400, 2500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 4, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42, 'n_jobs': -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(self.X_train_scaled, self.y_train, eval_set=[(self.X_test_scaled, self.y_test)], eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(self.X_test_scaled)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            return rmse

        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        print("\nBest trial found by Optuna:")
        print(f"  Value (RMSE): {study.best_value:.4f}")
        print("  Best Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        self.models['LightGBM'] = lgb.LGBMRegressor(**best_params)
        print("\n--- LightGBM model in trainer has been updated with optimized hyperparameters. ---")

    def train(self):
        """Trains all models defined in the constructor."""
        print("--- Model Training Started ---")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            if "LightGBM" in name and 'n_estimators' not in model.get_params():
                 model.fit(self.X_train_scaled, self.y_train, eval_set=[(self.X_test_scaled, self.y_test)], eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                model.fit(self.X_train_scaled, self.y_train)
            self.trained_models[name] = model
        print("--- All models trained. ---")

    def evaluate(self):
        """Evaluates all trained models and prints the results."""
        results = pd.DataFrame(columns=['MAE', 'RMSE', 'R2'])
        for name, model in self.trained_models.items():
            preds = model.predict(self.X_test_scaled)
            results.loc[name] = [
                mean_absolute_error(self.y_test, preds),
                np.sqrt(mean_squared_error(self.y_test, preds)),
                r2_score(self.y_test, preds)
            ]
        print("\n--- Model Performance Metrics ---")
        self.results = results.sort_values(by='RMSE')
        print(self.results)
        return self.results

    def get_best_model(self):
        """Identifies and returns the best performing model."""
        best_model_name = self.results.index[0]
        print(f"\nBest model identified: {best_model_name}")
        return self.trained_models[best_model_name]

    def save_artifacts(self, model, path='models'):
        """Saves the best model and the scaler."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(model, os.path.join(path, 'energy_forecasting_model.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'feature_scaler.pkl'))
        print(f"Model and scaler saved to '{path}' directory.")

    def generate_visualizations(self, model, model_name, path='report/figures'):
        """Generates and saves all evaluation plots."""
        os.makedirs(path, exist_ok=True)
        print("\n--- Generating Visualizations ---")
        
        # Plot Predictions vs Actuals
        preds = model.predict(self.X_test_scaled)
        plt.figure(figsize=(15, 7))
        plt.plot(self.y_test.index, self.y_test.values, label='Actuals', alpha=0.8)
        plt.plot(self.y_test.index, preds, label='Predictions', linestyle='--', alpha=0.8)
        plt.title(f'{model_name} - Predictions vs. Actuals')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'predictions_vs_actuals.png'))
        plt.close()
        print(f"  - Saved predictions_vs_actuals.png")

        # Plot Feature Importances
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=importances)
            plt.title(f'Feature Importances ({model_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'feature_importance.png'))
            plt.close()
            print(f"  - Saved feature_importance.png")

        # Plot Learning Curves
        tscv = TimeSeriesSplit(n_splits=5)
        sizes, train_scores, test_scores = learning_curve(model, self.X_train_scaled, self.y_train, cv=tscv, scoring='r2', n_jobs=-1)
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
        plt.plot(sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-Validation Score')
        plt.title(f'Learning Curves ({model_name})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'learning_curves.png'))
        plt.close()
        print(f"  - Saved learning_curves.png")
        
        #SHAP ANALYSIS
        if "LightGBM" in model_name or "Random Forest" in model_name:
            print(f"--- Generating SHAP visualizations for {model_name} ---")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test_scaled)
            
            # Use the unscaled X_test DataFrame for better plot interpretability
            X_test_for_shap = self.X_test_df.copy()

            # SHAP Summary Plot (like a better feature importance)
            shap.summary_plot(shap_values, X_test_for_shap, plot_type="bar", show=False, max_display=20)
            plt.title(f'SHAP Feature Importance ({model_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'shap_summary_bar.png'))
            plt.close()
            print(f"  - Saved shap_summary_bar.png")
            
            # SHAP Beeswarm Plot (shows feature value impact)
            shap.summary_plot(shap_values, X_test_for_shap, show=False, max_display=20)
            # Get the current figure and adjust layout
            fig = plt.gcf()
            fig.suptitle(f'SHAP Summary Plot ({model_name})', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'shap_summary_beeswarm.png'))
            plt.close()
            print(f"  - Saved shap_summary_beeswarm.png")