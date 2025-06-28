from src.data_handler import DataHandler
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from datetime import datetime
import os


def main():
    """Main function to orchestrate the final ML pipeline with Optuna tuning."""
    print(
        "--- Starting Quantile Regression Pipeline with Hyperparameter Optimization ---"
    )

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("report", "figures", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created unique directory for this run: {run_dir}")

    data_handler = DataHandler(file_path="data/energydata_complete.csv")
    df = data_handler.load_data()

    if df is None:
        return

    feature_engineer = FeatureEngineer(df)
    df_engineered = feature_engineer.engineer_features()

    TARGET = "Appliances"
    FEATURES = [col for col in df_engineered.columns if col != TARGET]

    trainer = ModelTrainer(df=df_engineered, features=FEATURES, target=TARGET)
    trainer.prepare_data()

    # Optimization step to tune the quantile models
    trainer.optimize_hyperparameters(n_trials=50)

    # Train the final models using the best parameters found by Optuna
    trainer.train()

    # Evaluate the final models on the unseen test set
    trainer.evaluate()

    # Save the final models and scalers
    trainer.save_artifacts()

    # Generate all visualizations for the report
    trainer.generate_visualizations(figures_path=run_dir)

    print(f"\n--- Pipeline Finished Successfully. All figures saved in {run_dir} ---")


if __name__ == "__main__":
    main()
