# train.py
from src.data_handler import DataHandler
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from datetime import datetime
import os


def main():
    print("--- Starting Quantile Regression Pipeline ---")
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("report", "figures", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created unique directory for this run: {run_dir}")

    data_handler = DataHandler(file_path="data/energydata_complete.csv")
    df = data_handler.load_data()
    if df is None:
        return

    # Use the feature_engineer with lag features
    feature_engineer = FeatureEngineer(df)
    df_engineered = feature_engineer.engineer_features()

    TARGET = "Appliances"
    FEATURES = [col for col in df_engineered.columns if col != TARGET]

    trainer = ModelTrainer(df=df_engineered, features=FEATURES, target=TARGET)
    trainer.prepare_data()
    trainer.train()
    trainer.evaluate()
    trainer.save_artifacts()
    trainer.generate_visualizations(figures_path=run_dir)

    print(f"\n--- Pipeline Finished. All figures saved in {run_dir} ---")


if __name__ == "__main__":
    main()
