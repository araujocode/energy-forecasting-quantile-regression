# train.py
from src.data_handler import DataHandler
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer


def main():
    """Main function to orchestrate the ML pipeline."""
    print("--- Starting ML Pipeline ---")

    # 1. Load Data
    data_handler = DataHandler(file_path="data/energydata_complete.csv")
    df = data_handler.load_data()

    if df is None:
        print("Halting pipeline due to data loading error.")
        return

    # 2. Engineer Features
    feature_engineer = FeatureEngineer(df)
    df_engineered = feature_engineer.engineer_features()

    # 3. Prepare for Training
    TARGET = "Appliances"
    FEATURES = [col for col in df_engineered.columns if col != TARGET]

    trainer = ModelTrainer(df=df_engineered, features=FEATURES, target=TARGET)
    trainer.prepare_data()

    # vvvvvvvvvvvvvvvv NEW LINE vvvvvvvvvvvvvvvv
    # 4. Optimize Hyperparameters before training
    trainer.optimize_hyperparameters(n_trials=50)  # You can change n_trials
    # ^^^^^^^^^^^^^^^^ END OF NEW LINE ^^^^^^^^^^^^^^^^

    # 5. Train Final Models
    trainer.train()

    # 6. Evaluate and Finalize
    trainer.evaluate()
    best_model = trainer.get_best_model()
    trainer.save_artifacts(best_model)

    # 7. Generate Visualizations for the report
    best_model_name = trainer.results.index[0]
    trainer.generate_visualizations(best_model, best_model_name)

    print("\n--- ML Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()
