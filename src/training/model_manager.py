import os
import joblib
import pandas as pd
import logging

class ModelManager:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, filename='best_model.pkl'):
        if model is None:
            raise ValueError("Model is None, cannot save.")
        file_path = os.path.join(self.model_dir, filename)
        joblib.dump(model, file_path)
        logging.info(f"Model saved to {file_path}")

    def load_model(self, filename='best_model.pkl'):
        file_path = os.path.join(self.model_dir, filename)
        try:
            model = joblib.load(file_path)
            logging.info(f"Model loaded from {file_path}")
            return model
        except FileNotFoundError:
            logging.error(f"File {file_path} not found.")
            return None

    def get_feature_importance(self, model, features, model_name):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            importances = model.get_feature_importance()
        else:
            raise ValueError("Model does not support feature importance.")

        df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
        file_path = os.path.join(self.model_dir, f"feature_importance_{model_name}.csv")
        df.to_csv(file_path, index=False)
        return df