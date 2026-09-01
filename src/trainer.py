import configparser
import logging
import os
import pandas as pd
from src.training.tuner import ModelTuner
from src.training.evaluator import ModelEvaluator
from src.training.model_manager import ModelManager

class ModelTrainer:
    def __init__(self, config_path='config/config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')

        self.features = self.config['DATA']['features'].split(',')
        self.target = self.config['DATA']['target']
        self.testpath = self.config['DATA']['testpath']
        self.trainpath = self.config['DATA']['trainpath']
        
        n_iter = int(self.config['TRAINING']['n_iter'])
        cv = int(self.config['TRAINING']['cv'])
        
        self.train = None
        self.test = None
        self.model = None

        # Khởi tạo các module con
        self.tuner = ModelTuner(n_iter=n_iter, cv=cv)
        self.evaluator = ModelEvaluator()
        self.manager = ModelManager()

    def load_data(self):
        self.train = pd.read_csv(self.trainpath)
        self.test = pd.read_csv(self.testpath)

    def optimize_params(self, model_name):
        X_train = self.train[self.features]
        y_train = self.train[self.target]
        self.model, score = self.tuner.optimize(model_name, X_train, y_train)
        return self.model, score

    def train_predict(self):
        X_test = self.test[self.features]
        y_test = self.test[self.target]
        y_pred = self.model.predict(X_test)
        report = self.evaluator.evaluate_predictions(y_test, y_pred)
        return y_pred, y_test, report

    def auto_select_model(self):
        models = ['CatBoost', 'LightGBM', 'RandomForest', 'XGBoost']
        best_score = -1
        best_model = None
        best_name = ''
        summary_results = []

        for m in models:
            try:
                model, score = self.optimize_params(m)
                self.model = model
                y_pred, y_test, report = self.train_predict()
                
                self.manager.get_feature_importance(self.model, self.features, m)
                self.evaluator.save_metrics(m, self.model, self.test[self.features], y_test, report, f'evaluation_{m}.json')
                
                summary_results.append({
                    "Model": m, "Best_CV_Score": score, "Accuracy": report['accuracy'],
                    "Macro_Avg_F1": report['macro avg']['f1-score']
                })

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = m
            except Exception as e:
                logging.error(f"Failed to train {m}: {e}")

        if summary_results:
            df_sum = pd.DataFrame(summary_results).sort_values(by="Accuracy", ascending=False)
            df_sum.to_csv(os.path.join(self.manager.model_dir, "model_comparison_summary.csv"), index=False)

        if best_model is not None:
            self.model = best_model
            self.manager.save_model(best_model, 'best_model.pkl')
            print(f"\nTHE BEST MODEL IS: {best_name} with score {best_score:.4f}")
        else:
            raise ValueError("All models failed.")

    def plot_evaluation_results(self):
        self.evaluator.plot_evaluation_results()