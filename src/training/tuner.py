import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

class ModelTuner:
    def __init__(self, n_iter=10, cv=5, random_state=42):
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state

    def get_model_and_params(self, model_name):
        """Trả về base model và không gian tham số tương ứng."""
        if model_name == "CatBoost":
            model = CatBoostClassifier(verbose=0, random_state=self.random_state)
            params = {
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [200, 500]
            }
        elif model_name == "LightGBM":
            model = LGBMClassifier(verbose=-1, boosting_type="gbdt", random_state=self.random_state)
            params = {
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [200, 500]
            }
        elif model_name == "RandomForest":
            model = RandomForestClassifier(random_state=self.random_state)
            params = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False]
            }
        elif model_name == 'XGBoost':
            model = XGBClassifier(eval_metric='logloss', random_state=self.random_state)
            params = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'subsample': [0.7, 0.8, 1.0]
            }
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        return model, params

    def optimize(self, model_name, X_train, y_train):
        """Thực hiện RandomizedSearchCV để tìm tham số tốt nhất."""
        logging.info(f"Tuning hyperparameters for {model_name}...")
        base_model, param_dist = self.get_model_and_params(model_name)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        
        logging.info(f"Optimization complete. Best Params: {random_search.best_params_}")
        logging.info(f"Best CV Score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_score_