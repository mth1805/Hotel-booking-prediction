import os
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import logging

class ModelEvaluator:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir

    @staticmethod
    def evaluate_predictions(y_test, y_pred):
        return metrics.classification_report(y_test, y_pred, output_dict=True)

    def save_metrics(self, model_name, model, X_test, y_test, report, filename):
        roc_data = {}
        cm = []
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            fpr, tpr, _ = metrics.roc_curve(y_test, y_proba[:, 1])
            roc_auc = metrics.auc(fpr, tpr)
            roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
            cm = metrics.confusion_matrix(y_test, model.predict(X_test))

        eval_data = {
            "model_name": model_name,
            "best_cv_score": model.best_score_ if hasattr(model, 'best_score_') else None,
            "report": report,
            "confusion_matrix": cm.tolist() if len(cm) > 0 else [],
            "roc_curve_data": roc_data
        }
        
        os.makedirs(self.model_dir, exist_ok=True)
        file_path = os.path.join(self.model_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(eval_data, f, indent=4)
        logging.info(f"Metrics for {model_name} saved to {file_path}")

    def plot_evaluation_results(self, file_pattern="evaluation_*.json"):
        search_path = os.path.join(os.path.abspath(self.model_dir), file_pattern)
        files = glob.glob(search_path)
        if not files:
            logging.error(f"Không tìm thấy file kết quả nào tại: {search_path}")
            return
        
        results = []
        for file in files:
            with open(file, 'r') as f:
                results.append(json.load(f))

        # 1. BARPLOT ACCURACY
        model_names = [res['model_name'] for res in results]
        accuracies = [res['report']['accuracy'] for res in results]
        df_scores = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies}).sort_values(by='Accuracy', ascending=False)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_scores, x='Model', y='Accuracy', palette='viridis', hue='Model', legend=False)
        plt.ylim(0, 1.1)
        plt.title('Model Accuracy Comparison', fontsize=16)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')
        plt.savefig(os.path.join(self.model_dir, 'comparison_barplot.png'))
        plt.close()

        # 2. ROC CURVE
        plt.figure(figsize=(10, 8))
        for res in results:
            roc_data = res.get('roc_curve_data')
            if roc_data and roc_data.get('fpr'):
                plt.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2, label=f"{res['model_name']} (AUC = {roc_data['auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.model_dir, 'comparison_roc_curve.png'))
        plt.close()