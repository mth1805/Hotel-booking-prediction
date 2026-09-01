import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Import module ML
from src.preprocessing import DataPreprocessor
from src.trainer import ModelTrainer

# 2. Import module EDA
from src.eda_processor import DataProcessor as EDAProcessor
from src.eda_visualizer import Visualizer
from src.eda_reporter import HTMLReporter
from config.settings import DATA_PATH as EDA_DATA_PATH, setup_logging as setup_eda_logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def run_eda():
    """Thực thi luồng phân tích dữ liệu trực quan (EDA)"""
    setup_eda_logging()
    logging.info("--- BẮT ĐẦU QUÁ TRÌNH EDA ---")
    if not os.path.exists(EDA_DATA_PATH):
        logging.critical(f"Không tìm thấy dữ liệu tại: {EDA_DATA_PATH}")
        return

    # Load và xử lý dữ liệu cho EDA
    processor = EDAProcessor(EDA_DATA_PATH)
    df = processor.load_and_process()
    
    # Vẽ biểu đồ
    visualizer = Visualizer(df)
    plots_data = visualizer.generate_all_plots()
    
    # Xuất báo cáo HTML
    reporter = HTMLReporter(df, plots_data, processor.target_cat_cols)
    reporter.build_and_save_report()
    logging.info("--- QUÁ TRÌNH EDA HOÀN TẤT ---")

def run_preprocessing(config_path):
    """Đọc dữ liệu raw, tiền xử lý qua preprocessing.py và lưu thành train/test processed."""
    import configparser
    config = configparser.ConfigParser()
    config.read(config_path)
    
    input_path = config['PREPROCESSING']['inputpath']
    target = config['DATA']['target']
    test_size = float(config['TRAINING']['test_size'])
    random_state = int(config['TRAINING']['random_state'])
    
    logging.info(f"Đang đọc dữ liệu raw từ: {input_path}")
    df = pd.read_csv(input_path)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # Chia tập Train/Test tránh Data Leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logging.info("Bắt đầu tiền xử lý tập TRAIN (fit_transform)...")
    preprocessor = DataPreprocessor(target_col=target)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    logging.info("Bắt đầu tiền xử lý tập TEST (transform)...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Lưu file
    preprocessor.save_processed_data(X_train_processed, 'train_processed.csv', y=y_train)
    preprocessor.save_processed_data(X_test_processed, 'test_processed.csv', y=y_test)
    
    os.makedirs('models', exist_ok=True)
    preprocessor.save_preprocessor('models/preprocessor.joblib')
    logging.info("Hoàn tất quy trình Tiền xử lý dữ liệu!")

def run_training(config_path, tune, model_name):
    """Thực thi quá trình huấn luyện và chọn mô hình."""
    trainer = ModelTrainer(config_path=config_path)
    
    try:
        trainer.load_data()
    except Exception as e:
        logging.error(f"Lỗi khi load data: {e}")
        print("Gợi ý: Hãy chạy lệnh với cờ --preprocess trước để tạo file dữ liệu processed.")
        return
    
    if tune:
        if model_name == 'Auto':
            trainer.auto_select_model()
            trainer.plot_evaluation_results()
        else:
            print(f"Bắt đầu tinh chỉnh siêu tham số cho {model_name}...")
            trainer.optimize_params(model_name=model_name)
            trainer.train_predict()
            try:
                trainer.get_feature_importance(model_name)
            except: 
                pass
            trainer.save_model(f"best_{model_name}.pkl")
    else:
        print("Chưa chọn hành động huấn luyện. Sử dụng --tune để bắt đầu training.")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="End-to-End ML Pipeline cho Hotel Booking")
    
    parser.add_argument('--config', type=str, default='config/config.ini', help='Đường dẫn file config')
    parser.add_argument('--eda', action='store_true', help='Chạy phân tích dữ liệu EDA và xuất báo cáo HTML')
    parser.add_argument('--preprocess', action='store_true', help='Chạy tiền xử lý dữ liệu ML')
    parser.add_argument('--tune', action='store_true', help='Huấn luyện và tinh chỉnh mô hình')
    parser.add_argument('--model', type=str, default='Auto', 
                        choices=['CatBoost', 'LightGBM', 'XGBoost' , 'RandomForest', 'Auto'], 
                        help='Chọn thuật toán')

    args = parser.parse_args()

    # 1. Chạy phân tích EDA
    if args.eda:
        run_eda()

    # 2. Chạy tiền xử lý Machine Learning
    if args.preprocess:
        run_preprocessing(args.config)
        
    # 3. Chạy huấn luyện và đánh giá mô hình
    if args.tune:
        run_training(args.config, args.tune, args.model)