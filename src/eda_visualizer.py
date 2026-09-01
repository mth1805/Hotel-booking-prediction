import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
from config.settings import IMG_DIR

class Visualizer:
    def __init__(self, df):
        self.df = df
        self.img_count = 0
        self.plots_data = []

    def save_plot(self, title_vn, filename_slug):
        self.img_count += 1
        filename = f"{self.img_count:02d}_{filename_slug}.png"
        filepath = os.path.join(IMG_DIR, filename)
        
        try:
            plt.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close() 
            logging.info(f"Đã lưu biểu đồ {self.img_count}: {title_vn}")
            return filename, title_vn
        except Exception as e:
            logging.error(f"Lỗi lưu ảnh {filename}: {str(e)}")
            return None, title_vn

    def generate_all_plots(self):
        logging.info("Đang vẽ các biểu đồ phân tích...")

        # 1. Boxplot Outlier
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        cols_box = ['lead_time', 'adr', 'total_nights']
        for i, col in enumerate(cols_box):
            limit = self.df[col].quantile(0.99)
            data_sub = self.df[self.df[col] < limit]
            sns.boxplot(data=data_sub, x='is_canceled', y=col, ax=axes[i], palette="Set2")
            axes[i].set_title(f"Boxplot: {col}")
        plt.tight_layout()
        fname, title = self.save_plot("Biểu đồ Hộp (Boxplot) phát hiện Outlier", "boxplot_outliers")
        comment = "<b>Phân tích Outlier:</b> Khách đặt càng sớm (lead_time cao) thì nguy cơ hủy phòng càng lớn."
        self.plots_data.append((fname, title, comment))

        # 2. Histogram Phân phối biến số
        plt.figure(figsize=(14, 8))
        cols = ['lead_time', 'adr', 'total_nights', 'total_guests']
        for i, col in enumerate(cols, 1):
            plt.subplot(2, 2, i)
            data_plot = self.df[self.df[col] < self.df[col].quantile(0.99)]
            sns.histplot(data=data_plot, x=col, kde=True, bins=30, color="#3498db")
            plt.title(f"Phân bố {col}")
        plt.tight_layout()
        fname, title = self.save_plot("Phân phối các biến số (Histogram)", "histogram_vars")
        comment = "<b>Phân phối:</b> Thời gian lead_time lệch phải; adr tập trung quanh mức trung bình 100 USD."
        self.plots_data.append((fname, title, comment))

        # 3. Countplot biến phân loại
        plt.figure(figsize=(16, 10))
        cats = ['hotel', 'arrival_date_month', 'market_segment', 'customer_type']
        for i, col in enumerate(cats, 1):
            plt.subplot(2, 2, i)
            sns.countplot(data=self.df, y=col, order=self.df[col].value_counts().index, palette="viridis")
            plt.title(f"Phân bố {col}")
        plt.tight_layout()
        fname, title = self.save_plot("Thống kê các nhóm phân loại", "countplot_cats")
        comment = "<b>Phân loại:</b> City Hotel chiếm đa số, kênh Online TA là kênh đặt phòng chính."
        self.plots_data.append((fname, title, comment))

        # 4. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        numeric_cols_heat = ['is_canceled', 'lead_time', 'arrival_date_year', 'adults', 'children', 
                             'previous_cancellations', 'booking_changes', 'adr', 'total_of_special_requests']
        valid_cols_heat = [c for c in numeric_cols_heat if c in self.df.columns]
        numeric_df = self.df[valid_cols_heat].copy()
        numeric_df['is_canceled'] = self.df['is_canceled_int']
        
        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
        sns.heatmap(numeric_df.corr(), mask=mask, cmap="coolwarm", center=0, vmax=1, vmin=-1, linewidths=0.5, annot=True, fmt=".2f")
        plt.title("Ma trận tương quan")
        fname, title = self.save_plot("Ma trận tương quan (Correlation Matrix)", "heatmap_corr")
        comment = "<b>Tương quan:</b> Lead time có tương quan dương với việc hủy phòng."
        self.plots_data.append((fname, title, comment))

        return self.plots_data