import pandas as pd
import numpy as np
import logging

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.target_cat_cols = [
            'hotel', 'is_canceled', 'arrival_date_year', 'arrival_date_month', 'meal', 
            'country', 'market_segment', 'distribution_channel', 'is_repeated_guest', 
            'reserved_room_type', 'assigned_room_type', 'deposit_type', 'agent', 
            'company', 'customer_type', 'reservation_status', 
            'name', 'email', 'phone-number', 'credit_card'
        ]

    def load_and_process(self):
        logging.info("Đang đọc dữ liệu cho EDA...")
        self.df = pd.read_csv(self.filepath)
        
        try:
            # Feature Engineering phục vụ vẽ biểu đồ
            self.df['total_nights'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
            self.df['total_guests'] = self.df['adults'] + self.df['children'].fillna(0) + self.df['babies']
            self.df['has_requests'] = self.df['total_of_special_requests'] > 0
            self.df['is_canceled_int'] = pd.to_numeric(self.df['is_canceled'], errors='coerce')
            
            # Ép kiểu dữ liệu phân loại
            for col in self.target_cat_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda x: str(x) if pd.notnull(x) else np.nan)
            
            logging.info(f"Đã xử lý xong dữ liệu EDA. Kích thước: {self.df.shape}")
            return self.df
        except Exception as e:
            logging.error(f"Lỗi tiền xử lý EDA: {str(e)}")
            raise e