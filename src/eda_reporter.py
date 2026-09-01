import os
import logging
from config.settings import REPORT_DIR, COLORS

class HTMLReporter:
    def __init__(self, df, plots_data, target_cat_cols):
        self.df = df
        self.plots_data = plots_data
        self.target_cat_cols = target_cat_cols
        self.main_color = COLORS["main_color"]
        self.bg_color = COLORS["bg_color"]

    def generate_info_missing_table(self):
        html = f"""
        <h3 style="color: {self.main_color};">Thông tin chi tiết các cột</h3>
        <table style="width: 100%; border-collapse: collapse; border: 2px solid {self.main_color}; font-family: sans-serif;">
            <tr style="background-color: {self.bg_color}; color: {self.main_color}; font-weight: bold;">
                <th style="padding: 10px; border: 1px solid {self.main_color};">Column</th>
                <th style="padding: 10px; border: 1px solid {self.main_color};">Dtype</th>
                <th style="padding: 10px; border: 1px solid {self.main_color};">Missing Count</th>
                <th style="padding: 10px; border: 1px solid {self.main_color};">NUnique</th>
            </tr>
        """
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{col}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{str(self.df[col].dtype)}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{missing:,}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{self.df[col].nunique():,}</td>
            </tr>
            """
        html += "</table>"
        return html

    def generate_overview_section(self):
        n_rows, n_cols = self.df.shape
        return f"""
        <h2 style="color: {self.main_color}; border-left: 5px solid {self.main_color};">1. TỔNG QUAN DỮ LIỆU</h2>
        <div style="background-color: {self.bg_color}; padding: 15px; border-radius: 5px;">
            <p><b>Kích thước:</b> {n_rows:,} dòng, {n_cols} cột.</p>
        </div>
        {self.generate_info_missing_table()}
        """

    def generate_toc(self):
        html = "<h3>MỤC LỤC BIỂU ĐỒ</h3><ul>"
        for i, (_, title, _) in enumerate(self.plots_data, 1):
            html += f"<li><a href='#chart_{i}' style='text-decoration:none; color:#2980b9;'>{i}. {title}</a></li>"
        html += "</ul>"
        return html

    def build_and_save_report(self):
        logging.info("Đang tạo báo cáo HTML...")
        html = f"""
        <!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>Full EDA Report</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            img {{ width: 100%; border-radius: 5px; border: 1px solid #ddd; }}
        </style>
        </head><body>
        <div class="container">
            <h1>BÁO CÁO PHÂN TÍCH DỮ LIỆU TOÀN DIỆN</h1>
            {self.generate_overview_section()}
            <hr>
            <h2 style="color: {self.main_color};">2. BIỂU ĐỒ TRỰC QUAN HÓA & INSIGHTS</h2>
            {self.generate_toc()}
            <br>
        """
        
        for i, (filename, title, comment) in enumerate(self.plots_data, 1):
            html += f"""
            <div id="chart_{i}" style="margin-bottom: 50px;">
                <h3>{i}. {title}</h3>
                <img src="images/{filename}" alt="{title}">
                <div style="background-color: {self.bg_color}; padding: 12px; margin-top: 10px; border-left: 5px solid {self.main_color};">
                    {comment}
                </div>
            </div>
            """
        html += "</div></body></html>"
        
        report_path = os.path.join(REPORT_DIR, "FULL_EDA_REPORT.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info(f"Đã xuất báo cáo tại: {report_path}")