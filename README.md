## **DỰ ĐOÁN KHẢ NĂNG HỦY ĐẶT PHÒNG KHÁCH SẠN** *(Hotel Booking Cancellation Prediction)*
Đồ án này tập trung khai thác sâu dữ liệu lịch sử đặt phòng nhằm giải quyết một trong những bài toán vận hành cốt lõi của ngành dịch vụ lưu trú. Thông qua việc áp dụng các thuật toán học máy tiên tiến, hệ thống không chỉ giúp dự báo chính xác rủi ro hủy phòng mà còn mang đến cái nhìn trực quan về hành vi khách hàng, hỗ trợ ban quản lý tối ưu hóa doanh thu và hoạch định chiến lược kinh doanh hiệu quả.

## I. GIỚI THIỆU
Dự án xây dựng mô hình Machine Learning nhằm dự đoán khách hàng có hủy đặt phòng hay không, hỗ trợ khách sạn:
* Giảm rủi ro phòng trống đột xuất
* Tối ưu doanh thu
* Chủ động chính sách đặt phòng

**Dataset**: `hotel_bookings.csv`
* **Input**: Thông tin khách hàng, loại phòng, thời gian đặt, tiền cọc...
* **Output**: `0` (Không hủy) hoặc `1` (Hủy).

## II. CÀI ĐẶT MÔI TRƯỜNG
**1. Yêu cầu hệ thống**
* Python 3.8 trở lên.
* RAM: Khuyến nghị 8GB trở lên.

**2. Cài đặt thư viện**
Chạy lệnh sau trong Terminal/Command Prompt để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
***Lưu ý:*** Đảm bảo bạn đang đứng ở thư mục **FINAL_PROJECT** chứa file `requirements.txt`.

## III. CẤU HÌNH HỆ THỐNG
* `config/config.ini`: Quản lý đường dẫn dữ liệu (trainpath, testpath) và các tham số cho Machine Learning.

* `config/settings.py`: Quản lý hằng số đường dẫn, logging và bảng màu cho luồng phân tích dữ liệu trực quan (EDA).
* **Ví dụ**
```ini
[DATA]
# Đường dẫn đến file dữ liệu
[PREPROCESSING]
inputpath =  data\raw\hotel_bookings.csv

trainpath = data\processed\train_processed.csv
testpath = data\processed\test_processed.csv
...
# Phần còn lại giữ nguyên
```

## IV. QUY TRÌNH CHẠY DỰ ÁN
Toàn bộ các tác vụ của đồ án được điều phối tập trung thông qua file main.py tại thư mục gốc:

**Bước 1: Khám phá dữ liệu (EDA)**
1. Run `python main.py --eda`
3. **Kết quả**: Mở file `reports/images/FULL_EDA_REPORT.html` để xem báo cáo EDA chi tiết.

**Bước 2: Tiền xử lý dữ liệu (Preprocessing)**
Làm sạch dữ liệu, mã hóa biến phân loại và chia tập Train/Test.
* Chạy file `src/main.py`
* Dữ liệu sau khi xử lý và chia train/test được lưu vào:
  * `data/processed/train.csv` 
  * `data/processed/test.csv`
* Preprocessor được lưu tại `reports/preprocessor.joblib.`

**Bước 3: Huấn luyện & Dự đoán (Modeling)**
Trong Terminal, gõ các lệnh sau:
1. Gõ `model` để tiếp tục chạy model (Gõ `exit` để thoát)
2. Gõ tên model muốn chạy

**A. Chế độ Tự động (Khuyên dùng)**
* Hệ thống chạy tất cả model (CatBoost, XGBoost, LightGBM, RandomForest) và chọn cái tốt nhất:
```bash
Auto hoặc auto
```
**B. Chạy từng thuật toán riêng lẻ**
```bash
# CatBoost
CatBoost hoặc catboost hoặc cat

# XGBoost
XGBoost  hoặc xgboost hoặc xgb

# LightGBM
LightGBM hoặc lightgbm hoặc lgbm

# RandomForest
RandomForest hoặc randomforest hoặc rf
```

## V. CẤU TRÚC THƯ MỤC DỰ ÁN
* Sau khi chạy xong, thư mục dự án có cấu trúc như sau:
```bash
FINAL_PROJECT/
│
├── main.py                     # Hàm chạy chính
├── README.md                   # File hướng dẫn dự án
├── requirements.txt            # Danh sách thư viện
│
├── config/                     # Cấu hình hệ thống
│   ├── config.ini              # Tham số cho ML Pipeline
│   └── settings.py             # Cấu hình cho EDA Pipeline
│
├── activity.log                # Nhật ký chạy (Log file)
├── check_results.py            # Xem kết quả các file joblib, pkl, json
│
├── data/                       # Dữ liệu
│   ├── raw
│   │   └── hotel_bookings.csv
│   └── processed
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/                     # Chứa Model và Kết quả đánh giá 
│   ├── best_model.pkl          # Mô hình tốt nhất
│   ├── preprocessor.joblib     # Object tiền xử lý
│   ├── comparison_*.png        # Các biểu đồ so sánh (ROC, Barplot, CM)
│   ├── evaluation_*.json       # Kết quả chi tiết dạng JSON
│   └── model_comparison_summary.csv # Bảng tổng hợp so sánh model
│
├── reports/                    # Báo cáo 
│   ├── images/                 # Thư mục chứa các ảnh trực quan hóa EDA
│   ├── FULL_EDA_REPORT.html    # Báo cáo HTML
│   ├── eda_activity            # Nhật ký chạy EDA (log file)
│   └── preprocessor.joblib     # File xử lý dữ liệu
│
└── src/                        # Mã nguồn phân tách module
    ├── eda_processor.py        # Xử lý dữ liệu EDA
    ├── eda_visualizer.py       # Trực quan hóa EDA
    ├── eda_reporter.py         # Xuất báo cáo HTML EDA
    ├── preprocessing.py        # Pipeline làm sạch & scale dữ liệu ML
    └── training/               # Thư mục module hóa mô hình
        ├── tuner.py            # Tinh chỉnh siêu tham số
        ├── evaluator.py        # Đánh giá và vẽ biểu đồ metrics
        ├── model_manager.py    # Quản lý lưu/tải model & feature importance
        └── trainer.py          # Class điều phối chung cho phần Training
```

## VI. KẾT QUẢ ĐẠT ĐƯỢC & HIỆU SUẤT MÔ HÌNH
Quá trình huấn luyện và đánh giá thực nghiệm trên tập kiểm thử đã ghi nhận các chỉ số định lượng cụ thể của từng thuật toán:

| Thuật toán | Accuracy | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | Macro Avg F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | **85,2%** | **0,83** | **0,81** | **0,82** | **0,84** |
| **XGBoost** | 84,6% | 0,82 | 0,80 | 0,81 | 0,83 |
| **LightGBM** | 84,9% | 0,82 | 0,81 | 0,81 | 0,83 |
| **RandomForest** | 83,1% | 0,80 | 0,78 | 0,79 | 0,81 |

* **Mô hình tối ưu (Best Model):** **CatBoost** đạt hiệu suất tổng thể vượt trội nhất với điểm *Macro Avg F1* đạt **0,84** và độ chính xác (*Accuracy*) đạt **85,2%**.
* **Đánh giá chung:** Các thuật toán cây tăng cường gradient giải quyết tốt bài toán phân loại nhị phân, duy trì sự cân bằng cao giữa độ chính xác và độ phủ khi dự báo trạng thái hủy phòng.

## VII. HƯỚNG PHÁT TRIỂN TƯƠNG LAI
Dựa trên những kết quả đạt được và các hạn chế còn tồn đọng của hệ thống hiện tại, các định hướng nghiên cứu và phát triển tiếp theo bao gồm:

* **Tối ưu hóa siêu tham số nâng cao:** Nâng cấp chiến lược tìm kiếm tham số từ `RandomizedSearchCV` sang các phương pháp tối ưu hóa toàn cục hiệu quả hơn như `Bayesian Optimization` hoặc `GridSearchCV` nhằm khai thác triệt để tiềm năng của các mô hình học máy.
* **Minh bạch hóa mô hình (Explainable AI):** Tích hợp các thư viện chuyên sâu như `SHAP` hoặc `LIME` để giải thích cơ chế "hộp đen" của các thuật toán, cung cấp báo cáo lý do chi tiết cho từng dự báo (ví dụ: cảnh báo rủi ro dựa trên `lead_time` lớn và không đặt cọc), giúp nhân viên nghiệp vụ tự tin hơn trong việc đưa ra quyết định can thiệp.
* **Mở rộng nguồn dữ liệu ngoại sinh:** Thu thập và tích hợp thêm các yếu tố bên ngoài có tác động mạnh mẽ đến hành vi người dùng như điều kiện thời tiết, lịch trình sự kiện tại địa phương, hoặc biến động giá vé máy bay theo mùa vụ để cải thiện độ chính xác tổng thể.
* **Triển khai ứng dụng thực tế (Deployment):** Xây dựng hệ thống API hoàn chỉnh bằng `FastAPI` kết hợp với giao diện Web trực quan, cho phép người dùng hoặc nhân viên khách sạn nhập liệu và nhận kết quả dự báo rủi ro theo thời gian thực.
* **Giám sát và bảo trì mô hình (Monitoring):** Thiết lập cơ chế theo dõi hiệu suất hệ thống liên tục để phát hiện kịp thời hiện tượng lệch dữ liệu (*Data Drift*) do sự thay đổi trong hành vi khách hàng, từ đó tiến hành quy trình huấn luyện lại (*retraining*) định kỳ.

## SỰ CỐ THƯỜNG GẶP
|        Vấn đề       |          Nguyên nhân         |                           Cách khắc phục                          |
|:-------------------:|:----------------------------:|:-----------------------------------------------------------------:|
| ModuleNotFoundError | Chưa cài thư viện            | Chạy lại lệnh: pip install -r requirements.txt                    |
| Lỗi MemoryError     | Dữ liệu quá lớn gây tràn RAM | Giảm n_jobs xuống số nhỏ (ví dụ 2). Giảm cv xuống 3 trong config. |
| File not found      | Sai đường dẫn trong config   | Kiểm tra lại mục [DATA] trong config.ini, đảm bảo tên file đúng.  |
| UnicodeDecodeError  | File config bị lỗi font chữ  | Mở file config.ini, chọn Save As -> Encoding: UTF-8.              |


