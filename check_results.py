import joblib
import json
import os
import sys

# Thêm đường dẫn hiện tại
sys.path.append('.')

def in_du_lieu_de_quy(data, indent=0):
    """
    - Tự động thụt đầu dòng theo cấp độ.
    - Hiển thị 5 phần tử đầu và ẩn phần còn lại nếu danh sách quá dài.
    """
    spacing = " " * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            # In Key trước
            print(f"{spacing}- {key}:", end="")
            
            if isinstance(value, (dict, list)):
                if isinstance(value, list) and len(value) <= 5:
                    print(f" {value}")
                else:
                    print() # Xuống dòng
                    in_du_lieu_de_quy(value, indent + 4) # Tăng thụt lề
            else:
                # Value đơn giản (số, chuỗi), in ra màn hình luôn
                print(f" {value}")
                
    elif isinstance(data, list):
        if len(data) > 5:
            # In 5 phần tử đầu 
            preview = data[:5]
            preview_str = str(preview)[:-1] 
            
            type_name = type(data[0]).__name__ if data else "unknown"
            
            print(f"{spacing}{preview_str}, ... <Còn {len(data) - 5} phần tử dạng '{type_name}'>]")
        else:
            # Danh sách ngắn thì in ra 
            print(f"{spacing}{data}")
            
    else:
        # Các kiểu dữ liệu khác
        print(f"{spacing}{data}")

def xem_file_pkl_joblib(duong_dan):
    print("\n" + "-"*30)
    print(f"ĐANG ĐỌC FILE MODEL/PKL: {duong_dan}")
    
    try:
        data = joblib.load(duong_dan)
        print("*** Đã load xong!")
        print(f"Loại dữ liệu: {type(data)}")
        
        # 1. NẾU LÀ MODEL
        if hasattr(data, 'get_params'):
            print("\n[THAM SỐ CỦA MODEL]:")
            params = data.get_params()
            in_du_lieu_de_quy(params)

        # 2. FEATURE IMPORTANCES
        if hasattr(data, 'feature_importances_'):
            print("\n[TOP 5 ĐẶC TRƯNG QUAN TRỌNG]:")
            scores = data.feature_importances_
            
            feature_names = []
            if hasattr(data, 'feature_names_in_'):
                feature_names = data.feature_names_in_
            elif hasattr(data, 'feature_names_'):
                feature_names = data.feature_names_
            
            danh_sach = []
            for i in range(len(scores)):
                diem = scores[i]
                ten = feature_names[i] if len(feature_names) > i else f"Feature_{i}"
                danh_sach.append((ten, diem))
            
            danh_sach_sap_xep = sorted(danh_sach, key=lambda x: x[1], reverse=True)
            
            for i in range(5):
                if i < len(danh_sach_sap_xep):
                    print(f"  {i+1}. {danh_sach_sap_xep[i][0]}: {danh_sach_sap_xep[i][1]:.4f}")

        # 3. THÔNG TIN KHÁC
        if not hasattr(data, 'fit'): 
            print("\n[THÔNG TIN OBJECT]:")
            try:
                in_du_lieu_de_quy(vars(data))
            except:
                print("Không đọc được chi tiết biến bên trong.")

    except Exception as loi:
        print(f"*** Có lỗi xảy ra khi đọc pkl: {loi}")

def xem_file_json(duong_dan):
    print("\n" + "-"*30)
    print(f"ĐANG ĐỌC FILE JSON: {duong_dan}")
    print("-" * 30)
    
    try:
        with open(duong_dan, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Dùng hàm thông minh mới viết để in
        in_du_lieu_de_quy(data)
                
    except Exception as loi:
        print(f"*** Lỗi đọc JSON: {loi}")

def lay_danh_sach_file():
    ds_file = []
    cac_thu_muc = ['models', 'reports']
    
    for thu_muc in cac_thu_muc:
        if os.path.exists(thu_muc):
            file_trong_folder = os.listdir(thu_muc)
            for ten_file in file_trong_folder:
                if ten_file.endswith('.pkl') or ten_file.endswith('.joblib') or ten_file.endswith('.json'):
                    duong_dan_day_du = os.path.join(thu_muc, ten_file)
                    ds_file.append(duong_dan_day_du)
    return ds_file

def main():
    while True:
        print("\n" + "-"*40)
        print(" KIỂM TRA FILE KẾT QUẢ")
        print("-"*40)
        
        danh_sach = lay_danh_sach_file()
        
        if len(danh_sach) == 0:
            print("*** Không tìm thấy file kết quả nào trong 'models/' hoặc 'reports/'")
            break

        for i in range(len(danh_sach)):
            print(f"[{i + 1}] {danh_sach[i]}")
        print("[0] Thoát")

        chon = input("\n*** Nhập số file muốn xem: ")

        if chon == '0':
            print("Đã thoát.")
            break
        
        try:
            so_thu_tu = int(chon) - 1
            if 0 <= so_thu_tu < len(danh_sach):
                file_duoc_chon = danh_sach[so_thu_tu]
                
                if file_duoc_chon.endswith('.json'):
                    xem_file_json(file_duoc_chon)
                else:
                    xem_file_pkl_joblib(file_duoc_chon)
                
                input("\n(Ấn Enter để quay lại menu...)")
            else:
                print("*** Số nhập không đúng!")
        except ValueError:
            print("*** Vui lòng nhập số!")

if __name__ == "__main__":
    main()
    
    # Nhập lệnh sau trong Terminal
    # python check_results.py