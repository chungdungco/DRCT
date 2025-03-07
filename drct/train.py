import os
import os.path as osp
import sys

# Thêm đường dẫn DRCT vào PYTHONPATH
sys.path.append('/content/DRCT')

# Import các module từ DRCT
from drct.archs import *
from drct.data import *
from drct.models import *

# Import từ BasicSR
from basicsr.train import train_pipeline

if __name__ == '__main__':
    # Đường dẫn đến thư mục gốc của DRCT
    root_path = '/content/DRCT'
    
    # Đường dẫn đến file cấu hình YAML
    config_path = osp.join(root_path, 'options', 'train', 'train_DRCT_SRx2_from_scratch.yml')
    
    # Đường dẫn để lưu kết quả huấn luyện
    experiments_root = '/content/DRCT/experiments'
    
    # Tạo thư mục experiments nếu chưa tồn tại
    os.makedirs(experiments_root, exist_ok=True)
    
    # In ra đường dẫn experiments_root để kiểm tra
    print(f"experiments_root: {experiments_root}")
    
    # Chạy training pipeline
    train_pipeline(root_path)
