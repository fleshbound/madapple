import os
import sys
from datetime import datetime


def set_logger():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Логи в файл
    log_file = open(f"{log_dir}/training.log", "w")
    sys.stdout = log_file

