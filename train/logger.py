#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(log_file_path=None):
    """
    Настройка логгера с выводом в консоль и файл.
    
    Args:
        log_file_path (str): Путь к файлу логов
    
    Returns:
        logging.Logger: Настроенный логгер
    """
    # Отключаем буферизацию для sys.stdout
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    # Настройка корневого логгера
    logging.basicConfig(level=logging.INFO)
    
    # Создание логгера
    logger = logging.getLogger('apple_detector')
    logger.setLevel(logging.INFO)
    
    # Явно указываем, что сообщения должны передаваться родительскому логгеру
    logger.propagate = True
    
    # Очистка обработчиков, если они уже были созданы
    if logger.handlers:
        logger.handlers = []
    
    # Формат сообщений логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Добавление обработчика для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Принудительная очистка буфера после каждого сообщения
    console_handler.flush()
    
    # Добавление обработчика для вывода в файл, если задан путь
    if log_file_path:
        # Создание директории для логов, если не существует
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Создание обработчика для файла логов с ротацией (до 5 файлов по 5 МБ)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=5 * 1024 * 1024,  # 5 МБ
            backupCount=5,
            encoding='utf-8',
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Проверка настройки логгера
    logger.info("Logger setup complete. This message should appear in both console and log file.")
    
    return logger