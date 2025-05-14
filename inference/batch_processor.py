#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для работы с изображениями в форме батча.
"""

import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from image_processor import preprocess_image
from detector import detect_apples
from config import MODEL_INPUT_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD

def process_image_batch(image_paths, model, threshold=0.5, device="cpu"):
    """
    Обрабатывает пакет изображений для обнаружения яблок.
    
    Args:
        image_paths (list): Список путей к изображениям
        model: Модель Faster R-CNN
        threshold (float): Порог уверенности
        device (str): Устройство для вычислений
    
    Returns:
        list: Список результатов обнаружения для каждого изображения
    """
    results = []
    
    # Переводим модель в режим оценки
    model = model.to(device)
    model.eval()
    
    # Обрабатываем каждое изображение
    for image_path in image_paths:
        try:
            # Предобработка изображения
            image_tensor, original_image, (height, width) = preprocess_image(image_path)
            image_tensor = image_tensor.to(device)
            
            # Обнаружение яблок
            detection_result = detect_apples(model, image_tensor, (height, width), threshold, device)
            
            # Добавляем путь к изображению в результаты
            detection_result['image_path'] = image_path
            
            results.append(detection_result)
        except Exception as e:
            # В случае ошибки добавляем пустой результат
            print(f"Ошибка при обработке изображения {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([]),
                'error': str(e)
            })
    
    return results

def create_batch_transform(target_size=MODEL_INPUT_SIZE):
    """
    Создает трансформацию для батча изображений.
    
    Args:
        target_size (int): Целевой размер изображения (квадратный)
    
    Returns:
        albumentations.Compose: Трансформация для изображений
    """
    transform = A.Compose([
        A.Resize(target_size, target_size),
        A.ToFloat(max_value=255.0),
       #A.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ToTensorV2()
    ])
    
    return transform

def process_directory(directory_path, model, threshold=0.5, device="cpu", recursive=False):
    """
    Обрабатывает все изображения в указанной директории.
    
    Args:
        directory_path (str): Путь к директории с изображениями
        model: Модель Faster R-CNN
        threshold (float): Порог уверенности
        device (str): Устройство для вычислений
        recursive (bool): Обрабатывать поддиректории рекурсивно
    
    Returns:
        dict: Словарь с результатами обнаружения для каждого изображения
    """
    # Поддерживаемые расширения файлов изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # Список путей к изображениям
    image_paths = []
    
    # Функция для добавления файлов из директории
    def add_files_from_dir(dir_path):
        for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            
            # Проверяем, является ли файл изображением
            if os.path.isfile(full_path) and filename.lower().endswith(image_extensions):
                image_paths.append(full_path)
            
            # Рекурсивная обработка поддиректорий
            elif recursive and os.path.isdir(full_path):
                add_files_from_dir(full_path)
    
    # Добавляем файлы из указанной директории
    add_files_from_dir(directory_path)
    
    # Обрабатываем изображения
    results = process_image_batch(image_paths, model, threshold, device)
    
    # Преобразуем список результатов в словарь для удобства доступа
    results_dict = {result['image_path']: result for result in results}
    
    return results_dict