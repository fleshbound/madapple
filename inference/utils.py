#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль с утилитами для работы с программой обнаружения яблок.
"""

import os
import torch
import numpy as np
import time

class Timer:
    """
    Класс для замера времени выполнения кода.
    """
    def __init__(self, name="Operation", verbose=True):
        self.name = name
        self.verbose = verbose
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.verbose:
            print(f"{self.name} выполнена за {self.elapsed_time:.4f} секунд")

def ensure_dir(directory):
    """
    Создает директорию, если она не существует.
    
    Args:
        directory (str): Путь к директории
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_device():
    """
    Определяет доступное устройство для вычислений.
    
    Returns:
        torch.device: Устройство (CPU или CUDA)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_iou(box1, box2):
    """
    Вычисляет IoU (Intersection over Union) для двух боксов.
    
    Args:
        box1 (np.ndarray): Первый бокс в формате [x1, y1, x2, y2]
        box2 (np.ndarray): Второй бокс в формате [x1, y1, x2, y2]
    
    Returns:
        float: Значение IoU (0-1)
    """
    # Определяем координаты пересечения боксов
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Вычисляем площадь пересечения
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Вычисляем площади боксов
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Вычисляем IoU
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0:
        return 0
    
    return intersection_area / union_area

def count_apples_by_type(results):
    """
    Подсчитывает количество яблок каждого типа в результатах обнаружения.
    
    Args:
        results (dict): Результаты обнаружения
    
    Returns:
        tuple: (unripe_count, ripe_count, total_count)
    """
    if 'labels' not in results:
        return 0, 0, 0
    
    unripe_count = sum(1 for label in results['labels'] if label == 1)
    ripe_count = sum(1 for label in results['labels'] if label == 2)
    total_count = unripe_count + ripe_count
    
    return unripe_count, ripe_count, total_count