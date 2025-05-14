#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для предобработки изображений перед детекцией.
"""

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess_image(image_path):
    """
    Загружает и предобрабатывает изображение для модели.
    
    Args:
        image_path (str): Путь к изображению
    
    Returns:
        tuple: (tensor, original_image, dimensions)
    """
    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    
    # Сохраняем оригинальные размеры
    height, width, _ = original_image.shape
    
    # Создаем трансформации (без аугментаций, только нормализация)
    transform = A.Compose([
        A.Resize(512, 512),
        A.ToFloat(max_value=255.0),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Применяем трансформации
    transformed = transform(image=original_image)
    image_tensor = transformed["image"]
    
    # Проверка типа и диапазона значений
    if not image_tensor.is_floating_point():
        image_tensor = image_tensor.float()
    
    # Возвращаем тензор, оригинальное изображение и размеры
    return image_tensor, original_image, (height, width)