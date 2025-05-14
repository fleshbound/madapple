#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для обнаружения яблок на изображении.
"""

import torch

def detect_apples(model, image_tensor, original_size, threshold=0.5, device="cpu"):
    """
    Обнаруживает яблоки на изображении и масштабирует боксы к исходному размеру.
    
    Args:
        model: Модель Faster R-CNN
        image_tensor (torch.Tensor): Тензор изображения
        original_size (tuple): Исходные размеры изображения (height, width)
        threshold (float): Порог уверенности
        device (str): Устройство для вычислений
    
    Returns:
        dict: Словарь с результатами обнаружения
    """
    # Перемещаем модель и данные на нужное устройство
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Добавляем размерность батча
    image_tensor = image_tensor.unsqueeze(0)
    
    # Получаем предсказания
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    # Фильтруем предсказания по порогу уверенности
    mask = prediction['scores'] >= threshold
    boxes = prediction['boxes'][mask].cpu().numpy()
    scores = prediction['scores'][mask].cpu().numpy()
    labels = prediction['labels'][mask].cpu().numpy()
    
    # Масштабируем боксы к исходному размеру
    original_height, original_width = original_size
    # Коэффициенты масштабирования
    width_scale = original_width / 512
    height_scale = original_height / 512
    
    # Применяем масштабирование к каждому боксу
    scaled_boxes = boxes.copy()
    scaled_boxes[:, 0] *= width_scale  # x1
    scaled_boxes[:, 1] *= height_scale  # y1
    scaled_boxes[:, 2] *= width_scale  # x2
    scaled_boxes[:, 3] *= height_scale  # y2
    
    return {
        'boxes': scaled_boxes,
        'scores': scores,
        'labels': labels
    }