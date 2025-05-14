#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для настройки констант и конфигураций.
"""

# Размер входного изображения для модели
MODEL_INPUT_SIZE = 512

# Классы яблок
APPLE_CLASSES = {
    0: "Background",
    1: "Unripe apple",
    2: "Ripe apple"
}

# Параметры нормализации (значения ImageNet)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Цвета для визуализации (в RGB)
VISUALIZATION_COLORS = {
    1: (255, 255, 0),  # Желтый для незрелых яблок
    2: (255, 0, 0)     # Красный для зрелых яблок
}