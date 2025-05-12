#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import random
import albumentations as A

def simulate_overcast(image, intensity=0.5):
    """
    Имитация пасмурной погоды.
    
    Args:
        image: PIL.Image или numpy.ndarray
        intensity: Интенсивность эффекта (0.0 - 1.0)
    
    Returns:
        PIL.Image или numpy.ndarray: Изображение с эффектом пасмурной погоды
    """
    # Преобразование в PIL.Image, если это numpy массив
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
        is_numpy = True
    else:
        pil_image = image
        is_numpy = False
    
    # Уменьшение контраста
    contrast_factor = 1.0 - intensity * 0.5  # От 1.0 до 0.5
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)
    
    # Уменьшение яркости
    brightness_factor = 1.0 - intensity * 0.3  # От 1.0 до 0.7
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_factor)
    
    # Уменьшение насыщенности
    saturation_factor = 1.0 - intensity * 0.4  # От 1.0 до 0.6
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(saturation_factor)
    
    # Добавление размытия для имитации тумана
    blur_radius = intensity * 1.5  # От 0.0 до 1.5
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Возвращаем результат в том же формате, что и входное изображение
    if is_numpy:
        return np.array(pil_image)
    else:
        return pil_image

def simulate_fog(image, intensity=0.5):
    """
    Имитация тумана.
    
    Args:
        image: PIL.Image или numpy.ndarray
        intensity: Интенсивность эффекта (0.0 - 1.0)
    
    Returns:
        PIL.Image или numpy.ndarray: Изображение с эффектом тумана
    """
    # Преобразование в numpy, если это PIL.Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        is_pil = True
    else:
        image_np = image.copy()
        is_pil = False
    
    # Создание тумана
    fog_transform = A.Compose([
        A.RandomFog(fog_coef_lower=intensity*0.2, fog_coef_upper=intensity*0.5, alpha_coef=0.1)
    ])
    
    # Применение трансформации
    image_fogged = fog_transform(image=image_np)["image"]
    
    # Возвращаем результат в том же формате, что и входное изображение
    if is_pil:
        return Image.fromarray(image_fogged)
    else:
        return image_fogged

def simulate_rain(image, intensity=0.5):
    """
    Имитация дождя.
    
    Args:
        image: PIL.Image или numpy.ndarray
        intensity: Интенсивность эффекта (0.0 - 1.0)
    
    Returns:
        PIL.Image или numpy.ndarray: Изображение с эффектом дождя
    """
    # Преобразование в numpy, если это PIL.Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        is_pil = True
    else:
        image_np = image.copy()
        is_pil = False
    
    # Создание дождя
    rain_transform = A.Compose([
        A.RandomRain(
            slant_lower=-10, 
            slant_upper=10, 
            drop_length=int(20 * intensity), 
            drop_width=1, 
            drop_color=(200, 200, 200), 
            blur_value=3,
            rain_type='drizzle' if intensity < 0.5 else None
        )
    ])
    
    # Применение трансформации
    image_rainy = rain_transform(image=image_np)["image"]
    
    # Возвращаем результат в том же формате, что и входное изображение
    if is_pil:
        return Image.fromarray(image_rainy)
    else:
        return image_rainy

def simulate_low_light(image, intensity=0.5):
    """
    Имитация низкой освещенности.
    
    Args:
        image: PIL.Image или numpy.ndarray
        intensity: Интенсивность эффекта (0.0 - 1.0)
    
    Returns:
        PIL.Image или numpy.ndarray: Изображение с эффектом низкой освещенности
    """
    # Преобразование в PIL.Image, если это numpy массив
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
        is_numpy = True
    else:
        pil_image = image
        is_numpy = False
    
    # Уменьшение яркости
    brightness_factor = 1.0 - intensity * 0.7  # От 1.0 до 0.3
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_factor)
    
    # Увеличение контраста
    contrast_factor = 1.0 + intensity * 0.2  # От 1.0 до 1.2
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)
    
    # Возвращаем результат в том же формате, что и входное изображение
    if is_numpy:
        return np.array(pil_image)
    else:
        return pil_image

def visualize_weather_effects(image):
    """
    Визуализация различных погодных эффектов.
    
    Args:
        image: PIL.Image или numpy.ndarray
    
    Returns:
        matplotlib.figure.Figure: Фигура с визуализацией
    """
    # Преобразуем в numpy, если это не так
    if isinstance(image, Image.Image):
        original = np.array(image)
    else:
        original = image.copy()
    
    # Применяем различные эффекты
    overcast_light = simulate_overcast(original, intensity=0.3)
    overcast_heavy = simulate_overcast(original, intensity=0.7)
    
    fog_light = simulate_fog(original, intensity=0.3)
    fog_heavy = simulate_fog(original, intensity=0.7)
    
    rain_light = simulate_rain(original, intensity=0.3)
    rain_heavy = simulate_rain(original, intensity=0.7)
    
    low_light = simulate_low_light(original, intensity=0.6)
    
    # Автоматическая аугментация погодных условий
    aug = A.Compose([
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            A.RandomRain(p=0.3),
            # A.RandomSnow(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
        ], p=1.0)
    ])
    
    albumentations_effect = aug(image=original)["image"]
    
    # Визуализация
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Оригинальное изображение")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(overcast_light)
    axes[0, 1].set_title("Легкая облачность")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(overcast_heavy)
    axes[0, 2].set_title("Сильная облачность")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(fog_light)
    axes[1, 0].set_title("Легкий туман")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fog_heavy)
    axes[1, 1].set_title("Сильный туман")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(rain_light)
    axes[1, 2].set_title("Легкий дождь")
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(rain_heavy)
    axes[2, 0].set_title("Сильный дождь")
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(low_light)
    axes[2, 1].set_title("Низкая освещенность")
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(albumentations_effect)
    axes[2, 2].set_title("Albumentations автоэффект")
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def get_weather_augmentation(weather_prob=0.5):
    """
    Создание трансформации для аугментации погодных условий.
    
    Args:
        weather_prob (float): Вероятность применения погодных эффектов
    
    Returns:
        albumentations.Compose: Трансформация для аугментации погодных условий
    """
    return A.Compose([
        A.OneOf([
            # Туман
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.4),
            
            # Дождь
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=8, drop_width=1, 
                         drop_color=(200, 200, 200), p=0.3),
            
            # Снег
            # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.0, p=0.1),
            
            # Изменение освещения
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), contrast_limit=(-0.2, 0.2), p=0.3),
            
            # Тени
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.2),
            
            # Солнечные блики
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, 
                            num_flare_circles_lower=1, num_flare_circles_upper=3, p=0.1),
        ], p=weather_prob)
    ])