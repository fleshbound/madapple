#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

def normalize_imagenet(image):
    """
    Нормализация изображения с использованием средних значений и стандартных отклонений ImageNet.
    
    Args:
        image (PIL.Image или numpy.ndarray): Входное изображение
    
    Returns:
        numpy.ndarray или torch.Tensor: Нормализованное изображение
    """
    # Средние значения и стандартные отклонения каналов RGB для ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Проверка типа входного изображения
    if isinstance(image, Image.Image):
        # Преобразование PIL Image в тензор PyTorch
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        return transform(image)
    
    elif isinstance(image, np.ndarray):
        # Нормализация numpy массива
        transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        # Убедимся, что у изображения правильный формат (H, W, C)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return transform(image=image)["image"]
    
    elif isinstance(image, torch.Tensor):
        # Если это уже тензор PyTorch, просто нормализуем его
        # Предполагается, что формат (C, H, W) и значения [0, 1]
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        mean_tensor = torch.tensor(mean, device=image.device).view(-1, 1, 1)
        std_tensor = torch.tensor(std, device=image.device).view(-1, 1, 1)
        
        return (image - mean_tensor) / std_tensor
    
    else:
        raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")

def normalize_dataset_stats(images):
    """
    Вычисление среднего и стандартного отклонения для набора изображений.
    
    Args:
        images (list): Список изображений (numpy.ndarray или torch.Tensor)
    
    Returns:
        tuple: (mean, std) - средние значения и стандартные отклонения по каналам
    """
    # Преобразование всех изображений в тензоры, если это необходимо
    tensors = []
    for img in images:
        if isinstance(img, np.ndarray):
            # Предполагается формат (H, W, C) и значения [0, 255]
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        elif isinstance(img, torch.Tensor):
            # Предполагается формат (C, H, W) или (H, W, C)
            if img.shape[0] != 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        else:
            continue
        tensors.append(img)
    
    if not tensors:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # Значения ImageNet по умолчанию
    
    # Объединение всех тензоров
    batch = torch.stack(tensors)
    
    # Вычисление среднего и стандартного отклонения
    mean = batch.mean(dim=(0, 2, 3)).tolist()
    std = batch.std(dim=(0, 2, 3)).tolist()
    
    return mean, std

def get_transform(train=True, augmentation_prob=0.5, normalize=True, custom_mean=None, custom_std=None):
    """
    Создание трансформаций для обучения или валидации с поддержкой нормализации.
    
    Args:
        train (bool): Флаг режима трансформаций (обучение/валидация)
        augmentation_prob (float): Вероятность применения аугментаций
        normalize (bool): Применять ли нормализацию
        custom_mean (list, optional): Пользовательские средние значения для нормализации
        custom_std (list, optional): Пользовательские стандартные отклонения для нормализации
    
    Returns:
        albumentations.Compose: Набор трансформаций
    """
    # Средние значения и стандартные отклонения для нормализации
    mean = custom_mean if custom_mean is not None else [0.485, 0.456, 0.406]  # ImageNet по умолчанию
    std = custom_std if custom_std is not None else [0.229, 0.224, 0.225]  # ImageNet по умолчанию
    
    if train:
        transforms = [
            # Геометрические преобразования
            A.HorizontalFlip(p=augmentation_prob),
            A.VerticalFlip(p=augmentation_prob * 0.3),  # Вертикальный флип реже
            A.Rotate(limit=30, p=augmentation_prob * 0.7, border_mode=0),  # Поворот с обработкой границ
            
            # Преобразования цвета и контраста
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=augmentation_prob),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, 
                                p=augmentation_prob * 0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=augmentation_prob * 0.3),
            
            # Имитация разного освещения
            A.RandomShadow(p=augmentation_prob * 0.3),
            A.RandomSunFlare(p=augmentation_prob * 0.1),
            
            # Добавление шума
            A.GaussNoise(var_limit=(10, 50), p=augmentation_prob * 0.2),
            
            # Масштабирование и обрезка
            A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=augmentation_prob * 0.5),
        ]
        
        # Добавляем имитацию пасмурной погоды
        weather_transform = A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
            A.RandomRain(p=0.3),
            A.RandomSnow(p=0.2),
        ], p=augmentation_prob * 0.4)
        
        transforms.append(weather_transform)
        
    else:  # Валидация
        transforms = [
            A.Resize(512, 512),
        ]
    
    # Добавление нормализации, если требуется
    if normalize:
        transforms.append(A.Normalize(mean=mean, std=std))
    
    # Финальное преобразование в тензор
    transforms.append(ToTensorV2())
    
    return A.Compose(
        transforms, 
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )

def visualize_normalization(image):
    """
    Визуализация эффекта нормализации на изображении.
    
    Args:
        image (PIL.Image или numpy.ndarray): Входное изображение
    """
    # Преобразуем в numpy, если это не так
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # Создаем копии для разных преобразований
    image_normalized = normalize_imagenet(image_np)
    
    # Создаем трансформации с разными параметрами нормализации
    transform_imagenet = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transform_zero_one = A.Compose([
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Просто масштабирование до [0, 1]
        ToTensorV2()
    ])
    
    transform_custom = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Масштабирование до [-1, 1]
        ToTensorV2()
    ])
    
    # Применяем трансформации
    image_imagenet = transform_imagenet(image=image_np)["image"]
    image_zero_one = transform_zero_one(image=image_np)["image"]
    image_custom = transform_custom(image=image_np)["image"]
    
    # Преобразуем тензоры обратно для визуализации
    def tensor_to_img(tensor):
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        return tensor.numpy()
    
    # Денормализация для корректного отображения
    def denormalize(tensor, mean, std):
        if tensor.shape[0] == 3:
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
        return tensor
    
    image_imagenet_vis = tensor_to_img(denormalize(image_imagenet.clone(), 
                                                 [0.485, 0.456, 0.406], 
                                                 [0.229, 0.224, 0.225]))
    image_zero_one_vis = tensor_to_img(image_zero_one)
    image_custom_vis = tensor_to_img(denormalize(image_custom.clone(), 
                                                [0.5, 0.5, 0.5], 
                                                [0.5, 0.5, 0.5]))
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Оригинальное изображение")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image_imagenet_vis)
    axes[0, 1].set_title("ImageNet нормализация\nmean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(image_zero_one_vis)
    axes[1, 0].set_title("Масштабирование [0, 1]\nmean=[0, 0, 0], std=[1, 1, 1]")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image_custom_vis)
    axes[1, 1].set_title("Масштабирование [-1, 1]\nmean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig