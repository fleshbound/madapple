#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from pycocotools.coco import COCO
from train.normalization_utils import get_transform, normalize_dataset_stats
from train.weather_augmentation import get_weather_augmentation

class AppleDataset(Dataset):
    def __init__(self, root_dir, annotations_path, transforms=None, weather_aug=None, is_train=True, normalize=True):
        """
        Датасет изображений яблонь с аннотациями в формате COCO.
        
        Args:
            root_dir (str): Директория с изображениями
            annotations_path (str): Путь к файлу аннотаций в формате COCO
            transforms: Преобразования изображений
            weather_aug: Дополнительные погодные преобразования
            is_train (bool): Флаг режима работы (обучение/тестирование)
            normalize (bool): Применять ли нормализацию изображений
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.weather_aug = weather_aug
        self.is_train = is_train
        self.normalize = normalize
        
        # Загрузка аннотаций COCO
        self.coco = COCO(annotations_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Получаем соответствие категорий их ID
        self.categories = {cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())}
        
        # Фильтрация изображений без аннотаций
        ids_with_annotations = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                ids_with_annotations.append(img_id)
        self.ids = ids_with_annotations
        
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Загрузка изображения
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size
        
        # Получение аннотаций для изображения
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Получение bounding boxes и меток классов
        boxes = []
        labels = []
        
        for ann in annotations:
            # Формат COCO: [x, y, width, height]
            # Преобразуем в формат [x1, y1, x2, y2]
            x, y, width, height = ann["bbox"]
            x_min = max(0, x)
            y_min = max(0, y)
            # Проверка и обрезка координат x_max и y_max до границ изображения
            x_max = min(img_width, x + width)
            y_max = min(img_height, y + height)
            
            # Проверяем, что box валидный
            if width > 0 and height > 0 and x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                # Категории в COCO начинаются с 1, переводим к формату (0 - фон)
                labels.append(ann["category_id"])
        
        # Если нет боксов, добавляем фиктивный бокс, чтобы не было ошибок
        if len(boxes) == 0:
            boxes.append([0, 0, 10, 10])  # Фиктивный маленький бокс
            labels.append(1)  # Предполагаем, что 1 - это класс "незрелое яблоко"
        
        # Преобразование в тензоры
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Создание словаря аннотаций
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        # Подготовка изображения к формату numpy array для применения аугментаций
        img_np = np.array(img)
        
        # Применение погодных аугментаций (если включены и в режиме обучения)
        if self.is_train and self.weather_aug is not None:
            img_np = self.weather_aug(image=img_np)["image"]
        
        # Применение основных аугментаций и нормализации
        if self.transforms is not None:
            # Нормализация координат боксов для Albumentations (требуется формат [0, 1])
            if len(boxes) > 0:
                # Нормализация координат до [0, 1]
                norm_boxes = boxes.clone()
                norm_boxes[:, [0, 2]] = norm_boxes[:, [0, 2]] / img_width
                norm_boxes[:, [1, 3]] = norm_boxes[:, [1, 3]] / img_height
                
                # Принудительное ограничение значений в диапазоне [0, 1]
                norm_boxes = torch.clamp(norm_boxes, min=0.0, max=1.0)
                
                # Подготовка формата для albumentations (включая bbox)
                transformed = self.transforms(
                    image=img_np, 
                    bboxes=norm_boxes.numpy(),
                    labels=labels.numpy()
                )
            else:
                # Если боксов нет, просто трансформируем изображение
                transformed = self.transforms(image=img_np)
            
            img_tensor = transformed["image"]
            
            if "bboxes" in transformed and len(transformed["bboxes"]) > 0:
                # Преобразование нормализованных координат обратно в абсолютные
                denorm_boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                denorm_boxes[:, [0, 2]] = denorm_boxes[:, [0, 2]] * img_width
                denorm_boxes[:, [1, 3]] = denorm_boxes[:, [1, 3]] * img_height
                
                target["boxes"] = denorm_boxes
                target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)
            else:
                # Если после аугментаций не осталось боксов, создаем пустой тензор
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(0, dtype=torch.int64)
        else:
            # Преобразование в тензор без аугментаций
            # ВАЖНОЕ ИСПРАВЛЕНИЕ: Преобразуем в float32 и нормализуем в диапазон [0, 1]
            img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
        
        # ПРОВЕРКА: Убедиться, что тензор имеет тип float и значения в диапазоне [0, 1]
        if not img_tensor.is_floating_point():
            img_tensor = img_tensor.float() / 255.0
        
        # Дополнительная проверка диапазона значений
        if img_tensor.min() < 0 or img_tensor.max() > 1:
            # Если значения вне диапазона [0, 1], нормализуем их
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-6)
        
        return img_tensor, target
    
    def __len__(self):
        return len(self.ids)

def compute_dataset_stats(data_path, annotations_path, sample_size=100):
    """
    Вычисление статистики датасета для нормализации.
    
    Args:
        data_path (str): Путь к директории с изображениями
        annotations_path (str): Путь к файлу аннотаций
        sample_size (int): Размер выборки для вычисления статистики
    
    Returns:
        tuple: (mean, std) - средние значения и стандартные отклонения по каналам
    """
    # Создаем COCO объект для загрузки изображений
    coco = COCO(annotations_path)
    
    # Получаем список всех изображений
    image_ids = list(coco.imgs.keys())
    
    # Выбираем случайную выборку изображений
    if len(image_ids) > sample_size:
        sample_ids = random.sample(image_ids, sample_size)
    else:
        sample_ids = image_ids
    
    # Загружаем изображения
    images = []
    for img_id in sample_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(data_path, img_info["file_name"])
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            images.append(img_np)
        except Exception as e:
            print(f"Ошибка при загрузке изображения {img_path}: {e}")
    
    # Вычисляем среднее и стандартное отклонение
    return normalize_dataset_stats(images)

def get_data_loaders(data_path, annotations_path, batch_size=8, val_size=0.2, num_workers=4, 
                     augmentation_prob=0.5, weather_prob=0.3, normalize=True, use_custom_stats=False):
    """
    Создает загрузчики данных для обучения и валидации с учетом нормализации и погодных аугментаций.
    
    Args:
        data_path (str): Путь к данным
        annotations_path (str): Путь к файлу аннотаций
        batch_size (int): Размер батча
        val_size (float): Доля данных для валидации
        num_workers (int): Количество рабочих потоков
        augmentation_prob (float): Вероятность применения аугментаций
        weather_prob (float): Вероятность применения погодных эффектов
        normalize (bool): Применять ли нормализацию
        use_custom_stats (bool): Использовать ли статистику датасета вместо ImageNet
    
    Returns:
        tuple: Загрузчики для обучения и валидации
    """
    # Вычисление статистики датасета, если требуется
    if normalize and use_custom_stats:
        print("Вычисление статистики датасета для нормализации...")
        mean, std = compute_dataset_stats(data_path, annotations_path)
        print(f"Статистика датасета: mean={mean}, std={std}")
    else:
        # Используем стандартные значения ImageNet
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if normalize:
            print(f"Использование стандартной статистики ImageNet: mean={mean}, std={std}")
    
    # Создание погодных аугментаций
    weather_aug = get_weather_augmentation(weather_prob=weather_prob) if weather_prob > 0 else None
    
    # Создание трансформаций для обучения и валидации
    train_transforms = get_transform(train=True, augmentation_prob=augmentation_prob, 
                                    normalize=normalize, custom_mean=mean, custom_std=std)
    val_transforms = get_transform(train=False, normalize=normalize, 
                                  custom_mean=mean, custom_std=std)
    
    # Создание всего датасета
    full_dataset = AppleDataset(
        root_dir=data_path,
        annotations_path=annotations_path,
        transforms=None,  # Трансформации применим позже
        weather_aug=None,  # Погодные аугментации применим позже
        is_train=True,
        normalize=normalize
    )
    
    # Разделение на обучающую и валидационную выборки
    val_size = int(len(full_dataset) * val_size)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Обновление трансформаций для каждого датасета
    train_dataset.dataset.transforms = train_transforms
    train_dataset.dataset.weather_aug = weather_aug
    val_dataset.dataset.transforms = val_transforms
    val_dataset.dataset.weather_aug = None  # Без погодных аугментаций для валидации
    
    # Создание загрузчиков данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """
    Функция для объединения элементов в батч.
    Требуется для корректной работы DataLoader с данными разного размера.
    
    Args:
        batch: Батч данных
    
    Returns:
        tuple: Кортеж изображений и целей
    """
    return tuple(zip(*batch))

# def get_transform(train=True, augmentation_prob=0.5, normalize=True, custom_mean=None, custom_std=None):
    # """
    # Создание трансформаций для обучения или валидации с поддержкой нормализации.
    
    # Args:
        # train (bool): Флаг режима трансформаций (обучение/валидация)
        # augmentation_prob (float): Вероятность применения аугментаций
        # normalize (bool): Применять ли нормализацию
        # custom_mean (list, optional): Пользовательские средние значения для нормализации
        # custom_std (list, optional): Пользовательские стандартные отклонения для нормализации
    
    # Returns:
        # albumentations.Compose: Набор трансформаций
    # """
    # # Средние значения и стандартные отклонения для нормализации
    # mean = custom_mean if custom_mean is not None else [0.485, 0.456, 0.406]  # ImageNet по умолчанию
    # std = custom_std if custom_std is not None else [0.229, 0.224, 0.225]  # ImageNet по умолчанию
    
    # if train:
        # transforms = [
            # # Геометрические преобразования
            # A.HorizontalFlip(p=augmentation_prob),
            # A.VerticalFlip(p=augmentation_prob * 0.3),  # Вертикальный флип реже
            # A.Rotate(limit=30, p=augmentation_prob * 0.7, border_mode=0),  # Поворот с обработкой границ
            
            # # Преобразования цвета и контраста
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=augmentation_prob),
            # A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, 
                                # p=augmentation_prob * 0.5),
            # A.GaussianBlur(blur_limit=(3, 5), p=augmentation_prob * 0.3),
            
            # # Имитация разного освещения
            # A.RandomShadow(p=augmentation_prob * 0.3),
            # A.RandomSunFlare(p=augmentation_prob * 0.1),
            
            # # Добавление шума
            # A.GaussNoise(var_limit=(10, 50), p=augmentation_prob * 0.2),
            
            # # Масштабирование и обрезка
            # A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=augmentation_prob * 0.5),
        # ]
        
        # # Добавляем имитацию пасмурной погоды
        # weather_transform = A.OneOf([
            # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
            # A.RandomRain(p=0.3),
            # A.RandomSnow(p=0.2),
        # ], p=augmentation_prob * 0.4)
        
        # transforms.append(weather_transform)
        
    # else:  # Валидация
        # transforms = [
            # A.Resize(512, 512),
        # ]
    
    # # ВАЖНОЕ ИЗМЕНЕНИЕ: Преобразование к типу float32 и нормализация
    # # Сначала преобразуем к диапазону [0,1]
    # transforms.append(A.ToFloat(max_value=255.0))
    
    # # Добавление нормализации, если требуется
    # #if normalize:
    # #    transforms.append(A.Normalize(mean=mean, std=std))
    
    # # Финальное преобразование в тензор
    # transforms.append(ToTensorV2())
    
    # return A.Compose(
        # transforms, 
        # bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    # )   