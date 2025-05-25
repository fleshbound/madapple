#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Простая программа для просмотра случайного изображения из датасета
с аугментациями и визуализацией bounding boxes.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from train.dataset import AppleDataset
from train.normalization_utils import get_transform

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Денормализация изображения для корректного отображения.
    
    Args:
        tensor: Нормализованный тензор изображения (C, H, W)
        mean: Средние значения для денормализации
        std: Стандартные отклонения для денормализации
    
    Returns:
        numpy.ndarray: Денормализованное изображение (H, W, C)
    """
    # Клонируем тензор чтобы не изменить оригинал
    img = tensor.clone()
    
    # Денормализация
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    
    # Ограничиваем значения диапазоном [0, 1]
    img = torch.clamp(img, 0, 1)
    
    # Преобразуем в numpy и меняем порядок каналов
    img = img.permute(1, 2, 0).numpy()
    
    return img

def visualize_sample(data_path, annotations_file, use_augmentations=True, sample_index=None):
    """
    Визуализирует случайное изображение из датасета с bounding boxes.
    
    Args:
        data_path (str): Путь к директории с изображениями
        annotations_file (str): Имя файла аннотаций COCO
        use_augmentations (bool): Применять ли аугментации
        sample_index (int, optional): Индекс конкретного изображения (если None, выбирается случайно)
    """
    
    # Создаем трансформации
    if use_augmentations:
        transforms = get_transform(
            train=True, 
            augmentation_prob=0.6,  # Высокая вероятность для демонстрации
            normalize=True
        )
        title_suffix = "с аугментациями"
    else:
        transforms = get_transform(
            train=False,  # Без аугментаций
            normalize=True
        )
        title_suffix = "без аугментаций"
    
    # Создаем датасет
    dataset = AppleDataset(
        root_dir=data_path,
        annotations_path=os.path.join(data_path, annotations_file),
        transforms=transforms,
        is_train=use_augmentations,
        normalize=True
    )
    
    print(f"Загружен датасет с {len(dataset)} изображениями")
    
    # Выбираем случайный индекс, если не указан конкретный
    if sample_index is None:
        sample_index = random.randint(0, len(dataset) - 1)
    else:
        sample_index = min(sample_index, len(dataset) - 1)
    
    print(f"Показываем изображение #{sample_index}")
    
    # Получаем изображение и аннотации
    image_tensor, target = dataset[sample_index]
    
    # Денормализуем изображение для отображения
    # image = denormalize_image(image_tensor)
    image = image_tensor.permute(1, 2, 0).numpy()
    
    # Получаем данные о bounding boxes
    boxes = target['boxes'].numpy() if len(target['boxes']) > 0 else []
    labels = target['labels'].numpy() if len(target['labels']) > 0 else []
    
    # Создаем фигуру
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Отображаем изображение
    ax.imshow(image)
    ax.set_title(f'Изображение из датасета ({title_suffix})', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Цвета и названия классов
    colors = {
        1: 'black',    # Незрелое яблоко
        2: 'white'        # Зрелое яблоко
    }
    
    class_names = {
        1: 'Незрелое',
        2: 'Зрелое'
    }
    
    # Счетчики для статистики
    unripe_count = 0
    ripe_count = 0
    
    # Рисуем bounding boxes
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Создаем прямоугольник
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3,
            edgecolor=colors.get(label, 'white'),
            facecolor='none',
            linestyle='-' if label == 2 else '--'
        )
        ax.add_patch(rect)
        
        # Добавляем текстовую метку
        #class_name = class_names.get(label, f'Class {label}')
        #ax.text(
        #    x1, y1 - 5,
        #    class_name,
        #    color=colors.get(label, 'white'),
        #    fontsize=12,
        #    fontweight='bold',
        #    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7)
        #)
        
        # Подсчитываем объекты
        if label == 1:
            unripe_count += 1
        elif label == 2:
            ripe_count += 1
    
    # Добавляем статистику в заголовок
    #total_count = unripe_count + ripe_count
    #stats_text = f"Найдено: {unripe_count} незрелых, {ripe_count} зрелых (всего: {total_count})"
    
    # Добавляем текст со статистикой
    #ax.text(
    #    0.02, 0.98, stats_text,
    #    transform=ax.transAxes,
    #    fontsize=12,
    #    verticalalignment='top',
    #    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    #)
    
    # Создаем легенду
    #legend_elements = []
    #for label, color in colors.items():
    #    legend_elements.append(
    #        patches.Patch(color=color, label=class_names[label])
    #    )
    
    #ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Выводим дополнительную информацию
    print(f"\nИнформация об изображении:")
    print(f"Размер тензора: {image_tensor.shape}")
    print(f"Размер изображения для отображения: {image.shape}")
    print(f"Количество объектов: {len(boxes)}")
    print(f"Незрелых яблок: {unripe_count}")
    print(f"Зрелых яблок: {ripe_count}")
    
    if len(boxes) > 0:
        print(f"\nДетали bounding boxes:")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            print(f"  Объект {i+1}: {class_names.get(label, f'Class {label}')} "
                  f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                  f"(размер: {x2-x1:.1f}x{y2-y1:.1f})")

def main():
    """Основная функция программы."""
    
    # Настройки по умолчанию
    data_path = '../data/train'  # Путь к изображениям
    annotations_file = '_annotations.coco.json'  # Файл аннотаций
    
    print("=" * 60)
    print("ПРОСТОЙ ПРОСМОТРЩИК ДАТАСЕТА ЯБЛОК")
    print("=" * 60)
    
    # Проверяем существование файлов
    if not os.path.exists(data_path):
        print(f"❌ Ошибка: Директория {data_path} не найдена!")
        print("Убедитесь, что путь к данным указан правильно.")
        return
    
    annotations_path = os.path.join(data_path, annotations_file)
    if not os.path.exists(annotations_path):
        print(f"❌ Ошибка: Файл аннотаций {annotations_path} не найден!")
        print("Убедитесь, что файл аннотаций существует.")
        return
    
    try:
        # Устанавливаем seed для воспроизводимости
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        print("🎯 Показываем случайное изображение С аугментациями...")
        visualize_sample(data_path, annotations_file, use_augmentations=True)
        
        input("\n👆 Нажмите Enter, чтобы показать то же изображение БЕЗ аугментаций...")
        
        # Сброс seed для получения того же изображения
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        print("🎯 Показываем то же изображение БЕЗ аугментаций...")
        visualize_sample(data_path, annotations_file, use_augmentations=False)
        
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        print("\nВозможные причины:")
        print("1. Неправильный путь к данным")
        print("2. Поврежденный файл аннотаций")
        print("3. Отсутствуют зависимости (dataset.py, train/normalization_utils.py)")

if __name__ == "__main__":
    main()