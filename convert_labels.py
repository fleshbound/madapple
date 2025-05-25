#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Конвертер аннотаций из формата COCO JSON в формат YOLO TXT.

Использование:
    python convert_coco_to_yolo.py --coco_json _annotations.coco.json --images_dir data/train --output_dir data/train_yolo

Формат YOLO TXT:
- Каждая строка: label_idx x_center y_center width height
- Координаты нормализованы к [0, 1]
- label_idx начинается с 0
"""

import os
import json
import argparse
from pathlib import Path
from pycocotools.coco import COCO


def convert_bbox_coco_to_yolo(coco_bbox, img_width, img_height):
    """
    Конвертирует bounding box из формата COCO в формат YOLO.
    
    Args:
        coco_bbox (list): [x, y, width, height] в формате COCO (абсолютные координаты)
        img_width (int): Ширина изображения
        img_height (int): Высота изображения
    
    Returns:
        tuple: (x_center, y_center, width, height) в формате YOLO (нормализованные координаты [0, 1])
    """
    x, y, width, height = coco_bbox
    
    # Вычисляем центр bbox
    x_center = x + width / 2
    y_center = y + height / 2
    
    # Нормализуем координаты к диапазону [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def create_classes_file(coco_file, output_dir):
    """
    Создает файл classes.names на основе категорий из COCO файла.
    
    Args:
        coco_file (COCO): Объект COCO
        output_dir (str): Директория для сохранения файла classes.names
    
    Returns:
        dict: Словарь соответствия category_id -> class_index
    """
    # Получаем все категории из COCO файла
    categories = coco_file.loadCats(coco_file.getCatIds())
    
    # Сортируем категории по ID для стабильного порядка
    categories = sorted(categories, key=lambda x: x['id'])
    
    # Создаем файл classes.names
    classes_file = os.path.join(output_dir, 'classes.names')
    category_mapping = {}
    
    with open(classes_file, 'w', encoding='utf-8') as f:
        for idx, category in enumerate(categories):
            f.write(f"{category['name']}\n")
            category_mapping[category['id']] = idx
    
    print(f"Создан файл классов: {classes_file}")
    print(f"Классы: {[cat['name'] for cat in categories]}")
    
    return category_mapping


def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, classes_file=None):
    """
    Конвертирует аннотации из формата COCO JSON в формат YOLO TXT.
    
    Args:
        coco_json_path (str): Путь к файлу аннотаций COCO JSON
        images_dir (str): Директория с изображениями
        output_dir (str): Директория для сохранения TXT файлов
        classes_file (str, optional): Путь к файлу с именами классов
    """
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем COCO аннотации
    print(f"Загрузка COCO аннотаций из: {coco_json_path}")
    coco = COCO(coco_json_path)
    
    # Создаем или загружаем соответствие категорий
    if classes_file and os.path.exists(classes_file):
        # Загружаем существующий файл классов
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Создаем соответствие category_id -> class_index
        categories = coco.loadCats(coco.getCatIds())
        category_mapping = {}
        for category in categories:
            try:
                class_index = class_names.index(category['name'])
                category_mapping[category['id']] = class_index
            except ValueError:
                print(f"Предупреждение: Класс '{category['name']}' не найден в файле классов")
                continue
        
        print(f"Использован существующий файл классов: {classes_file}")
    else:
        # Создаем новый файл классов
        category_mapping = create_classes_file(coco, output_dir)
    
    # Получаем все изображения
    image_ids = coco.getImgIds()
    
    print(f"Найдено {len(image_ids)} изображений для конвертации")
    
    converted_count = 0
    skipped_count = 0
    
    # Обрабатываем каждое изображение
    for img_id in image_ids:
        # Получаем информацию об изображении
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Проверяем существование файла изображения
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"Пропускаем: изображение {img_filename} не найдено в {images_dir}")
            skipped_count += 1
            continue
        
        # Получаем аннотации для этого изображения
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # Создаем имя файла аннотаций (.txt)
        base_name = os.path.splitext(img_filename)[0]
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        # Записываем аннотации в YOLO формате
        with open(txt_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                category_id = ann['category_id']
                
                # Проверяем, есть ли соответствие для этой категории
                if category_id not in category_mapping:
                    print(f"Предупреждение: Пропускаем аннотацию с неизвестной категорией {category_id}")
                    continue
                
                class_index = category_mapping[category_id]
                coco_bbox = ann['bbox']  # [x, y, width, height]
                
                # Конвертируем bbox в формат YOLO
                x_center, y_center, width, height = convert_bbox_coco_to_yolo(
                    coco_bbox, img_width, img_height
                )
                
                # Проверяем корректность координат
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                        0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Предупреждение: Некорректные координаты для {img_filename}: "
                          f"{x_center:.4f}, {y_center:.4f}, {width:.4f}, {height:.4f}")
                    continue
                
                # Записываем в формате YOLO
                f.write(f"{class_index-1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
        
        # Выводим прогресс
        if converted_count % 100 == 0:
            print(f"Конвертировано: {converted_count}/{len(image_ids)}")
    
    print(f"\nКонвертация завершена!")
    print(f"Успешно конвертировано: {converted_count} изображений")
    print(f"Пропущено: {skipped_count} изображений")
    print(f"TXT файлы сохранены в: {output_dir}")


def verify_conversion(images_dir, txt_dir, sample_size=5):
    """
    Проверяет корректность конвертации на нескольких образцах.
    
    Args:
        images_dir (str): Директория с изображениями
        txt_dir (str): Директория с TXT аннотациями
        sample_size (int): Количество файлов для проверки
    """
    print(f"\nПроверка корректности конвертации...")
    
    # Получаем список TXT файлов
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt') and f != 'classes.names']
    
    if len(txt_files) == 0:
        print("Не найдено TXT файлов для проверки")
        return
    
    # Выбираем случайные файлы для проверки
    import random
    sample_files = random.sample(txt_files, min(sample_size, len(txt_files)))
    
    for txt_file in sample_files:
        print(f"\nПроверка файла: {txt_file}")
        
        # Проверяем существование соответствующего изображения
        base_name = os.path.splitext(txt_file)[0]
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_found = False
        
        for ext in image_extensions:
            img_file = f"{base_name}{ext}"
            img_path = os.path.join(images_dir, img_file)
            if os.path.exists(img_path):
                print(f"  Соответствующее изображение: {img_file}")
                image_found = True
                break
        
        if not image_found:
            print(f"  Предупреждение: Изображение для {txt_file} не найдено")
            continue
        
        # Проверяем содержимое TXT файла
        txt_path = os.path.join(txt_dir, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"  Количество аннотаций: {len(lines)}")
        
        # Проверяем первые несколько строк
        for i, line in enumerate(lines[:3]):
            parts = line.strip().split()
            if len(parts) == 5:
                class_idx, x_center, y_center, width, height = parts
                print(f"    Аннотация {i+1}: класс={class_idx}, "
                      f"центр=({x_center}, {y_center}), размер=({width}, {height})")
            else:
                print(f"    Некорректная строка {i+1}: {line.strip()}")


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Конвертация аннотаций из формата COCO JSON в формат YOLO TXT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--coco_json', type=str, required=True,
                       help='Путь к файлу аннотаций COCO JSON')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Директория с изображениями')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Директория для сохранения TXT файлов')
    parser.add_argument('--classes_file', type=str, default=None,
                       help='Путь к существующему файлу с именами классов (опционально)')
    parser.add_argument('--verify', action='store_true',
                       help='Выполнить проверку корректности конвертации')
    parser.add_argument('--verify_samples', type=int, default=5,
                       help='Количество файлов для проверки')
    
    return parser.parse_args()


def main():
    """Основная функция."""
    args = parse_arguments()
    
    print("="*80)
    print("КОНВЕРТЕР АННОТАЦИЙ ИЗ COCO JSON В YOLO TXT")
    print("="*80)
    print(f"COCO JSON файл: {args.coco_json}")
    print(f"Директория изображений: {args.images_dir}")
    print(f"Выходная директория: {args.output_dir}")
    print("="*80)
    
    # Проверяем существование входных файлов
    if not os.path.exists(args.coco_json):
        print(f"Ошибка: COCO JSON файл не найден: {args.coco_json}")
        return
    
    if not os.path.exists(args.images_dir):
        print(f"Ошибка: Директория изображений не найдена: {args.images_dir}")
        return
    
    try:
        # Выполняем конвертацию
        convert_coco_to_yolo(
            coco_json_path=args.coco_json,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            classes_file=args.classes_file
        )
        
        # Выполняем проверку, если требуется
        if args.verify:
            verify_conversion(args.images_dir, args.output_dir, args.verify_samples)
        
        print("\n" + "="*80)
        print("КОНВЕРТАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print("="*80)
        
    except Exception as e:
        print(f"\nОшибка при конвертации: {e}")
        raise


if __name__ == "__main__":
    main()