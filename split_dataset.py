#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для разделения датасета на обучающую и валидационную выборки.
20% изображений идет в train.txt, 80% в valid.txt с префиксом "data/custom/images/".

Использование:
    python split_dataset.py --images_dir data/train --output_dir . --val_split 0.2
"""

import os
import argparse
import random
from pathlib import Path


def get_image_files(images_dir, extensions=None):
    """
    Получает список файлов изображений из директории.
    
    Args:
        images_dir (str): Путь к директории с изображениями
        extensions (list): Список допустимых расширений файлов
    
    Returns:
        list: Список имен файлов изображений
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    image_files = []
    
    for filename in os.listdir(images_dir):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_files.append(filename)
    
    return sorted(image_files)


def split_dataset(images_dir, output_dir, val_split=0.2, prefix="data/custom/images/", seed=42):
    """
    Разделяет датасет на обучающую и валидационную выборки.
    
    Args:
        images_dir (str): Директория с изображениями
        output_dir (str): Директория для сохранения файлов train.txt и valid.txt
        val_split (float): Доля изображений для валидации (по умолчанию 0.2 = 20%)
        prefix (str): Префикс для путей к изображениям
        seed (int): Seed для воспроизводимости
    """
    # Устанавливаем seed для воспроизводимости
    random.seed(seed)
    
    # Получаем список файлов изображений
    image_files = get_image_files(images_dir)
    
    if not image_files:
        print(f"Не найдено изображений в директории: {images_dir}")
        return
    
    print(f"Найдено {len(image_files)} изображений")
    
    # Перемешиваем список файлов
    random.shuffle(image_files)
    
    # Вычисляем количество файлов для валидации
    val_count = int(len(image_files) * val_split)
    train_count = len(image_files) - val_count
    
    # Разделяем на train и validation
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    
    print(f"Обучающая выборка: {train_count} изображений ({(1-val_split)*100:.1f}%)")
    print(f"Валидационная выборка: {val_count} изображений ({val_split*100:.1f}%)")
    
    # Создаем выходную директорию, если не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Записываем train.txt
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, 'w', encoding='utf-8') as f:
        for filename in train_files:
            f.write(f"{prefix}{filename}\n")
    
    print(f"Создан файл: {train_file}")
    
    # Записываем valid.txt
    valid_file = os.path.join(output_dir, "valid.txt")
    with open(valid_file, 'w', encoding='utf-8') as f:
        for filename in val_files:
            f.write(f"{prefix}{filename}\n")
    
    print(f"Создан файл: {valid_file}")
    
    # Выводим примеры содержимого файлов
    print(f"\nПример содержимого {train_file} (первые 5 строк):")
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.strip()}")
    
    print(f"\nПример содержимого {valid_file} (первые 5 строк):")
    with open(valid_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.strip()}")


def verify_split(train_file, valid_file):
    """
    Проверяет корректность разделения датасета.
    
    Args:
        train_file (str): Путь к файлу train.txt
        valid_file (str): Путь к файлу valid.txt
    """
    print("\nПроверка разделения датасета:")
    
    # Читаем файлы
    train_images = set()
    valid_images = set()
    
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            train_images = set(line.strip() for line in f)
    
    if os.path.exists(valid_file):
        with open(valid_file, 'r', encoding='utf-8') as f:
            valid_images = set(line.strip() for line in f)
    
    # Проверяем пересечения
    intersection = train_images.intersection(valid_images)
    
    print(f"Изображений в train.txt: {len(train_images)}")
    print(f"Изображений в valid.txt: {len(valid_images)}")
    print(f"Общее количество изображений: {len(train_images) + len(valid_images)}")
    
    if intersection:
        print(f"⚠️  ВНИМАНИЕ: Найдено {len(intersection)} пересекающихся изображений!")
        print("Первые 5 пересекающихся файлов:")
        for i, img in enumerate(list(intersection)[:5]):
            print(f"  {img}")
    else:
        print("✅ Пересечений не найдено - разделение корректно")
    
    # Вычисляем пропорции
    total = len(train_images) + len(valid_images)
    if total > 0:
        train_percent = (len(train_images) / total) * 100
        valid_percent = (len(valid_images) / total) * 100
        print(f"Пропорции: train={train_percent:.1f}%, valid={valid_percent:.1f}%")


def create_yolo_structure(images_dir, output_dir, val_split=0.2, prefix="data/custom/images/", 
                         copy_files=False, seed=42):
    """
    Создает полную структуру для YOLO с опциональным копированием файлов.
    
    Args:
        images_dir (str): Директория с изображениями
        output_dir (str): Выходная директория
        val_split (float): Доля для валидации
        prefix (str): Префикс для путей
        copy_files (bool): Копировать ли файлы в структуру YOLO
        seed (int): Seed для воспроизводимости
    """
    print("Создание структуры YOLO...")
    
    if copy_files:
        # Создаем директории для YOLO структуры
        train_img_dir = os.path.join(output_dir, "images", "train")
        valid_img_dir = os.path.join(output_dir, "images", "valid")
        train_lbl_dir = os.path.join(output_dir, "labels", "train")
        valid_lbl_dir = os.path.join(output_dir, "labels", "valid")
        
        for dir_path in [train_img_dir, valid_img_dir, train_lbl_dir, valid_lbl_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"Созданы директории:")
        print(f"  {train_img_dir}")
        print(f"  {valid_img_dir}")
        print(f"  {train_lbl_dir}")
        print(f"  {valid_lbl_dir}")
    
    # Создаем файлы train.txt и valid.txt
    split_dataset(images_dir, output_dir, val_split, prefix, seed)
    
    if copy_files:
        import shutil
        
        # Читаем списки файлов
        train_file = os.path.join(output_dir, "train.txt")
        valid_file = os.path.join(output_dir, "valid.txt")
        
        # Копируем файлы изображений
        print("\nКопирование файлов изображений...")
        
        # Train images
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                filename = os.path.basename(line.strip())
                src_path = os.path.join(images_dir, filename)
                dst_path = os.path.join(train_img_dir, filename)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
        
        # Valid images
        with open(valid_file, 'r', encoding='utf-8') as f:
            for line in f:
                filename = os.path.basename(line.strip())
                src_path = os.path.join(images_dir, filename)
                dst_path = os.path.join(valid_img_dir, filename)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
        
        print("Копирование изображений завершено")
        print("Не забудьте скопировать соответствующие файлы аннотаций (.txt) в папки labels/!")


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Разделение датасета на обучающую и валидационную выборки',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Директория с изображениями')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Директория для сохранения train.txt и valid.txt')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Доля изображений для валидации (0.0-1.0)')
    parser.add_argument('--prefix', type=str, default='data/custom/images/',
                       help='Префикс для путей к изображениям')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed для воспроизводимости')
    parser.add_argument('--verify', action='store_true',
                       help='Проверить корректность разделения')
    parser.add_argument('--yolo_structure', action='store_true',
                       help='Создать полную структуру директорий для YOLO')
    parser.add_argument('--copy_files', action='store_true',
                       help='Копировать файлы в структуру YOLO (используется с --yolo_structure)')
    
    return parser.parse_args()


def main():
    """Основная функция."""
    args = parse_arguments()
    
    print("="*80)
    print("РАЗДЕЛЕНИЕ ДАТАСЕТА НА ОБУЧАЮЩУЮ И ВАЛИДАЦИОННУЮ ВЫБОРКИ")
    print("="*80)
    print(f"Директория изображений: {args.images_dir}")
    print(f"Выходная директория: {args.output_dir}")
    print(f"Доля для валидации: {args.val_split*100:.1f}%")
    print(f"Префикс путей: {args.prefix}")
    print(f"Seed: {args.seed}")
    print("="*80)
    
    # Проверяем существование директории с изображениями
    if not os.path.exists(args.images_dir):
        print(f"Ошибка: Директория изображений не найдена: {args.images_dir}")
        return
    
    # Проверяем валидность val_split
    if not 0.0 <= args.val_split <= 1.0:
        print(f"Ошибка: val_split должен быть в диапазоне [0.0, 1.0], получен: {args.val_split}")
        return
    
    try:
        if args.yolo_structure:
            # Создаем полную структуру YOLO
            create_yolo_structure(
                args.images_dir,
                args.output_dir,
                args.val_split,
                args.prefix,
                args.copy_files,
                args.seed
            )
        else:
            # Обычное разделение
            split_dataset(
                args.images_dir,
                args.output_dir,
                args.val_split,
                args.prefix,
                args.seed
            )
        
        # Проверка, если требуется
        if args.verify:
            train_file = os.path.join(args.output_dir, "train.txt")
            valid_file = os.path.join(args.output_dir, "valid.txt")
            verify_split(train_file, valid_file)
        
        print("\n" + "="*80)
        print("РАЗДЕЛЕНИЕ ДАТАСЕТА УСПЕШНО ЗАВЕРШЕНО!")
        print("="*80)
        
    except Exception as e:
        print(f"\nОшибка при разделении датасета: {e}")
        raise


if __name__ == "__main__":
    main()