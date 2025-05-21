#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для визуализации результатов обнаружения яблок.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def count_apples(labels):
    unripe_count = sum(1 for label in labels if label == 1)
    ripe_count = sum(1 for label in labels if label == 2)
    return unripe_count, ripe_count, unripe_count + ripe_count

def print_detections(results):
    """
    Выводит информацию о обнаруженных яблоках в консоль.
    
    Args:
        results (dict): Результаты обнаружения
    """
    # Подсчет типов яблок
    unripe_count, ripe_count, total = count_apples(results['labels'])
    
    print("\nОбнаруженные объекты:")
    print(f"{'Класс':<15} {'Уверенность':<15} {'x1':<10} {'y1':<10} {'x2':<10} {'y2':<10}")
    print("-" * 70)
    
    for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
        class_name = "Незрелое яблоко" if label == 1 else "Зрелое яблоко" if label == 2 else "Неизвестно"
        print(f"{class_name:<15} {score:<10.4f} {box[0]:<10.1f} {box[1]:<10.1f} {box[2]:<10.1f} {box[3]:<10.1f}")
    
    print("\nИТОГО")
    print(f"Незрелых яблок: {unripe_count}")
    print(f"Зрелых яблок: {ripe_count}")
    print(f"Всего яблок: {unripe_count + ripe_count}")

def visualize_results(image, results, output_path=None, show_image=True, show_labels=True):
    """
    Визуализирует результаты обнаружения и выводит статистику.
    
    Args:
        image (np.ndarray): Оригинальное изображение (в формате RGB)
        results (dict): Результаты обнаружения
        output_path (str, optional): Путь для сохранения результата
        show_image (bool): Показывать ли изображение на экране
    """
    # Создаем копию изображения для визуализации
    vis_image = image.copy()
    
    # Определяем цвета для классов (в RGB)
    colors = {
        1: (255, 255, 0),  # Желтый для незрелых яблок (RGB)
        2: (255, 0, 0)     # Красный для зрелых яблок (RGB)
    }
    
    # Названия классов
    class_names = {
        1: "Незрелый",
        2: "Зрелый"
    }
    
    # Счетчики яблок
    unripe_count = 0
    ripe_count = 0
    
    # Используем PIL для рисования
    pil_image = Image.fromarray(vis_image)
    draw = ImageDraw.Draw(pil_image)
    
    # Пытаемся использовать стандартный шрифт
    try:
        # Пробуем разные шрифты, какой есть в системе
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/Library/Fonts/Arial Unicode.ttf",                 # macOS
            "C:/Windows/Fonts/arial.ttf",                       # Windows
            "C:/Windows/Fonts/calibri.ttf",                     # Windows
            None  # Использовать шрифт по умолчанию
        ]
        
        font = None
        for font_path in font_paths:
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, 15)
                    break
                else:
                    font = ImageFont.load_default()
                    break
            except Exception:
                continue
                
        if font is None:
            font = ImageFont.load_default()
    except Exception as e:
        # Используем шрифт по умолчанию
        font = ImageFont.load_default()
    
    # Отрисовка боксов и меток
    for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
        # Округляем координаты до целых чисел
        x1, y1, x2, y2 = box.astype(int)
        
        # Получаем цвет для класса
        color_rgb = colors.get(label, (255, 255, 255))
        
        # Рисуем рамку
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)
        
        if show_labels:
            # Текст с меткой и уверенностью
            label_text = f"{class_names.get(label, 'Unknown')}: {score:.2f}"
            
            # Рисуем фон для текста для лучшей видимости
            text_size = draw.textbbox((0, 0), label_text, font=font)[2:4]
            draw.rectangle([x1, y1 - text_size[1] - 5, x1 + text_size[0], y1], fill=color_rgb)
            draw.text((x1, y1 - text_size[1] - 5), label_text, fill=(0, 0, 0), font=font)
        
        # Подсчет яблок
        if label == 1:
            unripe_count += 1
        elif label == 2:
            ripe_count += 1
    
    # Конвертируем обратно в numpy array
    vis_image = np.array(pil_image)
    
    # Создаем фигуру matplotlib только если нужно показать или сохранить
    if show_image or output_path:
        plt.figure(figsize=(12, 10))
        plt.imshow(vis_image)
        plt.axis('off')
        # plt.title(f"Найдено: {unripe_count} незрелых и {ripe_count} зрелых плодов", fontsize=14)
        plt.title(f"Найдено: {ripe_count} зрелых плодов", fontsize=14)
        
        # Добавляем легенду
        import matplotlib.patches as mpatches
        unripe_patch = mpatches.Patch(color='yellow', label='Незрелые')
        ripe_patch = mpatches.Patch(color='red', label='Зрелые')
        plt.legend(handles=[unripe_patch, ripe_patch], loc='upper right', fontsize=12)
        
        # Сохраняем результат, если указан путь
        if output_path is not None:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Результат сохранен в: {output_path}")
        
        # Отображаем изображение если нужно
        if show_image:
            plt.show()
        else:
            plt.close()
    
    if show_image:
    # Выводим информацию о количестве яблок
        print_detections(results)
    
    return vis_image
