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
    
def draw_styled_rectangle(draw, bbox, color, line_style='solid', width=2):
    """
    Рисует прямоугольник с заданным стилем линии.
    
    Args:
        draw: объект ImageDraw
        bbox: координаты рамки [x1, y1, x2, y2]
        color: цвет линии (RGB кортеж)
        line_style: стиль линии ('solid', 'dashed', 'dotted')
        width: толщина линии
    """
    x1, y1, x2, y2 = bbox
    
    if line_style == 'solid':
        # Обычный сплошной прямоугольник
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    elif line_style == 'dashed':
        # Пунктирная линия
        dash_length = 8
        gap_length = 4
        
        # Верхняя линия
        draw_dashed_line(draw, x1, y1, x2, y1, color, dash_length, gap_length, width)
        # Правая линия
        draw_dashed_line(draw, x2, y1, x2, y2, color, dash_length, gap_length, width)
        # Нижняя линия
        draw_dashed_line(draw, x2, y2, x1, y2, color, dash_length, gap_length, width)
        # Левая линия
        draw_dashed_line(draw, x1, y2, x1, y1, color, dash_length, gap_length, width)
    
    elif line_style == 'dotted':
        # Точечная линия
        dot_size = 2
        gap_length = 3
        
        # Верхняя линия
        draw_dotted_line(draw, x1, y1, x2, y1, color, dot_size, gap_length, width)
        # Правая линия
        draw_dotted_line(draw, x2, y1, x2, y2, color, dot_size, gap_length, width)
        # Нижняя линия
        draw_dotted_line(draw, x2, y2, x1, y2, color, dot_size, gap_length, width)
        # Левая линия
        draw_dotted_line(draw, x1, y2, x1, y1, color, dot_size, gap_length, width)

def draw_dashed_line(draw, x1, y1, x2, y2, color, dash_length, gap_length, width):
    """Рисует пунктирную линию между двумя точками."""
    import math
    
    # Вычисляем длину и направление линии
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    
    if length == 0:
        return
    
    # Нормализуем направление
    dx /= length
    dy /= length
    
    # Рисуем пунктиры
    pos = 0
    while pos < length:
        # Начало пунктира
        start_x = x1 + dx * pos
        start_y = y1 + dy * pos
        
        # Конец пунктира
        end_pos = min(pos + dash_length, length)
        end_x = x1 + dx * end_pos
        end_y = y1 + dy * end_pos
        
        # Рисуем пунктир
        draw.line([start_x, start_y, end_x, end_y], fill=color, width=width)
        
        # Переходим к следующему пунктиру
        pos += dash_length + gap_length

def draw_dotted_line(draw, x1, y1, x2, y2, color, dot_size, gap_length, width):
    """Рисует точечную линию между двумя точками."""
    import math
    
    # Вычисляем длину и направление линии
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    
    if length == 0:
        return
    
    # Нормализуем направление
    dx /= length
    dy /= length
    
    # Рисуем точки
    pos = 0
    step = dot_size + gap_length
    while pos < length:
        # Позиция точки
        dot_x = x1 + dx * pos
        dot_y = y1 + dy * pos
        
        # Рисуем точку как маленький эллипс
        draw.ellipse([dot_x - width//2, dot_y - width//2, 
                     dot_x + width//2, dot_y + width//2], 
                    fill=color, outline=color)
        
        pos += step

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
    
    line_style = {
        1: 'dashed',
        2: 'solid'
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
        draw_styled_rectangle(draw, box, color_rgb, line_style.get(label, 'solid'))
        # draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)
        
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
