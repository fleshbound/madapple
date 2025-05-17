#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для пакетной обработки изображений через командную строку.
Позволяет обрабатывать целые директории с изображениями.

Использование:
    python batch_main.py --dir path/to/directory --model path/to/model.pth --output path/to/output
"""

import os
import argparse
import time
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.model import safely_load_model
from batch_processor import process_directory
from visualizer import visualize_results
from utils.utils import ensure_dir, count_apples_by_type

def parse_args():
    parser = argparse.ArgumentParser(description='Пакетное распознавание яблок на изображениях')
    parser.add_argument('--dir', type=str, required=True, help='Путь к директории с изображениями')
    parser.add_argument('--model', type=str, default='apple_detector.pth', help='Путь к файлу модели')
    parser.add_argument('--output', type=str, default='results', help='Директория для сохранения результатов')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог уверенности')
    parser.add_argument('--recursive', action='store_true', help='Обрабатывать поддиректории рекурсивно')
    parser.add_argument('--quiet', action='store_true', help='Не выводить прогресс обработки')
    parser.add_argument('--summary', action='store_true', help='Создать итоговый отчет')
    parser.add_argument('--skip-visualization', action='store_true', help='Пропустить визуализацию')
    parser.add_argument('--time', action='store_true', help='Замерить время выполнения')
    parser.add_argument('--no_labels', action='store_true', help='Выводить рамки без текстовых меток')
    return parser.parse_args()

def main():
    # Парсинг аргументов
    args = parse_args()
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.quiet:
        print(f"Используется устройство: {device}")
    
    # Загружаем модель с безопасной загрузкой
    if not args.quiet:
        print(f"Загрузка модели из: {args.model}")
    model = safely_load_model(args.model, num_classes=3, device=device, quiet=args.quiet)
    
    # Создаем выходную директорию
    ensure_dir(args.output)
    
    # Проверяем существование директории с изображениями
    if not os.path.exists(args.dir):
        print(f"Ошибка: Директория {args.dir} не существует")
        return
    
    # Измеряем время выполнения, если требуется
    start_time = time.time() if args.time else None
    
    # Обрабатываем директорию с изображениями
    if not args.quiet:
        print(f"Обработка изображений из директории: {args.dir}")
    
    results_dict = process_directory(
        args.dir, 
        model, 
        threshold=args.threshold, 
        device=device, 
        recursive=args.recursive
    )
    
    if args.time:
        elapsed_time = time.time() - start_time
        time_per_image = elapsed_time / len(results_dict) if results_dict else 0
        if not args.quiet:
            print(f"Общее время обработки: {elapsed_time:.2f} секунд")
            print(f"Среднее время на изображение: {time_per_image:.4f} секунд")
    
    # Создаем директорию для визуализации результатов
    vis_dir = os.path.join(args.output, 'visualizations')
    if not args.skip_visualization:
        ensure_dir(vis_dir)
    
    # Директория для JSON-результатов
    json_dir = os.path.join(args.output, 'json')
    ensure_dir(json_dir)
    
    # Обрабатываем результаты
    total_unripe = 0
    total_ripe = 0
    image_results = []
    
    # Используем tqdm для отображения прогресса
    iterator = tqdm(results_dict.items(), desc="Обработка результатов", disable=args.quiet)
    
    for image_path, result in iterator:
        # Получаем имя файла без пути
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Подсчитываем яблоки
        unripe_count, ripe_count, total_count = count_apples_by_type(result)
        total_unripe += unripe_count
        total_ripe += ripe_count
        
        # Сохраняем JSON с результатами
        json_path = os.path.join(json_dir, f"{name_without_ext}.json")
        json_result = {
            'image_path': image_path,
            'boxes': result['boxes'].tolist() if 'boxes' in result and len(result['boxes']) > 0 else [],
            'scores': result['scores'].tolist() if 'scores' in result and len(result['scores']) > 0 else [],
            'labels': result['labels'].tolist() if 'labels' in result and len(result['labels']) > 0 else [],
            'unripe_count': unripe_count,
            'ripe_count': ripe_count,
            'total_count': total_count
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        # Добавляем в общий список для итогового отчета
        image_results.append({
            'filename': filename,
            'unripe_count': unripe_count,
            'ripe_count': ripe_count,
            'total_count': total_count
        })
        
        # Визуализация, если не пропускаем
        if not args.skip_visualization:
            try:
                # Загружаем исходное изображение
                from image_processor import preprocess_image
                _, original_image, _ = preprocess_image(image_path)
                
                # Сохраняем визуализацию
                vis_path = os.path.join(vis_dir, f"{name_without_ext}_result.png")
                visualize_results(original_image, result, vis_path, show_image=False)
            except Exception as e:
                print(f"Ошибка при визуализации {image_path}: {e}")
    
    # Создаем итоговый отчет
    if args.summary:
        # Создаем директорию для отчетов
        report_dir = os.path.join(args.output, 'reports')
        ensure_dir(report_dir)
        
        # Отчет в формате JSON
        summary_json = {
            'total_images': len(results_dict),
            'total_unripe_apples': total_unripe,
            'total_ripe_apples': total_ripe,
            'total_apples': total_unripe + total_ripe,
            'average_unripe_per_image': total_unripe / len(results_dict) if results_dict else 0,
            'average_ripe_per_image': total_ripe / len(results_dict) if results_dict else 0,
            'average_apples_per_image': (total_unripe + total_ripe) / len(results_dict) if results_dict else 0,
            'image_results': image_results
        }
        
        # Добавляем время выполнения, если замеряли
        if args.time:
            summary_json.update({
                'total_processing_time': elapsed_time,
                'average_time_per_image': time_per_image
            })
        
        # Сохраняем отчет в JSON
        with open(os.path.join(report_dir, 'summary.json'), 'w') as f:
            json.dump(summary_json, f, indent=2)
        
        # Создаем графический отчет
        plt.figure(figsize=(12, 8))
        
        # График количества яблок на изображении
        plt.subplot(2, 1, 1)
        image_names = [res['filename'] for res in image_results[:20]]  # Ограничиваем для читаемости
        unripe_counts = [res['unripe_count'] for res in image_results[:20]]
        ripe_counts = [res['ripe_count'] for res in image_results[:20]]
        
        x = range(len(image_names))
        plt.bar(x, unripe_counts, width=0.4, label='Незрелые яблоки', color='yellow', alpha=0.7)
        plt.bar(x, ripe_counts, width=0.4, label='Зрелые яблоки', color='red', alpha=0.7, bottom=unripe_counts)
        
        plt.xlabel('Изображения')
        plt.ylabel('Количество яблок')
        plt.title('Распределение яблок по изображениям (первые 20)')
        plt.xticks(x, image_names, rotation=90)
        plt.legend()
        
        # График суммарного количества яблок
        plt.subplot(2, 1, 2)
        plt.pie([total_unripe, total_ripe], labels=['Незрелые яблоки', 'Зрелые яблоки'],
                autopct='%1.1f%%', colors=['yellow', 'red'], startangle=90)
        plt.axis('equal')
        plt.title('Соотношение типов яблок')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'summary_chart.png'), dpi=300, bbox_inches='tight')
        
        if not args.quiet:
            print(f"Итоговый отчет сохранен в {report_dir}")
    
    # Выводим итоговую статистику
    if not args.quiet:
        print("\nИтоговая статистика:")
        print(f"Обработано изображений: {len(results_dict)}")
        print(f"Обнаружено незрелых яблок: {total_unripe}")
        print(f"Обнаружено зрелых яблок: {total_ripe}")
        print(f"Всего яблок: {total_unripe + total_ripe}")
        print(f"Результаты сохранены в {args.output}")

if __name__ == "__main__":
    main()