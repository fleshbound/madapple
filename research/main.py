#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Программа анализа влияния яркости на обнаружение яблок

Данная программа анализирует, как изменение яркости влияет на обнаружение зрелых яблок на изображениях.
Она обрабатывает несколько тестовых изображений с различными уровнями яркости и записывает:
1. Количество обнаруженных зрелых яблок
2. Время обработки

Результаты сохраняются как индивидуально для каждого изображения, так и в виде агрегированной статистики.
"""

import os
import time
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
import statistics
import tempfile

# Импорт необходимых пользовательских модулей из предоставленного кода
from model.model import safely_load_model
from inference.image_processor import preprocess_image
from inference.detector import detect_apples


def adjust_brightness(image, brightness_factor):
    """
    Регулировка яркости изображения.
    
    Аргументы:
        image (numpy.ndarray): Исходное изображение
        brightness_factor (float): Коэффициент регулировки яркости
                                  (< 1: темнее, > 1: ярче)
    
    Возвращает:
        numpy.ndarray: Изображение с измененной яркостью
    """
    # Создаем HSV-версию изображения (Оттенок, Насыщенность, Яркость)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Регулируем канал V (яркость)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
    
    # Преобразуем обратно в RGB
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return adjusted_image


def count_ripe_apples(results):
    """
    Подсчет количества зрелых яблок в результатах обнаружения.
    
    Аргументы:
        results (dict): Результаты обнаружения, содержащие 'boxes', 'scores', 'labels'
    
    Возвращает:
        int: Количество обнаруженных зрелых яблок
    """
    # В нашем наборе данных класс 2 представляет зрелые яблоки (класс 1 - незрелые)
    if 'labels' not in results or len(results['labels']) == 0:
        return 0
        
    return np.sum(results['labels'] == 2).item()


def process_image_with_brightness(image_path, model, brightness_factors, device="cpu"):
    """
    Обработка изображения с разными уровнями яркости и сбор результатов.
    
    Аргументы:
        image_path (str): Путь к изображению
        model: Обученная модель обнаружения
        brightness_factors (list): Список коэффициентов регулировки яркости
        device (str): Вычислительное устройство (cpu/cuda)
    
    Возвращает:
        list: [(процент_изменения_яркости, количество_зрелых_яблок, время_обнаружения), ...]
    """
    results = []
    
    try:
        # Загрузка исходного изображения
        original_image = np.array(Image.open(image_path).convert("RGB"))
        original_height, original_width = original_image.shape[:2]
        
        # Обработка каждого коэффициента яркости
        for factor in brightness_factors:
            # Вычисляем процент изменения яркости для отчета
            brightness_change_percent = (factor - 1.0) * 100
            
            # Регулируем яркость
            adjusted_image = adjust_brightness(original_image, factor)
            
            # Создаем временный файл для обработанного изображения
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_filename = temp_file.name
                # Сохраняем изображение с измененной яркостью
                temp_img = Image.fromarray(adjusted_image)
                temp_img.save(temp_filename)
            
            try:
                # Используем функцию предобработки с файлом, а не с объектом изображения
                image_tensor, processed_image, (height, width) = preprocess_image(temp_filename)
                image_tensor = image_tensor.to(device)
                
                # Измеряем время обнаружения
                start_time = time.time()
                detection_results = detect_apples(model, image_tensor, (original_height, original_width), threshold=0.5, device=device)
                detection_time = time.time() - start_time
                
                # Подсчитываем зрелые яблоки
                ripe_count = count_ripe_apples(detection_results)
                
                # Сохраняем результаты
                results.append((brightness_change_percent, ripe_count, detection_time))
                
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
    
    return results


def save_individual_results(image_path, results, output_dir):
    """
    Сохранение результатов для отдельного изображения в текстовый файл.
    
    Аргументы:
        image_path (str): Путь к исходному изображению
        results (list): Список кортежей (изменение_яркости, количество_зрелых, время_обнаружения)
        output_dir (str): Директория для сохранения результатов
    """
    # Извлекаем имя файла без расширения
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    
    # Создаем путь к выходному файлу
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    
    # Записываем результаты в файл
    with open(output_path, 'w') as f:
        for brightness_change, ripe_count, detection_time in results:
            f.write(f"{brightness_change:.1f} {ripe_count} {detection_time:.6f}\n")


def save_aggregate_results(all_results, brightness_factors, output_dir):
    """
    Сохранение агрегированных результатов по всем изображениям.
    
    Аргументы:
        all_results (dict): Словарь, сопоставляющий коэффициенты яркости со списками результатов
        brightness_factors (list): Список коэффициентов регулировки яркости
        output_dir (str): Директория для сохранения результатов
    """
    # Вычисляем агрегированную статистику
    aggregate_results = []
    
    for factor in brightness_factors:
        brightness_change = (factor - 1.0) * 100
        results_for_factor = all_results[factor]
        
        if results_for_factor:
            ripe_counts = [result[1] for result in results_for_factor]
            detection_times = [result[2] for result in results_for_factor]
            
            avg_ripe_count = statistics.mean(ripe_counts)
            avg_detection_time = statistics.mean(detection_times)
            
            aggregate_results.append((brightness_change, avg_ripe_count, avg_detection_time))
    
    # Сохраняем в файл
    with open(os.path.join(output_dir, "total_result.txt"), 'w') as f:
        for brightness_change, avg_ripe_count, avg_detection_time in aggregate_results:
            f.write(f"{brightness_change:.1f} {avg_ripe_count:.2f} {avg_detection_time:.6f}\n")
    
    # Создаем и сохраняем диаграммы
    create_charts(aggregate_results, output_dir)


def create_charts(aggregate_results, output_dir):
    """
    Создание и сохранение диаграмм, визуализирующих результаты.
    
    Аргументы:
        aggregate_results (list): Список кортежей (изменение_яркости, среднее_количество_зрелых, среднее_время_обнаружения)
        output_dir (str): Директория для сохранения диаграмм
    """
    # Извлекаем данные
    brightness_changes = [result[0] for result in aggregate_results]
    avg_ripe_counts = [result[1] for result in aggregate_results]
    avg_detection_times = [result[2] for result in aggregate_results]
    
    # Создаем диаграмму количества зрелых яблок
    plt.figure(figsize=(12, 6))
    plt.bar(brightness_changes, avg_ripe_counts, color='red')
    plt.xlabel('Изменение яркости (%)')
    plt.ylabel('Среднее количество зрелых яблок')
    plt.title('Влияние яркости на обнаружение зрелых яблок')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'ripe_apple_count_vs_brightness.png'), dpi=300, bbox_inches='tight')
    
    # Создаем диаграмму времени обнаружения
    plt.figure(figsize=(12, 6))
    plt.bar(brightness_changes, avg_detection_times, color='blue')
    plt.xlabel('Изменение яркости (%)')
    plt.ylabel('Среднее время обнаружения (секунды)')
    plt.title('Влияние яркости на время обнаружения')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'detection_time_vs_brightness.png'), dpi=300, bbox_inches='tight')


def main():
    # Пути и настройки
    test_dir = "..\\data\\test"
    model_path = "..\\train\\apple_detector.pth"  # Измените, если ваша модель находится в другом месте
    output_dir = "result"
    
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем коэффициенты регулировки яркости
    # Значения < 1 делают изображение темнее, значения > 1 делают его ярче
    brightness_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загружаем модель
    print("Загрузка модели...")
    model = safely_load_model(model_path, num_classes=3, device=device)
    model.eval()
    
    # Получаем все файлы изображений
    image_files = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                 glob.glob(os.path.join(test_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(test_dir, "*.png"))
    
    print(f"Найдено {len(image_files)} тестовых изображений")
    
    # Обрабатываем все изображения
    all_results = {factor: [] for factor in brightness_factors}
    
    for image_path in tqdm(image_files, desc="Обработка изображений"):
        results = process_image_with_brightness(image_path, model, brightness_factors, device)
        
        # Сохраняем индивидуальные результаты
        save_individual_results(image_path, results, output_dir)
        
        # Собираем результаты для агрегированного анализа
        for i, factor in enumerate(brightness_factors):
            if i < len(results):
                all_results[factor].append(results[i])
    
    # Сохраняем агрегированные результаты и создаем диаграммы
    save_aggregate_results(all_results, brightness_factors, output_dir)
    
    print(f"Анализ завершен. Результаты сохранены в {output_dir}")
    
    # Выводим сводку результатов
    print("\nСводка результатов:")
    with open(os.path.join(output_dir, "total_result.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            brightness, count, time = line.strip().split()
            print(f"Изменение яркости: {brightness}%, Среднее количество зрелых яблок: {float(count):.2f}, Среднее время обнаружения: {float(time):.6f}с")


if __name__ == "__main__":
    main()