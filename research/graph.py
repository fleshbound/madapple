#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Программа для визуализации результатов анализа влияния яркости на обнаружение яблок.

Данная программа читает файл total_result.txt с результатами анализа и создает
графики, показывающие зависимость среднего количества обнаруженных зрелых яблок
и времени обнаружения от изменения яркости изображения.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def read_results(file_path):
    """
    Чтение результатов из файла.
    
    Аргументы:
        file_path (str): Путь к файлу total_result.txt
    
    Возвращает:
        tuple: (brightness_changes, avg_ripe_counts, avg_detection_times)
    """
    brightness_changes = []
    avg_ripe_counts = []
    avg_detection_times = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Разбиваем строку на значения (изменение яркости, среднее количество, среднее время)
                values = line.strip().split()
                if len(values) == 3:
                    try:
                        brightness_change = float(values[0])
                        avg_ripe_count = float(values[1])
                        avg_detection_time = float(values[2])
                        
                        brightness_changes.append(brightness_change)
                        avg_ripe_counts.append(avg_ripe_count)
                        avg_detection_times.append(avg_detection_time)
                    except ValueError as e:
                        print(f"Ошибка при преобразовании значений: {e}")
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
    
    return brightness_changes, avg_ripe_counts, avg_detection_times

def create_ripe_count_chart(brightness_changes, avg_ripe_counts, save_path=None):
    """
    Создание диаграммы зависимости количества зрелых яблок от яркости.
    
    Аргументы:
        brightness_changes (list): Проценты изменения яркости
        avg_ripe_counts (list): Среднее количество зрелых яблок
        save_path (str, optional): Путь для сохранения диаграммы
    """
    plt.figure(figsize=(12, 6))
    
    # Сортируем данные по изменению яркости (от меньшего к большему)
    sorted_data = sorted(zip(brightness_changes, avg_ripe_counts))
    sorted_brightness = [x[0] for x in sorted_data]
    sorted_counts = [x[1] for x in sorted_data]
    
    # Создаем столбчатую диаграмму
    bars = plt.bar(sorted_brightness, sorted_counts, color='red', width=5)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Настраиваем оси и заголовок
    plt.xlabel('Изменение яркости (%)', fontsize=12)
    plt.ylabel('Среднее количество зрелых яблок', fontsize=12)
    plt.title('Влияние яркости на обнаружение зрелых яблок', fontsize=14)
    
    # Добавляем сетку для лучшей читаемости
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Устанавливаем подписи оси X для каждого значения
    plt.xticks(sorted_brightness, [f"{b:.1f}%" for b in sorted_brightness], rotation=45)
    
    # Оптимизируем расположение элементов
    plt.tight_layout()
    
    # Сохраняем диаграмму, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Показываем график
    plt.show()

def create_detection_time_chart(brightness_changes, avg_detection_times, save_path=None):
    """
    Создание диаграммы зависимости времени обнаружения от яркости.
    
    Аргументы:
        brightness_changes (list): Проценты изменения яркости
        avg_detection_times (list): Среднее время обнаружения
        save_path (str, optional): Путь для сохранения диаграммы
    """
    plt.figure(figsize=(12, 6))
    
    # Сортируем данные по изменению яркости (от меньшего к большему)
    sorted_data = sorted(zip(brightness_changes, avg_detection_times))
    sorted_brightness = [x[0] for x in sorted_data]
    sorted_times = [x[1] for x in sorted_data]
    
    # Создаем столбчатую диаграмму
    bars = plt.bar(sorted_brightness, sorted_times, color='blue', width=5)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                 f'{height:.4f}', ha='center', va='bottom', rotation=45)
    
    # Настраиваем оси и заголовок
    plt.xlabel('Изменение яркости (%)', fontsize=12)
    plt.ylabel('Среднее время обнаружения (секунды)', fontsize=12)
    plt.title('Влияние яркости на время обнаружения', fontsize=14)
    
    # Добавляем сетку для лучшей читаемости
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Устанавливаем подписи оси X для каждого значения
    plt.xticks(sorted_brightness, [f"{b:.1f}%" for b in sorted_brightness], rotation=45)
    
    # Оптимизируем расположение элементов
    plt.tight_layout()
    
    # Сохраняем диаграмму, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Показываем график
    plt.show()

def create_combined_chart(brightness_changes, avg_ripe_counts, avg_detection_times, save_path=None):
    """
    Создание совмещенной диаграммы с двумя осями Y для наглядного сравнения.
    
    Аргументы:
        brightness_changes (list): Проценты изменения яркости
        avg_ripe_counts (list): Среднее количество зрелых яблок
        avg_detection_times (list): Среднее время обнаружения
        save_path (str, optional): Путь для сохранения диаграммы
    """
    # Сортируем данные по изменению яркости (от меньшего к большему)
    sorted_data = sorted(zip(brightness_changes, avg_ripe_counts, avg_detection_times))
    sorted_brightness = [x[0] for x in sorted_data]
    sorted_counts = [x[1] for x in sorted_data]
    sorted_times = [x[2] for x in sorted_data]
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Настраиваем первую ось Y (количество яблок)
    color = 'red'
    ax1.set_xlabel('Изменение яркости (%)', fontsize=12)
    ax1.set_ylabel('Среднее количество зрелых яблок', color=color, fontsize=12)
    line1 = ax1.plot(sorted_brightness, sorted_counts, color=color, marker='o', linewidth=2, label='Количество яблок')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Создаем вторую ось Y (время обнаружения)
    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('Среднее время обнаружения (секунды)', color=color, fontsize=12)
    line2 = ax2.plot(sorted_brightness, sorted_times, color=color, marker='s', linewidth=2, label='Время обнаружения')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Добавляем заголовок
    plt.title('Влияние яркости на обнаружение зрелых яблок и время обработки', fontsize=14)
    
    # Добавляем сетку
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Добавляем легенду
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    # Устанавливаем подписи оси X для каждого значения
    plt.xticks(sorted_brightness, [f"{b:.1f}%" for b in sorted_brightness])
    
    # Оптимизируем расположение элементов
    fig.tight_layout()
    
    # Сохраняем диаграмму, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Показываем график
    plt.show()

def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Построение графиков на основе файла total_result.txt')
    parser.add_argument('--file', type=str, default='result/total_result.txt',
                        help='Путь к файлу total_result.txt')
    parser.add_argument('--save', action='store_true', help='Сохранить графики')
    parser.add_argument('--output-dir', type=str, default='графики',
                        help='Директория для сохранения графиков')
    
    args = parser.parse_args()
    
    # Проверяем существование файла
    if not os.path.exists(args.file):
        print(f"Ошибка: Файл {args.file} не найден")
        return
    
    # Создаем директорию для сохранения, если требуется
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Читаем данные из файла
    brightness_changes, avg_ripe_counts, avg_detection_times = read_results(args.file)
    
    if not brightness_changes:
        print("Ошибка: Не удалось прочитать данные из файла")
        return
    
    # Выводим считанные данные
    print("Данные из файла:")
    print(f"{'Изменение яркости (%)':<20} {'Среднее количество яблок':<25} {'Среднее время обнаружения (сек)':<30}")
    print("-" * 75)
    for brightness, count, time in zip(brightness_changes, avg_ripe_counts, avg_detection_times):
        print(f"{brightness:<20.1f} {count:<25.2f} {time:<30.6f}")
    
    # Пути для сохранения
    ripe_count_chart_path = None
    detection_time_chart_path = None
    combined_chart_path = None
    if args.save:
        ripe_count_chart_path = os.path.join(args.output_dir, 'ripe_apple_count_vs_brightness.png')
        detection_time_chart_path = os.path.join(args.output_dir, 'detection_time_vs_brightness.png')
        combined_chart_path = os.path.join(args.output_dir, 'combined_chart.png')
    
    # Создаем и показываем графики
    print("\nСоздание графика количества зрелых яблок...")
    create_ripe_count_chart(brightness_changes, avg_ripe_counts, ripe_count_chart_path)
    
    print("Создание графика времени обнаружения...")
    create_detection_time_chart(brightness_changes, avg_detection_times, detection_time_chart_path)
    
    print("Создание совмещенного графика...")
    create_combined_chart(brightness_changes, avg_ripe_counts, avg_detection_times, combined_chart_path)
    
    if args.save:
        print(f"\nГрафики сохранены в директории {args.output_dir}")


if __name__ == "__main__":
    main()