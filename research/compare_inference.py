#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Программа для сравнения времени инференса моделей детекции объектов.
Тестирует Faster R-CNN, FCOS и RetinaNet на изображениях без разметки.
"""

import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Импорты моделей
import torchvision
from torchvision.models.detection import (
    fcos_resnet50_fpn, 
    retinanet_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights
)
from torchvision.models.detection.fcos import FCOSHead
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Импорты из проекта
from model.model import get_faster_rcnn_model


def load_faster_rcnn_model(model_path, device):
    """Загружает обученную модель Faster R-CNN."""
    model = get_faster_rcnn_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def create_fcos_model(num_classes):
    """Создает модель FCOS."""
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights)
    
    in_channels = model.backbone.out_channels
    model.head = FCOSHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_anchors=1
    )
    return model


def create_retinanet_model(num_classes):
    """Создает модель RetinaNet."""
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights)
    
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head = RetinaNetHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    return model


def load_fcos_model(model_path, device):
    """Загружает обученную модель FCOS."""
    model = create_fcos_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_retinanet_model(model_path, device):
    """Загружает обученную модель RetinaNet."""
    model = create_retinanet_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """
    Предобработка изображения для инференса.
    
    Args:
        image_path (str): Путь к изображению
        
    Returns:
        torch.Tensor: Предобработанное изображение
    """
    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Трансформации (без аугментаций, только нормализация)
    transform = A.Compose([
        A.Resize(512, 512),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ])
    
    # Применение трансформаций
    transformed = transform(image=image_np)
    image_tensor = transformed["image"]
    
    # Проверка типа данных
    if not image_tensor.is_floating_point():
        image_tensor = image_tensor.float()
    
    return image_tensor


def measure_inference_time(model, image_tensor, device, warmup_runs=5, test_runs=10):
    """
    Измеряет время инференса модели.
    
    Args:
        model: Модель для тестирования
        image_tensor: Предобработанное изображение
        device: Устройство для вычислений
        warmup_runs: Количество прогревочных запусков
        test_runs: Количество тестовых запусков
        
    Returns:
        float: Среднее время инференса в секундах
    """
    model.eval()
    image_tensor = image_tensor.to(device).unsqueeze(0)  # Добавляем batch dimension
    
    # Прогревочные запуски
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(image_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Тестовые запуски с замером времени
    times = []
    with torch.no_grad():
        for _ in range(test_runs):
            start_time = time.time()
            _ = model(image_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    return np.mean(times)


def find_image_files(directory):
    """
    Находит все файлы изображений в директории.
    
    Args:
        directory (str): Путь к директории
        
    Returns:
        list: Список путей к изображениям
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)


def benchmark_models(models_config, image_files, device, max_images=50):
    """
    Проводит бенчмарк всех моделей.
    
    Args:
        models_config (dict): Конфигурация моделей {название: путь_к_модели}
        image_files (list): Список путей к изображениям
        device: Устройство для вычислений
        max_images (int): Максимальное количество изображений для тестирования
        
    Returns:
        dict: Результаты бенчмарка
    """
    results = {}
    
    # Ограничиваем количество изображений для ускорения тестирования
    test_images = image_files[:max_images] if len(image_files) > max_images else image_files
    print(f"Тестирование на {len(test_images)} изображениях")
    
    # Загрузка и тестирование каждой модели
    for model_name, model_path in models_config.items():
        print(f"\nТестирование модели: {model_name}")
        print(f"Загрузка модели из: {model_path}")
        
        # Загрузка модели
        try:
            if model_name == 'faster_rcnn':
                model = load_faster_rcnn_model(model_path, device)
            elif model_name == 'fcos':
                model = load_fcos_model(model_path, device)
            elif model_name == 'retinanet':
                model = load_retinanet_model(model_path, device)
            else:
                print(f"Неизвестная модель: {model_name}")
                continue
                
            print(f"Модель {model_name} успешно загружена")
        except Exception as e:
            print(f"Ошибка при загрузке модели {model_name}: {e}")
            continue
        
        # Измерение времени на каждом изображении
        inference_times = []
        
        for image_path in tqdm(test_images, desc=f"Тестирование {model_name}"):
            try:
                # Предобработка изображения
                image_tensor = preprocess_image(image_path)
                
                # Измерение времени инференса
                inference_time = measure_inference_time(model, image_tensor, device)
                inference_times.append(inference_time)
                
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")
                continue
        
        # Вычисление статистики
        if inference_times:
            results[model_name] = {
                'mean_time': np.mean(inference_times),
                'std_time': np.std(inference_times),
                'min_time': np.min(inference_times),
                'max_time': np.max(inference_times),
                'median_time': np.median(inference_times),
                'all_times': inference_times,
                'images_processed': len(inference_times)
            }
            
            print(f"Среднее время инференса {model_name}: {results[model_name]['mean_time']:.4f} ± {results[model_name]['std_time']:.4f} сек")
        else:
            print(f"Не удалось обработать изображения для модели {model_name}")
        
        # Освобождение памяти
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results


def create_benchmark_plot(results, output_path):
    """
    Создает столбчатую диаграмму результатов бенчмарка.
    
    Args:
        results (dict): Результаты бенчмарка
        output_path (str): Путь для сохранения графика
    """
    if not results:
        print("Нет результатов для построения графика")
        return
    
    # Настройка стиля для черно-белой совместимости
    plt.style.use('default')
    
    # Данные для графика
    model_names = list(results.keys())
    mean_times = [results[name]['mean_time'] * 1000 for name in model_names]  # Конвертируем в миллисекунды
    std_times = [results[name]['std_time'] * 1000 for name in model_names]
    
    # Цвета и узоры для черно-белой совместимости
    colors = ['lightgray', 'gray', 'darkgray']
    hatches = ['', '///', '...']
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Столбчатая диаграмма с планками ошибок
    bars = ax.bar(
        model_names, 
        mean_times, 
        yerr=std_times,
        color=colors[:len(model_names)],
        hatch=hatches[:len(model_names)],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8,
        capsize=5
    )
    
    # Добавление значений на столбцы
    for bar, mean_time, std_time in zip(bars, mean_times, std_times):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + std_time + height*0.02,
            f'{mean_time:.1f}ms',
            ha='center', 
            va='bottom', 
            fontweight='bold',
            fontsize=12
        )
    
    # Настройка осей и заголовка
    ax.set_ylabel('Время инференса (миллисекунды)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Модели', fontweight='bold', fontsize=12)
    ax.set_title('Сравнение времени инференса моделей детекции объектов', 
                fontweight='bold', fontsize=14)
    
    # Сетка для лучшей читаемости
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Улучшение читаемости названий моделей
    model_labels = {
        'faster_rcnn': 'Faster R-CNN',
        'fcos': 'FCOS', 
        'retinanet': 'RetinaNet'
    }
    ax.set_xticklabels([model_labels.get(name, name) for name in model_names])
    
    # Настройка макета
    plt.tight_layout()
    
    # Сохранение графика
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    
    print(f"График сохранен в: {output_path}")


def save_benchmark_results(results, output_path):
    """
    Сохраняет результаты бенчмарка в JSON файл.
    
    Args:
        results (dict): Результаты бенчмарка
        output_path (str): Путь для сохранения JSON файла
    """
    # Подготовка данных для JSON (удаляем numpy массивы)
    json_results = {}
    for model_name, stats in results.items():
        json_results[model_name] = {
            'mean_time_seconds': stats['mean_time'],
            'std_time_seconds': stats['std_time'],
            'min_time_seconds': stats['min_time'],
            'max_time_seconds': stats['max_time'],
            'median_time_seconds': stats['median_time'],
            'images_processed': stats['images_processed'],
            'mean_time_ms': stats['mean_time'] * 1000,
            'std_time_ms': stats['std_time'] * 1000
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты сохранены в: {output_path}")


def main():
    """Основная функция программы."""
    parser = argparse.ArgumentParser(description='Бенчмарк времени инференса моделей детекции')
    parser.add_argument('--data_dir', type=str, default='../data/train',
                       help='Директория с тестовыми изображениями')
    parser.add_argument('--faster_rcnn_model', type=str, default='../train/apple_detector.pth',
                       help='Путь к модели Faster R-CNN')
    parser.add_argument('--fcos_model', type=str, default='model_comparison_results/fcos/fcos_detector.pth',
                       help='Путь к модели FCOS')
    parser.add_argument('--retinanet_model', type=str, default='model_comparison_results/retinanet/retinanet_detector.pth',
                       help='Путь к модели RetinaNet')
    parser.add_argument('--output_dir', type=str, default='inference_benchmark_results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--max_images', type=int, default=50,
                       help='Максимальное количество изображений для тестирования')
    
    args = parser.parse_args()
    
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Поиск изображений
    print(f"Поиск изображений в директории: {args.data_dir}")
    image_files = find_image_files(args.data_dir)
    
    if not image_files:
        print(f"Не найдено изображений в директории: {args.data_dir}")
        return
    
    print(f"Найдено {len(image_files)} изображений")
    
    # Конфигурация моделей
    models_config = {}
    
    if os.path.exists(args.faster_rcnn_model):
        models_config['faster_rcnn'] = args.faster_rcnn_model
    else:
        print(f"Модель Faster R-CNN не найдена: {args.faster_rcnn_model}")
    
    if os.path.exists(args.fcos_model):
        models_config['fcos'] = args.fcos_model
    else:
        print(f"Модель FCOS не найдена: {args.fcos_model}")
    
    if os.path.exists(args.retinanet_model):
        models_config['retinanet'] = args.retinanet_model
    else:
        print(f"Модель RetinaNet не найдена: {args.retinanet_model}")
    
    if not models_config:
        print("Не найдено ни одной модели для тестирования")
        return
    
    print(f"Будут протестированы модели: {', '.join(models_config.keys())}")
    
    # Запуск бенчмарка
    print("\nНачало бенчмарка...")
    results = benchmark_models(models_config, image_files, device, args.max_images)
    
    if not results:
        print("Не удалось получить результаты бенчмарка")
        return
    
    # Сохранение результатов
    json_path = os.path.join(args.output_dir, 'benchmark_results.json')
    save_benchmark_results(results, json_path)
    
    # Создание графика
    plot_path = os.path.join(args.output_dir, 'inference_time_comparison.png')
    create_benchmark_plot(results, plot_path)
    
    # Вывод итоговой статистики
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print("="*60)
    
    for model_name, stats in results.items():
        print(f"\nМодель: {model_name}")
        print(f"  Среднее время: {stats['mean_time']*1000:.2f} ± {stats['std_time']*1000:.2f} мс")
        print(f"  Медианное время: {stats['median_time']*1000:.2f} мс")
        print(f"  Мин/Макс время: {stats['min_time']*1000:.2f} / {stats['max_time']*1000:.2f} мс")
        print(f"  Обработано изображений: {stats['images_processed']}")
    
    # Определение самой быстрой модели
    fastest_model = min(results.items(), key=lambda x: x[1]['mean_time'])
    print(f"\nСамая быстрая модель: {fastest_model[0]} ({fastest_model[1]['mean_time']*1000:.2f} мс)")
    
    print("="*60)
    print(f"Результаты сохранены в директории: {args.output_dir}")


if __name__ == "__main__":
    main()