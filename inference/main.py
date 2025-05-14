#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основной модуль для распознавания яблок на изображении.
Использование:
    python main.py --image path/to/image.jpg --model path/to/model.pth
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import argparse
import time
import torch
import numpy as np

from model import load_model
from image_processor import preprocess_image
from detector import detect_apples
from visualizer import visualize_results, print_detections


def parse_args():
    parser = argparse.ArgumentParser(description='Распознавание яблок на изображении')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--model', type=str, default='apple_detector.pth', help='Путь к файлу модели')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог уверенности')
    parser.add_argument('--output', type=str, default=None, help='Путь для сохранения результата (опционально)')
    parser.add_argument('--quiet', action='store_true', help='Не выводить изображение на экран')
    parser.add_argument('--raw', action='store_true', help='Вывести только список bbox без отрисовки')
    parser.add_argument('--time', action='store_true', help='Замерить время выполнения детекции')
    parser.add_argument('--no_labels', action='store_true', help='Выводить рамки без текстовых меток')
    return parser.parse_args()


def main():
    # Парсинг аргументов
    args = parse_args()

    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.quiet and not args.raw:
        print(f"Используется устройство: {device}")

    # Загружаем модель с безопасной загрузкой
    if not args.quiet and not args.raw:
        print(f"Загрузка модели из: {args.model}")
    model = load_model(args.model, num_classes=3, device=device, quiet=args.quiet or args.raw)

    # Загружаем и обрабатываем изображение
    if not args.quiet and not args.raw:
        print(f"Обработка изображения: {args.image}")
    try:
        image_tensor, original_image, (height, width) = preprocess_image(args.image)
        if not args.quiet and not args.raw:
            print(f"Размеры изображения: {width}x{height}")
    except Exception as e:
        if not args.quiet:
            print(f"Ошибка при загрузке изображения: {e}")
        return

    # Обнаруживаем яблоки с замером времени, если требуется
    if not args.quiet and not args.raw:
        print("Обнаружение яблок...")

    # Замер времени выполнения детекции, если требуется
    if args.time:
        start_time = time.time()
        results = detect_apples(model, image_tensor, (height, width), threshold=args.threshold, device=device)
        elapsed_time = time.time() - start_time
        if not args.quiet:
            print(f"Время обнаружения: {elapsed_time:.4f} секунд")
    else:
        results = detect_apples(model, image_tensor, (height, width), threshold=args.threshold, device=device)

    # Определяем путь для сохранения результата
    output_path = args.output
    if output_path is None and args.image and not args.raw:
        # Если путь для сохранения не указан, создаем его на основе пути к исходному изображению
        base_name = os.path.splitext(args.image)[0]
        output_path = f"{base_name}_result.png"

    # Обработка результатов
    if args.raw:
        # Только вывод списка bbox без отрисовки
        if not args.quiet:
            print_detections(results)
    else:
        # Визуализация результатов
        if not args.quiet:
            print("Визуализация результатов...")
        visualize_results(original_image, results, output_path, show_image=not args.quiet, show_labels=not args.no_labels)


if __name__ == "__main__":
    main()