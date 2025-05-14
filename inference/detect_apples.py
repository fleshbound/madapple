#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Программа для распознавания и визуализации яблок на изображении.
Исправлена проблема с цветовыми каналами при визуализации.
Использование:
    python detect_apples.py --image path/to/image.jpg --model path/to/model.pth
"""

import os
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Подавление предупреждений Albumentations
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

def get_faster_rcnn_model(num_classes):
    """
    Создает модель Faster R-CNN на основе предобученного ResNet-50 backbone.

    Args:
        num_classes (int): Количество классов (включая фон)

    Returns:
        model (torchvision.models.detection.FasterRCNN): Модель Faster R-CNN
    """
    # Загрузка предтренированной модели с весами
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Получаем количество входных признаков для классификатора
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Заменяем выходной слой на новый с нужным количеством классов
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Настройка параметров модели для лучшего обучения на маленьком датасете
    # Чувствительность NMS (Non-Maximum Suppression)
    model.roi_heads.nms_thresh = 0.3

    # Минимальный размер обнаруживаемого объекта
    model.roi_heads.box_roi_pool.output_size = (7, 7)

    # Настройка минимального порога доверия для предложений области интереса
    model.rpn.nms_thresh = 0.6
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.post_nms_top_n_test = 1000

    # Добавляем дополнительные параметры для маленьких объектов
    model.rpn.anchor_generator.sizes = ((16,), (32,), (64,), (128,), (256,))
    model.rpn.anchor_generator.aspect_ratios = ((0.5, 1.0, 2.0),) * 5

    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Распознавание яблок на изображении')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--model', type=str, default='..\\train\\apple_detector.pth', help='Путь к файлу модели')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог уверенности')
    parser.add_argument('--output', type=str, default=None, help='Путь для сохранения результата (опционально)')
    parser.add_argument('--fpn_version', type=str, default='v1', choices=['v1', 'v2'], 
                        help='Версия FPN архитектуры (v1 или v2)')
    return parser.parse_args()

def get_model(num_classes=3, fpn_version='v1'):
    """
    Создает модель Faster R-CNN с нужным количеством классов.
    
    Args:
        num_classes (int): Количество классов (включая фон)
        fpn_version (str): Версия FPN архитектуры ('v1' или 'v2')
    
    Returns:
        torch.nn.Module: Модель Faster R-CNN
    """
    try:
        if fpn_version == 'v2':
            # Версия FPN v2 (новая архитектура)
            try:
                # Для новых версий PyTorch (>= 1.13)
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
            except:
                # Для более старых версий
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
        else:
            # Версия FPN v1 (старая архитектура) - обычно используется для compatibility
            try:
                # Для новых версий PyTorch
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            except:
                # Для старых версий
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        
        # Получаем количество входных признаков для предиктора
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Заменяем предиктор на новый с нужным количеством классов
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return model
    except Exception as e:
        print(f"Ошибка при создании модели: {e}")
        # Если все не удалось, возвращаем базовую модель
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

def load_model_safely(model_path, num_classes=3, device="cpu"):
    """
    Безопасная загрузка модели с попыткой определить правильную архитектуру.
    
    Args:
        model_path (str): Путь к файлу модели
        num_classes (int): Количество классов
        device (str): Устройство для загрузки
    
    Returns:
        torch.nn.Module: Загруженная модель
    """
    # Пробуем разные архитектуры, начиная с v1 (более старой и распространенной)
    #architectures = ['v1', 'v2']
    
    #for arch in architectures:
    #    try:
            #print(f"Пробуем загрузить модель с архитектурой FPN {arch}...")
    print(f"Пробуем загрузить модель")
    #model = get_model(num_classes=num_classes, fpn_version=arch)
    model = get_faster_rcnn_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Модель успешно загружена") # с архитектурой FPN {arch}")
    return model
    #    except Exception as e:
    #        print(f"Не удалось загрузить с архитектурой {arch}: {e}")
    
    # Если не удалось загрузить ни одну из архитектур, 
    # попробуем загрузить с полной моделью (включая архитектуру)
    try:
        print("Пробуем загрузить полную модель...")
        model = torch.load(model_path, map_location=device)
        print("Полная модель успешно загружена")
        return model
    except Exception as e:
        print(f"Не удалось загрузить полную модель: {e}")
    
    # Последняя попытка - создаем новую модель и предупреждаем пользователя
    print("ВНИМАНИЕ: Не удалось загрузить модель. Создана новая модель без предобученных весов.")
    return get_model(num_classes=num_classes, fpn_version='v1')

def preprocess_image(image_path):
    """
    Загружает и предобрабатывает изображение для модели.
    
    Args:
        image_path (str): Путь к изображению
    
    Returns:
        tuple: (tensor, original_image, dimensions)
    """
    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    
    # Сохраняем оригинальные размеры
    height, width, _ = original_image.shape
    
    # Создаем трансформации (без аугментаций, только нормализация)
    transform = A.Compose([
        A.Resize(512, 512),
        A.ToFloat(max_value=255.0),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Применяем трансформации
    transformed = transform(image=original_image)
    image_tensor = transformed["image"]
    
    # Проверка типа и диапазона значений
    if not image_tensor.is_floating_point():
        image_tensor = image_tensor.float()
    
    # Возвращаем тензор, оригинальное изображение и размеры
    return image_tensor, original_image, (height, width)


def detect_apples(model, image_tensor, original_size, threshold=0.5, device="cpu"):
    """
    Обнаруживает яблоки на изображении и масштабирует боксы к исходному размеру.

    Args:
        model: Модель Faster R-CNN
        image_tensor (torch.Tensor): Тензор изображения
        original_size (tuple): Исходные размеры изображения (height, width)
        threshold (float): Порог уверенности
        device (str): Устройство для вычислений

    Returns:
        dict: Словарь с результатами обнаружения
    """
    # Перемещаем модель и данные на нужное устройство
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    # Переводим модель в режим оценки
    model.eval()

    # Добавляем размерность батча
    image_tensor = image_tensor.unsqueeze(0)

    # Получаем предсказания
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Фильтруем предсказания по порогу уверенности
    mask = prediction['scores'] >= threshold
    boxes = prediction['boxes'][mask].cpu().numpy()
    scores = prediction['scores'][mask].cpu().numpy()
    labels = prediction['labels'][mask].cpu().numpy()

    # Масштабируем боксы к исходному размеру
    original_height, original_width = original_size
    # Коэффициенты масштабирования
    width_scale = original_width / 512
    height_scale = original_height / 512

    # Применяем масштабирование к каждому боксу
    scaled_boxes = boxes.copy()
    scaled_boxes[:, 0] *= width_scale  # x1
    scaled_boxes[:, 1] *= height_scale  # y1
    scaled_boxes[:, 2] *= width_scale  # x2
    scaled_boxes[:, 3] *= height_scale  # y2

    return {
        'boxes': scaled_boxes,
        'scores': scores,
        'labels': labels
    }

def visualize_results(image, results, output_path=None):
    """
    Визуализирует результаты обнаружения и выводит статистику.
    Исправлена проблема с отображением кириллических символов.
    
    Args:
        image (np.ndarray): Оригинальное изображение (в формате RGB)
        results (dict): Результаты обнаружения
        output_path (str, optional): Путь для сохранения результата
    """
    # Создаем копию изображения для визуализации
    vis_image = image.copy()  # Работаем с RGB изображением
    
    # Определяем цвета для классов (в RGB)
    colors = {
        1: (255, 255, 0),  # Желтый для незрелых яблок (RGB)
        2: (255, 0, 0)     # Красный для зрелых яблок (RGB)
    }
    
    # Названия классов
    class_names = {
        1: "Unripe apple",  # Используем английские названия для OpenCV
        2: "Ripe apple"
    }
    
    # Счетчики яблок
    unripe_count = 0
    ripe_count = 0
    
    # Используем PIL для рисования, так как она лучше обрабатывает разные кодировки
    from PIL import Image, ImageDraw, ImageFont
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
        print(f"Не удалось загрузить шрифт: {e}")
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
    
    # Отображаем результаты
    plt.figure(figsize=(12, 10))
    plt.imshow(vis_image)
    plt.axis('off')
    plt.title(f"Found: {unripe_count} unripe and {ripe_count} ripe apples", fontsize=14)
    
    # Добавляем легенду
    import matplotlib.patches as mpatches
    unripe_patch = mpatches.Patch(color='yellow', label='Unripe apples')
    ripe_patch = mpatches.Patch(color='red', label='Ripe apples')
    plt.legend(handles=[unripe_patch, ripe_patch], loc='upper right', fontsize=12)
    
    # Выводим информацию о количестве яблок
    print(f"Обнаружено на изображении:")
    print(f"- Незрелых яблок: {unripe_count}")
    print(f"- Зрелых яблок: {ripe_count}")
    print(f"- Всего яблок: {unripe_count + ripe_count}")
    
    # Сохраняем результат, если указан путь
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Результат сохранен в: {output_path}")
    
    # Отображаем изображение
    plt.show()

def main():
    # Парсинг аргументов
    args = parse_args()
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загружаем модель с безопасной загрузкой
    print(f"Загрузка модели из: {args.model}")
    model = load_model_safely(args.model, num_classes=3, device=device)
    
    # Загружаем и обрабатываем изображение
    print(f"Обработка изображения: {args.image}")
    try:
        image_tensor, original_image, (height, width) = preprocess_image(args.image)
        print(f"Размеры изображения: {width}x{height}")
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return
    
    # Обнаруживаем яблоки
    print("Обнаружение яблок...")
    results = detect_apples(model, image_tensor, (height, width), threshold=args.threshold, device=device)
    
    # Определяем путь для сохранения результата
    output_path = args.output
    if output_path is None and args.image:
        # Если путь для сохранения не указан, создаем его на основе пути к исходному изображению
        base_name = os.path.splitext(args.image)[0]
        output_path = f"{base_name}_result.png"
    
    # Визуализируем результаты
    print("Визуализация результатов...")
    visualize_results(original_image, results, output_path)

if __name__ == "__main__":
    main()