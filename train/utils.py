#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import json


def seed_everything(seed):
    """
    Устанавливает seed для воспроизводимости результатов.

    Args:
        seed (int): Значение seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_map_simple(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Упрощенный и надежный расчет Mean Average Precision (mAP).
    
    Args:
        predictions (list): Список предсказаний модели
        targets (list): Список целевых аннотаций
        iou_threshold (float): Порог IoU для определения правильного обнаружения
        score_threshold (float): Порог уверенности для фильтрации предсказаний
    
    Returns:
        float: Значение mAP (0-1)
    """
    try:
        # Собираем все классы из целевых данных
        class_ids = set()
        for target in targets:
            if "labels" in target and len(target["labels"]) > 0:
                for label in target["labels"].cpu().numpy():
                    class_ids.add(int(label))
        
        # Если нет классов, возвращаем 0
        if not class_ids:
            return 0.0
        
        # Вычисляем AP для каждого класса
        aps = []
        
        for class_id in class_ids:
            # Собираем все предсказания и метки для данного класса
            all_detections = []  # (image_idx, confidence, is_true_positive)
            num_gt_per_image = []  # Количество gt объектов в каждом изображении
            
            # Для каждого изображения
            for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
                # Подсчет gt объектов данного класса
                if "labels" in target and len(target["labels"]) > 0:
                    gt_labels = target["labels"].cpu()
                    gt_count = torch.sum(gt_labels == class_id).item()
                else:
                    gt_count = 0
                
                num_gt_per_image.append(gt_count)
                
                # Если нет предсказаний, пропускаем
                if "boxes" not in pred or len(pred["boxes"]) == 0:
                    continue
                
                # Получаем предсказания для текущего класса
                pred_boxes = pred["boxes"].cpu()
                pred_scores = pred["scores"].cpu()
                pred_labels = pred["labels"].cpu()
                
                # Фильтрация по классу и порогу уверенности
                class_mask = pred_labels == class_id
                score_mask = pred_scores >= score_threshold
                mask = class_mask & score_mask
                
                if torch.sum(mask) == 0:
                    continue
                
                filtered_boxes = pred_boxes[mask]
                filtered_scores = pred_scores[mask]
                
                # Если нет gt объектов, все предсказания - ложноположительные
                if gt_count == 0:
                    for score in filtered_scores:
                        all_detections.append((img_idx, score.item(), False))
                    continue
                
                # Получаем gt боксы для текущего класса
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()
                gt_mask = gt_labels == class_id
                gt_boxes_filtered = gt_boxes[gt_mask]
                
                # Используем torchvision.ops.box_iou для расчета IoU
                iou_matrix = box_iou(filtered_boxes, gt_boxes_filtered)
                
                # Отслеживаем, какие gt объекты уже были сопоставлены
                gt_matched = [False] * len(gt_boxes_filtered)
                
                # Для каждого предсказания проверяем, является ли оно TP или FP
                for det_idx, score in enumerate(filtered_scores):
                    # Получаем IoU с gt боксами
                    ious = iou_matrix[det_idx]
                    
                    # Если нет gt с IoU > порога, это FP
                    if torch.max(ious).item() < iou_threshold:
                        all_detections.append((img_idx, score.item(), False))
                        continue
                    
                    # Находим лучший gt бокс
                    best_gt_idx = torch.argmax(ious).item()
                    
                    # Если этот gt уже был сопоставлен, это FP
                    if gt_matched[best_gt_idx]:
                        all_detections.append((img_idx, score.item(), False))
                    else:
                        # Иначе это TP
                        all_detections.append((img_idx, score.item(), True))
                        gt_matched[best_gt_idx] = True
            
            # Вычисление AP для текущего класса
            # Сортировка по уверенности (убывание)
            all_detections.sort(key=lambda x: x[1], reverse=True)
            
            # Вычисление precision-recall
            tp = 0
            fp = 0
            total_gt = sum(num_gt_per_image)
            
            if total_gt == 0:
                continue  # Пропускаем класс, если нет gt объектов
            
            precisions = []
            recalls = []
            
            for _, _, is_tp in all_detections:
                if is_tp:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp)
                recall = tp / total_gt
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Если нет предсказаний, AP = 0
            if not precisions:
                aps.append(0.0)
                continue
            
            # Расчет AP (площадь под precision-recall кривой)
            # Используем метод интерполяции 11 точек (PASCAL VOC)
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(np.array(recalls) >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.array(precisions)[np.array(recalls) >= t])
                ap += p / 11.0
            
            aps.append(ap)
        
        # Вычисление mAP как среднего AP по всем классам
        map_value = np.mean(aps) if aps else 0.0
        
        # Проверка корректности
        if map_value < 0.0 or map_value > 1.0:
            print(f"Внимание: Вычисленное значение mAP ({map_value}) вне диапазона [0,1]. Ограничиваем.")
            map_value = max(0.0, min(1.0, map_value))
        
        return map_value
    
    except Exception as e:
        print(f"Ошибка при вычислении mAP: {e}")
        return 0.0

def calculate_f1_score_simple(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Упрощенный и надежный расчет F1-score.
    
    Args:
        predictions (list): Список предсказаний модели
        targets (list): Список целевых аннотаций
        iou_threshold (float): Порог IoU для определения правильного обнаружения
        score_threshold (float): Порог уверенности для фильтрации предсказаний
    
    Returns:
        float: Значение F1-score (0-1)
    """
    try:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Обработка каждой пары (предсказание, цель)
        for pred, target in zip(predictions, targets):
            # Проверка наличия предсказаний
            if "boxes" not in pred or len(pred["boxes"]) == 0:
                # Если нет предсказаний, но есть цели, все цели — FN
                if "boxes" in target and len(target["boxes"]) > 0:
                    false_negatives += len(target["boxes"])
                continue
            
            # Проверка наличия целей
            if "boxes" not in target or len(target["boxes"]) == 0:
                # Если нет целей, но есть предсказания после фильтрации, все предсказания — FP
                filtered_preds = [score for score in pred["scores"] if score >= score_threshold]
                false_positives += len(filtered_preds)
                continue
            
            # Фильтрация предсказаний по порогу уверенности
            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu() if "labels" in pred else None
            
            score_mask = pred_scores >= score_threshold
            filtered_boxes = pred_boxes[score_mask]
            filtered_labels = pred_labels[score_mask] if pred_labels is not None else None
            
            if len(filtered_boxes) == 0:
                # Если после фильтрации не осталось предсказаний, все цели — FN
                false_negatives += len(target["boxes"])
                continue
            
            # Получение целевых данных
            target_boxes = target["boxes"].cpu()
            target_labels = target["labels"].cpu() if "labels" in target else None
            
            # Вычисление IoU между всеми парами боксов
            iou_matrix = box_iou(filtered_boxes, target_boxes)
            
            # Отслеживание сопоставленных объектов
            target_matched = torch.zeros(len(target_boxes), dtype=torch.bool)
            pred_matched = torch.zeros(len(filtered_boxes), dtype=torch.bool)
            
            # Сопоставление предсказаний с целями
            for pred_idx in range(len(filtered_boxes)):
                best_iou, best_target_idx = torch.max(iou_matrix[pred_idx], dim=0)
                
                # Проверка порога IoU и совпадения классов (если есть)
                if best_iou >= iou_threshold:
                    if (filtered_labels is None or target_labels is None or 
                        filtered_labels[pred_idx] == target_labels[best_target_idx]):
                        if not target_matched[best_target_idx]:
                            # True Positive
                            true_positives += 1
                            target_matched[best_target_idx] = True
                            pred_matched[pred_idx] = True
            
            # Подсчет FP и FN
            false_positives += torch.sum(~pred_matched).item()
            false_negatives += torch.sum(~target_matched).item()
        
        # Вычисление precision и recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Вычисление F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Проверка корректности
        if f1 < 0 or f1 > 1:
            print(f"Внимание: F1 ({f1}) вне диапазона [0,1]. Ограничиваем.")
            f1 = max(0, min(1, f1))
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
        
        return f1
    
    except Exception as e:
        print(f"Ошибка при вычислении F1-score: {e}")
        return 0.0
        
def calculate_map(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Расчет Mean Average Precision (mAP) для предсказаний.
    
    Args:
        predictions (list): Список предсказаний модели
        targets (list): Список целевых аннотаций
        iou_threshold (float): Порог IoU для определения правильного обнаружения
        score_threshold (float): Порог уверенности для фильтрации предсказаний
    
    Returns:
        float: Значение mAP
    """
    # Вызываем упрощенную и надежную функцию вычисления mAP
    return calculate_map_simple(predictions, targets, iou_threshold, score_threshold)


def calculate_f1_score(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Расчет F1-score для предсказаний.
    
    Args:
        predictions (list): Список предсказаний модели
        targets (list): Список целевых аннотаций
        iou_threshold (float): Порог IoU для определения правильного обнаружения
        score_threshold (float): Порог уверенности для фильтрации предсказаний
    
    Returns:
        float: Значение F1-score
    """
    # Вызываем упрощенную и надежную функцию вычисления F1-score
    return calculate_f1_score_simple(predictions, targets, iou_threshold, score_threshold)


def create_coco_format(data, is_gt=True):
    """
    Преобразование данных в формат COCO.
    
    Args:
        data (list): Список данных (предсказаний или целей)
        is_gt (bool): Флаг, указывающий на тип данных (ground truth или предсказания)
    
    Returns:
        dict: Данные в формате COCO
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "unripe_apple"},
            {"id": 2, "name": "ripe_apple"}
        ]
    }
    
    ann_id = 1
    for i, item in enumerate(data):
        try:
            # В PyTorch 2.7.0 формат предсказаний может отличаться
            # Проверяем наличие нужных ключей и адаптируем код
            
            # Получаем image_id
            if "image_id" in item:
                if isinstance(item["image_id"], torch.Tensor):
                    image_id = int(item["image_id"].cpu().numpy()[0])
                else:
                    image_id = item["image_id"]
            else:
                # Если нет image_id, используем индекс
                image_id = i + 1
            
            # Добавление информации об изображении
            coco_format["images"].append({
                "id": image_id,
                "width": 512,  # предполагаем, что все изображения имеют размер 512x512
                "height": 512,
                "file_name": f"image_{image_id}.jpg"
            })
            
            # Адаптация для разных форматов предсказаний/целей
            if is_gt:
                # Ground truth
                if "boxes" in item and len(item["boxes"]) > 0:
                    boxes = item["boxes"].cpu().numpy()
                    labels = item["labels"].cpu().numpy()
                    
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = box.tolist()
                        width = x2 - x1
                        height = y2 - y1
                        
                        coco_format["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        })
                        ann_id += 1
            else:
                # Предсказания - адаптация для PyTorch 2.7.0
                if "boxes" in item and len(item["boxes"]) > 0:
                    # Стандартный формат
                    boxes = item["boxes"].cpu().numpy()
                    scores = item["scores"].cpu().numpy()
                    labels = item["labels"].cpu().numpy()
                elif isinstance(item, dict) and "instances" in item:
                    # Альтернативный формат, может использоваться в PyTorch 2.7.0
                    boxes = item["instances"]["boxes"].cpu().numpy()
                    scores = item["instances"]["scores"].cpu().numpy()
                    labels = item["instances"]["labels"].cpu().numpy()
                elif isinstance(item, dict) and "pred_boxes" in item:
                    # Еще один возможный формат
                    boxes = item["pred_boxes"].cpu().numpy()
                    scores = item["scores"].cpu().numpy() if "scores" in item else np.ones(len(boxes))
                    labels = item["pred_classes"].cpu().numpy() if "pred_classes" in item else np.ones(len(boxes))
                else:
                    # Пропускаем, если формат неизвестен
                    continue
                
                for box, score, label in zip(boxes, scores, labels):
                    # Преобразуем box в формат [x1, y1, x2, y2], если это еще не сделано
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                    else:
                        # Предполагаем формат центр-ширина-высота
                        cx, cy, w, h = box
                        x1, y1 = cx - w/2, cy - h/2
                        x2, y2 = cx + w/2, cy + h/2
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 0 and height > 0:
                        coco_format["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, width, height],
                            "area": width * height,
                            "score": float(score),
                            "iscrowd": 0
                        })
                        ann_id += 1
        except Exception as e:
            print(f"Ошибка при обработке элемента {i}: {e}")
            continue
    
    return coco_format

def visualize_results(image, prediction, category_names, threshold=0.5):
    """
    Визуализация результатов обнаружения на изображении.

    Args:
        image: Исходное изображение (тензор)
        prediction: Предсказание модели
        category_names: Словарь соответствия id классов их названиям
        threshold: Порог уверенности для отображения предсказаний

    Returns:
        plt.Figure: Объект Figure с визуализацией
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import to_rgba

    # Преобразование изображения из тензора в numpy array
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Нормализация для отображения
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_np)

    # Цвета для разных классов
    colors = {
        1: 'yellow',  # незрелое яблоко
        2: 'red'  # зрелое яблоко
    }

    # Отображение предсказанных боксов
    if "boxes" in prediction:
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score > threshold:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                # Получение цвета для класса
                color = colors.get(label, 'white')

                # Добавление прямоугольника
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=to_rgba(color, 0.3)
                )
                ax.add_patch(rect)

                # Добавление метки
                class_name = category_names.get(label, f"Class {label}")
                ax.text(
                    x1, y1 - 5,
                    f"{class_name}: {score:.2f}",
                    color='white',
                    fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
                )

    plt.axis('off')
    plt.tight_layout()

    return fig