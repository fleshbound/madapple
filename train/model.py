#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


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


def save_model(model, path):
    """
    Сохранение модели.

    Args:
        model: Модель для сохранения
        path (str): Путь для сохранения модели
    """
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """
    Загрузка модели.

    Args:
        model: Модель для загрузки весов
        path (str): Путь к сохраненной модели
        device: Устройство для загрузки модели (cuda/cpu)

    Returns:
        model: Загруженная модель
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model