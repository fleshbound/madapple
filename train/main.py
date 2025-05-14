#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import argparse
import torch
import random
import numpy as np

from trainer import Trainer
from model import get_faster_rcnn_model
from dataset import AppleDataset, get_data_loaders
from logger import setup_logger
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description='Обучение Faster R-CNN для определения количества зрелых плодов на изображениях')
    parser.add_argument('--data_path', type=str, default='..\\data\\train', help='Путь к данным для обучения')
    parser.add_argument('--annotations_file', type=str, default='_annotations.coco.json',
                        help='Имя файла аннотаций в формате COCO')
    parser.add_argument('--output_model', type=str, default='apple_detector.pth',
                        help='Имя файла для сохранения модели')
    parser.add_argument('--history_path', type=str, default='training_history',
                        help='Путь для сохранения истории обучения')
    parser.add_argument('--log_dir', type=str, default='logs', help='Директория для логов')
    parser.add_argument('--batch_size', type=int, default=8, help='Размер батча')
    parser.add_argument('--num_epochs', type=int, default=30, help='Количество эпох обучения')
    parser.add_argument('--lr', type=float, default=0.001, help='Начальная скорость обучения')
    parser.add_argument('--momentum', type=float, default=0.9, help='Момент для оптимизатора')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Коэффициент регуляризации L2')
    parser.add_argument('--num_workers', type=int, default=4, help='Количество рабочих для загрузки данных')
    parser.add_argument('--seed', type=int, default=42, help='Seed для воспроизводимости')
    parser.add_argument('--val_size', type=float, default=0.2, help='Размер валидационной выборки')
    parser.add_argument('--early_stopping', type=int, default=5, help='Количество эпох раннего останова')
    parser.add_argument('--augmentation_prob', type=float, default=0.5, help='Вероятность применения аугментаций')
    parser.add_argument('--normalize', action='store_true', help='Apply normalization')
    parser.add_argument('--use_dataset_stats', action='store_true', help='Use dataset statistics for normalization')
    parser.add_argument('--weather_prob', type=float, default=0.3, help='Probability of applying weather effects')
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Создаем директории если не существуют
    os.makedirs(args.history_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Настройка логгера
    logger = setup_logger(os.path.join(args.log_dir, 'training.log'))
    logger.info(f"Старт обучения с параметрами: {vars(args)}")

    # Установка seed для воспроизводимости
    seed_everything(args.seed)
    logger.info(f"Установлен seed: {args.seed}")

    # Проверка наличия CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")

    # Загрузка и подготовка данных
    logger.info("Загрузка и подготовка данных...")
    annotations_path = os.path.join(args.data_path, args.annotations_file)

    train_loader, val_loader = get_data_loaders(
        data_path=args.data_path,
        annotations_path=annotations_path,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        augmentation_prob=args.augmentation_prob,
        weather_prob=args.weather_prob,      # Новый параметр
        normalize=args.normalize,            # Новый параметр
        use_custom_stats=args.use_dataset_stats  # Новый параметр
    )
    logger.info(f"Загружено данных: {len(train_loader.dataset)} для обучения, {len(val_loader.dataset)} для валидации")

    # Получение модели
    logger.info("Инициализация модели...")
    model = get_faster_rcnn_model(num_classes=3)  # 3 класса: фон (0), незрелое яблоко (1), зрелое яблоко (2)
    model = model.to(device)

    # Настройка оптимизатора
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Настройка планировщика скорости обучения
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Для метрики mAP, которую нужно максимизировать
        factor=0.1,
        patience=3,
    )

    # Инициализация тренера и обучение модели
    logger.info("Начало процесса обучения...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        output_model_path=args.output_model,
        history_path=args.history_path,
        early_stopping=args.early_stopping,
        logger=logger
    )

    # Запуск обучения
    best_model, history = trainer.fit(epochs=args.num_epochs)

    logger.info(f"Обучение завершено. Лучшая модель сохранена в {args.output_model}")
    logger.info(f"История обучения сохранена в {args.history_path}")


if __name__ == "__main__":
    main()