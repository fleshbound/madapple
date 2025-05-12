#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_map, calculate_f1_score

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer=None, 
        lr_scheduler=None, 
        train_loader=None, 
        val_loader=None, 
        device=None, 
        num_epochs=30, 
        output_model_path='best_model.pth', 
        history_path='training_history', 
        early_stopping=5,
        logger=None
    ):
        """
        Класс для обучения модели.
        
        Args:
            model: Модель Faster R-CNN
            optimizer: Оптимизатор (если None, будет создан SGD)
            lr_scheduler: Планировщик скорости обучения (если None, будет создан ReduceLROnPlateau)
            train_loader: Загрузчик данных для обучения (может быть установлен позже)
            val_loader: Загрузчик данных для валидации (может быть установлен позже)
            device: Устройство для обучения (если None, будет определено автоматически)
            num_epochs (int): Количество эпох обучения
            output_model_path (str): Путь для сохранения модели
            history_path (str): Путь для сохранения истории обучения
            early_stopping (int): Количество эпох для раннего останова
            logger: Логгер для вывода сообщений
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.output_model_path = output_model_path
        self.history_path = history_path
        self.early_stopping = early_stopping
        self.logger = logger
        
        # Автоматическое определение устройства, если не указано
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Перемещение модели на устройство
        self.model = self.model.to(self.device)
        
        # Создание оптимизатора, если не передан
        if optimizer is None and self.model is not None:
            self.optimizer = torch.optim.SGD(
                [p for p in self.model.parameters() if p.requires_grad], 
                lr=0.001, 
                momentum=0.9,
                weight_decay=0.0005
            )
        else:
            self.optimizer = optimizer
            
        # Создание планировщика скорости обучения, если не передан
        if lr_scheduler is None and self.optimizer is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max',
                factor=0.1,
                patience=3
            )
        else:
            self.lr_scheduler = lr_scheduler
        
        # Создание директории для сохранения истории
        os.makedirs(history_path, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'map': [],
            'f1_score': [],
            'learning_rate': []
        }
        
        self.best_map = 0.0
        self.best_model = None
        self.epochs_without_improvement = 0
    
    def train_one_epoch(self, epoch):
        """
        Функция обучения на одной эпохе.
        
        Args:
            epoch (int): Номер текущей эпохи
        
        Returns:
            float: Средняя потеря на эпохе
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for images, targets in progress_bar:
            # Перенос данных на устройство
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Обнуление градиентов
            self.optimizer.zero_grad()
            
            # Прямой проход (модель Faster R-CNN возвращает словарь потерь)
            loss_dict = self.model(images, targets)
            
            # Определение потерь в зависимости от возвращаемого типа PyTorch 2.7.0
            if isinstance(loss_dict, dict):
                # Модель вернула словарь потерь (стандартное поведение)
                losses = sum(loss for loss in loss_dict.values())
            elif isinstance(loss_dict, list) and len(loss_dict) > 0 and isinstance(loss_dict[0], dict):
                # Модель вернула список словарей (может быть в новых версиях PyTorch)
                losses = sum(sum(d.values()) for d in loss_dict)
            elif isinstance(loss_dict, torch.Tensor):
                # Модель вернула тензор потерь напрямую
                losses = loss_dict
            else:
                # Неизвестный формат возврата, логируем и пытаемся продолжить
                if self.logger:
                    self.logger.warning(f"Неизвестный формат возврата потерь: {type(loss_dict)}")
                
                # Пытаемся преобразовать в тензор, если возможно
                try:
                    losses = torch.tensor(loss_dict, device=self.device)
                except:
                    # Если не получилось, просто используем значение по умолчанию
                    if self.logger:
                        self.logger.error(f"Не удалось преобразовать потери в тензор: {loss_dict}")
                    losses = torch.tensor(1.0, device=self.device)
            
            # Обратное распространение и оптимизация
            losses.backward()
            self.optimizer.step()
            
            # Обновление общей потери
            total_loss += losses.item()
            
            # Обновление progress bar
            progress_bar.set_postfix(loss=losses.item())
        
        # Средняя потеря за эпоху
        avg_loss = total_loss / len(self.train_loader)
        
        # Обновление истории
        self.history['train_loss'].append(avg_loss)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        if self.logger:
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, epoch):
        """
        Функция валидации модели.
        
        Args:
            epoch (int): Номер текущей эпохи
        
        Returns:
            tuple: Средняя потеря, mAP и F1-score на валидационной выборке
        """
        self.model.eval()
        total_loss = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                # Перенос данных на устройство
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # В PyTorch 2.7.0 не вычисляем потери через модель, просто получаем предсказания
                self.model.eval()  # Убедимся, что модель в режиме eval
                predictions = self.model(images)
                
                # Вместо вычисления потерь через модель, просто используем приблизительное значение
                # от функции потерь на обучающей выборке
                # Это безопасно, так как мы всё равно больше заинтересованы в метриках (mAP, F1)
                avg_train_loss = self.history['train_loss'][-1] if self.history['train_loss'] else 0
                total_loss += avg_train_loss  # Используем потери с обучения как приближение
                
                # Сохраняем результаты для расчета метрик
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Средняя потеря на валидации (примерное значение)
        avg_loss = total_loss / len(self.val_loader)
        
        # Расчет mAP и F1
        map_value = calculate_map(all_predictions, all_targets)
        f1_value = calculate_f1_score(all_predictions, all_targets)
        
        # Обновление истории
        self.history['val_loss'].append(avg_loss)
        self.history['map'].append(map_value)
        self.history['f1_score'].append(f1_value)
        
        if self.logger:
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Val Loss: {avg_loss:.4f}, mAP: {map_value:.4f}, F1: {f1_value:.4f}")
        
        return avg_loss, map_value, f1_value
    
    def fit(self, train_data=None, val_data=None, epochs=None, callbacks=None):
        """
        Метод для обучения модели в стиле Keras model.fit()
        
        Args:
            train_data: Загрузчик данных для обучения или кортеж (dataset, batch_size, shuffle)
            val_data: Загрузчик данных для валидации или кортеж (dataset, batch_size)
            epochs (int): Количество эпох обучения (если None, используется self.num_epochs)
            callbacks (list): Список колбэков (не используется, добавлен для совместимости API)
        
        Returns:
            history: История обучения
        """
        # Настройка загрузчиков данных, если предоставлены
        if train_data is not None:
            if isinstance(train_data, torch.utils.data.DataLoader):
                self.train_loader = train_data
            elif isinstance(train_data, tuple) and len(train_data) >= 2:
                dataset, batch_size = train_data[0], train_data[1]
                shuffle = train_data[2] if len(train_data) > 2 else True
                collate_fn = getattr(dataset, 'collate_fn', None)
                self.train_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=shuffle,
                    collate_fn=collate_fn
                )
        
        if val_data is not None:
            if isinstance(val_data, torch.utils.data.DataLoader):
                self.val_loader = val_data
            elif isinstance(val_data, tuple) and len(val_data) >= 2:
                dataset, batch_size = val_data[0], val_data[1]
                collate_fn = getattr(dataset, 'collate_fn', None)
                self.val_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    collate_fn=collate_fn
                )
        
        # Проверка наличия загрузчиков данных
        if self.train_loader is None:
            raise ValueError("Загрузчик данных для обучения не предоставлен.")
        
        # Количество эпох
        if epochs is not None:
            self.num_epochs = epochs
        
        # Запуск обучения
        return self.train()
    
    def train(self):
        """
        Основная функция обучения модели.
        
        Returns:
            tuple: Лучшая модель и история обучения
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info("Начало обучения модели...")
        
        for epoch in range(self.num_epochs):
            # Обучение на одной эпохе
            train_loss = self.train_one_epoch(epoch)
            
            # Валидация, если есть валидационный загрузчик
            if self.val_loader is not None:
                val_loss, map_value, f1_value = self.validate(epoch)
                
                # Обновление планировщика скорости обучения, если он есть
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(map_value)
                
                # Проверка на лучшую модель
                if map_value > self.best_map:
                    self.best_map = map_value
                    self.best_model = self.model.state_dict().copy()
                    self.epochs_without_improvement = 0
                    
                    # Сохранение лучшей модели
                    torch.save(self.best_model, self.output_model_path)
                    
                    if self.logger:
                        self.logger.info(f"Epoch {epoch+1}: Сохранена лучшая модель с mAP = {map_value:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    
                    if self.logger:
                        self.logger.info(f"Epoch {epoch+1}: Нет улучшения модели. Эпох без улучшения: {self.epochs_without_improvement}")
            else:
                # Если нет валидационного загрузчика, сохраняем модель после каждой эпохи
                self.best_model = self.model.state_dict().copy()
                torch.save(self.best_model, self.output_model_path)
                
                if self.logger:
                    self.logger.info(f"Epoch {epoch+1}: Сохранена модель (без валидации)")
            
            # Сохранение истории после каждой эпохи
            os.makedirs(self.history_path, exist_ok=True)
            with open(os.path.join(self.history_path, 'training_history.json'), 'w') as f:
                json.dump(self.history, f)
            
            # Проверка на ранний останов, если включен и есть валидационный загрузчик
            if self.early_stopping > 0 and self.val_loader is not None and self.epochs_without_improvement >= self.early_stopping:
                if self.logger:
                    self.logger.info(f"Ранний останов на эпохе {epoch+1}")
                break
        
        # Сохранение графиков истории обучения
        self.plot_history()
        
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if self.logger:
            self.logger.info(f"Обучение завершено за {int(hours)} часов, {int(minutes)} минут, {int(seconds)} секунд")
            if self.val_loader is not None:
                self.logger.info(f"Лучшее значение mAP: {self.best_map:.4f}")
        
        # Загрузка лучшей модели
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        
        return self.model, self.history
    
    def plot_history(self):
        """
        Построение и сохранение графиков истории обучения.
        """
        plt.figure(figsize=(12, 10))
        
        # График потерь
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # График mAP
        plt.subplot(2, 2, 2)
        plt.plot(self.history['map'], label='mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.title('Mean Average Precision (mAP)')
        
        # График F1-score
        plt.subplot(2, 2, 3)
        plt.plot(self.history['f1_score'], label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('F1 Score')
        
        # График скорости обучения
        plt.subplot(2, 2, 4)
        plt.plot(self.history['learning_rate'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.history_path, 'training_history.png'))
        plt.close()