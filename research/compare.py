#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Программа для сравнения обучения различных моделей детекции объектов.
Поддерживает: Faster R-CNN, FCOS, RetinaNet
Замеряет время обучения и сохраняет графики метрик.
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import argparse
import torch
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Импорты из существующего проекта
from train.trainer import Trainer
from train.dataset import get_data_loaders
from train.logger import setup_logger
from utils.utils import seed_everything, Timer

# Новые импорты для дополнительных моделей
import torchvision
from torchvision.models.detection import (
    fcos_resnet50_fpn, 
    retinanet_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights
)
from torchvision.models.detection.fcos import FCOSHead
from torchvision.models.detection.retinanet import RetinaNetHead


def get_fcos_model(num_classes):
    """
    Создает модель FCOS на основе предобученного ResNet-50 backbone.
    
    Args:
        num_classes (int): Количество классов (включая фон)
    
    Returns:
        model: Модель FCOS
    """
    # Загрузка предтренированной модели FCOS
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights)
    
    # Получаем количество входных признаков
    in_channels = model.backbone.out_channels
    
    # Заменяем голову классификации
    model.head = FCOSHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_anchors=1  # FCOS использует anchor-free подход
    )
    
    return model


def get_retinanet_model(num_classes):
    """
    Создает модель RetinaNet на основе предобученного ResNet-50 backbone.
    
    Args:
        num_classes (int): Количество классов (включая фон)
    
    Returns:
        model: Модель RetinaNet
    """
    # Загрузка предтренированной модели RetinaNet
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights)
    
    # Получаем количество входных признаков
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Заменяем голову классификации
    model.head = RetinaNetHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model


def get_faster_rcnn_model(num_classes):
    """
    Импортируем функцию из существующего проекта.
    """
    from model.model import get_faster_rcnn_model as get_frcnn
    return get_frcnn(num_classes)


class ModelComparisonTrainer:
    """
    Класс для сравнительного обучения различных моделей детекции.
    """
    
    def __init__(self, args, logger):
        """
        Инициализация тренера для сравнения моделей.
        
        Args:
            args: Аргументы командной строки
            logger: Логгер для вывода сообщений
        """
        self.args = args
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Результаты обучения для всех моделей
        self.results = {}
        
        # Создаем выходные директории
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Настройка стилей для графиков
        self.setup_plot_styles()
    
    def setup_plot_styles(self):
        """Настройка стилей для графиков."""
        plt.style.use('default')
        self.colors = ['blue', 'red', 'green', 'orange', 'purple']
        self.linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        self.markers = ['o', 's', '^', 'v', 'D']
    
    def get_model(self, model_name, num_classes):
        """
        Получение модели по имени.
        
        Args:
            model_name (str): Название модели
            num_classes (int): Количество классов
            
        Returns:
            torch.nn.Module: Модель
        """
        model_functions = {
            'faster_rcnn': get_faster_rcnn_model,
            'fcos': get_fcos_model,
            'retinanet': get_retinanet_model
        }
        
        if model_name not in model_functions:
            raise ValueError(f"Неподдерживаемая модель: {model_name}")
        
        return model_functions[model_name](num_classes)
    
    def train_model(self, model_name):
        """
        Обучение одной модели.
        
        Args:
            model_name (str): Название модели
            
        Returns:
            dict: Результаты обучения
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ: {model_name.upper()}")
        self.logger.info(f"=" * 80)
        
        # Создание загрузчиков данных
        self.logger.info("Создание загрузчиков данных...")
        annotations_path = os.path.join(self.args.data_path, self.args.annotations_file)
        
        train_loader, val_loader = get_data_loaders(
            data_path=self.args.data_path,
            annotations_path=annotations_path,
            batch_size=self.args.batch_size,
            val_size=self.args.val_size,
            num_workers=self.args.num_workers,
            augmentation_prob=self.args.augmentation_prob,
            weather_prob=self.args.weather_prob,
            normalize=self.args.normalize,
            use_custom_stats=self.args.use_dataset_stats
        )
        
        # Создание модели
        self.logger.info(f"Создание модели {model_name}...")
        model = self.get_model(model_name, num_classes=3)
        model = model.to(self.device)
        
        # Настройка оптимизатора
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        # Настройка планировщика скорости обучения
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=3,
        )
        
        # Пути для сохранения результатов модели
        model_output_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        model_path = os.path.join(model_output_dir, f"{model_name}_detector.pth")
        history_path = os.path.join(model_output_dir, "training_history")
        
        # Создание тренера
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            num_epochs=self.args.num_epochs,
            output_model_path=model_path,
            history_path=history_path,
            early_stopping=self.args.early_stopping,
            logger=self.logger
        )
        
        # Замер времени обучения
        start_time = time.time()
        
        with Timer(f"Обучение модели {model_name}", verbose=True):
            # Запуск обучения
            best_model, history = trainer.train()
        
        training_time = time.time() - start_time
        
        # Сохранение результатов
        result = {
            'model_name': model_name,
            'training_time': training_time,
            'history': history,
            'model_path': model_path,
            'best_map': trainer.best_map,
            'final_metrics': {
                'train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
                'val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
                'map': history['map'][-1] if history['map'] else 0,
                'f1_score': history['f1_score'][-1] if history['f1_score'] else 0
            }
        }
        
        self.logger.info(f"Обучение модели {model_name} завершено за {training_time:.2f} секунд")
        self.logger.info(f"Лучший mAP: {trainer.best_map:.4f}")
        
        return result
    
    def train_all_models(self, models_to_train=None):
        """
        Обучение всех указанных моделей.
        
        Args:
            models_to_train (list): Список названий моделей для обучения
        """
        if models_to_train is None:
            models_to_train = self.args.models
        
        self.logger.info(f"Запуск обучения моделей: {', '.join(models_to_train)}")
        
        # Обучение каждой модели
        for model_name in models_to_train:
            try:
                result = self.train_model(model_name)
                self.results[model_name] = result
                
                # Сохранение промежуточных результатов
                self.save_intermediate_results()
                
            except Exception as e:
                self.logger.error(f"Ошибка при обучении модели {model_name}: {e}")
                continue
        
        # Создание сравнительных графиков
        self.create_comparison_plots()
        
        # Сохранение итогового отчета
        self.save_final_report()
    
    def save_intermediate_results(self):
        """Сохранение промежуточных результатов."""
        results_path = os.path.join(self.output_dir, 'intermediate_results.json')
        
        # Подготовка данных для JSON (конвертация numpy типов)
        json_results = {}
        for model_name, result in self.results.items():
            json_result = result.copy()
            # Конвертируем numpy типы для JSON сериализации
            json_result = self._convert_for_json(json_result)
            json_results[model_name] = json_result
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    def _convert_for_json(self, obj):
        """Конвертация numpy типов для JSON сериализации."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj
    
    def create_comparison_plots(self):
        """Создание сравнительных графиков для всех моделей."""
        if not self.results:
            self.logger.warning("Нет результатов для создания графиков")
            return
        
        # График 1: Сравнение времени обучения
        self.plot_training_time_comparison()
        
        # График 2: Сравнение финальных метрик
        self.plot_final_metrics_comparison()
        
        # График 3: Динамика обучения (loss, mAP, F1)
        self.plot_training_dynamics()
        
        # График 4: Детальное сравнение метрик
        self.plot_detailed_metrics()
    
    def plot_training_time_comparison(self):
        """График сравнения времени обучения."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(self.results.keys())
        times = [self.results[model]['training_time'] for model in models]
        
        bars = ax.bar(models, times, color=self.colors[:len(models)], alpha=0.7, edgecolor='black')
        
        # Добавляем значения на столбцы
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Время обучения (секунды)', fontweight='bold')
        ax.set_title('Сравнение времени обучения моделей', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_final_metrics_comparison(self):
        """График сравнения финальных метрик."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        
        # mAP
        map_values = [self.results[model]['final_metrics']['map'] for model in models]
        bars1 = ax1.bar(models, map_values, color=self.colors[0], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('mAP', fontweight='bold')
        ax1.set_title('Mean Average Precision (mAP)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars1, map_values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-Score
        f1_values = [self.results[model]['final_metrics']['f1_score'] for model in models]
        bars2 = ax2.bar(models, f1_values, color=self.colors[1], alpha=0.7, edgecolor='black')
        ax2.set_ylabel('F1-Score', fontweight='bold')
        ax2.set_title('F1-Score', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars2, f1_values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Train Loss
        train_loss_values = [self.results[model]['final_metrics']['train_loss'] for model in models]
        bars3 = ax3.bar(models, train_loss_values, color=self.colors[2], alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Train Loss', fontweight='bold')
        ax3.set_title('Финальная потеря на обучении', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars3, train_loss_values):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + val*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Validation Loss
        val_loss_values = [self.results[model]['final_metrics']['val_loss'] for model in models]
        bars4 = ax4.bar(models, val_loss_values, color=self.colors[3], alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Validation Loss', fontweight='bold')
        ax4.set_title('Финальная потеря на валидации', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars4, val_loss_values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + val*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'final_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_training_dynamics(self):
        """График динамики обучения для всех моделей."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, (model_name, result) in enumerate(self.results.items()):
            history = result['history']
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Train Loss
            ax1.plot(epochs, history['train_loss'], color=color, linestyle=linestyle, 
                    marker=marker, markersize=4, label=f'{model_name}', linewidth=2)
            
            # Validation Loss
            if history['val_loss']:
                val_epochs = range(1, len(history['val_loss']) + 1)
                ax2.plot(val_epochs, history['val_loss'], color=color, linestyle=linestyle,
                        marker=marker, markersize=4, label=f'{model_name}', linewidth=2)
            
            # mAP
            if history['map']:
                map_epochs = range(1, len(history['map']) + 1)
                ax3.plot(map_epochs, history['map'], color=color, linestyle=linestyle,
                        marker=marker, markersize=4, label=f'{model_name}', linewidth=2)
            
            # F1-Score
            if history['f1_score']:
                f1_epochs = range(1, len(history['f1_score']) + 1)
                ax4.plot(f1_epochs, history['f1_score'], color=color, linestyle=linestyle,
                        marker=marker, markersize=4, label=f'{model_name}', linewidth=2)
        
        # Настройка осей
        ax1.set_xlabel('Эпоха', fontweight='bold')
        ax1.set_ylabel('Train Loss', fontweight='bold')
        ax1.set_title('Динамика потерь на обучении', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Эпоха', fontweight='bold')
        ax2.set_ylabel('Validation Loss', fontweight='bold')
        ax2.set_title('Динамика потерь на валидации', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Эпоха', fontweight='bold')
        ax3.set_ylabel('mAP', fontweight='bold')
        ax3.set_title('Динамика Mean Average Precision', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('Эпоха', fontweight='bold')
        ax4.set_ylabel('F1-Score', fontweight='bold')
        ax4.set_title('Динамика F1-Score', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_detailed_metrics(self):
        """Детальный график сравнения метрик."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.2
        
        # Получаем метрики
        map_values = [self.results[model]['final_metrics']['map'] for model in models]
        f1_values = [self.results[model]['final_metrics']['f1_score'] for model in models]
        best_maps = [self.results[model]['best_map'] for model in models]
        
        # Нормализуем время обучения для отображения на том же графике
        times = [self.results[model]['training_time'] for model in models]
        max_time = max(times)
        normalized_times = [t / max_time for t in times]  # Нормализация к [0, 1]
        
        # Построение столбцов
        bars1 = ax.bar(x - width*1.5, map_values, width, label='Final mAP', 
                      color=self.colors[0], alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x - width*0.5, f1_values, width, label='Final F1-Score',
                      color=self.colors[1], alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width*0.5, best_maps, width, label='Best mAP',
                      color=self.colors[2], alpha=0.8, edgecolor='black')
        bars4 = ax.bar(x + width*1.5, normalized_times, width, label='Время (норм.)',
                      color=self.colors[3], alpha=0.8, edgecolor='black')
        
        # Добавляем значения на столбцы
        for bars, values in [(bars1, map_values), (bars2, f1_values), (bars3, best_maps)]:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Для времени показываем реальные значения
        for bar, time_val in zip(bars4, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{time_val:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Модели', fontweight='bold')
        ax.set_ylabel('Значение метрики', fontweight='bold')
        ax.set_title('Детальное сравнение моделей по всем метрикам', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем вторую ось Y для времени
        ax2 = ax.twinx()
        ax2.set_ylabel('Время обучения (секунды)', fontweight='bold', color=self.colors[3])
        ax2.tick_params(axis='y', labelcolor=self.colors[3])
        ax2.set_ylim(0, max_time * 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detailed_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def save_final_report(self):
        """Сохранение итогового отчета."""
        report_path = os.path.join(self.output_dir, 'final_report.json')
        
        # Подготовка итогового отчета
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_parameters': {
                'num_epochs': self.args.num_epochs,
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.lr,
                'device': str(self.device)
            },
            'models_trained': list(self.results.keys()),
            'results_summary': {},
            'detailed_results': self.results
        }
        
        # Создание сводки результатов
        for model_name, result in self.results.items():
            report['results_summary'][model_name] = {
                'training_time_seconds': result['training_time'],
                'best_map': result['best_map'],
                'final_map': result['final_metrics']['map'],
                'final_f1': result['final_metrics']['f1_score'],
                'final_train_loss': result['final_metrics']['train_loss'],
                'final_val_loss': result['final_metrics']['val_loss']
            }
        
        # Конвертируем для JSON
        json_report = self._convert_for_json(report)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Итоговый отчет сохранен в {report_path}")
        
        # Выводим краткую сводку в лог
        self.logger.info("\n" + "="*80)
        self.logger.info("ИТОГОВАЯ СВОДКА СРАВНЕНИЯ МОДЕЛЕЙ")
        self.logger.info("="*80)
        
        for model_name, summary in report['results_summary'].items():
            self.logger.info(f"\nМодель: {model_name}")
            self.logger.info(f"  Время обучения: {summary['training_time_seconds']:.2f} сек")
            self.logger.info(f"  Лучший mAP: {summary['best_map']:.4f}")
            self.logger.info(f"  Финальный mAP: {summary['final_map']:.4f}")
            self.logger.info(f"  Финальный F1: {summary['final_f1']:.4f}")
        
        # Определяем лучшую модель
        best_model_by_map = max(report['results_summary'].items(), 
                               key=lambda x: x[1]['best_map'])
        fastest_model = min(report['results_summary'].items(),
                           key=lambda x: x[1]['training_time_seconds'])
        
        self.logger.info(f"\nЛучшая модель по mAP: {best_model_by_map[0]} ({best_model_by_map[1]['best_map']:.4f})")
        self.logger.info(f"Самая быстрая модель: {fastest_model[0]} ({fastest_model[1]['training_time_seconds']:.2f} сек)")
        self.logger.info("="*80)


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Сравнительное обучение моделей детекции объектов',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Параметры данных
    parser.add_argument('--data_path', type=str, default='../data/train',
                       help='Путь к директории с изображениями')
    parser.add_argument('--annotations_file', type=str, default='_annotations.coco.json',
                       help='Имя файла аннотаций COCO')
    
    # Параметры моделей для обучения
    parser.add_argument('--models', nargs='+', 
                       choices=['faster_rcnn', 'fcos', 'retinanet'],
                       default=['fcos', 'retinanet'],
                       help='Модели для обучения')
    
    # Параметры обучения
    parser.add_argument('--num_epochs', type=int, default=40,
                       help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Начальная скорость обучения')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Момент для оптимизатора')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Коэффициент регуляризации L2')
    parser.add_argument('--early_stopping', type=int, default=7,
                       help='Количество эпох раннего останова')
    
    # Параметры данных
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Размер валидационной выборки')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Количество рабочих для загрузки данных')
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                       help='Вероятность применения аугментаций')
    parser.add_argument('--weather_prob', type=float, default=0.3,
                       help='Вероятность применения погодных эффектов')
    parser.add_argument('--normalize', action='store_true',
                       help='Применять нормализацию')
    parser.add_argument('--use_dataset_stats', action='store_true',
                       help='Использовать статистику датасета для нормализации')
    
    # Параметры вывода
    parser.add_argument('--output_dir', type=str, default='model_comparison_results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Директория для логов')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed для воспроизводимости')
    
    return parser.parse_args()


def main():
    """Основная функция."""
    args = parse_arguments()
    
    # Создаем директории если не существуют
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Настройка логгера
    log_file = os.path.join(args.log_dir, f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("НАЧАЛО СРАВНИТЕЛЬНОГО ОБУЧЕНИЯ МОДЕЛЕЙ")
    logger.info("="*80)
    logger.info(f"Параметры: {vars(args)}")
    
    # Установка seed для воспроизводимости
    seed_everything(args.seed)
    logger.info(f"Установлен seed: {args.seed}")
    
    # Проверка наличия CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")
    
    # Проверка существования файлов данных
    if not os.path.exists(args.data_path):
        logger.error(f"Директория с данными не найдена: {args.data_path}")
        return
    
    annotations_path = os.path.join(args.data_path, args.annotations_file)
    if not os.path.exists(annotations_path):
        logger.error(f"Файл аннотаций не найден: {annotations_path}")
        return
    
    logger.info(f"Будут обучены модели: {', '.join(args.models)}")
    
    try:
        # Создание тренера для сравнения моделей
        trainer = ModelComparisonTrainer(args, logger)
        
        # Запуск обучения всех моделей
        trainer.train_all_models()
        
        logger.info("="*80)
        logger.info("СРАВНИТЕЛЬНОЕ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        logger.info("="*80)
        logger.info(f"Результаты сохранены в: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении сравнительного обучения: {e}")
        raise


if __name__ == "__main__":
    main()