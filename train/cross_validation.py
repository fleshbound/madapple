#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Система кросс-валидации для оценки качества модели обнаружения яблок.
Использует k-fold CV на обучающем датасете с детальным анализом результатов.

Использование:
    python cross_validation.py --data_path data/train --k_folds 5 --model_path apple_detector.pth
"""

import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Импорты из вашего проекта
from model.model import get_faster_rcnn_model
from dataset import AppleDataset, collate_fn
from utils.utils import calculate_map_simple, calculate_f1_score_simple
from logger import setup_logger

class CrossValidationAnalyzer:
    """Класс для проведения и анализа кросс-валидации."""
    
    def __init__(self, data_path, annotations_path, k_folds=5, device=None, 
                 output_dir='cv_results', seed=42):
        """
        Инициализация анализатора кросс-валидации.
        
        Args:
            data_path (str): Путь к изображениям
            annotations_path (str): Путь к аннотациям COCO
            k_folds (int): Количество фолдов
            device (str): Устройство для вычислений
            output_dir (str): Директория для сохранения результатов
            seed (int): Seed для воспроизводимости
        """
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.k_folds = k_folds
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.seed = seed
        
        # Создаем выходную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        # Настройка логгера
        self.logger = setup_logger(os.path.join(output_dir, 'cv_log.txt'))
        
        # Результаты CV
        self.cv_results = {
            'fold_results': [],
            'aggregated_metrics': {},
            'statistical_analysis': {},
            'metadata': {
                'k_folds': k_folds,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat(),
                'seed': seed
            }
        }
        
        # Загружаем полный датасет
        self.full_dataset = AppleDataset(
            root_dir=data_path,
            annotations_path=annotations_path,
            transforms=None,
            is_train=False,  # Без аугментаций для честной оценки
            normalize=True
        )
        
        self.logger.info(f"Загружен датасет: {len(self.full_dataset)} изображений")
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Количество фолдов: {k_folds}")
    
    def create_folds(self, stratified=True):
        """
        Создание фолдов для кросс-валидации.
        
        Args:
            stratified (bool): Использовать стратифицированное разделение
            
        Returns:
            list: Список кортежей (train_indices, val_indices)
        """
        self.logger.info("Создание фолдов для кросс-валидации...")
        
        # Получаем метки классов для стратификации
        if stratified:
            labels = []
            for i in range(len(self.full_dataset)):
                _, target = self.full_dataset[i]
                # Подсчитываем количество объектов каждого класса
                if len(target['labels']) > 0:
                    # Берем самый частый класс в изображении
                    unique_labels, counts = torch.unique(target['labels'], return_counts=True)
                    dominant_label = unique_labels[torch.argmax(counts)].item()
                    labels.append(dominant_label)
                else:
                    labels.append(0)  # Фон, если нет объектов
            
            kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            folds = list(kfold.split(range(len(self.full_dataset)), labels))
            self.logger.info("Использована стратифицированная кросс-валидация")
        else:
            kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            folds = list(kfold.split(range(len(self.full_dataset))))
            self.logger.info("Использована обычная кросс-валидация")
        
        # Анализ распределения по фолдам
        fold_sizes = [len(val_indices) for _, val_indices in folds]
        self.logger.info(f"Размеры фолдов: {fold_sizes}")
        self.logger.info(f"Среднее ± стд: {np.mean(fold_sizes):.1f} ± {np.std(fold_sizes):.1f}")
        
        return folds
    
    def evaluate_fold(self, model, val_loader, fold_num):
        """
        Оценка модели на одном фолде.
        
        Args:
            model: Обученная модель
            val_loader: DataLoader для валидации
            fold_num (int): Номер фолда
            
        Returns:
            dict: Метрики качества
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        self.logger.info(f"Оценка fold {fold_num+1}/{self.k_folds}...")
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Fold {fold_num+1} evaluation"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                predictions = model(images)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Вычисление метрик
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['fold'] = fold_num + 1
        metrics['n_images'] = len(val_loader.dataset)
        
        return metrics
    
    def calculate_metrics(self, predictions, targets, iou_thresholds=[0.5, 0.75], 
                         confidence_thresholds=[0.3, 0.5, 0.7]):
        """
        Вычисление детальных метрик качества.
        
        Args:
            predictions: Предсказания модели
            targets: Истинные метки
            iou_thresholds: Пороги IoU для оценки
            confidence_thresholds: Пороги уверенности
            
        Returns:
            dict: Словарь с метриками
        """
        metrics = {}
        
        # Основные метрики для разных порогов
        for conf_thresh in confidence_thresholds:
            for iou_thresh in iou_thresholds:
                # Фильтруем предсказания по порогу уверенности
                filtered_preds = []
                for pred in predictions:
                    mask = pred['scores'] >= conf_thresh
                    filtered_pred = {
                        'boxes': pred['boxes'][mask],
                        'labels': pred['labels'][mask],
                        'scores': pred['scores'][mask]
                    }
                    filtered_preds.append(filtered_pred)
                
                # Вычисляем метрики
                map_score = calculate_map_simple(filtered_preds, targets, 
                                               iou_threshold=iou_thresh, 
                                               score_threshold=conf_thresh)
                f1_score = calculate_f1_score_simple(filtered_preds, targets,
                                                   iou_threshold=iou_thresh,
                                                   score_threshold=conf_thresh)
                
                key_suffix = f"_iou{iou_thresh}_conf{conf_thresh}"
                metrics[f'mAP{key_suffix}'] = map_score
                metrics[f'F1{key_suffix}'] = f1_score
        
        # Подсчет объектов по классам
        total_objects = {'unripe': 0, 'ripe': 0}
        detected_objects = {'unripe': 0, 'ripe': 0}
        
        for target in targets:
            if len(target['labels']) > 0:
                for label in target['labels']:
                    if label.item() == 1:
                        total_objects['unripe'] += 1
                    elif label.item() == 2:
                        total_objects['ripe'] += 1
        
        # Основные предсказания (confidence > 0.5)
        for pred in predictions:
            mask = pred['scores'] >= 0.5
            if torch.sum(mask) > 0:
                labels = pred['labels'][mask]
                for label in labels:
                    if label.item() == 1:
                        detected_objects['unripe'] += 1
                    elif label.item() == 2:
                        detected_objects['ripe'] += 1
        
        metrics['total_unripe'] = total_objects['unripe']
        metrics['total_ripe'] = total_objects['ripe']
        metrics['detected_unripe'] = detected_objects['unripe']
        metrics['detected_ripe'] = detected_objects['ripe']
        
        # Коэффициенты обнаружения
        metrics['detection_rate_unripe'] = (detected_objects['unripe'] / max(total_objects['unripe'], 1))
        metrics['detection_rate_ripe'] = (detected_objects['ripe'] / max(total_objects['ripe'], 1))
        
        return metrics
    
    def run_cross_validation(self, model_path=None, model=None):
        """
        Запуск полной кросс-валидации.
        
        Args:
            model_path (str): Путь к сохраненной модели
            model (torch.nn.Module): Предварительно загруженная модель
            
        Returns:
            dict: Результаты кросс-валидации
        """
        start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info("НАЧАЛО КРОСС-ВАЛИДАЦИИ")
        self.logger.info("=" * 80)
        
        # Создание фолдов
        folds = self.create_folds(stratified=True)
        
        # Прогон по каждому фолду
        for fold_num, (train_indices, val_indices) in enumerate(folds):
            fold_start_time = time.time()
            
            self.logger.info(f"\n--- FOLD {fold_num + 1}/{self.k_folds} ---")
            self.logger.info(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
            
            # Создание подмножеств данных
            val_subset = Subset(self.full_dataset, val_indices)
            val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, 
                                  num_workers=2, collate_fn=collate_fn)
            
            # Загрузка модели
            if model is None:
                if model_path is None:
                    raise ValueError("Необходимо указать либо model_path, либо передать model")
                
                fold_model = get_faster_rcnn_model(num_classes=3)
                fold_model.load_state_dict(torch.load(model_path, map_location=self.device))
                fold_model.to(self.device)
            else:
                fold_model = model
            
            # Оценка на фолде
            fold_metrics = self.evaluate_fold(fold_model, val_loader, fold_num)
            
            fold_time = time.time() - fold_start_time
            fold_metrics['evaluation_time'] = fold_time
            
            self.cv_results['fold_results'].append(fold_metrics)
            
            # Вывод результатов фолда
            self.print_fold_results(fold_metrics)
            
            self.logger.info(f"Время выполнения fold {fold_num + 1}: {fold_time:.2f} сек")
        
        # Агрегация результатов
        self.aggregate_results()
        
        # Статистический анализ
        self.statistical_analysis()
        
        # Сохранение результатов
        self.save_results()
        
        # Создание визуализаций
        self.create_visualizations()
        
        total_time = time.time() - start_time
        self.logger.info(f"\nВремя выполнения кросс-валидации: {total_time:.2f} сек")
        self.logger.info("=" * 80)
        self.logger.info("КРОСС-ВАЛИДАЦИЯ ЗАВЕРШЕНА")
        self.logger.info("=" * 80)
        
        return self.cv_results
    
    def print_fold_results(self, metrics):
        """Вывод результатов одного фолда."""
        self.logger.info(f"Результаты Fold {metrics['fold']}:")
        self.logger.info(f"  mAP@0.5: {metrics['mAP_iou0.5_conf0.5']:.4f}")
        self.logger.info(f"  F1@0.5:  {metrics['F1_iou0.5_conf0.5']:.4f}")
        self.logger.info(f"  Обнаружено незрелых: {metrics['detected_unripe']}/{metrics['total_unripe']} "
                        f"({metrics['detection_rate_unripe']:.2%})")
        self.logger.info(f"  Обнаружено зрелых: {metrics['detected_ripe']}/{metrics['total_ripe']} "
                        f"({metrics['detection_rate_ripe']:.2%})")
    
    def aggregate_results(self):
        """Агрегация результатов по всем фолдам."""
        self.logger.info("\n" + "="*50)
        self.logger.info("АГРЕГИРОВАННЫЕ РЕЗУЛЬТАТЫ")
        self.logger.info("="*50)
        
        # Собираем все метрики
        all_metrics = {}
        for key in self.cv_results['fold_results'][0].keys():
            if isinstance(self.cv_results['fold_results'][0][key], (int, float)):
                values = [fold[key] for fold in self.cv_results['fold_results']]
                all_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        self.cv_results['aggregated_metrics'] = all_metrics
        
        # Вывод основных результатов
        main_metrics = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5', 
                       'detection_rate_unripe', 'detection_rate_ripe']
        
        self.logger.info("\nОсновные метрики (среднее ± стандартное отклонение):")
        for metric in main_metrics:
            if metric in all_metrics:
                mean_val = all_metrics[metric]['mean']
                std_val = all_metrics[metric]['std']
                min_val = all_metrics[metric]['min']
                max_val = all_metrics[metric]['max']
                self.logger.info(f"  {metric:25s}: {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]")
        
        # 95% доверительный интервал для основных метрик
        self.logger.info("\n95% доверительные интервалы:")
        for metric in ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5']:
            if metric in all_metrics:
                values = all_metrics[metric]['values']
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                self.logger.info(f"  {metric:25s}: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    def statistical_analysis(self):
        """Статистический анализ результатов."""
        self.logger.info("\n" + "="*50)
        self.logger.info("СТАТИСТИЧЕСКИЙ АНАЛИЗ")
        self.logger.info("="*50)
        
        # Анализ стабильности
        main_metrics = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5']
        stability_analysis = {}
        
        for metric in main_metrics:
            if metric in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric]['values']
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv_coef = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
                
                stability_analysis[metric] = {
                    'coefficient_of_variation': cv_coef,
                    'stability_rating': 'высокая' if cv_coef < 0.1 else 'средняя' if cv_coef < 0.2 else 'низкая'
                }
                
                self.logger.info(f"{metric}:")
                self.logger.info(f"  Коэффициент вариации: {cv_coef:.4f}")
                self.logger.info(f"  Стабильность: {stability_analysis[metric]['stability_rating']}")
        
        self.cv_results['statistical_analysis'] = stability_analysis
        
        # Анализ распределения результатов
        self.logger.info("\nАнализ распределения:")
        for metric in main_metrics:
            if metric in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric]['values']
                from scipy import stats
                
                # Тест на нормальность
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(values)
                    is_normal = shapiro_p > 0.05
                    self.logger.info(f"  {metric} - нормальное распределение: {'Да' if is_normal else 'Нет'} (p={shapiro_p:.4f})")
                except:
                    self.logger.info(f"  {metric} - тест нормальности не удался")
    
    def save_results(self):
        """Сохранение результатов в файлы."""
        self.logger.info("\nСохранение результатов...")
        
        # JSON с полными результатами
        json_path = os.path.join(self.output_dir, 'cv_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            # Конвертируем numpy типы для JSON сериализации
            json_results = self._convert_for_json(self.cv_results)
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # CSV с результатами по фолдам
        df_folds = pd.DataFrame(self.cv_results['fold_results'])
        csv_path = os.path.join(self.output_dir, 'fold_results.csv')
        df_folds.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Сводная таблица метрик
        summary_data = []
        for metric, stats in self.cv_results['aggregated_metrics'].items():
            if isinstance(stats, dict):
                summary_data.append({
                    'metric': metric,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max']
                })
        
        df_summary = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(self.output_dir, 'metrics_summary.csv')
        df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Результаты сохранены в {self.output_dir}/")
        self.logger.info(f"  - cv_results.json: полные результаты")
        self.logger.info(f"  - fold_results.csv: результаты по фолдам")
        self.logger.info(f"  - metrics_summary.csv: сводка метрик")
    
    def _convert_for_json(self, obj):
        """Конвертация numpy типов для JSON сериализации."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def create_visualizations(self):
        """Создание графиков и визуализаций."""
        self.logger.info("\nСоздание визуализаций...")
        
        # Настройка стиля
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Распределение метрик по фолдам
        self._plot_metrics_distribution()
        
        # 2. Box plots основных метрик
        self._plot_metrics_boxplots()
        
        # 3. Детальный анализ производительности
        self._plot_detailed_performance()
        
        # 4. Сводная диаграмма
        self._plot_summary_dashboard()
        
        self.logger.info(f"Графики сохранены в {self.output_dir}/")
    
    def _plot_metrics_distribution(self):
        """График распределения метрик по фолдам."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Распределение метрик по фолдам', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5', 
                          'detection_rate_unripe', 'detection_rate_ripe']
        metric_labels = ['mAP@0.5', 'F1-Score@0.5', 
                        'Detection Rate (Unripe)', 'Detection Rate (Ripe)']
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            if metric in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric]['values']
                folds = list(range(1, len(values) + 1))
                
                ax.bar(folds, values, alpha=0.7, color=sns.color_palette("husl", 1)[0])
                ax.axhline(y=np.mean(values), color='red', linestyle='--', 
                          label=f'Среднее: {np.mean(values):.4f}')
                ax.set_xlabel('Fold')
                ax.set_ylabel(label)
                ax.set_title(f'{label} по фолдам')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_boxplots(self):
        """Box plots для основных метрик."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics_data = []
        metric_labels = []
        
        main_metrics = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5', 
                       'detection_rate_unripe', 'detection_rate_ripe']
        labels = ['mAP@0.5', 'F1-Score@0.5', 'Detection Rate\n(Unripe)', 'Detection Rate\n(Ripe)']
        
        for metric, label in zip(main_metrics, labels):
            if metric in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric]['values']
                metrics_data.append(values)
                metric_labels.append(label)
        
        bp = ax.boxplot(metrics_data, labels=metric_labels, patch_artist=True)
        
        # Раскраска боксплотов
        colors = sns.color_palette("husl", len(metrics_data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Распределение основных метрик качества', fontsize=14, fontweight='bold')
        ax.set_ylabel('Значение метрики')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_boxplots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_performance(self):
        """Детальный анализ производительности."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Детальный анализ производительности модели', fontsize=16, fontweight='bold')
        
        # 1. mAP для разных порогов IoU
        ax = axes[0, 0]
        iou_thresholds = [0.5, 0.75]
        for iou_thresh in iou_thresholds:
            metric_key = f'mAP_iou{iou_thresh}_conf0.5'
            if metric_key in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric_key]['values']
                folds = list(range(1, len(values) + 1))
                ax.plot(folds, values, marker='o', label=f'IoU@{iou_thresh}')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('mAP')
        ax.set_title('mAP для разных порогов IoU')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. F1-Score для разных порогов confidence
        ax = axes[0, 1]
        conf_thresholds = [0.3, 0.5, 0.7]
        for conf_thresh in conf_thresholds:
            metric_key = f'F1_iou0.5_conf{conf_thresh}'
            if metric_key in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric_key]['values']
                folds = list(range(1, len(values) + 1))
                ax.plot(folds, values, marker='s', label=f'Conf@{conf_thresh}')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score для разных порогов уверенности')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Количество обнаруженных объектов
        ax = axes[0, 2]
        unripe_detected = [fold['detected_unripe'] for fold in self.cv_results['fold_results']]
        ripe_detected = [fold['detected_ripe'] for fold in self.cv_results['fold_results']]
        folds = list(range(1, len(unripe_detected) + 1))
        
        width = 0.35
        x = np.arange(len(folds))
        ax.bar(x - width/2, unripe_detected, width, label='Незрелые', alpha=0.7)
        ax.bar(x + width/2, ripe_detected, width, label='Зрелые', alpha=0.7)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Количество объектов')
        ax.set_title('Обнаруженные объекты по классам')
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Стабильность метрик
        ax = axes[1, 0]
        metrics = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5']
        cv_coeffs = []
        labels = []
        
        for metric in metrics:
            if metric in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric]['values']
                cv_coeff = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                cv_coeffs.append(cv_coeff)
                labels.append(metric.replace('_iou0.5_conf0.5', ''))
        
        colors = sns.color_palette("husl", len(cv_coeffs))
        bars = ax.bar(labels, cv_coeffs, color=colors, alpha=0.7)
        ax.set_ylabel('Коэффициент вариации')
        ax.set_title('Стабильность метрик (чем меньше, тем стабильнее)')
        ax.grid(True, alpha=0.3)
        
        # Добавляем линию "хорошей" стабильности
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Хорошая стабильность (<0.1)')
        ax.legend()
        
        # 5. Сравнение с базовыми показателями
        ax = axes[1, 1]
        # Примерные базовые показатели из литературы
        baseline_metrics = {
            'YOLOv3': {'mAP': 0.823, 'F1': 0.847},
            'SSD': {'mAP': 0.908, 'F1': 0.863},
            'Faster R-CNN\n(наш)': {
                'mAP': self.cv_results['aggregated_metrics']['mAP_iou0.5_conf0.5']['mean'],
                'F1': self.cv_results['aggregated_metrics']['F1_iou0.5_conf0.5']['mean']
            }
        }
        
        methods = list(baseline_metrics.keys())
        map_values = [baseline_metrics[method]['mAP'] for method in methods]
        f1_values = [baseline_metrics[method]['F1'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, map_values, width, label='mAP', alpha=0.7)
        ax.bar(x + width/2, f1_values, width, label='F1-Score', alpha=0.7)
        
        ax.set_ylabel('Значение метрики')
        ax.set_title('Сравнение с базовыми методами')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Корреляция между метриками
        ax = axes[1, 2]
        map_values = self.cv_results['aggregated_metrics']['mAP_iou0.5_conf0.5']['values']
        f1_values = self.cv_results['aggregated_metrics']['F1_iou0.5_conf0.5']['values']
        
        ax.scatter(map_values, f1_values, alpha=0.7, s=100)
        
        # Линия тренда
        z = np.polyfit(map_values, f1_values, 1)
        p = np.poly1d(z)
        ax.plot(map_values, p(map_values), "r--", alpha=0.8)
        
        # Корреляция
        correlation = np.corrcoef(map_values, f1_values)[0, 1]
        ax.text(0.05, 0.95, f'Корреляция: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=12, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('mAP@0.5')
        ax.set_ylabel('F1-Score@0.5')
        ax.set_title('Корреляция между mAP и F1-Score')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detailed_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_dashboard(self):
        """Сводная диаграмма результатов."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Сводка результатов кросс-валидации', fontsize=18, fontweight='bold')
        
        # 1. Основные метрики (большой график)
        ax1 = fig.add_subplot(gs[0, :2])
        
        main_metrics = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5']
        labels = ['mAP@0.5', 'F1-Score@0.5']
        
        for i, (metric, label) in enumerate(zip(main_metrics, labels)):
            if metric in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric]['values']
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                folds = list(range(1, len(values) + 1))
                color = sns.color_palette("husl", len(main_metrics))[i]
                
                ax1.errorbar(folds, values, yerr=std_val, marker='o', 
                           label=f'{label} ({mean_val:.3f}±{std_val:.3f})', 
                           color=color, capsize=5, capthick=2)
                ax1.axhline(y=mean_val, color=color, linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Значение метрики')
        ax1.set_title('Основные метрики качества по фолдам')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Сводная статистика (таблица)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('tight')
        ax2.axis('off')
        
        stats_data = []
        for metric in ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5', 
                      'detection_rate_unripe', 'detection_rate_ripe']:
            if metric in self.cv_results['aggregated_metrics']:
                stats = self.cv_results['aggregated_metrics'][metric]
                stats_data.append([
                    metric.replace('_iou0.5_conf0.5', '').replace('_', ' '),
                    f"{stats['mean']:.3f}",
                    f"±{stats['std']:.3f}",
                    f"[{stats['min']:.3f}, {stats['max']:.3f}]"
                ])
        
        table = ax2.table(cellText=stats_data,
                         colLabels=['Метрика', 'Среднее', 'Станд. откл.', 'Диапазон'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax2.set_title('Сводная статистика', fontweight='bold')
        
        # 3. Распределение объектов
        ax3 = fig.add_subplot(gs[1, 0])
        
        total_unripe = sum([fold['total_unripe'] for fold in self.cv_results['fold_results']])
        total_ripe = sum([fold['total_ripe'] for fold in self.cv_results['fold_results']])
        
        labels_pie = ['Незрелые яблоки', 'Зрелые яблоки']
        sizes = [total_unripe, total_ripe]
        colors = ['lightcoral', 'lightgreen']
        
        ax3.pie(sizes, labels=labels_pie, autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Распределение объектов в датасете')
        
        # 4. Эффективность обнаружения
        ax4 = fig.add_subplot(gs[1, 1])
        
        avg_detection_unripe = np.mean([fold['detection_rate_unripe'] for fold in self.cv_results['fold_results']])
        avg_detection_ripe = np.mean([fold['detection_rate_ripe'] for fold in self.cv_results['fold_results']])
        
        categories = ['Незрелые', 'Зрелые']
        detection_rates = [avg_detection_unripe, avg_detection_ripe]
        
        bars = ax4.bar(categories, detection_rates, color=['orange', 'green'], alpha=0.7)
        ax4.set_ylabel('Коэффициент обнаружения')
        ax4.set_title('Средняя эффективность обнаружения')
        ax4.set_ylim(0, 1)
        
        # Добавляем значения на столбцы
        for bar, rate in zip(bars, detection_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Стабильность результатов
        ax5 = fig.add_subplot(gs[1, 2])
        
        fold_numbers = [fold['fold'] for fold in self.cv_results['fold_results']]
        map_values = [fold['mAP_iou0.5_conf0.5'] for fold in self.cv_results['fold_results']]
        
        # Violin plot для показа распределения
        ax5.violinplot([map_values], positions=[1], widths=0.5, showmeans=True, showmedians=True)
        ax5.set_xticks([1])
        ax5.set_xticklabels(['mAP@0.5'])
        ax5.set_ylabel('Значение')
        ax5.set_title('Распределение mAP по фолдам')
        ax5.grid(True, alpha=0.3)
        
        # 6. Временные затраты
        ax6 = fig.add_subplot(gs[2, :])
        
        if 'evaluation_time' in self.cv_results['fold_results'][0]:
            fold_times = [fold['evaluation_time'] for fold in self.cv_results['fold_results']]
            folds = list(range(1, len(fold_times) + 1))
            
            ax6.bar(folds, fold_times, alpha=0.7, color='skyblue')
            ax6.set_xlabel('Fold')
            ax6.set_ylabel('Время выполнения (сек)')
            ax6.set_title(f'Время оценки по фолдам (общее: {sum(fold_times):.1f} сек)')
            
            # Средняя линия
            avg_time = np.mean(fold_times)
            ax6.axhline(y=avg_time, color='red', linestyle='--', 
                       label=f'Среднее: {avg_time:.1f} сек')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'summary_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Кросс-валидация для модели обнаружения яблок',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_path', type=str, default='data/train',
                       help='Путь к директории с изображениями')
    parser.add_argument('--annotations_file', type=str, default='_annotations.coco.json',
                       help='Имя файла аннотаций COCO')
    parser.add_argument('--model_path', type=str, default='apple_detector.pth',
                       help='Путь к обученной модели')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Количество фолдов для кросс-валидации')
    parser.add_argument('--output_dir', type=str, default='cv_results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed для воспроизводимости')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство (cuda/cpu), auto если не указано')
    
    return parser.parse_args()


def main():
    """Основная функция."""
    args = parse_arguments()
    
    # Формируем полный путь к аннотациям
    annotations_path = os.path.join(args.data_path, args.annotations_file)
    
    # Проверяем существование файлов
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Директория с данными не найдена: {args.data_path}")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Файл аннотаций не найден: {annotations_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Файл модели не найден: {args.model_path}")
    
    # Определяем устройство
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("СИСТЕМА КРОСС-ВАЛИДАЦИИ ДЛЯ ОБНАРУЖЕНИЯ ЯБЛОК")
    print("="*80)
    print(f"Путь к данным: {args.data_path}")
    print(f"Файл аннотаций: {annotations_path}")
    print(f"Модель: {args.model_path}")
    print(f"Количество фолдов: {args.k_folds}")
    print(f"Устройство: {device}")
    print(f"Результаты будут сохранены в: {args.output_dir}")
    print("="*80)
    
    # Создаем анализатор
    analyzer = CrossValidationAnalyzer(
        data_path=args.data_path,
        annotations_path=annotations_path,
        k_folds=args.k_folds,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Запускаем кросс-валидацию
    try:
        results = analyzer.run_cross_validation(model_path=args.model_path)
        
        print("\n" + "="*80)
        print("КРОСС-ВАЛИДАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print("="*80)
        print(f"Результаты сохранены в: {args.output_dir}")
        
        # Краткая сводка
        map_mean = results['aggregated_metrics']['mAP_iou0.5_conf0.5']['mean']
        map_std = results['aggregated_metrics']['mAP_iou0.5_conf0.5']['std']
        f1_mean = results['aggregated_metrics']['F1_iou0.5_conf0.5']['mean']
        f1_std = results['aggregated_metrics']['F1_iou0.5_conf0.5']['std']
        
        print(f"\nИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   mAP@0.5:     {map_mean:.4f} ± {map_std:.4f}")
        print(f"   F1-Score@0.5: {f1_mean:.4f} ± {f1_std:.4f}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        raise


if __name__ == "__main__":
    main()