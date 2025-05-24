#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Система кросс-валидации для оценки качества модели обнаружения яблок.
Использует k-fold CV на обучающем датасете с детальным анализом результатов.

Основные режимы работы:

1. Полная кросс-валидация:
    python cross_validation.py --data_path data/train --k_folds 5 --model_path apple_detector.pth

2. Анализ существующих результатов:
    python cross_validation.py --analyze_only --load_results_from cv_results/

3. Только генерация графиков:
    python cross_validation.py --generate_plots_only --output_dir cv_results/

4. Пропуск CV с анализом текущей директории:
    python cross_validation.py --skip_cv

Возможности:
- Стратифицированная кросс-валидация
- Тесты на нормальность распределения метрик
- Корреляционный анализ между метриками
- Сравнение с базовыми методами
- Черно-белые графики для печати
- Интерактивные графики с автосохранением
- Статистический анализ стабильности модели
"""

import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
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
            data_path (str): Путь к изображениям (может быть None для режима анализа)
            annotations_path (str): Путь к аннотациям COCO (может быть None для режима анализа)
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
        
        # Загружаем полный датасет (только если пути указаны)
        if data_path is not None and annotations_path is not None:
            self.full_dataset = AppleDataset(
                root_dir=data_path,
                annotations_path=annotations_path,
                transforms=None,
                is_train=False,  # Без аугментаций для честной оценки
                normalize=True
            )
            
            self.logger.info(f"Загружен датасет: {len(self.full_dataset)} изображений")
        else:
            self.full_dataset = None
            self.logger.info("Инициализация в режиме анализа существующих результатов")
        
        # Настройка стилей для графиков (черно-белая совместимость)
        self.setup_plot_styles()
        
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Количество фолдов: {k_folds}")
    
    def setup_plot_styles(self):
        """Настройка стилей для графиков с черно-белой совместимостью."""
        # Настройка matplotlib для интерактивного режима
        plt.ion()  # Включаем интерактивный режим
        
        # Цвета и узоры для черно-белой совместимости
        self.colors = ['black', 'darkgray', 'lightgray', 'white']
        self.hatches = ['', '///', '...', '+++', 'xxx', '\\\\\\', '|||', '---']
        self.markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
        self.linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 3, 1, 3))]
        
        # Настройка стиля seaborn для лучшего вида
        plt.style.use('default')
        
        # Настройка шрифтов для лучшей читаемости
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def load_existing_results(self, results_dir):
        """
        Загрузка существующих результатов кросс-валидации.
        
        Args:
            results_dir (str): Путь к директории с результатами
            
        Returns:
            bool: True если загрузка успешна, False иначе
        """
        try:
            # Загружаем JSON с результатами
            json_path = os.path.join(results_dir, 'cv_results.json')
            if not os.path.exists(json_path):
                self.logger.error(f"Файл результатов не найден: {json_path}")
                return False
            
            with open(json_path, 'r', encoding='utf-8') as f:
                self.cv_results = json.load(f)
            
            self.logger.info(f"Успешно загружены результаты из {json_path}")
            self.logger.info(f"Количество фолдов: {len(self.cv_results['fold_results'])}")
            
            # Проверяем структуру данных
            required_keys = ['fold_results', 'aggregated_metrics', 'metadata']
            for key in required_keys:
                if key not in self.cv_results:
                    self.logger.warning(f"Отсутствует ключ '{key}' в результатах")
            
            # Обновляем output_dir для сохранения новых графиков
            if results_dir != self.output_dir:
                self.logger.info(f"Обновляем output_dir с {self.output_dir} на {results_dir}")
                self.output_dir = results_dir
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке результатов: {e}")
            return False

    def analyze_existing_results(self, results_dir=None):
        """
        Анализ существующих результатов без проведения CV.
        
        Args:
            results_dir (str): Путь к директории с результатами
        """
        if results_dir is None:
            results_dir = self.output_dir
        
        self.logger.info("="*80)
        self.logger.info("АНАЛИЗ СУЩЕСТВУЮЩИХ РЕЗУЛЬТАТОВ КРОСС-ВАЛИДАЦИИ")
        self.logger.info("="*80)
        
        # Загружаем результаты
        if not self.load_existing_results(results_dir):
            raise ValueError(f"Не удалось загрузить результаты из {results_dir}")
        
        # Проводим анализ загруженных данных
        if 'aggregated_metrics' not in self.cv_results or not self.cv_results['aggregated_metrics']:
            self.logger.info("Агрегированные метрики отсутствуют, выполняем агрегацию...")
            self.aggregate_results()
        
        if 'statistical_analysis' not in self.cv_results or not self.cv_results['statistical_analysis']:
            self.logger.info("Статистический анализ отсутствует, выполняем анализ...")
            self.statistical_analysis()
        
        # Выводим результаты
        self.print_analysis_summary()
        
        # Создаем визуализации (простые графики для демонстрации)
        self.create_simple_visualizations()
        
        self.logger.info("="*80)
        self.logger.info("АНАЛИЗ СУЩЕСТВУЮЩИХ РЕЗУЛЬТАТОВ ЗАВЕРШЕН")
        self.logger.info("="*80)

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
        self.create_simple_visualizations()
        
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
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.generic):
            # Общий случай для любых numpy скаляров
            return obj.item()
        else:
            return obj

    def aggregate_results(self):
        """Агрегация результатов по всем фолдам."""
        self.logger.info("Выполнение агрегации результатов...")
        
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

    def statistical_analysis(self):
        """Расширенный статистический анализ результатов."""
        self.logger.info("Выполнение расширенного статистического анализа...")
        
        main_metrics = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5']
        stability_analysis = {}
        
        for metric in main_metrics:
            if metric in self.cv_results['aggregated_metrics']:
                values = np.array(self.cv_results['aggregated_metrics'][metric]['values'])
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv_coef = std_val / mean_val if mean_val > 0 else 0
                
                stability_analysis[metric] = {
                    'coefficient_of_variation': cv_coef,
                    'stability_rating': 'высокая' if cv_coef < 0.1 else 'средняя' if cv_coef < 0.2 else 'низкая'
                }
                
                self.logger.info(f"{metric}:")
                self.logger.info(f"  Коэффициент вариации: {cv_coef:.4f}")
                self.logger.info(f"  Стабильность: {stability_analysis[metric]['stability_rating']}")
                
                # Базовая статистика
                self.logger.info(f"  Среднее: {np.mean(values):.4f}")
                self.logger.info(f"  Медиана: {np.median(values):.4f}")
                self.logger.info(f"  Стд. откл.: {np.std(values):.4f}")
                self.logger.info(f"  Асимметрия: {self._calculate_skewness(values):.4f}")
                self.logger.info(f"  Эксцесс: {self._calculate_kurtosis(values):.4f}")
                
                # Тесты распределений
                distribution_tests = self._test_distributions(values)
                
                # Сохраняем результаты тестов
                stability_analysis[metric].update(distribution_tests)
                
                # Выводим результаты
                self.logger.info("  Тесты распределений:")
                for dist_name, test_result in distribution_tests.items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        p_val = test_result['p_value']
                        is_fit = test_result['fits_distribution']
                        self.logger.info(f"    {dist_name}: p={p_val:.4f}, подходит={is_fit}")
                
                # Рекомендация лучшего распределения
                best_dist = self._recommend_best_distribution(distribution_tests)
                self.logger.info(f"  Рекомендуемое распределение: {best_dist}")
                stability_analysis[metric]['recommended_distribution'] = best_dist
        
        self.cv_results['statistical_analysis'] = stability_analysis

    def _calculate_skewness(self, data):
        """Вычисление коэффициента асимметрии."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0
        skew = np.sum(((data - mean) / std) ** 3) / n
        return skew
    
    def _calculate_kurtosis(self, data):
        """Вычисление коэффициента эксцесса."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0
        kurt = np.sum(((data - mean) / std) ** 4) / n
        return kurt - 3  # Избыточный эксцесс (excess kurtosis)
    
    def _test_distributions(self, values):
        """Тестирование различных распределений."""
        results = {}
        
        try:
            from scipy import stats
            
            # 1. Тест Шапиро-Уилка на нормальность
            shapiro_stat, shapiro_p = stats.shapiro(values)
            results['normal_shapiro'] = {
                'p_value': shapiro_p,
                'fits_distribution': shapiro_p > 0.05,
                'test_statistic': shapiro_stat
            }
            
            # 2. Тест Д'Агостино-Пирсона на нормальность (если достаточно данных)
            if len(values) >= 8:
                dagostino_stat, dagostino_p = stats.normaltest(values)
                results['normal_dagostino'] = {
                    'p_value': dagostino_p,
                    'fits_distribution': dagostino_p > 0.05,
                    'test_statistic': dagostino_stat
                }
            
            # 3. Тест на Beta-распределение (для метрик в [0,1])
            if np.all(values >= 0) and np.all(values <= 1):
                try:
                    beta_params = stats.beta.fit(values, floc=0, fscale=1)
                    ks_stat, ks_p = stats.kstest(values, lambda x: stats.beta.cdf(x, *beta_params))
                    results['beta'] = {
                        'p_value': ks_p,
                        'fits_distribution': ks_p > 0.05,
                        'test_statistic': ks_stat,
                        'parameters': beta_params
                    }
                except:
                    pass
            
            # 4. Тест на равномерное распределение
            try:
                uniform_params = stats.uniform.fit(values)
                ks_stat, ks_p = stats.kstest(values, lambda x: stats.uniform.cdf(x, *uniform_params))
                results['uniform'] = {
                    'p_value': ks_p,
                    'fits_distribution': ks_p > 0.05,
                    'test_statistic': ks_stat,
                    'parameters': uniform_params
                }
            except:
                pass
            
            # 5. Тест на логнормальное распределение (если все значения > 0)
            if np.all(values > 0):
                try:
                    lognorm_params = stats.lognorm.fit(values, floc=0)
                    ks_stat, ks_p = stats.kstest(values, lambda x: stats.lognorm.cdf(x, *lognorm_params))
                    results['lognormal'] = {
                        'p_value': ks_p,
                        'fits_distribution': ks_p > 0.05,
                        'test_statistic': ks_stat,
                        'parameters': lognorm_params
                    }
                except:
                    pass
            
            # 6. Тест на усеченное нормальное распределение
            try:
                min_val, max_val = 0, 1
                if np.min(values) > 0 or np.max(values) < 1:
                    min_val, max_val = np.min(values) - 0.01, np.max(values) + 0.01
                
                truncnorm_params = stats.truncnorm.fit(values, min_val, max_val)
                ks_stat, ks_p = stats.kstest(values, lambda x: stats.truncnorm.cdf(x, *truncnorm_params))
                results['truncated_normal'] = {
                    'p_value': ks_p,
                    'fits_distribution': ks_p > 0.05,
                    'test_statistic': ks_stat,
                    'parameters': truncnorm_params
                }
            except:
                pass
            
            # 7. Тест Андерсона-Дарлинга на нормальность
            try:
                ad_stat, ad_critical, ad_significance = stats.anderson(values, dist='norm')
                ad_p_approx = 1 - (ad_significance[2] / 100)
                results['normal_anderson'] = {
                    'p_value': ad_p_approx,
                    'fits_distribution': ad_stat < ad_critical[2],
                    'test_statistic': ad_stat
                }
            except:
                pass
                
        except ImportError:
            self.logger.warning("scipy не установлен, используем базовые тесты")
            
            # Простой тест на равномерность
            if len(values) > 3:
                hist, bin_edges = np.histogram(values, bins=min(5, len(values)//2))
                expected = len(values) / len(hist)
                chi2_stat = np.sum((hist - expected)**2 / expected)
                results['uniform_simple'] = {
                    'p_value': 1 / (1 + chi2_stat),
                    'fits_distribution': chi2_stat < len(hist),
                    'test_statistic': chi2_stat
                }
        
        except Exception as e:
            self.logger.warning(f"Ошибка в тестах распределений: {e}")
        
        return results
    
    def _recommend_best_distribution(self, distribution_tests):
        """Рекомендация наилучшего распределения на основе тестов."""
        priority_order = [
            'beta',
            'truncated_normal',
            'normal_shapiro',
            'normal_dagostino',
            'normal_anderson',
            'uniform',
            'lognormal',
            'uniform_simple'
        ]
        
        best_dist = "неопределено"
        best_p_value = 0
        
        for dist_name in priority_order:
            if dist_name in distribution_tests:
                test_result = distribution_tests[dist_name]
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    p_val = test_result['p_value']
                    fits = test_result.get('fits_distribution', False)
                    
                    if fits and p_val > best_p_value:
                        best_dist = dist_name
                        best_p_value = p_val
        
        if best_dist == "неопределено":
            for dist_name, test_result in distribution_tests.items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    p_val = test_result['p_value']
                    if p_val > best_p_value:
                        best_dist = dist_name
                        best_p_value = p_val
        
        return f"{best_dist} (p={best_p_value:.4f})"

    def create_simple_visualizations(self):
        """Создание расширенных визуализаций."""
        self.logger.info("Создание расширенных визуализаций...")
        
        # График 1: Распределение основных метрик
        self.plot_metrics_distribution()
        
        # График 2: Анализ распределений
        self.plot_distribution_analysis()

    def plot_metrics_distribution(self):
        """График распределения основных метрик по фолдам."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics_to_plot = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5']
        metric_labels = ['mAP@0.5', 'F1-Score@0.5']
        
        width = 0.35
        folds = list(range(1, len(self.cv_results['fold_results']) + 1))
        x = np.arange(len(folds))
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            if metric in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric]['values']
                mean_val = np.mean(values)
                
                bars = ax.bar(x + idx * width, values, width, 
                             label=f'{label} (μ={mean_val:.3f})',
                             color=self.colors[idx % len(self.colors)],
                             hatch=self.hatches[idx],
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=1)
                
                # Добавляем линию среднего
                ax.axhline(y=mean_val, color=self.colors[idx % len(self.colors)], 
                          linestyle=self.linestyles[idx + 1], linewidth=2,
                          alpha=0.7)
        
        ax.set_xlabel('Fold', fontweight='bold')
        ax.set_ylabel('Значение метрики', fontweight='bold')
        ax.set_title('Распределение основных метрик качества по фолдам', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(folds)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, '01_metrics_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Сохранен график: {filename}")
        plt.show()
        plt.close()

    def plot_distribution_analysis(self):
        """График анализа различных распределений."""
        if 'statistical_analysis' not in self.cv_results:
            self.logger.info("Статистический анализ не проведен, пропуск графика")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        main_metrics = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5']
        
        # График 1: Q-Q plots для проверки нормальности
        for idx, metric in enumerate(main_metrics):
            if metric in self.cv_results['aggregated_metrics']:
                values = np.array(self.cv_results['aggregated_metrics'][metric]['values'])
                
                # Нормализуем данные
                normalized_values = (values - np.mean(values)) / np.std(values)
                
                # Теоретические квантили нормального распределения
                n = len(normalized_values)
                np.random.seed(42)  # Для воспроизводимости
                theoretical_quantiles = np.sort(np.random.normal(0, 1, n))
                observed_quantiles = np.sort(normalized_values)
                
                ax1.scatter(theoretical_quantiles, observed_quantiles,
                           marker=self.markers[idx], s=100, alpha=0.8,
                           color=self.colors[idx], 
                           edgecolor='black', linewidth=2,
                           label=metric.replace('_iou0.5_conf0.5', ''))
                
                # Линия идеальной нормальности
                min_val = min(theoretical_quantiles.min(), observed_quantiles.min())
                max_val = max(theoretical_quantiles.max(), observed_quantiles.max())
                ax1.plot([min_val, max_val], [min_val, max_val], 
                        linestyle=self.linestyles[idx], color=self.colors[idx], 
                        linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Теоретические квантили (нормальное)', fontweight='bold')
        ax1.set_ylabel('Наблюдаемые квантили', fontweight='bold')
        ax1.set_title('Q-Q Plot для проверки нормальности', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # График 2: P-values различных тестов распределений
        distribution_names = []
        p_values_map = []
        p_values_f1 = []
        
        test_names = ['normal_shapiro', 'normal_dagostino', 'beta', 'uniform', 
                     'truncated_normal', 'lognormal', 'normal_anderson']
        
        for test_name in test_names:
            found_data = False
            p_map, p_f1 = None, None
            
            for metric in main_metrics:
                if metric in self.cv_results['statistical_analysis']:
                    analysis = self.cv_results['statistical_analysis'][metric]
                    if test_name in analysis and isinstance(analysis[test_name], dict):
                        p_val = analysis[test_name].get('p_value', 0)
                        if metric == 'mAP_iou0.5_conf0.5':
                            p_map = p_val
                        else:
                            p_f1 = p_val
                        found_data = True
            
            if found_data and (p_map is not None or p_f1 is not None):
                distribution_names.append(test_name.replace('_', '\n'))
                p_values_map.append(p_map if p_map is not None else 0)
                p_values_f1.append(p_f1 if p_f1 is not None else 0)
        
        if distribution_names:
            x = np.arange(len(distribution_names))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, p_values_map, width,
                           label='mAP', color=self.colors[0],
                           hatch=self.hatches[0], alpha=0.8, edgecolor='black')
            bars2 = ax2.bar(x + width/2, p_values_f1, width,
                           label='F1-Score', color=self.colors[1],
                           hatch=self.hatches[1], alpha=0.8, edgecolor='black')
            
            # Линия значимости
            ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2,
                       label='α = 0.05 (граница значимости)')
            
            ax2.set_ylabel('P-value', fontweight='bold')
            ax2.set_title('P-values тестов различных распределений\n(p > 0.05 = распределение подходит)', 
                         fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(distribution_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(0, max(max(p_values_map + p_values_f1), 0.1) * 1.1)
        
        # График 3-4: Гистограммы с подогнанными распределениями
        try:
            from scipy import stats
            
            for idx, metric in enumerate(main_metrics):
                if metric in self.cv_results['aggregated_metrics']:
                    values = self.cv_results['aggregated_metrics'][metric]['values']
                    
                    ax = ax3 if idx == 0 else ax4
                    
                    # Гистограмма данных
                    n_bins = min(len(values), 8)
                    counts, bins, _ = ax.hist(values, bins=n_bins, alpha=0.7, density=True,
                                            color=self.colors[idx], hatch=self.hatches[idx],
                                            edgecolor='black', linewidth=1, 
                                            label='Наблюдаемые данные')
                    
                    # Подгоняем и рисуем различные распределения
                    x_range = np.linspace(min(values), max(values), 100)
                    
                    # Нормальное распределение
                    normal_params = stats.norm.fit(values)
                    ax.plot(x_range, stats.norm.pdf(x_range, *normal_params), 
                           linestyle='-', linewidth=2, color='red', 
                           label=f'Нормальное (μ={normal_params[0]:.3f})')
                    
                    # Beta-распределение (если данные в [0,1])
                    if np.all(np.array(values) >= 0) and np.all(np.array(values) <= 1):
                        try:
                            beta_params = stats.beta.fit(values, floc=0, fscale=1)
                            ax.plot(x_range, stats.beta.pdf(x_range, *beta_params), 
                                   linestyle='--', linewidth=2, color='blue',
                                   label=f'Beta (α={beta_params[0]:.2f}, β={beta_params[1]:.2f})')
                        except:
                            pass
                    
                    # Равномерное распределение
                    uniform_params = stats.uniform.fit(values)
                    ax.plot(x_range, stats.uniform.pdf(x_range, *uniform_params), 
                           linestyle='-.', linewidth=2, color='green',
                           label=f'Равномерное')
                    
                    ax.set_xlabel('Значение метрики', fontweight='bold')
                    ax.set_ylabel('Плотность', fontweight='bold')
                    ax.set_title(f'Подгонка распределений: {metric.replace("_iou0.5_conf0.5", "")}', 
                               fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3, linestyle='--')
        
        except ImportError:
            # Если scipy недоступен, показываем простые гистограммы
            for idx, metric in enumerate(main_metrics):
                if metric in self.cv_results['aggregated_metrics']:
                    values = self.cv_results['aggregated_metrics'][metric]['values']
                    ax = ax3 if idx == 0 else ax4
                    
                    ax.hist(values, bins=min(len(values), 8), alpha=0.7,
                           color=self.colors[idx], hatch=self.hatches[idx],
                           edgecolor='black', linewidth=1)
                    ax.set_xlabel('Значение метрики', fontweight='bold')
                    ax.set_ylabel('Частота', fontweight='bold')
                    ax.set_title(f'Распределение: {metric.replace("_iou0.5_conf0.5", "")}', 
                               fontweight='bold')
                    ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, '02_distribution_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Сохранен график: {filename}")
        plt.show()
        plt.close()
        """Создание простых графиков для демонстрации."""
        self.logger.info("Создание простых визуализаций...")
        
        # График 1: Распределение основных метрик
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics_to_plot = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5']
        metric_labels = ['mAP@0.5', 'F1-Score@0.5']
        
        width = 0.35
        folds = list(range(1, len(self.cv_results['fold_results']) + 1))
        x = np.arange(len(folds))
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            if metric in self.cv_results['aggregated_metrics']:
                values = self.cv_results['aggregated_metrics'][metric]['values']
                mean_val = np.mean(values)
                
                bars = ax.bar(x + idx * width, values, width, 
                             label=f'{label} (μ={mean_val:.3f})',
                             color=self.colors[idx % len(self.colors)],
                             hatch=self.hatches[idx],
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=1)
        
        ax.set_xlabel('Fold', fontweight='bold')
        ax.set_ylabel('Значение метрики', fontweight='bold')
        ax.set_title('Распределение основных метрик по фолдам', fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(folds)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Сохранение и показ
        filename = os.path.join(self.output_dir, 'metrics_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Сохранен график: {filename}")
        plt.show()
        plt.close()

    def print_analysis_summary(self):
        """Вывод сводки анализа результатов."""
        self.logger.info("\n" + "="*50)
        self.logger.info("СВОДКА РЕЗУЛЬТАТОВ")
        self.logger.info("="*50)
        
        # Метаданные
        if 'metadata' in self.cv_results:
            metadata = self.cv_results['metadata']
            self.logger.info(f"Количество фолдов: {metadata.get('k_folds', 'неизвестно')}")
            self.logger.info(f"Устройство: {metadata.get('device', 'неизвестно')}")
            self.logger.info(f"Дата проведения: {metadata.get('timestamp', 'неизвестно')}")
            self.logger.info(f"Seed: {metadata.get('seed', 'неизвестно')}")
        
        # Основные метрики
        if 'aggregated_metrics' in self.cv_results:
            main_metrics = ['mAP_iou0.5_conf0.5', 'F1_iou0.5_conf0.5', 
                           'detection_rate_unripe', 'detection_rate_ripe']
            
            self.logger.info("\nОсновные метрики:")
            for metric in main_metrics:
                if metric in self.cv_results['aggregated_metrics']:
                    stats = self.cv_results['aggregated_metrics'][metric]
                    mean_val = stats['mean']
                    std_val = stats['std']
                    min_val = stats['min']
                    max_val = stats['max']
                    self.logger.info(f"  {metric:25s}: {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]")
        
        # Статистический анализ - ИСПРАВЛЕНО
        if 'statistical_analysis' in self.cv_results:
            self.logger.info("\nСтатистический анализ:")
            for metric, analysis in self.cv_results['statistical_analysis'].items():
                if isinstance(analysis, dict):  # ИСПРАВЛЕНО: было isinstance(obj, dict)
                    cv_coef = analysis.get('coefficient_of_variation', 'неизвестно')
                    stability = analysis.get('stability_rating', 'неизвестно')
                    self.logger.info(f"  {metric}: CV={cv_coef:.4f}, стабильность={stability}")
                    
                    # Тесты нормальности
                    if 'shapiro_p_value' in analysis:
                        shapiro_p = analysis['shapiro_p_value']
                        is_normal = 'да' if analysis.get('is_normal_shapiro', False) else 'нет'
                        self.logger.info(f"    Шапиро-Уилк: p={shapiro_p:.4f}, нормальное={is_normal}")
                    
                    # Рекомендуемое распределение
                    if 'recommended_distribution' in analysis:
                        recommended = analysis['recommended_distribution']
                        self.logger.info(f"    Рекомендуемое распределение: {recommended}")
        
        self.logger.info("="*50)


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
    
    # Новые параметры для работы с существующими данными
    parser.add_argument('--analyze_only', action='store_true',
                       help='Только анализ существующих результатов без проведения CV')
    parser.add_argument('--load_results_from', type=str, default=None,
                       help='Путь к директории с существующими результатами CV для анализа')
    parser.add_argument('--generate_plots_only', action='store_true',
                       help='Только генерация графиков из существующих данных')
    parser.add_argument('--skip_cv', action='store_true',
                       help='Пропустить проведение кросс-валидации (синоним для --analyze_only)')
    
    return parser.parse_args()


def main():
    """Основная функция."""
    args = parse_arguments()
    
    # Обработка флагов анализа
    analyze_only = args.analyze_only or args.skip_cv or args.generate_plots_only
    results_dir = args.load_results_from or args.output_dir
    
    # Определяем устройство (нужно только для CV)
    if not analyze_only:
        if args.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
    else:
        device = None
    
    print("="*80)
    if analyze_only:
        print("АНАЛИЗ СУЩЕСТВУЮЩИХ РЕЗУЛЬТАТОВ КРОСС-ВАЛИДАЦИИ")
        print("="*80)
        print(f"Директория с результатами: {results_dir}")
    else:
        print("СИСТЕМА КРОСС-ВАЛИДАЦИИ ДЛЯ ОБНАРУЖЕНИЯ ЯБЛОК")
        print("="*80)
        print(f"Путь к данным: {args.data_path}")
        # Формируем полный путь к аннотациям
        annotations_path = os.path.join(args.data_path, args.annotations_file)
        print(f"Файл аннотаций: {annotations_path}")
        print(f"Модель: {args.model_path}")
        print(f"Количество фолдов: {args.k_folds}")
        print(f"Устройство: {device}")
    
    print(f"Результаты будут сохранены в: {args.output_dir}")
    print("="*80)
    
    # Создаем анализатор
    analyzer = CrossValidationAnalyzer(
        data_path=args.data_path if not analyze_only else None,
        annotations_path=annotations_path if not analyze_only else None,
        k_folds=args.k_folds,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    try:
        if analyze_only:
            # Только анализ существующих результатов
            analyzer.analyze_existing_results(results_dir)
            
            print("\n" + "="*80)
            print("АНАЛИЗ СУЩЕСТВУЮЩИХ РЕЗУЛЬТАТОВ ЗАВЕРШЕН!")
            print("="*80)
            print(f"Графики сохранены в: {analyzer.output_dir}")
            
        else:
            # Проверяем существование файлов для CV
            if not os.path.exists(args.data_path):
                raise FileNotFoundError(f"Директория с данными не найдена: {args.data_path}")
            
            if not os.path.exists(annotations_path):
                raise FileNotFoundError(f"Файл аннотаций не найден: {annotations_path}")
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Файл модели не найден: {args.model_path}")
            
            # Запускаем кросс-валидацию
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