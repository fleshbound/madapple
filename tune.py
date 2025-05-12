import datetime
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import RandomHorizontalFlip, ColorJitter
from models.dataset import AppleDataset
from models.faster_rcnn import get_model
from models.transforms import get_transform
from torchmetrics.detection import MeanAveragePrecision
import matplotlib.pyplot as plt
import uuid
import logging

def collate_fn(batch):
    return tuple(zip(*batch))

# 2. Загрузка и подготовка данных
def prepare_data():
    # Аугментации
    transforms = get_transform(augment=True)
    # transforms.transforms.extend([
    #     RandomHorizontalFlip(p=0.5),
    #     ColorJitter(brightness=0.3, contrast=0.3)
    # ])

    # Полный датасет
    full_dataset = AppleDataset(
        root='data/train',
        transforms=transforms
    )

    # Разделение на train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    return train_loader, val_loader

# 3. Загрузка модели для дообучения
def load_model():
    model_path = "logs/20250512_020657/epoch_10.pth"  # Замените на путь к вашей модели
    checkpoint = torch.load(model_path, map_location=device)

    model = get_model(num_classes=3)  # Фон + 2 класса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Оптимизатор с меньшим LR
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    return model, optimizer


def calculate_metrics(predictions, targets):
    """Вычисление метрик с конвертацией типов данных"""
    metric = MeanAveragePrecision()

    # Конвертируем targets в нужный формат
    processed_targets = []
    for target in targets:
        processed_target = {
            'boxes': torch.stack([torch.as_tensor(box) for box in target['boxes']]),
            'labels': torch.as_tensor(target['labels']),
            'image_id': torch.as_tensor(target['image_id']),
            # 'area': torch.as_tensor(target['area']),
            # 'iscrowd': torch.as_tensor(target['iscrowd'])
        }
        processed_targets.append(processed_target)

    # Конвертируем predictions
    processed_preds = []
    for pred in predictions:
        processed_pred = {
            'boxes': torch.as_tensor(pred['boxes']),
            'scores': torch.as_tensor(pred['scores']),
            'labels': torch.as_tensor(pred['labels'])
        }
        processed_preds.append(processed_pred)

    metric.update(processed_preds, processed_targets)
    return metric.compute()

# 5. Цикл дообучения
def fine_tune(model, optimizer, train_loader, val_loader, num_epochs=3):
    best_map = 0.0
    train_losses = []
    no_improve = 0
    patience = 3  # Количество эпох без улучшения

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        i = 0
        for images, targets in train_loader:
            i += 1
            logging.info(f"Train epoch {epoch + 1}/{num_epochs}, image: {i}")
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Валидация
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]

                # Конвертируем targets перед отправкой в модель
                processed_targets = []
                for target in targets:
                    processed_target = {
                        'boxes': torch.stack([t.to(device) for t in target['boxes']]),
                        'labels': target['labels'].to(device),
                        'image_id': target['image_id'],
                        # 'area': target['area'],
                        # 'iscrowd': target['iscrowd']
                    }
                    processed_targets.append(processed_target)

                predictions = model(images)

                # Сохраняем предсказания и цели
                all_predictions.extend(predictions)
                all_targets.extend(processed_targets)

        val_metrics = calculate_metrics(all_predictions, all_targets)

        # Логирование
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Validation mAP: {val_metrics['map']:.4f}")

        if val_metrics['map'] > best_map:
            best_map = val_metrics['map']
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'metrics': val_metrics,
                'train_losses': train_losses
            }, f"logs/best_finetuned_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            logging.info("Saved best model")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Ранняя остановка на эпохе {epoch}")
                break

    return train_losses

# 6. Основной блок
if __name__ == "__main__":
    # Настройка logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_id = uuid.uuid4()
    logging.info(f"Starting run with ID: {run_id}")

    # 1. Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    try:
        train_loader, val_loader = prepare_data()
        model, optimizer = load_model()
        train_losses = fine_tune(model, optimizer, train_loader, val_loader, num_epochs=15)

        # Визуализация графика loss
        plt.plot(train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("Train Loss vs. Epoch")
        plt.savefig("logs/train_loss.png")
        plt.show()

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
