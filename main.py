import logging
import os
import sys
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from models.dataset import AppleDataset, TestDataset
from models.transforms import get_transform
from models.faster_rcnn import get_model
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.v2.functional import to_pil_image

def collate_fn(batch):
    return tuple(zip(*batch))

def setup_logging(log_dir="logs"):
    """Настройка системы логирования"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file


def setup_device():
    """Настройка устройства (GPU/CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используемое устройство: {device}")
    if torch.cuda.is_available():
        logging.info(f"Тип GPU: {torch.cuda.get_device_name(0)}")
    return device


def prepare_dataloaders(config):
    """Подготовка загрузчиков данных"""
    try:
        train_dataset = AppleDataset(
            root='data/train',
            transforms=get_transform(augment=True)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn  # Явное указание collate_fn
        )

        test_dataset = TestDataset(
            root='data/test',
            transforms=get_transform(augment=False)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        logging.info(f"Загружено {len(train_dataset)} тренировочных изображений")
        logging.info(f"Загружено {len(test_dataset)} тестовых изображений")
        return train_loader, test_loader

    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {str(e)}", exc_info=True)
        raise


def train_model(model, train_loader, config, device, log_dir):
    """Цикл обучения модели"""
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(config["num_epochs"]):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (images, targets) in enumerate(train_loader):
                try:
                    # Проверка данных перед передачей на устройство
                    if not images or not targets:
                        logging.warning(f"Пустой батч {batch_idx}, пропускаем")
                        continue

                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    optimizer.zero_grad()
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()

                    epoch_loss += losses.item()

                    # if batch_idx % 10 == 0:
                    logging.info(
                            f"Epoch: {epoch + 1}/{config['num_epochs']}, "
                            f"Batch: {batch_idx}, Loss: {losses.item():.4f}"
                        )

                except Exception as e:
                    logging.error(f"Ошибка в батче {batch_idx}: {str(e)}", exc_info=True)
                    continue

            avg_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1} завершена. Средний Loss: {avg_loss:.4f}")

            if (epoch + 1) % 2 == 0:
                model_path = f"{log_dir}/epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, model_path)
                logging.info(f"Модель сохранена: {model_path}")

            lr_scheduler.step()

    except Exception as e:
        logging.error(f"Критическая ошибка при обучении: {str(e)}", exc_info=True)
        raise


def main():
    # Конфигурация
    config = {
        "batch_size": 4,
        "num_epochs": 10,
        "learning_rate": 0.001,
        "num_classes": 3
    }

    try:
        # Настройка логирования
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = setup_logging(log_dir)

        logging.info("Инициализация обучения...")
        device = setup_device()

        logging.info("Подготовка данных...")
        train_loader, test_loader = prepare_dataloaders(config)

        logging.info("Инициализация модели...")
        model = get_model(num_classes=config["num_classes"])
        model.to(device)
        logging.info(f"Модель инициализирована. Количество классов: {config['num_classes']}")

        logging.info("Начало обучения модели...")
        train_model(model, train_loader, config, device, log_dir)

        logging.info("Обучение успешно завершено!")

    except Exception as e:
        logging.critical(f"Фатальная ошибка: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
