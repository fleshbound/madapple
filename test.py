import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader

from models.dataset import AppleDataset
from models.faster_rcnn import get_model
from models.transforms import get_transform


# Инициализация датасета
dataset = AppleDataset(root='data/train', transforms=get_transform(augment=True))

# Цвета и названия классов
class_colors = {
    1: (1, 1, 0),  # Желтый для незрелых (RGB)
    2: (1, 0, 0)  # Красный для зрелых
}
class_names = {
    1: 'immature',
    2: 'mature'
}

for i in range(min(5, len(dataset))):  # Проверяем первые 5 изображений
    try:
        img, target = dataset[i]
        print(f"\nИзображение {i}:")
        print(f"  Размер: {img.shape}")
        print(f"  Количество объектов: {len(target['boxes'])}")

        # Визуализация
        plt.figure(figsize=(12, 8))
        plt.imshow(img.permute(1, 2, 0))  # CHW -> HWC

        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.tolist()
            color = class_colors.get(label.item(), (0, 0, 1))  # Синий для неизвестных
            label_name = class_names.get(label.item(), 'unknown')

            # Рисуем прямоугольник
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.7
            )
            plt.gca().add_patch(rect)

            # Добавляем подпись
            plt.text(
                x1, y1 - 10,
                f"{label_name}",
                color='white',
                fontsize=12,
                bbox=dict(
                    facecolor=color,
                    alpha=0.9,
                    edgecolor='none',
                    boxstyle='round,pad=0.3'
                )
            )

        plt.title(f"Image {i} | Objects: {len(target['boxes'])}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ошибка при обработке изображения {i} ({dataset.image_info[dataset.ids[i]]['file_name']}): {str(e)}")
        continue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Проверка данных
img, target = dataset[0]
print("\nПроверка данных:")
print(f"Тип изображения: {type(img)} | Размер: {img.shape}")
print(f"Тип bbox: {type(target['boxes'])} | Количество: {len(target['boxes'])}")
print(f"Labels: {target['labels']}")

# Тест DataLoader
loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: tuple(zip(*x)))
batch = next(iter(loader))
print("\nТест DataLoader:")
print(f"Размер батча: {len(batch[0])}")
print(f"Тип первого изображения: {type(batch[0][0])}")
print(f"Размер первого изображения: {batch[0][0].shape}")


# Исправленная функция тестового обучения
def test_training_pipeline():
    model = get_model(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for images, targets in loader:
        try:
            # Перенос данных на устройство
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Тестовый forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            print(f"Тестовый шаг успешен! Loss: {losses.item():.4f}")
            break

        except Exception as e:
            print(f"Ошибка при тестовом обучении: {str(e)}")
            break


# Запуск теста
if torch.cuda.is_available():
    test_training_pipeline()
else:
    print("Предупреждение: CUDA недоступна, обучение будет на CPU")
    test_training_pipeline()