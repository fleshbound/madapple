from models.dataset import AppleDataset
from models.transforms import get_transform
import torch
from torchvision.transforms.v2 import functional as F
from pathlib import Path
import matplotlib.pyplot as plt


def validate_and_visualize_boxes(root_dir, output_dir="validation_results"):
    # Создаем безопасные трансформации
    transforms = [
        F.to_image(),
        F.to_dtype(torch.float32, scale=True),
        F.sanitize_bounding_boxes()  # Автоматически исправляет bboxes
    ]

    dataset = AppleDataset(root=root_dir, transforms=transforms)
    Path(output_dir).mkdir(exist_ok=True)

    invalid_count = 0
    empty_count = 0

    for idx in range(len(dataset)):
        try:
            img, target = dataset[idx]
            boxes = target.get('boxes', torch.zeros((0, 4)))

            if len(boxes) == 0:
                empty_count += 1
                continue

            # Визуализация для проверки
            img = (img * 255).byte().permute(1, 2, 0).numpy()
            plt.figure(figsize=(12, 8))
            plt.imshow(img)

            # Рисуем все bboxes
            for box in boxes:
                x1, y1, x2, y2 = box
                plt.gca().add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor='green', linewidth=2
                ))

            plt.savefig(f"{output_dir}/img_{idx}.png")
            plt.close()

        except Exception as e:
            invalid_count += 1
            print(f"Ошибка в изображении {idx}: {str(e)}")
            continue

    print("\nИтоговая статистика:")
    print(f"Всего изображений: {len(dataset)}")
    print(f"Корректных изображений с bboxes: {len(dataset) - empty_count - invalid_count}")
    print(f"Изображений без bboxes: {empty_count}")
    print(f"Проблемных изображений: {invalid_count}")


if __name__ == "__main__":
    validate_and_visualize_boxes('..\\data\\train')