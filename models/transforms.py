import torch
from torchvision.transforms import v2 as T
from typing import List, Union


def get_transform(augment: bool = True, max_rotation_angle: int = 15) -> T.Compose:
    # Базовые обязательные трансформации
    transforms_list: List[Union[T.ToImage, T.ToDtype]] = [
        T.ToImage(),  # Конвертация в тензор
        T.ToDtype(torch.float32, scale=True)  # Нормализация [0, 1]
    ]

    if augment:
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=(-max_rotation_angle, max_rotation_angle)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.SanitizeBoundingBoxes(),  # Очистка некорректных bbox
        ])

    return T.Compose(transforms_list)