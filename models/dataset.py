from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms import v2 as T


class AppleDataset(Dataset):
    def __init__(self, root: str, transforms: Optional[Any] = None) -> None:
        self.root = root
        self.transforms = transforms
        self.coco_data = self._load_coco_annotations()
        self.image_info = {
            img["id"]: img for img in self.coco_data["images"]
        }
        self.annotations = self._process_annotations()
        self.ids = list(sorted(self.image_info.keys()))

    def _load_coco_annotations(self) -> Dict[str, Any]:
        with open(os.path.join(self.root, "_annotations.coco.json"), "r") as f:
            data: Dict[str, Any] = json.load(f)
            return data

    def _process_annotations(self) -> Dict[int, List[Dict[str, Any]]]:
        annotations: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append(ann)
        return annotations

    def __getitem__(self, idx):
        # Загрузка изображения
        img_id = self.ids[idx]
        img_path = os.path.join(self.root, self.image_info[img_id]['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Загрузка аннотаций
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        # Конвертация в тензоры
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Обработка случая без объектов
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Создание tv_tensors
        img = T.functional.to_image(img)  # Новый метод вместо to_image_tensor
        img = T.functional.to_dtype(img, torch.float32, scale=True)

        boxes = BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=(img.shape[-2], img.shape[-1])  # (height, width)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)


class TestDataset(Dataset):
    def __init__(self, root: str, transforms: Optional[Any] = None) -> None:
        self.root = Path(root)
        self.transforms = transforms
        self.image_files = sorted(
            list(self.root.glob("*.jpg")) + list(self.root.glob("*.png"))
        )

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path)

        if self.transforms:
            img, _ = self.transforms(img, {})

        return img, filename

    def __len__(self) -> int:
        return len(self.image_files)


class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, self.image_files[idx]  # Возвращаем изображение и имя файла