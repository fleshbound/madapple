from typing import List, Dict, Optional, Any
import os
from pathlib import Path

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as T
from PIL import Image


def visualize_predictions(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader[Any],
        device: torch.device,
        save_dir: str = "predictions",
        num_images: Optional[int] = None,
        confidence_threshold: float = 0.5
) -> None:
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    class_names = {1: "immature", 2: "mature"}

    with torch.no_grad():
        for i, (images, filenames) in enumerate(dataloader):
            if num_images is not None and i >= num_images:
                break

            images_list = [img.to(device) for img in images]
            predictions: List[Dict[str, torch.Tensor]] = model(images_list)

            for img, pred, filename in zip(images_list, predictions, filenames):
                _save_prediction(
                    img.cpu(),
                    pred,
                    class_names,
                    Path(save_dir) / f"pred_{filename}",
                    confidence_threshold
                )


def _save_prediction(
        img: torch.Tensor,
        pred: Dict[str, torch.Tensor],
        class_names: Dict[int, str],
        save_path: Path,
        confidence_threshold: float
) -> None:
    img_uint8 = (img * 255).byte()
    boxes = pred["boxes"].cpu()
    labels = pred["labels"].cpu()
    scores = pred["scores"].cpu()

    keep = scores > confidence_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return

    captions = [
        f"{class_names.get(l.item(), 'unknown')} {s:.2f}"
        for l, s in zip(labels, scores)
    ]

    result_img = draw_bounding_boxes(
        img_uint8,
        boxes,
        captions,
        colors="red",
        width=2,
        font_size=14
    )

    T.ToPILImage()(result_img).save(save_path)