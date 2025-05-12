import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from models.dataset import UnlabeledDataset
from models.faster_rcnn import get_model
from models.transforms import get_transform
from torchmetrics.detection import MeanAveragePrecision


def save_predictions_with_metrics(model, test_loader, device, output_dir="test_predictions"):
    """Сохраняет предсказания в файлы и метрики в CSV"""
    os.makedirs(output_dir, exist_ok=True)
    class_names = {1: "immature", 2: "mature"}
    results = []

    # Инициализация метрик
    metric = MeanAveragePrecision()
    model.eval()

    with torch.no_grad():
        for images, filenames in test_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for img, pred, filename in zip(images, predictions, filenames):
                # Сохранение изображения с предсказаниями
                img_uint8 = (img * 255).byte().cpu()
                keep = pred['scores'] > 0.5
                boxes = pred['boxes'][keep].cpu()
                labels = pred['labels'][keep].cpu()
                scores = pred['scores'][keep].cpu()

                if len(boxes) > 0:
                    result_img = draw_bounding_boxes(
                        img_uint8,
                        boxes,
                        [f"{class_names[label.item()]} {score:.2f}" for label, score in zip(labels, scores)],
                        colors=["green" if label == 1 else "yellow" for label in labels],
                        width=3
                    )
                    to_pil_image(result_img).save(f"{output_dir}/pred_{filename}")

                # Запись результатов для метрик
                results.append({
                    'filename': filename,
                    'boxes': pred['boxes'].cpu().tolist(),
                    'scores': pred['scores'].cpu().tolist(),
                    'labels': pred['labels'].cpu().tolist()
                })

    # # Сохранение метрик в CSV (пример, нужны ground truth для реальных метрик)
    # metrics_df = pd.DataFrame([{
    #     'model': 'FasterRCNN',
    #     'precision': 'N/A',  # Замените на реальные значения
    #     'recall': 'N/A',
    #     'f1': 'N/A',
    #     'mAP': 'N/A'
    # }])
    #
    # metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)

    # Сохранение детальных предсказаний
    predictions_df = pd.DataFrame(results)
    predictions_df.to_csv(f"{output_dir}/predictions_details.csv", index=False)


# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
model_path = "logs/20250512_020657/epoch_10.pth"
checkpoint = torch.load(model_path, map_location=device)
model = get_model(num_classes=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Подготовка данных
test_dataset = UnlabeledDataset(
    root_dir="data/test",
    transforms=get_transform(augment=False)
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Запуск предсказания
save_predictions_with_metrics(model, test_loader, device)