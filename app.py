import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import cv2  # Для визуализации

from models.faster_rcnn import get_model


# 1. Загрузка модели
def load_model(model_path, num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = get_model(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Загрузите веса модели
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Указываем CPU, если нет GPU
    model.eval() # Переводим в режим eval
    return model

# 2. Загрузка и преобразование изображения
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()]) # Преобразуем в тензор
    img_tensor = transform(img)
    return img_tensor, img # возвращаем и PIL Image, пригодится для визуализации

# 3. Выполнение предсказания
def predict(model, image_tensor, threshold=0.5):
    with torch.no_grad(): # Отключаем вычисление градиентов
        prediction = model([image_tensor])

    pred_boxes = prediction[0]['boxes']
    pred_labels = prediction[0]['labels']
    pred_scores = prediction[0]['scores']

    # Фильтруем предсказания по порогу уверенности
    keep = pred_scores >= threshold
    pred_boxes = pred_boxes[keep].cpu().numpy()
    pred_labels = pred_labels[keep].cpu().numpy()
    pred_scores = pred_scores[keep].cpu().numpy()

    return pred_boxes, pred_labels, pred_scores

# 4. Визуализация предсказаний
def visualize_predictions(image, boxes, labels, scores, class_names=['__background__', 'apple', 'rotten_apple']):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # PIL -> OpenCV
    for i in range(len(boxes)):
        box = boxes[i].astype(int)
        label = labels[i]
        score = scores[i]
        class_name = class_names[label]

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text = f"{class_name}: {score:.2f}"
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 5. Основной блок
if __name__ == "__main__":
    import argparse
    import numpy as np # для cv2

    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Inference script for object detection")
    parser.add_argument("--model_path", required=True, help="Path to the trained model (.pth file)")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for predictions")
    args = parser.parse_args()

    # Загружаем модель
    model = load_model(args.model_path)

    # Загружаем и преобразуем изображение
    image_tensor, image_pil = load_image(args.image_path)

    # Выполняем предсказание
    pred_boxes, pred_labels, pred_scores = predict(model, image_tensor, args.threshold)

    # Визуализируем предсказания
    visualize_predictions(image_pil, pred_boxes, pred_labels, pred_scores)
