from train.model import get_faster_rcnn_model
import torch


def load_model(model_path, num_classes=3, device="cpu", quiet=False):
    """
    Безопасная загрузка модели с попыткой определить правильную архитектуру.
    
    Args:
        model_path (str): Путь к файлу модели
        num_classes (int): Количество классов
        device (str): Устройство для загрузки
        quiet (bool): Подавить вывод сообщений
    
    Returns:
        torch.nn.Module: Загруженная модель
    """
    if not quiet:
        print(f"Пробуем загрузить модель")
    
    model = get_faster_rcnn_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    if not quiet:
        print(f"Модель успешно загружена")
    
    return model