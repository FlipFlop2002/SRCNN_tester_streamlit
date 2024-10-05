import torch
from torchvision import transforms
import numpy as np

def pil_to_tensor(image):
    """
    Konwertuje obraz PIL Image na tensor PyTorch.

    Parametry:
    - image (PIL.Image): Obraz w formacie PIL Image.

    Zwraca:
    - Tensor PyTorch reprezentujący obraz.
    """
    # Definiowanie transformacji PIL -> Tensor
    transform = transforms.ToTensor()

    # Konwersja obrazu na tensor
    tensor = transform(image)

    return tensor


def tensor_to_pil(tensor):
    """
    Konwertuje tensor PyTorch na obraz w formacie PIL Image.

    Parametry:
    - tensor (torch.Tensor): Tensor PyTorch o kształcie (C, H, W), gdzie C to liczba kanałów, H to wysokość, W to szerokość.

    Zwraca:
    - Obraz w formacie PIL Image.
    """
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = torch.clamp(tensor, 0, 1)  # Przykład ograniczenia wartości tensora do zakresu [0, 1]
    # Definiowanie transformacji Tensor -> PIL Image
    transform = transforms.ToPILImage(mode="RGB")

    # Konwersja tensora na obraz PIL
    pil_image = transform(tensor)

    return pil_image