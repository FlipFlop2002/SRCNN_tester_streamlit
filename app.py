import streamlit as st
from PIL import Image
from functions import pil_to_tensor, tensor_to_pil
from model import SRCNN
import torch

# Tytuł aplikacji
st.title('SRCNN tester')

# Sekcja do wgrania obrazu
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
model_state_dict_path = "best_model_state_dict.pth"
scale_factor = 2

if uploaded_file is not None:
    # Otwórz obraz
    pil_oryginal_img = Image.open(uploaded_file)
    # Pobranie rozmiarów oryginalnego obrazu
    original_width, original_height = pil_oryginal_img.size
    # Obliczenie nowych rozmiarów (zmniejszenie)
    new_width = int(original_width / scale_factor)
    new_height = int(original_height / scale_factor)
    # Zmniejszenie obrazu
    pil_lr_image = pil_oryginal_img.resize((new_width, new_height), Image.BICUBIC)
    # Powiększenie z powrotem do oryginalnych rozmiarów za pomocą bikubicznej interpolacji
    pil_bicubic_image = pil_lr_image.resize((original_width, original_height), Image.BICUBIC)

    # zamiana obrazuów oryginalnego i bicubic na tensory
    tensor_original_img = pil_to_tensor(pil_oryginal_img)
    tensor_bicubic_img = pil_to_tensor(pil_bicubic_image)
    # inicjalizacja modelu SRCNN i przepuszczenie obrazu lr przez sieć
    model = SRCNN()
    model.load_state_dict(torch.load(model_state_dict_path, map_location='cpu'))
    model.eval()
    tensor_sr_img = model(tensor_bicubic_img.unsqueeze(0)).squeeze(0)
    tensor_sr_img = tensor_sr_img.clamp(0, 1)

    pil_sr_img = tensor_to_pil(tensor_sr_img.squeeze(0))

    # Wyświetl obraz czterokrotnie
    st.image(pil_lr_image, caption='LR image', use_column_width=False)
    st.image(pil_oryginal_img, caption='GT image', use_column_width=False)
    st.image(pil_bicubic_image, caption='Bicubic upscaled image (SRCNN input)', use_column_width=False)
    st.image(pil_sr_img, caption='SRCNN output', use_column_width=False)
else:
    st.write("Please upload an image file")