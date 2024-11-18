import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from training import (  # Asegúrate de que estos se importen correctamente
    CycleGANDataset, Generator)

# Definir el número de canales de entrada y salida
input_nc = 3  # Número de canales de entrada (por ejemplo, RGB)
output_nc = 3  # Número de canales de salida (por ejemplo, RGB)

# Inicializar el modelo del generador \( B \to A \)
generator_B_to_A = Generator(input_nc=input_nc, output_nc=output_nc)  # Pasa los argumentos requeridos
generator_B_to_A.load_state_dict(torch.load('../models/B2A/netG_B2A_epoch_100.pth', map_location=torch.device('cpu')))
generator_B_to_A.eval()

# Función para cargar la imagen y preprocesarla
def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizar a [-1, 1]
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Agregar dimensión batch

# Función para convertir tensor a imagen
def tensor_to_image(tensor):
    image = tensor.squeeze(0).detach().cpu().numpy()
    image = (image * 0.5 + 0.5) * 255  # Desnormalizar a [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8).transpose(1, 2, 0)  # Convertir a formato HWC
    return image

# Ruta de las imágenes
high_res_image_path = "./results/sample_5/ground_truth.png"
low_res_image_path = "./results/sample_5/input.png"

# Tamaño de imagen de entrada
image_size = 256

# Cargar la imagen de alta resolución
high_res_image = load_image(high_res_image_path, image_size)

# Cargar la imagen de baja resolución (para referencia)
low_res_image = load_image(low_res_image_path, image_size)

# Generar la imagen de baja resolución a partir de la de alta resolución
with torch.no_grad():
    generated_low_res_image = generator_B_to_A(high_res_image)

# Convertir los tensores a imágenes
high_res_img_np = tensor_to_image(high_res_image)
low_res_img_np = tensor_to_image(low_res_image)
generated_low_res_img_np = tensor_to_image(generated_low_res_image)

# Mostrar las imágenes
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Alta resolución (Original)")
plt.imshow(high_res_img_np)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Baja resolución (Original)")
plt.imshow(low_res_img_np)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Baja resolución (Generada)")
plt.imshow(generated_low_res_img_np)
plt.axis('off')

plt.tight_layout()
plt.show()
