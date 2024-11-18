import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from training import CycleGANDataset, Generator


def denormalize_image(image):
    """Convierte las imágenes de [-1,1] a [0,1] para visualización"""
    return (image + 1) / 2

def create_combined_image(images):
    """
    Crea una imagen combinada horizontal con todas las imágenes
    Args:
        images (list): Lista de imágenes numpy arrays
    Returns:
        numpy.ndarray: Imagen combinada
    """
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    total_width = sum(img.shape[1] for img in images)
    
    combined_image = np.zeros((max_height, total_width, 3))
    current_x = 0
    
    for img in images:
        h, w = img.shape[:2]
        combined_image[0:h, current_x:current_x+w] = img
        current_x += w
        
    return combined_image

def save_individual_results(input_img, ground_truth, generated, save_dir):
    """
    Guarda las imágenes individuales y la combinada en el directorio especificado
    
    Args:
        input_img (numpy.ndarray): Imagen de entrada
        ground_truth (numpy.ndarray): Imagen objetivo
        generated (numpy.ndarray): Imagen generada
        save_dir (str): Directorio donde guardar las imágenes
    """
    # Guardar imágenes individuales
    plt.imsave(os.path.join(save_dir, 'input.png'), input_img)
    plt.imsave(os.path.join(save_dir, 'ground_truth.png'), ground_truth)
    plt.imsave(os.path.join(save_dir, 'generated.png'), generated)
    
    # Crear y guardar imagen combinada
    combined = create_combined_image([input_img, ground_truth, generated])
    plt.imsave(os.path.join(save_dir, 'combined.png'), combined)

def visualize_results(model_path, data_path, device='cuda'):
    """
    Visualiza y guarda los resultados del modelo CycleGAN en carpetas individuales
    
    Args:
        model_path (str): Ruta al archivo .pth del modelo
        data_path (str): Ruta al archivo .npz con los datos
        device (str): Dispositivo a usar ('cuda' o 'cpu')
    """
    # Configurar el dispositivo
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Cargar el modelo
    model = Generator(3, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preparar el dataset
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CycleGANDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Crear directorio principal de resultados
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(dataloader):
            # Crear directorio para el conjunto actual de imágenes
            current_dir = os.path.join(results_dir, f'sample_{i+1}')
            os.makedirs(current_dir, exist_ok=True)
            
            # Mover datos al dispositivo
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Generar imagen falsa
            fake_B = model(real_A)
            
            # Convertir a CPU y numpy para visualización
            real_A_np = denormalize_image(real_A[0]).cpu().numpy().transpose(1, 2, 0)
            real_B_np = denormalize_image(real_B[0]).cpu().numpy().transpose(1, 2, 0)
            fake_B_np = denormalize_image(fake_B[0]).cpu().numpy().transpose(1, 2, 0)
            
            # Guardar todas las imágenes en el directorio actual
            save_individual_results(
                real_A_np,
                real_B_np,
                fake_B_np,
                current_dir
            )
            
            print(f"Processed and saved sample {i+1}")

if __name__ == "__main__":
    model_path = '../models/A2B/netG_A2B_epoch_100.pth'
    data_path = '../data/confocal_exper_paired_filt_validsetR_256.npz'
    
    visualize_results(
        model_path=model_path,
        data_path=data_path,
        device='cuda'
    )