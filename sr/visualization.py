import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from training import CycleGANDataset, Generator


def denormalize_image(image):
    """Convierte las imágenes de [-1,1] a [0,1] para visualización"""
    return (image + 1) / 2

def visualize_results(model_path, data_path, num_images=5, device='cuda'):
    """
    Visualiza los resultados del modelo CycleGAN
    
    Args:
        model_path (str): Ruta al archivo .pth del modelo
        data_path (str): Ruta al archivo .npz con los datos
        num_images (int): Número de imágenes a visualizar
        device (str): Dispositivo a usar ('cuda' o 'cpu')
    """
    # Configurar el dispositivo
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Cargar el modelo
    model = Generator(3, 3)  # 3 canales de entrada y salida
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preparar el dataset
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CycleGANDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Configurar la visualización
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    plt.subplots_adjust(hspace=0.3)
    
    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(dataloader):
            if i >= num_images:
                break
                
            # Mover datos al dispositivo
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Generar imagen falsa
            fake_B = model(real_A)
            
            # Convertir a CPU y numpy para visualización
            real_A_np = denormalize_image(real_A[0]).cpu().numpy().transpose(1, 2, 0)
            real_B_np = denormalize_image(real_B[0]).cpu().numpy().transpose(1, 2, 0)
            fake_B_np = denormalize_image(fake_B[0]).cpu().numpy().transpose(1, 2, 0)
            
            # Visualizar las imágenes
            axes[i, 0].imshow(real_A_np)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(real_B_np)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(fake_B_np)
            axes[i, 2].set_title('Generated')
            axes[i, 2].axis('off')
    
    plt.savefig('cyclegan_results.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Results saved as 'cyclegan_results.png'")

if __name__ == "__main__":

    model_path = '../models/A2B/netG_A2B_epoch_100.pth' 
    data_path = '../data/confocal_exper_paired_filt_validsetR_256.npz'  
    
    visualize_results(
        model_path=model_path,
        data_path=data_path,
        num_images=5,
        device='cuda'
    )