import os
from math import log10

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor

from training import CycleGANDataset, Generator


def denormalize_image(image):
    """Convierte las imágenes de [-1,1] a [0,1] para visualización"""
    return (image + 1) / 2


def calculate_metrics(ground_truth, generated, win_size=3):
    """
    Calcula las métricas SSIM y PSNR entre la imagen objetivo y la generada.
    Args:
        ground_truth (numpy.ndarray): Imagen objetivo (HR)
        generated (numpy.ndarray): Imagen generada
        win_size (int): Tamaño de la ventana para SSIM
    Returns:
        dict: Diccionario con valores de SSIM y PSNR
    """
    gt = (ground_truth * 255).astype(np.uint8)
    gen = (generated * 255).astype(np.uint8)

    if min(gt.shape[:2]) < win_size or min(gen.shape[:2]) < win_size:
        return {'SSIM': None, 'PSNR': None}

    ssim_value = ssim(gt, gen, multichannel=True, data_range=gen.max() - gen.min(), win_size=win_size)
    mse = np.mean((gt - gen) ** 2)
    psnr_value = 10 * log10(255**2 / mse) if mse != 0 else float('inf')

    return {'SSIM': ssim_value, 'PSNR': psnr_value}


def apply_bicubic_interpolation(lr_image, target_size):
    """
    Aplica interpolación bicúbica a la imagen de baja resolución.
    """
    return cv2.resize(lr_image, target_size, interpolation=cv2.INTER_CUBIC)


def apply_lanczos_interpolation(lr_image, target_size):
    """
    Aplica interpolación Lanczos a la imagen de baja resolución.
    """
    return cv2.resize(lr_image, target_size, interpolation=cv2.INTER_LANCZOS4)


def apply_srcnn(lr_image, model, device):
    """
    Aplica el modelo SRCNN a la imagen de baja resolución.
    Args:
        lr_image (numpy.ndarray): Imagen de baja resolución
        model (torch.nn.Module): Modelo SRCNN preentrenado
        device (torch.device): Dispositivo para ejecutar el modelo
    """
    lr_tensor = ToTensor()(lr_image).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu())
    return np.array(sr_image)


def create_combined_image(images, titles):
    """
    Combina varias imágenes en una sola fila con títulos.
    """
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(titles[i], fontsize=8)
    fig.tight_layout()
    
    combined_path = os.path.join("temp_combined.png")
    plt.savefig(combined_path)
    plt.close()
    return combined_path


def evaluate_methods(model_path, srcnn_model_path, data_path, device='cuda'):
    """
    Procesa imágenes con el modelo entrenado, Bicubic, Lanczos y SRCNN, y guarda resultados combinados.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Cargar el modelo entrenado
    model = Generator(3, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Cargar el modelo SRCNN
    srcnn = torch.load(srcnn_model_path, map_location=device)
    srcnn.to(device)
    srcnn.eval()

    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CycleGANDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results_dir = 'results_with_lanczos_and_srcnn'
    os.makedirs(results_dir, exist_ok=True)
    
    all_metrics_model = {'SSIM': [], 'PSNR': []}
    all_metrics_bicubic = {'SSIM': [], 'PSNR': []}
    all_metrics_lanczos = {'SSIM': [], 'PSNR': []}
    all_metrics_srcnn = {'SSIM': [], 'PSNR': []}

    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B = model(real_A)

            real_A_np = denormalize_image(real_A[0]).cpu().numpy().transpose(1, 2, 0)
            real_B_np = denormalize_image(real_B[0]).cpu().numpy().transpose(1, 2, 0)
            fake_B_np = denormalize_image(fake_B[0]).cpu().numpy().transpose(1, 2, 0)

            # Métodos adicionales
            bicubic_image = apply_bicubic_interpolation(real_A_np, target_size=real_B_np.shape[:2][::-1])
            lanczos_image = apply_lanczos_interpolation(real_A_np, target_size=real_B_np.shape[:2][::-1])
            srcnn_image = apply_srcnn(real_A_np, srcnn, device)

            # Calcular métricas
            model_metrics = calculate_metrics(real_B_np, fake_B_np, win_size=3)
            bicubic_metrics = calculate_metrics(real_B_np, bicubic_image, win_size=3)
            lanczos_metrics = calculate_metrics(real_B_np, lanczos_image, win_size=3)
            srcnn_metrics = calculate_metrics(real_B_np, srcnn_image, win_size=3)

            # Guardar métricas globales
            for metric_set, metrics in zip(
                [all_metrics_model, all_metrics_bicubic, all_metrics_lanczos, all_metrics_srcnn],
                [model_metrics, bicubic_metrics, lanczos_metrics, srcnn_metrics]
            ):
                if metrics['SSIM'] is not None:
                    metric_set['SSIM'].append(metrics['SSIM'])
                    metric_set['PSNR'].append(metrics['PSNR'])

            current_dir = os.path.join(results_dir, f'sample_{i+1}')
            os.makedirs(current_dir, exist_ok=True)

            combined_path = create_combined_image(
                [real_A_np, fake_B_np, real_B_np, bicubic_image, lanczos_image, srcnn_image],
                ["Input", "Generated", "Ground Truth", "Bicubic", "Lanczos", "SRCNN"]
            )
            os.rename(combined_path, os.path.join(current_dir, 'combined.png'))

            # Guardar imágenes individuales
            plt.imsave(os.path.join(current_dir, 'input.png'), real_A_np)
            plt.imsave(os.path.join(current_dir, 'ground_truth.png'), real_B_np)
            plt.imsave(os.path.join(current_dir, 'generated_model.png'), fake_B_np)
            plt.imsave(os.path.join(current_dir, 'bicubic.png'), bicubic_image)
            plt.imsave(os.path.join(current_dir, 'lanczos.png'), lanczos_image)
            plt.imsave(os.path.join(current_dir, 'srcnn.png'), srcnn_image)

            # Guardar métricas en un archivo
            metrics_path = os.path.join(current_dir, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Metrics for Sample {i+1}:\n")
                f.write(f"  Model - SSIM: {model_metrics['SSIM']:.4f}, PSNR: {model_metrics['PSNR']:.2f}\n")
                f.write(f"  Bicubic - SSIM: {bicubic_metrics['SSIM']:.4f}, PSNR: {bicubic_metrics['PSNR']:.2f}\n")
                f.write(f"  Lanczos - SSIM: {lanczos_metrics['SSIM']:.4f}, PSNR: {lanczos_metrics['PSNR']:.2f}\n")
                f.write(f"  SRCNN - SSIM: {srcnn_metrics['SSIM']:.4f}, PSNR: {srcnn_metrics['PSNR']:.2f}\n")

            print(f"Processed sample {i+1}")

    # Guardar métricas globales
    summary_path = os.path.join(results_dir, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Overall Metrics Summary:\n")
        for method, metrics in zip(
            ["Model", "Bicubic", "Lanczos", "SRCNN"],
            [all_metrics_model, all_metrics_bicubic, all_metrics_lanczos, all_metrics_srcnn]
        ):
            f.write(f"\n{method} Metrics:\n")
            f.write(f"  SSIM - Mean: {np.mean(metrics['SSIM']):.4f}, STD: {np.std(metrics['SSIM']):.4f}, "
                    f"Max: {np.max(metrics['SSIM']):.4f}, Min: {np.min(metrics['SSIM']):.4f}\n")
            f.write(f"  PSNR - Mean: {np.mean(metrics['PSNR']):.2f}, STD: {np.std(metrics['PSNR']):.2f}, "
                    f"Max: {np.max(metrics['PSNR']):.2f}, Min: {np.min(metrics['PSNR']):.2f}\n")


if __name__ == "__main__":
    model_path = '../models/A2B/netG_A2B_epoch_100.pth'
    srcnn_model_path = '../models/Git/SRCNN.pth'  # Asegúrate de tener un modelo SRCNN preentrenado
    data_path = '../data/confocal_exper_paired_filt_validsetR_256.npz'
    
    evaluate_methods(
        model_path=model_path,
        srcnn_model_path=srcnn_model_path,
        data_path=data_path,
        device='cuda'
    )
