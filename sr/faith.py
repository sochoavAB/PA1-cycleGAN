import os
from math import log10

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from training import CycleGANDataset, Generator


class SRCNN(torch.nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = torch.nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.layer2 = torch.nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.layer3 = torch.nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def denormalize_image(image):
    """Convierte las imágenes de [-1,1] a [0,1] para visualización"""
    return (image + 1) / 2


def calculate_metrics(ground_truth, generated, win_size=3):
    """Calcula SSIM y PSNR entre ground truth y generated."""
    gt = (ground_truth * 255).astype(np.uint8)
    gen = (generated * 255).astype(np.uint8)
    if min(gt.shape[:2]) < win_size or min(gen.shape[:2]) < win_size:
        return {'SSIM': None, 'PSNR': None}
    ssim_value = ssim(gt, gen, multichannel=True, data_range=gen.max() - gen.min(), win_size=win_size)
    mse = np.mean((gt - gen) ** 2)
    psnr_value = 10 * log10(255**2 / mse) if mse != 0 else float('inf')
    return {'SSIM': ssim_value, 'PSNR': psnr_value}


def apply_interpolation(lr_image, target_size, method):
    """Aplica interpolación (Bicubic o Lanczos) a la imagen de baja resolución."""
    if method == "bicubic":
        return cv2.resize(lr_image, target_size, interpolation=cv2.INTER_CUBIC)
    elif method == "lanczos":
        return cv2.resize(lr_image, target_size, interpolation=cv2.INTER_LANCZOS4)


def apply_srcnn(lr_image, srcnn_model, device):
    """Aplica SRCNN a la imagen de baja resolución."""
    lr_tensor = transforms.ToTensor()(lr_image).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = srcnn_model(lr_tensor)
    sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
    return np.array(sr_image)


def evaluate_methods(model_path, data_path, srcnn_path, device='cuda'):
    """Procesa imágenes con todos los métodos y guarda resultados."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Cargar el modelo entrenado
    model = Generator(3, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Cargar SRCNN
    srcnn = SRCNN()
    state_dict = torch.load(srcnn_path, map_location="cpu")
    adjusted_state_dict = {
        "layer1.weight": state_dict["conv1.weight"].repeat(1, 3, 1, 1) / 3,
        "layer1.bias": state_dict["conv1.bias"],
        "layer2.weight": state_dict["conv2.weight"],
        "layer2.bias": state_dict["conv2.bias"],
        "layer3.weight": state_dict["conv3.weight"].repeat(3, 1, 1, 1),
        "layer3.bias": state_dict["conv3.bias"].repeat(3),
    }
    srcnn.load_state_dict(adjusted_state_dict)
    srcnn.to(device)
    srcnn.eval()

    # Preparar dataset
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CycleGANDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results_dir = 'results_all'
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = {
        'Input': {'SSIM': [], 'PSNR': []},
        'Model': {'SSIM': [], 'PSNR': []},
        'Bicubic': {'SSIM': [], 'PSNR': []},
        'Lanczos': {'SSIM': [], 'PSNR': []},
        'SRCNN': {'SSIM': [], 'PSNR': []},
    }

    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B = model(real_A)

            real_A_np = denormalize_image(real_A[0]).cpu().numpy().transpose(1, 2, 0)
            real_B_np = denormalize_image(real_B[0]).cpu().numpy().transpose(1, 2, 0)
            fake_B_np = denormalize_image(fake_B[0]).cpu().numpy().transpose(1, 2, 0)

            bicubic_image = apply_interpolation(real_A_np, target_size=real_B_np.shape[:2][::-1], method="bicubic")
            lanczos_image = apply_interpolation(real_A_np, target_size=real_B_np.shape[:2][::-1], method="lanczos")
            srcnn_image = apply_srcnn(real_A_np, srcnn, device)

            metrics = {
                'Input': calculate_metrics(real_B_np, real_A_np),
                'Model': calculate_metrics(real_B_np, fake_B_np),
                'Bicubic': calculate_metrics(real_B_np, bicubic_image),
                'Lanczos': calculate_metrics(real_B_np, lanczos_image),
                'SRCNN': calculate_metrics(real_B_np, srcnn_image),
            }

            for key in all_metrics:
                if metrics[key]['SSIM'] is not None:
                    all_metrics[key]['SSIM'].append(metrics[key]['SSIM'])
                    all_metrics[key]['PSNR'].append(metrics[key]['PSNR'])

            # Guardar resultados por muestra
            current_dir = os.path.join(results_dir, f'sample_{i+1}')
            os.makedirs(current_dir, exist_ok=True)

            for img, name in zip([real_A_np, fake_B_np, real_B_np, bicubic_image, lanczos_image, srcnn_image],
                                 ['input', 'generated_model', 'ground_truth', 'bicubic', 'lanczos', 'srcnn']):
                plt.imsave(os.path.join(current_dir, f'{name}.png'), img)

            metrics_path = os.path.join(current_dir, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Metrics for Sample {i+1}:\n")
                for key, value in metrics.items():
                    f.write(f"  {key} - SSIM: {value['SSIM']:.4f}, PSNR: {value['PSNR']:.2f}\n")

    # Guardar resumen de métricas
    summary_path = os.path.join(results_dir, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Overall Metrics Summary:\n")
        for key in all_metrics:
            f.write(f"\n{key} Metrics:\n")
            f.write(f"  SSIM - Mean: {np.mean(all_metrics[key]['SSIM']):.4f}, STD: {np.std(all_metrics[key]['SSIM']):.4f}, "
                    f"Max: {np.max(all_metrics[key]['SSIM']):.4f}, Min: {np.min(all_metrics[key]['SSIM']):.4f}\n")
            f.write(f"  PSNR - Mean: {np.mean(all_metrics[key]['PSNR']):.2f}, STD: {np.std(all_metrics[key]['PSNR']):.2f}, "
                    f"Max: {np.max(all_metrics[key]['PSNR']):.2f}, Min: {np.min(all_metrics[key]['PSNR']):.2f}\n")


if __name__ == "__main__":
    model_path = '../models/A2B/netG_A2B_epoch_100.pth'
    data_path = '../data/confocal_exper_paired_filt_validsetR_256.npz'
    srcnn_path = '../models/Git/SRCNN.pth'
    
    evaluate_methods(
        model_path=model_path,
        data_path=data_path,
        srcnn_path=srcnn_path,
        device='cuda'
    )
