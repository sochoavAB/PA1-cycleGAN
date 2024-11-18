import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_cutoff_frequency(image, threshold=0.1):
    """
    Calcula la frecuencia de corte de una imagen usando la transformada de Fourier
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        # Asegurarse de que la imagen está en uint8 [0, 255]
        image_uint8 = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
        
    # Convertir a float para el procesamiento FFT
    gray = gray.astype(float) / 255.0
        
    # Aplicar FFT2D
    f = fft.fft2(gray)
    fshift = fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Normalizar el espectro
    magnitude_spectrum = magnitude_spectrum / magnitude_spectrum.max()
    
    # Encontrar el radio donde la magnitud cae por debajo del umbral
    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    y, x = np.ogrid[-center_y:magnitude_spectrum.shape[0]-center_y, 
                    -center_x:magnitude_spectrum.shape[1]-center_x]
    distances = np.sqrt(x*x + y*y)
    
    max_distance = np.sqrt(center_x**2 + center_y**2)
    radial_profile = []
    
    for r in range(int(max_distance)):
        mask = (distances >= r) & (distances < (r + 1))
        if mask.any():
            radial_mean = magnitude_spectrum[mask].mean()
            radial_profile.append(radial_mean)
    
    # Encontrar la primera frecuencia que cae por debajo del umbral
    radial_profile = np.array(radial_profile)
    cutoff_idx = np.where(radial_profile < threshold)[0]
    if len(cutoff_idx) > 0:
        cutoff = cutoff_idx[0] / len(radial_profile)
    else:
        cutoff = 1.0
        
    return cutoff

def calculate_metrics(ground_truth, generated):
    """
    Calcula PSNR, SSIM y frecuencia de corte entre dos imágenes
    """
    # Asegurar que las imágenes están en el rango [0, 1]
    ground_truth = ground_truth.astype(np.float32)
    generated = generated.astype(np.float32)
    
    # Calcular PSNR
    psnr_value = psnr(ground_truth, generated, data_range=1.0)
    
    # Calcular SSIM
    ssim_value = ssim(ground_truth, generated, data_range=1.0, channel_axis=2)
    
    # Calcular frecuencia de corte
    cutoff_freq = calculate_cutoff_frequency(generated)
    
    return psnr_value, ssim_value, cutoff_freq

def create_metrics_visualization(combined_image, metrics, save_path):
    """
    Crea una visualización que incluye la imagen combinada y sus métricas
    """
    # Crear figura con espacio para el texto de métricas
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]}, figsize=(15, 10))
    
    # Mostrar métricas en el subplot superior
    metrics_text = f'PSNR: {metrics[0]:.2f} dB | SSIM: {metrics[1]:.4f} | Cutoff Frequency: {metrics[2]:.4f}'
    ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12)
    ax1.axis('off')
    
    # Mostrar imagen combinada en el subplot inferior
    ax2.imshow(combined_image)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def load_image(path):
    """
    Carga una imagen y la convierte al formato correcto
    """
    # Leer la imagen
    image = plt.imread(path)
    
    # Si la imagen está en el rango [0, 255], convertir a [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    return image

def process_results_directory(results_dir='results'):
    """
    Procesa todas las carpetas de resultados, calcula métricas y genera visualizaciones
    """
    # Crear directorio para métricas
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Lista para almacenar todas las métricas
    all_metrics = []
    
    # Procesar cada carpeta de muestra
    for sample_dir in sorted(os.listdir(results_dir)):
        if not sample_dir.startswith('sample_'):
            continue
            
        sample_path = os.path.join(results_dir, sample_dir)
        if not os.path.isdir(sample_path):
            continue
            
        print(f"Processing {sample_dir}...")
        
        try:
            # Cargar imágenes
            ground_truth = load_image(os.path.join(sample_path, 'ground_truth.png'))
            generated = load_image(os.path.join(sample_path, 'generated.png'))
            combined = load_image(os.path.join(sample_path, 'combined.png'))
            
            # Calcular métricas
            metrics = calculate_metrics(ground_truth, generated)
            
            # Guardar métricas en un archivo CSV
            metrics_dict = {
                'sample': sample_dir,
                'psnr': metrics[0],
                'ssim': metrics[1],
                'cutoff_freq': metrics[2]
            }
            all_metrics.append(metrics_dict)
            
            # Crear y guardar visualización con métricas
            metrics_image_path = os.path.join(sample_path, 'metrics_visualization.png')
            create_metrics_visualization(combined, metrics, metrics_image_path)
            
        except Exception as e:
            print(f"Error processing {sample_dir}: {str(e)}")
            continue
    
    # Guardar todas las métricas en un archivo CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(metrics_dir, 'all_metrics.csv'), index=False)
        
        # Calcular y guardar estadísticas generales
        stats = metrics_df.describe()
        stats.to_csv(os.path.join(metrics_dir, 'metrics_statistics.csv'))
        
        # Crear visualizaciones de distribución de métricas
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # PSNR distribution
        axes[0].hist(metrics_df['psnr'], bins=20)
        axes[0].set_title('PSNR Distribution')
        axes[0].set_xlabel('PSNR (dB)')
        
        # SSIM distribution
        axes[1].hist(metrics_df['ssim'], bins=20)
        axes[1].set_title('SSIM Distribution')
        axes[1].set_xlabel('SSIM')
        
        # Cutoff frequency distribution
        axes[2].hist(metrics_df['cutoff_freq'], bins=20)
        axes[2].set_title('Cutoff Frequency Distribution')
        axes[2].set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'metrics_distribution.png'))
        plt.close()

if __name__ == "__main__":
    process_results_directory('results')