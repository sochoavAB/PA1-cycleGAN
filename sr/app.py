import io
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image

from training import Generator

app = Flask(__name__)
CORS(app)  

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '../models/A2B/netG_A2B_epoch_100.pth' 

def load_model():
    """
    Carga el modelo pre-entrenado
    """
    model = Generator(3, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def prepare_image(image):
    """
    Prepara la imagen para el modelo
    Args:
        image: PIL Image
    Returns:
        torch.Tensor: Imagen normalizada y preparada para el modelo
    """
    # Asegurar que la imagen está en RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convertir a numpy array y normalizar a [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convertir a tensor y añadir dimensión de batch
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # Normalizar a [-1, 1] como espera el modelo
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img_tensor = transform(img_tensor)
    
    return img_tensor

def process_output(output_tensor):
    """
    Convierte el tensor de salida del modelo a una imagen PNG
    Args:
        output_tensor: torch.Tensor
    Returns:
        bytes: Imagen PNG en bytes
    """
    # Denormalizar de [-1, 1] a [0, 1]
    output = output_tensor * 0.5 + 0.5
    
    # Convertir a numpy array
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Asegurar que los valores están en [0, 1]
    output = np.clip(output, 0, 1)
    
    # Convertir a uint8 [0, 255]
    output = (output * 255).astype(np.uint8)
    
    # Convertir a imagen PIL
    img = Image.fromarray(output)
    
    # Guardar en buffer de bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

# Cargar el modelo al iniciar el servidor
print("Cargando modelo...")
model = load_model()
print(f"Modelo cargado y utilizando dispositivo: {DEVICE}")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Verificar si se recibió una imagen
        if 'image' not in request.files:
            return {'error': 'No se encontró la imagen'}, 400
        
        file = request.files['image']
        
        # Verificar que es un formato de imagen permitido
        allowed_formats = {'png', 'jpg', 'jpeg', 'webp'}
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension not in allowed_formats:
            return {'error': 'Formato no soportado. Use PNG, JPG, JPEG o WEBP'}, 400
        
        # Leer la imagen
        image = Image.open(file.stream)
        
        # Preparar la imagen para el modelo
        input_tensor = prepare_image(image).to(DEVICE)
        
        # Generar predicción
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Procesar la salida
        output_bytes = process_output(output_tensor)
        
        # Devolver la imagen generada
        return send_file(
            output_bytes, 
            mimetype='image/png',  # Siempre devolvemos PNG
            as_attachment=True, 
            download_name='generated.png'
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'error': str(e)}, 500
    
@app.route('/health', methods=['GET'])
def health_check():
    return {'status': 'ok', 'device': str(DEVICE)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)