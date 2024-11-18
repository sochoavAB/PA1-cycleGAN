import requests
from PIL import Image


def test_generation(image_path):
    """
    Prueba el endpoint de generación
    
    Args:
        image_path (str): Ruta a la imagen PNG de entrada
    """
    # Preparar la imagen
    files = {'image': open(image_path, 'rb')}
    
    # Hacer la petición
    response = requests.post('http://localhost:5000/generate', files=files)
    
    if response.status_code == 200:
        # Guardar la imagen generada
        output_path = 'generated_test.png'
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Imagen generada guardada en: {output_path}")
        
        # Mostrar la imagen si estás en un notebook
        Image.open(output_path).show()
    else:
        print(f"Error: {response.json()}")

# Ejemplo de uso
test_generation('./results/sample_5/input.png')