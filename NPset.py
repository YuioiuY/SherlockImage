import numpy as np
import os
from PIL import Image

class Create_NPset():
    def __init__(self, filename, name):
        self.image_folder = filename
        self.output_file = name
        pass

    def save_synthetic_images_to_file(self):
        # Список всех изображений в папке
        image_files = os.listdir(self.image_folder)
        images = []
        
        for image_file in image_files:
            if image_file.endswith(".png") or image_file.endswith(".jpg"):  # Проверка на формат изображения
                img_path = os.path.join(self.image_folder, image_file)
                img = Image.open(img_path).convert('L')  # Конвертировать в оттенки серого
                img = img.resize((28, 28))  # Привести к размеру 28x28 пикселей
                img_array = np.array(img).reshape(28, 28)  # Преобразовать изображение в массив
                images.append(img_array)
        
        # Преобразовать в numpy массив и сохранить в файл
        images_array = np.array(images)
        np.save(self.output_file, images_array)
        print(f"Ready {self.output_file}")
