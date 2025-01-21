import numpy as np
from PIL import Image
import joblib


class Check_Image():
    '''
    The class is intended for image verification. In this project, we only check dogs.
    '''
    def __init__(self):
        pass

    def load_image(self,filepath):
        # Загрузка изображения и приведение его к размеру 28x28
        img = Image.open(filepath).convert('L')  # Конвертируем в градации серого
        img = img.resize((28, 28))
        
        # Преобразование изображения в numpy массив
        img_array = np.array(img).reshape(1, -1)  # Преобразование в одномерный массив
        img_array = img_array / 255.0  # Нормализация
        
        return img_array

    def predict_image(self,model, img_array):
        # Используем модель для предсказания
        prediction = model.predict(img_array)
        return prediction[0]

    def Fake_or_Real(self, filename):
        '''
        Returns real or synthetic.
        '''
        # Загрузка обученной модели
        model = joblib.load('data/svm_model_real_vs_synthetic.pkl')

        # Проверка на вашем изображении
        image_path = filename  # Укажите путь к вашему изображению
        img_array = self.load_image(image_path)
        result = self.predict_image(model, img_array)

        print(f"The image is predicted as: {result}")
        return result

