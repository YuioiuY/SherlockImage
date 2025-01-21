import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
import requests
import joblib


class Fake_Or_Real_Dog_Model():
    '''
    This class is designed to quickly download ready-made image packages and train the model. It is IMPORTANT to note that the packs must be of the same size!

    As a result, you will receive a file stored in the folder data\svm_model_real_vs_synthetic.pkl
    '''
    def __init__(self):
        pass
    
    # Загрузка данных из файла
    def load_data_from_file(self,filepath):
        data = np.load(filepath)
        return data
    
    def train_and_save(self):

        categories = ['dog']

        try:
            real_data = {category: self.load_data_from_file(f"data/real_{category}.npy") for category in categories}
            synthetic_data = {category: self.load_data_from_file(f"data/synthetic_{category}.npy") for category in categories}
        except FileNotFoundError:
            print('File not found. End.')
            return

        X_real = np.concatenate(list(real_data.values()), axis=0)
        X_synthetic = np.concatenate(list(synthetic_data.values()), axis=0)
        X_real = X_real / 255.0
        X_synthetic = X_synthetic / 255.0

        if X_synthetic.shape[1:] != X_real.shape[1:]:
            X_synthetic = X_synthetic.reshape(X_synthetic.shape[0], 28, 28)

        
        X = np.concatenate([X_real, X_synthetic], axis=0)
        X = X.reshape(X.shape[0], -1)  
        Y_real = ['real'] * X_real.shape[0]
        Y_synthetic = ['synthetic'] * X_synthetic.shape[0]
        Y = np.array(Y_real + Y_synthetic)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, Y_train)

        joblib.dump(model, 'data/svm_model_real_vs_synthetic.pkl')

        Y_pred = model.predict(X_test)
        report = classification_report(Y_test, Y_pred)
        print(report)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y_test, Y_pred, labels=['real', 'synthetic']),
                                    display_labels=['Real', 'Synthetic'])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()


