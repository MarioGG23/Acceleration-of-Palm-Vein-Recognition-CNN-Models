import os
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def calculate_lbp_with_bins(image_path, bins):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    radius = 1 
    n_points = 8 * radius 
    lbp_image = feature.local_binary_pattern(image, n_points, radius, method='uniform')

    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=bins, range=(0, n_points+2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)


    return lbp_hist

def lectura_and_lbp(folder_parkinson, folder_no_parkinson, bins):
    histograms = []
    labels = []

    for filename in os.listdir(folder_parkinson):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_parkinson, filename)
            lbp_histogram = calculate_lbp_with_bins(image_path, bins)
            histograms.append(lbp_histogram)
            labels.append(1)

    for filename in os.listdir(folder_no_parkinson):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_no_parkinson, filename)
            lbp_histogram = calculate_lbp_with_bins(image_path, bins)
            histograms.append(lbp_histogram)
            labels.append(0)

    return histograms, labels

def sistema_lineal_sobredeterminista(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Convierte las listas en matrices NumPy
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    
    # Agregar un término independiente a la matriz X_train
    X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
    
    X1 = X_train
    y1 = np.array(y_train)
    start_time = time.time()
    
    coefficients1 = np.linalg.pinv(X1)
    coefficients = np.dot(coefficients1, y1)
    end_time = time.time()

    # Calcular la precisión en el conjunto de validación
    X_val = np.column_stack((X_val, np.ones(X_val.shape[0])))
    y_val_pred = (X_val.dot(coefficients) >= 0.5).astype(int)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    
    # Calcular la precisión en el conjunto de prueba
    X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))
    y_test_pred = (X_test.dot(coefficients) >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    training_time = end_time - start_time
    return val_accuracy, test_accuracy, training_time


if __name__ == "__main__":

    folder_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Enfermos"
    folder_no_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Sanos"

    bins_values = [10, 20, 40, 60, 80, 100]

    for i in bins_values:
        histograms, labels = lectura_and_lbp(folder_parkinson, folder_no_parkinson, i)
        val_accuracy, test_accuracy, tiempo = sistema_lineal_sobredeterminista(histograms, labels)
        print(f"Precisión usando SLS con {i} bins en el conjunto de validacion: {val_accuracy * 100:.2f}%, conjunto de test: {test_accuracy * 100:.2f}%, Tiempo de entrenamiento: {tiempo:.5f} segundos")