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

def lectura_and_lbp(folder_parkinson, folder_no_parkinson, bins, folder_parkinson_test, folder_no_parkinson_test):
    histograms = []
    labels = []

    histograms_test = []
    labels_test = []

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

    for filename in os.listdir(folder_parkinson_test):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_parkinson_test, filename)
            lbp_histogram = calculate_lbp_with_bins(image_path, bins)
            histograms_test.append(lbp_histogram)
            labels_test.append(1)

    for filename in os.listdir(folder_no_parkinson_test):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_no_parkinson_test, filename)
            lbp_histogram = calculate_lbp_with_bins(image_path, bins)
            histograms_test.append(lbp_histogram)
            labels_test.append(0)

    return histograms, labels, histograms_test, labels_test

def sistema_lineal_sobredeterminista(X, y, X_T, y_T):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    X1 = np.array(X_train)
    y1 = np.array(y_train)
    start_time = time.time()
    coefficients1 = np.linalg.pinv(X1) #Encontrar coeficientes del modelo lineal, multiplico por y1 para ajustar el modelo lineal a los datos
    coefficients =  np.dot(coefficients1, y1)
    end_time = time.time()
    X_test = np.array(X_test)
    y_pred = (X_test.dot(coefficients) >= 0.5).astype(int) #Compara cada valor en el vector de predicciones con el valor 0.5, creando una matriz de valores booleanos, despues se pasa a enteros con la logica true=1 y flase = 0
    accuracy = accuracy_score(y_test, y_pred)
    training_time = end_time - start_time

    X_T = np.array(X_T)
    y_P = (X_T.dot(coefficients) >= 0.5).astype(int) 
    accuracy_test = accuracy_score(y_T, y_P)
    return accuracy, training_time, accuracy_test


if __name__ == "__main__":

    folder_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris_test/Enfermos"
    folder_no_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris_test/Sanos"
    folder_parkinson_test = "/home/alumnos/mgonzalez/parkinson/Gris_test/test_Enfermos"
    folder_no_parkinson_test = "/home/alumnos/mgonzalez/parkinson/Gris_test/test_Sanos"
    
    bins_values = [10, 20, 40, 60, 80, 100]

    for i in bins_values:
        histograms, labels, histograms_test, labels_test = lectura_and_lbp(folder_parkinson, folder_no_parkinson, i, folder_no_parkinson_test, folder_no_parkinson_test)
        accuracy, tiempo, accuracy_test = sistema_lineal_sobredeterminista(histograms, labels, histograms_test, labels_test)
        print(f"Precisión usando SLS con {i} bins en el conjunto de validacion: {accuracy * 100:.2f}%, Tiempo de entrenamiento: {tiempo:.5f} segundos, Precisión en el conjunto de test: {accuracy_test * 100:.2f}%")