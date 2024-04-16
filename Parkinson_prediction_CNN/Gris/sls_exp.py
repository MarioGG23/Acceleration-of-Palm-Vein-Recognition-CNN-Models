import os
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# def calculate_lbp_with_bins(image_path, bins):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     radius = 1 
#     n_points = 8 * radius 
#     lbp_image = feature.local_binary_pattern(image, n_points, radius, method='uniform')

#     lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=bins, range=(0, n_points+2))
#     lbp_hist = lbp_hist.astype("float")
#     lbp_hist /= (lbp_hist.sum())


#     return lbp_hist

def calculate_lbp_with_bins(image_path, bins):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    radius = 1
    n_points = 100 * radius 
    lbp_image = feature.local_binary_pattern(image, n_points, radius, method='uniform') # default, ror

    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=bins, range=(0, n_points+2), density=True)
    lbp_hist = lbp_hist.astype("float")

    #Normalizar el histograma
    lbp_hist /= (lbp_hist.sum() + 1e-10)  # Añadir un pequeño valor epsilon para evitar la división por cero

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

def generate_quadratic_features(X):
    X_squared = np.power(X, 2)
    return np.concatenate((X, X_squared), axis=1)

def generate_cubic_features(X):
    X_squared = np.power(X, 2)
    X_cubed = np.power(X, 3)
    return np.concatenate((X, X_squared, X_cubed), axis=1)

def generate_quartic_features(X):
    X_squared = np.power(X, 2)
    X_cubed = np.power(X, 3)
    X_quartic = np.power(X, 4)
    return np.concatenate((X, X_squared, X_cubed, X_quartic), axis=1)



def sistema_no_lineal_cuadratico(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # X_train = generate_quartic_features(X_train)
    # X_val = generate_quartic_features(X_val)
    # X_test = generate_quartic_features(X_test)
    
    X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
    X_val = np.column_stack((X_val, np.ones(X_val.shape[0])))
    X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))

    coefficients1 = np.linalg.pinv(X_train) # Encontrar coeficientes del modelo 
    coefficients = np.dot(coefficients1, y_train)

    # Calcular la precisión en el conjunto de validación
    y_val_pred = (X_val.dot(coefficients) >= 0.5).astype(int)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Calcular la precisión en el conjunto de prueba
    y_test_pred = (X_test.dot(coefficients) >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return val_accuracy, test_accuracy

if __name__ == "__main__":
    prueba = []
    folder_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Enfermos"
    folder_no_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Sanos"

    bins_values = [10, 20, 40, 60, 80, 100]
    #bins_values = [10] 
    for i in bins_values:
        histograms, labels = lectura_and_lbp(folder_parkinson, folder_no_parkinson, i)
        histograms = np.array(histograms)
        val_accuracy, test_accuracy = sistema_no_lineal_cuadratico(histograms, labels)
        print(f"Precisión usando sistema no lineal con {i} bins en el conjunto de validación: {val_accuracy * 100:.2f}%, conjunto de test: {test_accuracy * 100:.2f}%")
        prueba.append(histograms[0])
    
    for i in prueba:
        print(i)
        print("\n")