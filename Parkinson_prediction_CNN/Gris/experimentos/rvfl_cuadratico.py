import os
import cv2
import numpy as np
from skimage import feature
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

def generate_quadratic_features(X):

    X_squared = np.power(X, 2)
    return np.concatenate((X, X_squared), axis=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rvfl_quadratic(histograms, labels):

    X = np.array(histograms)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Genera características cuadráticas
    X_train_quadratic = generate_quadratic_features(X_train)
    X_val_quadratic = generate_quadratic_features(X_val)
    X_test_quadratic = generate_quadratic_features(X_test)

    hidden_neurons = 10
    start_time = time.time()
    
    # Genera pesos de entrada de forma aleatoria
    input_weights = np.random.rand(X_train_quadratic.shape[1], hidden_neurons)
    
    # Capa oculta
    hidden_layer_output = sigmoid(np.dot(X_train_quadratic, input_weights))
    
    # Concatena características originales y de la capa oculta
    concatenated_features_train = np.concatenate((X_train_quadratic, hidden_layer_output), axis=1)
    
    # Calcula los pesos de salida utilizando la pseudoinversa
    output_weights = np.dot(np.linalg.pinv(concatenated_features_train), y_train)
    end_time = time.time()

    # Validación
    hidden_layer_output_val = sigmoid(np.dot(X_val_quadratic, input_weights))
    concatenated_features_val = np.concatenate((X_val_quadratic, hidden_layer_output_val), axis=1)
    predicted_output = np.dot(concatenated_features_val, output_weights)
    val_accuracy = accuracy_score(y_val, (predicted_output > 0.5).astype(int))

    # Test
    hidden_layer_output_test = sigmoid(np.dot(X_test_quadratic, input_weights))
    concatenated_features_test = np.concatenate((X_test_quadratic, hidden_layer_output_test), axis=1)
    predicted_output = np.dot(concatenated_features_test, output_weights)
    test_accuracy = accuracy_score(y_test, (predicted_output > 0.5).astype(int))

    training_time = end_time - start_time
    return val_accuracy, test_accuracy, training_time

if __name__ == "__main__":
    folder_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Enfermos"
    folder_no_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Sanos"

    bins_values = [10, 20, 40, 60, 80, 100]

    for i in bins_values:
        histograms, labels = lectura_and_lbp(folder_parkinson, folder_no_parkinson, i)
        val_accuracy, test_accuracy, tiempo = rvfl_quadratic(histograms, labels)
        print(f"Precisión usando RVFL con sistema no lineal cuadrático y {i} bins en el conjunto de validación: {val_accuracy * 100:.2f}%, conjunto de test: {test_accuracy * 100:.2f}%, Tiempo de entrenamiento: {tiempo:.5f} segundos")
