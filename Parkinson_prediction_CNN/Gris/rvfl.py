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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rvfl(histograms, labels):

    X = np.array(histograms)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    hidden_neurons = 10
    start_time = time.time()
    input_weights = np.random.rand(X_train.shape[1], hidden_neurons)
    hidden_layer_output = sigmoid(np.dot(X_train, input_weights))
    #hidden_layer_output = np.tanh(np.dot(X_train, input_weights))
    output_weights = np.dot(np.linalg.pinv(np.concatenate((X_train, hidden_layer_output), axis=1)), y_train)# creo que al multiplicar se ajusta la relación entre las características de entrada y las salidas deseadas.
    end_time = time.time()

    #Validación
    val_hidden_layer_output = sigmoid(np.dot(X_val, input_weights))
    #test_hidden_layer_output = np.tanh(np.dot(X_test, input_weights))
    predicted_output = np.dot(np.concatenate((X_val, val_hidden_layer_output), axis=1), output_weights)
    val_accuracy = accuracy_score(y_val, (predicted_output > 0.5).astype(int))

    #Test 
    test_hidden_layer_output = sigmoid(np.dot(X_test, input_weights))
    #test_hidden_layer_output = np.tanh(np.dot(X_test, input_weights))
    predicted_output = np.dot(np.concatenate((X_test, test_hidden_layer_output), axis=1), output_weights)
    test_accuracy = accuracy_score(y_test, (predicted_output > 0.5).astype(int))
    training_time = end_time - start_time
    return val_accuracy, test_accuracy, training_time

if __name__ == "__main__":

    folder_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Enfermos"
    folder_no_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Sanos"

    bins_values = [10, 20, 40, 60, 80, 100]

    for i in bins_values:
        histograms, labels = lectura_and_lbp(folder_parkinson, folder_no_parkinson, i)
        val_accuracy, test_accuracy, tiempo = rvfl(histograms, labels)
        print(f"Precisión usando RVFL con {i} bins en el conjunto de validacion: {val_accuracy * 100:.2f}%, conjunto de test: {test_accuracy * 100:.2f}%, Tiempo de entrenamiento: {tiempo:.5f} segundos")