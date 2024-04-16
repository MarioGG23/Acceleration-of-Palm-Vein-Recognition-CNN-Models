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

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def elm_quadratic(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
    # X_val = np.column_stack((X_val, np.ones(X_val.shape[0])))
    # X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))

    X_train_quadratic = generate_quadratic_features(X_train)
    X_val_quadratic = generate_quadratic_features(X_val)
    X_test_quadratic = generate_quadratic_features(X_test)

    hidden_units = 10
    start_time = time.time()

    input_weights = np.random.rand(X_train_quadratic.shape[1], hidden_units)

    biases = np.random.rand(hidden_units)

    hidden_layer_input_train = np.dot(X_train_quadratic, input_weights) + biases
    hidden_layer_output_train = sigmoide(hidden_layer_input_train)

    output_weights = np.dot(np.linalg.pinv(hidden_layer_output_train), y_train)
    end_time = time.time()

    hidden_layer_input_test = np.dot(X_val_quadratic, input_weights) + biases
    hidden_layer_output_test = sigmoide(hidden_layer_input_test)

    predicted_output = np.dot(hidden_layer_output_test, output_weights)
    val_accuracy = accuracy_score(y_val, (predicted_output > 0.5).astype(int))
    training_time = end_time - start_time

    hidden_layer_input_test = np.dot(X_test_quadratic, input_weights) + biases
    hidden_layer_output_test = sigmoide(hidden_layer_input_test)

    predicted_output = np.dot(hidden_layer_output_test, output_weights)
    test_accuracy = accuracy_score(y_test, (predicted_output > 0.5).astype(int))

    return val_accuracy, test_accuracy, training_time

if __name__ == "__main__":
    folder_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Enfermos"
    folder_no_parkinson = "/home/alumnos/mgonzalez/parkinson/Gris/Sanos"

    bins_values = [10, 20, 40, 60, 80, 100]

    for i in bins_values:

        histograms, labels = lectura_and_lbp(folder_parkinson, folder_no_parkinson, i)
        X = np.array(histograms)
        y = np.array(labels)
        val_accuracy, test_accuracy, tiempo = elm_quadratic(X, y)
        print(f"Precisión usando ELM con sistema no lineal cuadrático y {i} bins en el conjunto de validación: {val_accuracy * 100:.2f}%, conjunto de test: {test_accuracy * 100:.2f}%, Tiempo de entrenamiento: {tiempo:.5f} segundos")
