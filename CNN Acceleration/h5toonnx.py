import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

model_h5 = '/home/alumnos/mgonzalez/thadli/Modelos/pruned/casia.8sparsity.h5'
model = load_model(model_h5)