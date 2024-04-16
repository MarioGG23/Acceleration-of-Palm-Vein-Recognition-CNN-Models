import onnx
      
model = onnx.load('/home/alumnos/mgonzalez/onnx_models/vera.onnx')
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
onnx.save(model, '/home/alumnos/mgonzalez/onnx_models/vera_dynamic_batch.onnx')
onnx.checker.check_model(model)
#######################################################################################################################################
# # Carga el modelo ONNX
# model_path = 'C:\\Users\FALABELLA\Desktop\onnx_models\casia_dinamic.onnx'
# model = onnx.load(model_path)

# # Imprime el nombre de cada capa
# for i, node in enumerate(model.graph.node):
#     print(f'Capa {i+1}: {node.name}')
##############################################################################################################################
# import tensorflow as tf

# # Carga el modelo HDF5
# model_path = 'C:\\Users\FALABELLA\Desktop\palmvein-best_models\casia-best_model.h5'
# model = tf.keras.models.load_model(model_path)

# # Obtiene el nombre de la capa de entrada
# input_layer_name = model.layers[0].name

# print('Nombre de la capa de entrada:', input_layer_name)

###########################################################################################################################################

# import onnxruntime as ort

# # Cargar el modelo
# model = ort.InferenceSession('C:\\Users\FALABELLA\Desktop\onnx_models\casia_dinamic.onnx')

# # Obtener el nombre de la capa de entrada
# input_name = model.get_inputs()[0].name

# print("Nombre de la capa de entrada:", input_name)