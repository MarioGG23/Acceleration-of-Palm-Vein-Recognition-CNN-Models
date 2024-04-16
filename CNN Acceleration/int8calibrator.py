import tensorrt as trt
import numpy as np

# Definir la ruta del modelo ONNX
model_path = '/home/alumnos/mgonzalez/onnx_models/casia.onnx'

# Definir el conjunto de datos de calibración
calibration_data = [...]  # Datos de calibración

# Crear una clase personalizada de calibrador INT8
class EngineCalibrator(trt.IInt8Calibrator):
    def __init__(self, calibration_data):
        trt.IInt8Calibrator.__init__(self)
        self.calibration_data = calibration_data
        self.batch_index = 0

    def get_batch_size(self):
        return len(self.calibration_data)

    def get_batch(self, names):
        batch_data = self.calibration_data[self.batch_index]
        self.batch_index += 1
        return [batch_data]

    def read_calibration_cache(self, length):
        return None

    def write_calibration_cache(self, cache, length):
        pass

# Crear una instancia de logger
logger = trt.Logger(trt.Logger.WARNING)

# Crear un builder de TensorRT
builder = trt.Builder(logger)

# Crear una configuración de construcción para TensorRT
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MiB
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.INT8)

# Crear una red TensorRT
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Crear un analizador ONNX para poblar la red
parser = trt.OnnxParser(network, logger)

# Leer el archivo del modelo ONNX y procesar errores
success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not success:
    pass  # Manejo de errores aquí

# Crear el calibrador INT8
calibrator = EngineCalibrator(calibration_data)
config.int8_calibrator = calibrator

# Construir el motor TensorRT
engine = builder.build_engine(network, config)

# Guardar el motor en un archivo para uso futuro
serialized_engine = engine.serialize()
with open('sample.engine', 'wb') as f:
    f.write(serialized_engine)