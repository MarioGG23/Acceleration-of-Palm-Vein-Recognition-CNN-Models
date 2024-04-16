from tensorflow.python.compiler.tensorrt import trt_convert as trt
import premio_nobel_dela_programacion_V4 as pnp

testdata, nclases, imgsize = pnp.lectura_datos('/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/CASIA_MS_Palmprint_v1/augmented_roi/850/clahe/', "CASIA")
# for i in testdata:
#     print(i)
# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir="/home/alumnos/mgonzalez/practica/modelosh5/casia_saved_model",
   precision_mode=trt.TrtPrecisionMode.INT8,
   use_calibration=True
)
 
# Use data from the test/validation set to perform INT8 calibration
BATCH_SIZE=1
NUM_CALIB_BATCHES=10
x_test = testdata
def calibration_input_fn():
   for i in range(NUM_CALIB_BATCHES):
       start_idx = i * BATCH_SIZE
       end_idx = (i + 1) * BATCH_SIZE
       x = x_test[start_idx:end_idx, :]
       yield [x]
 
# Convert the model with valid calibration data
func = converter.convert(calibration_input_fn=calibration_input_fn)
 
# Input for dynamic shapes profile generation
MAX_BATCH_SIZE=128
def input_fn():
   batch_size = MAX_BATCH_SIZE
   x = x_test[0:batch_size, :]
   yield [x]
 
# Build the engine
converter.build(input_fn=input_fn)
 
OUTPUT_SAVED_MODEL_DIR="/home/alumnos/mgonzalez/practica/modelosh5/casia_saved_model"
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)
 
converter.summary()
 
# Run some inferences!
# for step in range(10):
#    start_idx = step * BATCH_SIZE
#    end_idx   = (step + 1) * BATCH_SIZE
 
#    print(f"Step: {step}")
#    x = x_test[start_idx:end_idx, :]
#    func(x)