import config.C as C
import tensorflow as tf
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(C.SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model
with open(C.TFLITE_MODEL, 'wb') as f:
    f.write(tflite_model)

writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file(C.TFLITE_MODEL), input_norm_mean=[0],
    input_norm_std=[255], label_file_paths=[C.LABELS_TXT])
writer_utils.save_file(writer.populate(), C.TFLITE_WITH_METADATA)