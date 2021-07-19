XML_DIR = 'dataset/bounding_boxes'

TRAIN_RECORD = 'dataset/tf data/training.record'
TEST_RECORD = 'dataset/tf data/testing.record'
CLASSES_FILE = 'dataset/tf data/classes.pbtxt'

CLASSES = {
    'Bulbasaur': 1,
    'Charmander': 2,
    'Squirtle': 3,
    'Pikachu': 4,
}

BASE = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

MODEL = f'model/exported model/{BASE}/saved_model'
NUM_CLASSES = 4
MIN_CONFIDENCE = 0.5

# TFLite
SAVED_MODEL = f'model/tflite saved model/{BASE}/saved_model'
TFLITE_MODEL = f'model/tflite saved model/{BASE}/model.tflite'
LABELS_TXT = f'dataset/tf data/labelmap.txt'
TFLITE_WITH_METADATA = f'model/tflite saved model/{BASE}/detect.tflite'