import config.C as C
import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import io
from PIL import Image   

def label_frames(image, model):
    cv2.imwrite('image.jpg', image)
    image = 'image.jpg'
    img_data = tf.io.gfile.GFile(image, 'rb').read()
    image = Image.open(io.BytesIO(img_data))
    image_np = np.array(image).astype(np.uint8)

    input_tensor = np.expand_dims(image_np, 0)
    detections = model(input_tensor)

    label_map = label_map_util.load_labelmap(C.CLASSES_FILE)
    categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=C.NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.4,
        agnostic_mode=False
    )
    image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
    cv2.imshow('Labeled Frames', image_np_with_detections)