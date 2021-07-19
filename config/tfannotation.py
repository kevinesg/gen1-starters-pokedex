from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import bytes_feature
from object_detection.utils.dataset_util import int64_feature

class TFAnnotation:
    def __init__(self):
        # Initialize the bounding box + label lists
        self.x_mins = []
        self.x_maxs = []
        self.y_mins = []
        self.y_maxs = []
        self.text_labels = []
        self.classes = []

        # Initialize additional variables, including the image
        # itself, spatial dimensions, encoding, and filename
        self.image = None
        self.width = None
        self.height = None
        self.filename = None
        self.encoding = None

    def build(self):
        # Construct the TensorFlow-compatible data dictionary
        data = {
            'image/height': int64_feature(self.height),
            'image/width': int64_feature(self.width),  
            'image/filename': bytes_feature(self.filename.encode('utf8')),
            'image/source_id': bytes_feature(self.filename.encode('utf8')),
            'image/encoded': bytes_feature(self.image),    
            'image/format': bytes_feature(self.encoding.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(self.x_mins),
            'image/object/bbox/xmax': float_list_feature(self.x_maxs),
            'image/object/bbox/ymin': float_list_feature(self.y_mins),
            'image/object/bbox/ymax': float_list_feature(self.y_maxs),     
            'image/object/class/text': bytes_list_feature(self.text_labels),
            'image/object/class/label': int64_list_feature(self.classes)
        }

        # Return the data dictionary
        return data