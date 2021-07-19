import config.C as C
from config.tfannotation import TFAnnotation
from bs4 import BeautifulSoup
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import io

def main(_):
    # Open the classes output file
    f = open(C.CLASSES_FILE, 'w')

    # Loop over the classes
    for (k, v) in C.CLASSES.items():
        # Construct the class information and write to file
        item = ('item {\n'
                    '\tid: ' + str(v) + '\n'
                    '\tname: "' + k + '"\n'
                    '}\n')
        f.write(item)
    
    # Close the output classes file
    f.close()

    # Generate train and test datasets
    X_train = []
    X_test = []
    for folder in os.listdir(C.XML_DIR):
        XML_LIST = []
        for xml_file in os.listdir(os.path.sep.join([C.XML_DIR, folder])):
            XML_LIST.append(os.path.sep.join([C.XML_DIR, folder, xml_file]))

        X_train_, X_test_ = train_test_split(
            XML_LIST, random_state=42, test_size=0.1
        )
        X_train.extend(X_train_)
        X_test.extend(X_test_)

    # Initialize the data split files
    datasets = [
        ('train', X_train, C.TRAIN_RECORD),
        ('test', X_test, C.TEST_RECORD)
    ]

    # Loop over the datasets
    for (dtype, xml_list, output_path) in datasets:
        print(f'[INFO] processing {dtype} set...')

        # Initialize the tensorflow writer and initialize the total
        # number of examples written to file
        writer = tf.io.TFRecordWriter(output_path)
        total = 0
            
        for xml in xml_list:
            # Build the soup
            contents = open(xml).read()
            soup = BeautifulSoup(contents, 'lxml')

            img_path = soup.find('path').text

            # Load the input image from disk as a TensorFlow object
            encoded = tf.io.gfile.GFile(img_path, 'rb').read()
            encoded_io = io.BytesIO(encoded)

            # Load the image from disk again, this time as a PIL
            # object
            img = Image.open(encoded_io)
            (w, h) = img.size[:2]

            # Parse the filename and encoding from the input path
            filename = img_path.split(os.path.sep)[-1]
            encoding = filename[filename.rfind('.') + 1:]
                
            # Initialize the annotation object used to store
            # information regarding the bounding box + labels
            tf_annot = TFAnnotation()
            tf_annot.image = encoded
            tf_annot.width = w
            tf_annot.height = h
            tf_annot.filename = filename
            tf_annot.encoding = encoding

            # Loop over the bounding boxes + labels associated with
            # the image
            for object in soup.find_all('object'):
                box = object.find('bndbox')

                # Extract the bounding box information + label,
                # ensuring that all bounding box dimensions fit
                # inside the image
                x_start = max(0, float(box.find('xmin').text))
                y_start = max(0, float(box.find('ymin').text))
                x_end = min(w, float(box.find('xmax').text))
                y_end = min(h, float(box.find('ymax').text))
                label = object.find('name').text

                # TensorFlow assumes all bounding boxes are in the
                # range [0, 1] so we need to scale them
                x_min = x_start / w
                x_max = x_end / w
                y_min = y_start / h
                y_max = y_end / h

                # Due to errors in annotation, it may be possible
                # that the minimum values are larger than the maximum
                # values -- in this case, treat it as an error during
                # annotation and ignore the bounding box
                if x_min > x_max or y_min > y_max:
                    continue
                    
                # Update the bounding boxes + labels lists
                tf_annot.x_mins.append(x_min)
                tf_annot.x_maxs.append(x_max)
                tf_annot.y_mins.append(y_min)
                tf_annot.y_maxs.append(y_max)
                tf_annot.text_labels.append(label.encode('utf8'))
                tf_annot.classes.append(C.CLASSES[label])
                    
                # Increment the total number of examples
                total += 1

            # Encode the data point attributes using the TensorFlow
            # helper functions
            features = tf.train.Features(feature=tf_annot.build())
            example = tf.train.Example(features=features)

            # Add the example to the writer
            writer.write(example.SerializeToString())

        # Close the writer and print diagnostic information to the
        # user
        writer.close()
        print(f'[INFO] {total} examples saved for {dtype} set')

# check to see if the main thread should be started
if __name__ == '__main__':
    tf.compat.v1.app.run()