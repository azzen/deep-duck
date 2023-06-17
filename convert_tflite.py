import os
import tensorflow as tf
from tensorflow import keras
from tflite_support.metadata_writers import image_classifier, writer_utils

LABEL_FILE = "labels-export.txt"
SAVE_TO_PATH = "model.tflite"
MODEL_PATH = 'saved_model'
class_names = [
    "Male allier white duck",
    "Female goosander",
    "Male gadwall",
    "Male Mallard Duck",
    "Male Mandarin duck",
    "Male Northern shoveler",
    "Male tufted duck",
    "Male Whistling duck",
]

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_FILE, 'w') as label_file:
    for label in class_names: # /!\ class_names variable must contains the names of the labels you have.
        label_file.write("{}\n".format(label))

tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()

with open(SAVE_TO_PATH, 'wb') as f:
    f.write(tflite_model)
    
ImageClassifierWriter = image_classifier.MetadataWriter

INPUT_NORM_MEAN = 127.5
INPUT_NORM_STD = 127.5

writer = ImageClassifierWriter.create_for_inference(
    writer_utils.load_file(SAVE_TO_PATH),
    [INPUT_NORM_MEAN],
    [INPUT_NORM_STD],
    [LABEL_FILE]
)

print(writer.get_metadata_json())

writer_utils.save_file(writer.populate(), SAVE_TO_PATH)