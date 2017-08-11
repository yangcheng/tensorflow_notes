from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import numpy as np
import tensorflow as tf

# data sets

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, 'wb') as f:
            f.write(raw)
    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, 'wb') as f:
            f.write(raw)

    # load datasets
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    print(feature_columns)

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir='/tmp/iris_model')

    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)

        return x, y

    classifier.fit(input_fn=get_train_inputs, steps=2000)

    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)

        return x, y
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    def new_samples():
    	return np.array([[6.4, 3.2, 4.5, 1.5],[5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    prediction = list(classifier.predict(input_fn = new_samples))

    print('New samples, : {}\n'.format(prediction))

if __name__ == "__main__":
    main()
