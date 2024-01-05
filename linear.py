from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])
tensor3 = tf.ones([3, 1, 2])

# Essentially the 1 represents the outer number of lists
# The 2 represents the number of lists within the outer list
# The 3 represents the number of elements in each of the inner lists

print(tensor1)
print(tensor2)
print(tensor3)

# Load dataset.
dftrain = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # gets a list of all unique values from given feature column
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))

print(feature_columns)


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True,
                  batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Creating the Linear Model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the model
linear_est.train(train_input_fn)

# Getting the model metrics/stats by testing on the test data
result = linear_est.evaluate(eval_input_fn)

clear_output()

# Printing the model summary
print(result['accuracy'])

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[4])
print(y_eval.loc[4])
print(result[4]['probabilities'][1])
