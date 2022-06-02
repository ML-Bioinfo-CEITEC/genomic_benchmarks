import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MaxPooling1D,
)
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

character_split_fn = lambda x: tf.strings.unicode_split(x, "UTF-8")
vectorize_layer = TextVectorization(output_mode="int", split=character_split_fn)

# one-hot encoding
onehot_layer = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.cast(x, "int64"), 4))

def get_basic_cnn_model_v0(num_classes):

    if num_classes == 2:
        num_classes = 1

    model = tf.keras.Sequential(
        [
            onehot_layer,
            Conv1D(32, kernel_size=8, data_format="channels_last", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(),
            Conv1D(16, kernel_size=8, data_format="channels_last", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(),
            Conv1D(4, kernel_size=8, data_format="channels_last", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(),
            Dropout(0.3),
            GlobalAveragePooling1D(),
            Dense(num_classes), # we don't need softmax, because entropy uses argmax and argmax of logits and probabilities are same.
        ]
    )

    if num_classes == 1:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        acc = tf.metrics.BinaryAccuracy()
        f1 = tfa.metrics.F1Score(num_classes=1, threshold=0.5, average="micro")
    else:
        loss = 'categorical_crossentropy'
        acc = tf.metrics.CategoricalAccuracy()
        f1 = tfa.metrics.F1Score(num_classes=num_classes, average="micro")

    model.compile(
        loss=loss,
        optimizer='adam',
        metrics=[acc, f1]
    )

    return model

