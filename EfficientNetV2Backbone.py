import tensorflow as tf
from tensorflow import keras
import keras_cv.models

def build_model(CFG, kbest_features):
    # image input
    image_input = tf.keras.layers.Input(name='images',shape=(*CFG.target_size,3))
    # Features Input
    features_input = tf.keras.layers.Input(name='features',shape=(len(kbest_features),)) 
    # Defining Backbone
    resbackbone = keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b2_imagenet")
    resbackbone.trainable = False
    # Layer = Image_input -> backbone
    image_layer = resbackbone(image_input)
    # Image_input -> backbone -> Global Layer
    global_layer = tf.keras.layers.GlobalAveragePooling2D(name='global_layer')(image_layer)
    # Feature input -> Dense 1
    dense1 = tf.keras.layers.Dense(256, activation='relu', name='Dense_1_Layer')(features_input)
    # Image_input -> backbone -> Dropout 
    dropout1 = tf.keras.layers.Dropout(0.3, name="Dropout_1")(dense1)
    # Image_input -> backbone -> Dropout -> Dense 2
    dense2 = tf.keras.layers.Dense(128, activation='relu', name='Dense_2_Layer')(dropout1)
    # Concat Image input and 
    concat_layer = tf.keras.layers.Concatenate(name='Concatenate')([dense2, global_layer])
    x = keras.layers.Dense(256, activation="relu", name="Dense_3_Layer")(concat_layer)
    x = keras.layers.Dropout(0.2, name="Dropout_2")(x)
    # Output layers
    out1 = keras.layers.Dense(CFG.num_classes, activation="softmax", name="head")(x)
    out2 = keras.layers.Dense(CFG.aux_num_classes, activation="softmax", name="aux_head")(x)
    out = {"head": out1, "aux_head": out2}
    model = tf.keras.Model([image_input, features_input], out)
    return model

# CFG = YourConfigurationClass()
# kbest_features = YourFeatureSelectionMethod()

model = build_model(CFG, kbest_features)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss={"head": R2Loss(use_mask=False),"aux_head": R2Loss(use_mask=True),},
    metrics={"head": R2Metric()}
)
# garbage collect
import gc
gc.collect()

