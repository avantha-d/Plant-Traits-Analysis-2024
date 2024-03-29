import os
import tensorflow as tf
import keras_cv
import pandas as pd
from tqdm.notebook import tqdm

# Set Keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Configuration settings
class CFG:
    verbose = 1
    seed = 42
    preset = "efficientnetv2_b2_imagenet"
    image_size = [224, 224]
    epochs = 12
    batch_size = 96
    lr_mode = "step"
    drop_remainder = True
    num_classes = 6
    num_folds = 5
    fold = 0
    class_names = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
    aux_class_names = [name.replace("mean", "sd") for name in class_names]
    aux_num_classes = len(aux_class_names)

# Set random seed
tf.random.set_seed(CFG.seed)

# Data paths
BASE_PATH = "/kaggle/input/planttraits2024"
train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
train_df['image_path'] = f'{BASE_PATH}/train_images/' + train_df['id'].astype(str) + '.jpeg'
train_df.loc[:, CFG.aux_class_names] = train_df.loc[:, CFG.aux_class_names].fillna(-1)

test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
test_df['image_path'] = f'{BASE_PATH}/test_images/' + test_df['id'].astype(str) + '.jpeg'

# Display data samples
display(train_df.head(2))
display(test_df.head(2))

# Function to build augmentation pipeline
def build_augmenter():
    aug_layers = [
        keras_cv.layers.RandomBrightness(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomContrast(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomSaturation(factor=(0.45, 0.55)),
        keras_cv.layers.RandomHue(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.15), width_factor=(0.06, 0.15)),
        keras_cv.layers.RandomFlip(mode="horizontal_and_vertical"),
        keras_cv.layers.RandomZoom(height_factor=(0.05, 0.15)),
        keras_cv.layers.RandomRotation(factor=(0.01, 0.05)),
    ]
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    augmenter = keras_cv.layers.Augmenter(aug_layers)
    
    def augment(inp, label):
        images = inp["images"]
        aug_data = {"images": images}
        aug_data = augmenter(aug_data)
        inp["images"] = aug_data["images"]
        return inp, label
    return augment

# Function to decode image and labels
def build_decoder(with_labels=True, target_size=CFG.image_size):
    def decode_image(inp):
        path = inp["images"]
        file_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(file_bytes)
        image = tf.image.resize(image, size=target_size, method="area")
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [*target_size, 3])
        inp["images"] = image
        return inp

    def decode_label(label, num_classes):
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [num_classes])
        return label

    def decode_with_labels(inp, labels=None):
        inp = decode_image(inp)
        label = decode_label(labels[0], CFG.num_classes)
        aux_label = decode_label(labels[1], CFG.aux_num_classes)
        return (inp, (label, aux_label))

    return decode_with_labels if with_labels else decode_image

def prepare_data(df, CFG):
    # Create separate bin for each trait
    for i, trait in enumerate(CFG.class_names):
        bin_edges = np.percentile(df[trait], np.linspace(0, 100, CFG.num_folds + 1))
        df[f"bin_{i}"] = np.digitize(df[trait], bin_edges)
    
    # Concatenate bins into a final bin
    df["final_bin"] = df[[f"bin_{i}" for i in range(len(CFG.class_names))]].astype(str).agg("".join, axis=1)
    df = df.reset_index(drop=True)
    
    # Perform stratified split using final bin
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df["final_bin"])):
        df.loc[valid_idx, "fold"] = fold
    
    return df

def preprocess_data(df, CFG):
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(df[FEATURE_COLS].values)
    
    # Split data into training and validation sets
    train_df = df[df.fold != CFG.fold]
    valid_df = df[df.fold == CFG.fold]

    return train_df, valid_df, features

def create_datasets(paths, features, labels, aux_labels, CFG, augment=False, cache_dir=""):
    train_ds = build_dataset(paths, features, labels, aux_labels,
                             batch_size=CFG.batch_size,
                             repeat=True, shuffle=True, augment=augment, cache_dir=cache_dir)
    
    valid_ds = build_dataset(paths, features, labels, aux_labels,
                             batch_size=CFG.batch_size,
                             repeat=False, shuffle=False, augment=False, cache_dir=cache_dir)
    
    return train_ds, valid_ds

def create_model(CFG):
    # Define input layers
    img_input = keras.Input(shape=(*CFG.image_size, 3), name="images")
    feat_input = keras.Input(shape=(len(FEATURE_COLS),), name="features")

    # Branch for image input
    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(CFG.preset)
    x1 = backbone(img_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)

    # Branch for tabular/feature input
    x2 = keras.layers.Dense(326, activation="selu")(feat_input)
    x2 = keras.layers.Dense(64, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)

    # Concatenate both branches
    concat = keras.layers.Concatenate()([x1, x2])

    # Output layer
    out1 = keras.layers.Dense(CFG.num_classes, activation=None, name="head")(concat)
    out2 = keras.layers.Dense(CFG.aux_num_classes, activation="relu", name="aux_head")(concat)
    out = {"head": out1, "aux_head":out2}

    # Build model
    model = keras.models.Model([img_input, feat_input], out)
    return model

def compile_model(model, CFG):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "head": R2Loss(use_mask=False),
            "aux_head": R2Loss(use_mask=True), # use_mask to ignore `NaN` auxiliary labels
        },
        loss_weights={"head": 1.0, "aux_head": 0.3},  # more weight to main task
        metrics={"head": R2Metric()} # evaluation metric only on main task
    )
    return model

# Prepare data
df = pd.read_csv(f'{BASE_PATH}/train.csv')
df['image_path'] = f'{BASE_PATH}/train_images/' + df['id'].astype(str) + '.jpeg'
df.loc[:, CFG.aux_class_names] = df.loc[:, CFG.aux_class_names].fillna(-1)
df = prepare_data(df, CFG)

# Preprocess data
train_df, valid_df, features = preprocess_data(df, CFG)

# Create datasets
train_ds, valid_ds = create_datasets(train_df.image_path.values, features,
                                      train_df[CFG.class_names].values,
                                      train_df[CFG.aux_class_names].values, CFG, augment=True)

# Create and compile model
model = create_model(CFG)
model = compile_model(model, CFG)

# Train model
history = model.fit(
    train_ds,
    epochs=CFG.epochs,
    callbacks=[lr_cb, ckpt_cb],
    steps_per_epoch=len(train_df) // CFG.batch_size,
    validation_data=valid_ds,
    verbose=CFG.verbose
)
