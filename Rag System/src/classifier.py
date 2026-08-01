import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB0
from src.config import MODEL_PATH, CLASS_NAMES, IMG_SIZE


def build_model_architecture():
    
    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights=None   # we are loading OUR trained weights, not ImageNet ones
    )

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    return model


def load_classifier():
    
    model = build_model_architecture()
    model.load_weights(str(MODEL_PATH))

    return model


def predict_style(model, image_path):


    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image).astype("float32")

    image_batch = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_batch, verbose=0)

    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)
    predicted_style = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    top_3 = []
    sorted_indices = np.argsort(probabilities)[::-1]

    for i in range(3):
        index = sorted_indices[i]
        style_name = CLASS_NAMES[index]
        style_confidence = float(probabilities[index])
        top_3.append((style_name, style_confidence))

    result = {
        "style": predicted_style,
        "confidence": confidence,
        "top_3": top_3
    }

    return result


# Quick test - only runs if this file is executed directly
if __name__ == "__main__":

    print("Loading model...")
    model = load_classifier()
    print("Model loaded.\n")

    TEST_IMAGE_PATH = "test_image.jpg"

    result = predict_style(model, TEST_IMAGE_PATH)

    print("Predicted style:", result["style"])
    print("Confidence:", round(result["confidence"] * 100, 1), "%")
    print("\nTop 3 candidates:")
    for style, conf in result["top_3"]:
        print(" ", style, "-", round(conf * 100, 1), "%")