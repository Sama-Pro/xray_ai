import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm

# -----------------------------
# 1. GENERATE HEATMAP (SAFE)
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Convert to numpy safely
    heatmap = heatmap.numpy()

    # Remove invalid values
    heatmap = np.nan_to_num(heatmap)

    # Normalize (VERY IMPORTANT for Render stability)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap


# -----------------------------
# 2. SAVE HEATMAP IMAGE (SAFE)
# -----------------------------
def save_gradcam(img_path, heatmap, output_path, alpha=0.4):

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert to colormap safely
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3]

    heatmap = np.uint8(255 * heatmap)

    # Blend images
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    cv2.imwrite(output_path, superimposed_img)