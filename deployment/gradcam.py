import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm

# -----------------------------
# 1. GENERATE HEATMAP (With Threshold)
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

    # Normalize and apply the professional 60% threshold
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap[heatmap < 0.85] = 0 

    return heatmap

# -----------------------------
# 2. SAVE HEATMAP IMAGE (The missing function)
# -----------------------------
def save_gradcam(img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(img_path)
    
    # Resize heatmap to original image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply JET colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)

    # Superimpose heatmap onto the original X-ray
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    # Save the result
    cv2.imwrite(output_path, superimposed_img)