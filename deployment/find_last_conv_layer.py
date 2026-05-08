from tensorflow.keras.models import load_model

# =========================
# MODEL PATH
# =========================

MODEL_PATH = r"C:\Users\sakia\Pictures\xray_ai\model\best_models.h5"

print(f"\n🚀 Loading model:\n{MODEL_PATH}\n")

try:
    model = load_model(MODEL_PATH)

    print("✅ MODEL LOADED SUCCESSFULLY!\n")

    print("🔍 Finding LAST convolution layer...\n")

    found = False

    # Search from end of model
    for layer in reversed(model.layers):

        layer_type = layer.__class__.__name__

        if "Conv" in layer_type:

            print("🔥 LAST CONVOLUTION LAYER FOUND:\n")

            print(f"Layer Name : {layer.name}")
            print(f"Layer Type : {layer_type}")

            found = True
            break

    if not found:
        print("❌ No convolution layer found.")

except Exception as e:
    print(f"\n❌ ERROR:\n{e}")