async function predict() {

    const input = document.getElementById("imageInput");

    const file = input.files[0];

    if (!file) {

        alert("Please select an image");

        return;
    }

    // -----------------------------
    // ORIGINAL IMAGE PREVIEW
    // -----------------------------
    const preview = document.getElementById("preview");

    preview.src = URL.createObjectURL(file);

    // -----------------------------
    // SEND TO BACKEND
    // -----------------------------
    const formData = new FormData();

    formData.append(
        "image",
        file
    );

    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        // -----------------------------
        // ERROR CHECK
        // -----------------------------
        if (data.error) {

            alert(data.error);

            return;
        }

        // -----------------------------
        // PREDICTION
        // -----------------------------
        document.getElementById("result").innerText =
            "Prediction: " + data.prediction;

        // -----------------------------
        // CONFIDENCE (FIXED)
        // -----------------------------
        document.getElementById("confidence").innerText =
            "Confidence: " + data.confidence.toFixed(2) + "%";

        // -----------------------------
        // GRAD-CAM DISPLAY
        // -----------------------------
        const gradcamImage = document.getElementById("gradcam");

        gradcamImage.src =
            data.gradcam + "?t=" + new Date().getTime();

    } catch (error) {

        console.error(error);

        alert("Error connecting to API");
    }
}