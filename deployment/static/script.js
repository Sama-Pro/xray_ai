async function predict() {
    const input = document.getElementById("imageInput");
    const file = input.files[0];

    if (!file) {
        alert("Please select an image");
        return;
    }

    // Show preview
    const preview = document.getElementById("preview");
    preview.src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("/predict", 
            {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        document.getElementById("result").innerText =
            "Prediction: " + data.prediction;

        document.getElementById("confidence").innerText =
            "Confidence: " + (data.confidence * 100).toFixed(2) + "%";

    } catch (error) {
        console.error(error);
        alert("Error connecting to API");
    }
}