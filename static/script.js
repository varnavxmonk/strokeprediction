function predictStroke() {
    // Collect input values from the form
    const data = {
        age: document.getElementById('age').value,
        hypertension: document.getElementById('hypertension').value,
        heart_disease: document.getElementById('heart_disease').value,
        avg_glucose_level: document.getElementById('avg_glucose_level').value,
        bmi: document.getElementById('bmi').value,
        gender: document.getElementById('gender').value,
        ever_married: document.getElementById('ever_married').value,
        work_type: document.getElementById('work_type').value,
        Residence_type: document.getElementById('Residence_type').value,  // Match backend key
        smoking_status: document.getElementById('smoking_status').value
    };

    // Show "Processing..." message
    document.getElementById("predictionOutput").innerHTML = "Processing...";

    // Send data to the backend for prediction
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        // Display the prediction result
        const predictionOutput = document.getElementById("predictionOutput");
        predictionOutput.innerHTML = result.stroke_prediction;

        // Add animation for better UX
        predictionOutput.classList.add("fade-in");
        setTimeout(() => predictionOutput.classList.remove("fade-in"), 1000);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("predictionOutput").innerHTML = "An error occurred. Please try again.";
    });
}