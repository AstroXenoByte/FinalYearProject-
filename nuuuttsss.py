from flask import Flask, request, jsonify, render_template_string
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import json

app = Flask(__name__)

# Config
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Model classes
CLASSES = [
    "early_rust_1",
    "late_leaf_spot_1",
    "nutrition_deficiency_1",
    "healthy_leaf_1",
    "early_leaf_spot_1",
    "rust_1"
]

# Disease info database with weather-based recommendations
DISEASE_INFO = {
    "early_rust_1": {
        "name": "Early Rust",
        "causes": [
            "Moderate temperatures (20-25°C) with high humidity",
            "Frequent dew or rainfall",
            "Wind-borne spores from nearby infected plants",
            "Nutrient deficiency weakening plant immunity"
        ],
        "cure": [
            "Apply Triadimefon 25% WP @ 1g/L",
            "Use Hexaconazole 5% SC @ 2ml/L",
            "Improve air circulation in the field",
            "Remove and destroy infected leaves"
        ],
        "prevention": [
            "Plant rust-resistant varieties",
            "Maintain good irrigation practices",
            "Balance soil nutrients",
            "Regularly inspect crops for early signs"
        ],
        "weather_alerts": {
            "high_humidity": "High humidity detected! Increase monitoring for rust spread."
        }
    },
    "late_leaf_spot_1": {
        "name": "Late Leaf Spot",
        "causes": [
            "High humidity (>85%)",
            "Cool to moderate temperatures (20-25°C)",
            "Prolonged leaf wetness",
            "Dense plant canopy restricting airflow"
        ],
        "cure": [
            "Apply Chlorothalonil 75% WP @ 2g/L",
            "Use Tebuconazole + Sulphur mix",
            "Alternate between contact and systemic fungicides"
        ],
        "prevention": [
            "Use resistant cultivars",
            "Increase plant spacing",
            "Irrigate in mornings for faster drying",
            "Remove infected crop debris"
        ],
        "weather_alerts": {
            "high_humidity": "High humidity may worsen leaf spot. Avoid overhead irrigation."
        }
    },
    "nutrition_deficiency_1": {
        "name": "Nutritional Deficiency",
        "causes": [
            "Imbalanced or inadequate fertilization",
            "Soil pH too low or too high",
            "Poor organic matter in soil",
            "Waterlogged or excessively dry conditions"
        ],
        "cure": [
            "Conduct soil testing",
            "Apply correct fertilizers based on deficiency",
            "Use foliar sprays for quick correction"
        ],
        "prevention": [
            "Maintain optimal soil pH",
            "Add compost or manure regularly",
            "Follow recommended fertilizer schedules"
        ],
        "weather_alerts": {
            "high_rain": "Heavy rain may leach nutrients. Check soil health."
        }
    },
    "healthy_leaf_1": {
        "name": "Healthy Leaf",
        "causes": [
            "Optimal growing conditions",
            "Balanced nutrition",
            "Good soil health",
            "Effective pest and disease management"
        ],
        "cure": [
            "Continue good agricultural practices",
            "Regular monitoring for early problem detection"
        ],
        "prevention": [
            "Use high-quality seeds",
            "Maintain crop rotation",
            "Avoid over- or under-irrigation"
        ],
        "weather_alerts": {
            "high_humidity": "Healthy leaves detected, but monitor for fungal risks in humid conditions."
        }
    },
    "early_leaf_spot_1": {
        "name": "Early Leaf Spot",
        "causes": [
            "High humidity and warm temperatures",
            "Poor air circulation",
            "Overhead irrigation causing wet leaves",
            "Infected crop residues in soil"
        ],
        "cure": [
            "Apply copper-based fungicides",
            "Use Propiconazole 25% EC",
            "Remove infected leaves immediately"
        ],
        "prevention": [
            "Use disease-resistant varieties",
            "Improve plant spacing",
            "Switch to drip irrigation"
        ],
        "weather_alerts": {
            "high_humidity": "High humidity increases leaf spot risk. Use drip irrigation."
        }
    },
    "rust_1": {
        "name": "Advanced Rust",
        "causes": [
            "Untreated early rust progression",
            "Persistent humid conditions",
            "Delayed disease control measures"
        ],
        "cure": [
            "Apply Propiconazole 25% EC",
            "Combine systemic and contact fungicides",
            "Increase spraying frequency"
        ],
        "prevention": [
            "Early detection and treatment",
            "Monitor weather for rust-favoring conditions",
            "Maintain optimal plant health"
        ],
        "weather_alerts": {
            "high_humidity": "Advanced rust detected! Urgent action needed in humid conditions."
        }
    }
}

# Load model with error handling
try:
    model = tf.keras.models.load_model('Groundnut_disease_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_file):
    try:
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_class(image_array):
    try:
        if model is None:
            raise Exception("Model not loaded")
        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASSES[predicted_class_idx]
        info = DISEASE_INFO.get(predicted_class, {})
        return {
            "predicted_class": info.get("name", predicted_class),
            "confidence": confidence,
            "causes": info.get("causes", []),
            "cure": info.get("cure", []),
            "prevention": info.get("prevention", []),
            "all_confidences": [float(x) for x in predictions[0]],
            "weather_alerts": info.get("weather_alerts", {})
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


# Weather fetch function using WeatherAPI.com
def get_weather(city="eMpangeni"):
    try:
        api_key = "f3ac3e90c50e44c29cb141812252509"
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return {
                "temp": data["current"]["temp_c"],  # Celsius
                "humidity": data["current"]["humidity"],
                "weather": data["current"]["condition"]["text"],
                "rain": "rain" in data["current"]["condition"]["text"].lower()
            }
        else:
            return {"error": data.get("error", {}).get("message", "Failed to fetch weather")}
    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🌱 Groundnut Health Classifier</title>
        <!-- Bootstrap CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <!-- Chart.js -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <!-- jsPDF for report export -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
        <style>
            body {
                background: #343a40;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #ffffff;
                overscroll-behavior: none;
            }
            .container {
                max-width: 1200px;
                padding: 20px;
            }
            .card {
                border-radius: 20px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
                background: #2c3e50;
                color: #ffffff;
                margin-bottom: 20px;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6);
            }
            .btn-primary {
                background: #34c759;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-weight: 600;
                color: #ffffff;
                transition: background 0.3s ease, transform 0.2s ease;
            }
            .btn-primary:hover {
                background: #2db44f;
                transform: scale(1.05);
            }
            .btn-secondary {
                background: #4b5e6a;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-weight: 600;
                color: #ffffff;
                transition: background 0.3s ease, transform 0.2s ease;
            }
            .btn-secondary:hover {
                background: #3f4e57;
                transform: scale(1.05);
            }
            .btn-info {
                background: #1abdd6;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-weight: 600;
                color: #ffffff;
                transition: background 0.3s ease, transform 0.2s ease;
            }
            .btn-info:hover {
                background: #16a3b8;
                transform: scale(1.05);
            }
            .input-group {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                justify-content: center;
            }
            .form-control {
                background: #495057;
                color: #ffffff;
                border-color: #6c757d;
            }
            .form-control::placeholder {
                color: #cccccc;
            }
            .result-section {
                margin-top: 30px;
                animation: fadeIn 0.5s ease-in;
            }
            .canvas-container {
                position: relative;
                height: 300px;
                margin-bottom: 20px;
            }
            #heatmapCanvas {
                width: 100%;
                height: 200px;
                border-radius: 10px;
                border: 1px solid #6c757d;
            }
            .heatmap-label {
                position: absolute;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
                text-align: center;
                pointer-events: none;
                transform: translate(-50%, -50%);
                z-index: 10;
            }
            #weatherInfo {
                margin-top: 20px;
                padding: 10px;
                background: #343a40;
                color: #ffffff;
                border-radius: 10px;
            }
            .alert-warning {
                background: #664d03;
                color: #ffffff;
                border-color: #997404;
            }
            .alert-danger {
                background: #58151c;
                color: #ffffff;
                border-color: #842029;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            h2, h4, h5, h6 {
                color: #ffffff;
                font-weight: 600;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                border: 1px solid #6c757d;
            }
            @media (max-width: 576px) {
                .container {
                    padding: 15px;
                }
                .input-group {
                    flex-direction: column;
                    align-items: stretch;
                }
                .btn {
                    width: 100%;
                    font-size: 1rem;
                }
                .canvas-container {
                    height: 250px;
                }
                h2 {
                    font-size: 1.5rem;
                }
                h4, h5 {
                    font-size: 1.2rem;
                }
                .card {
                    padding: 15px;
                }
                .heatmap-label {
                    font-size: 10px;
                }
            }
            @media (max-width: 768px) {
                .canvas-container {
                    height: 280px;
                }
            }
        </style>
    </head>
    <body class="container py-5">
        <div class="text-center mb-5">
            <h2 class="mb-4">🌱 Groundnut Health Classifier</h2>
            <div class="card p-4">
                <div class="input-group mb-3">
                    <input type="text" id="locationInput" class="form-control" placeholder="Enter city for weather (e.g., eMpangeni)">
                    <input type="file" id="fileUpload" accept="image/*" class="form-control">
                    <input type="file" id="cameraCapture" accept="image/*" capture="environment" class="form-control" style="display: none;">
                    <button class="btn btn-primary" onclick="uploadImage()">Analyze Image</button>
                    <button class="btn btn-secondary" onclick="captureImage()">Capture Photo</button>
                </div>
                <div id="preview" class="mb-3">
                    <img id="imagePreview" style="display: none;">
                </div>
                <div id="weatherInfo" class="text-start"></div>
            </div>
            <div id="result" class="result-section"></div>
            <div class="mt-3">
                <button class="btn btn-info" onclick="exportReport()">Export Report</button>
                <button class="btn btn-secondary" onclick="clearHistory()">Clear History</button>
            </div>
        </div>

        <!-- Dashboard -->
        <div class="mt-5">
            <h4>📊 Analysis Dashboard</h4>
            <div class="row">
                <div class="col-md-6 mb-4">
                    <div class="card p-3">
                        <h5>Confidence Trend</h5>
                        <div class="canvas-container">
                            <canvas id="confidenceChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-4">
                    <div class="card p-3">
                        <h5>Health Distribution</h5>
                        <div class="canvas-container">
                            <canvas id="healthPie"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-12 mb-4">
                    <div class="card p-3">
                        <h5>Category Confidence</h5>
                        <div class="canvas-container">
                            <canvas id="categoryBar"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-12 mb-4">
                    <div class="card p-3">
                        <h5>Confidence Heatmap</h5>
                        <div style="position: relative;">
                            <canvas id="heatmapCanvas"></canvas>
                            <div id="heatmapLabels"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let history = [];
            let currentImageFile = null;
            let chartInstances = {
                confidenceChart: null,
                healthPie: null,
                categoryBar: null
            };

            // Load history from localStorage
            function loadHistory() {
                const saved = localStorage.getItem('groundnutHistory');
                if (saved) {
                    history = JSON.parse(saved);
                    updateCharts();
                    if (history.length > 0) {
                        createHeatmap(history[history.length - 1].all_confidences, [
                            'Early Rust', 'Late Leaf Spot', 'Nutritional Deficiency',
                            'Healthy Leaf', 'Early Leaf Spot', 'Advanced Rust'
                        ]);
                    }
                }
            }

            // Save history to localStorage
            function saveHistory() {
                localStorage.setItem('groundnutHistory', JSON.stringify(history));
            }

            // Clear history
            function clearHistory() {
                history = [];
                saveHistory();
                updateCharts();
                const canvas = document.getElementById('heatmapCanvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                document.getElementById('heatmapLabels').innerHTML = '';
                document.getElementById('result').innerHTML = '';
                document.getElementById('imagePreview').style.display = 'none';
                currentImageFile = null;
                console.log('History cleared');
            }

            // Custom heatmap function with interactivity
            function createHeatmap(confidences, classNames) {
                const canvas = document.getElementById('heatmapCanvas');
                const ctx = canvas.getContext('2d');
                const labelsContainer = document.getElementById('heatmapLabels');

                // Reset canvas size
                canvas.width = canvas.offsetWidth;
                canvas.height = canvas.offsetHeight;

                // Clear canvas and labels
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                labelsContainer.innerHTML = '';

                const cols = 3;
                const rows = 2;
                const cellWidth = canvas.width / cols;
                const cellHeight = canvas.height / rows;

                const points = confidences.map((conf, i) => {
                    const row = Math.floor(i / cols);
                    const col = i % cols;
                    const x = col * cellWidth + cellWidth / 2;
                    const y = row * cellHeight + cellHeight / 2;
                    const intensity = conf * 100;

                    const gradient = ctx.createRadialGradient(x, y, 0, x, y, Math.min(cellWidth, cellHeight) / 2);
                    gradient.addColorStop(0, getHeatmapColor(intensity));
                    gradient.addColorStop(0.7, getHeatmapColor(intensity * 0.7));
                    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

                    ctx.beginPath();
                    ctx.arc(x, y, Math.min(cellWidth, cellHeight) / 2.5, 0, 2 * Math.PI);
                    ctx.fillStyle = gradient;
                    ctx.fill();

                    const label = document.createElement('div');
                    label.className = 'heatmap-label';
                    label.style.left = `${x}px`;
                    label.style.top = `${y}px`;
                    label.innerText = `${classNames[i]}\n${(conf * 100).toFixed(1)}%`;
                    labelsContainer.appendChild(label);

                    return { x, y, conf, className: classNames[i] };
                });

                // Add interactivity
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    const mouseY = e.clientY - rect.top;
                    points.forEach(point => {
                        const dx = mouseX - point.x;
                        const dy = mouseY - point.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        if (distance < cellWidth / 2.5) {
                            canvas.style.cursor = 'pointer';
                            labelsContainer.querySelectorAll('.heatmap-label').forEach(l => l.style.opacity = '0.5');
                            const hoveredLabel = Array.from(labelsContainer.children).find(l => 
                                Math.abs(parseFloat(l.style.left) - point.x) < 5 && 
                                Math.abs(parseFloat(l.style.top) - point.y) < 5
                            );
                            if (hoveredLabel) hoveredLabel.style.opacity = '1';
                        } else {
                            canvas.style.cursor = 'default';
                            labelsContainer.querySelectorAll('.heatmap-label').forEach(l => l.style.opacity = '1');
                        }
                    });
                });
            }

            // Heatmap color function
            function getHeatmapColor(value) {
                const t = value / 100;
                if (t < 0.2) return `hsl(${240 - t * 120}, 100%, 50%)`; // Blue to cyan
                if (t < 0.4) return `hsl(${180 - (t - 0.2) * 180}, 100%, 50%)`; // Cyan to green
                if (t < 0.6) return `hsl(${120 - (t - 0.4) * 120}, 100%, 50%)`; // Green to yellow
                if (t < 0.8) return `hsl(${60 - (t - 0.6) * 60}, 100%, 50%)`; // Yellow to orange
                return `hsl(${0}, 100%, ${50 + (t - 0.8) * 50}%)`; // Orange to red
            }

            // Preview image
            function previewImage(file) {
                const preview = document.getElementById('imagePreview');
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
                currentImageFile = file;
                console.log('Image preview loaded:', file.name);
            }

            // Capture image
            function captureImage() {
                document.getElementById('cameraCapture').click();
            }

            // Handle file uploads
            document.getElementById('fileUpload').addEventListener('change', function(e) {
                if (e.target.files && e.target.files[0]) {
                    previewImage(e.target.files[0]);
                } else {
                    console.error('No file selected for upload');
                }
            });

            document.getElementById('cameraCapture').addEventListener('change', function(e) {
                if (e.target.files && e.target.files[0]) {
                    previewImage(e.target.files[0]);
                } else {
                    console.error('No file selected for capture');
                }
            });

            // Fetch weather
            function fetchWeather() {
                const city = document.getElementById('locationInput').value || 'eMpangeni';
                fetch(`/weather?city=${encodeURIComponent(city)}`)
                    .then(res => res.json())
                    .then(data => {
                        if (data.error) {
                            document.getElementById('weatherInfo').innerHTML = `<div class="alert alert-warning">${data.error}</div>`;
                        } else {
                            document.getElementById('weatherInfo').innerHTML = `
                                <p><strong>🌤️ Weather in ${city}:</strong> ${data.weather} | Temp: ${data.temp}°C | Humidity: ${data.humidity}%</p>
                            `;
                        }
                    })
                    .catch(err => {
                        document.getElementById('weatherInfo').innerHTML = `<div class="alert alert-warning">Unable to fetch weather data</div>`;
                        console.error('Weather fetch error:', err);
                    });
            }

            // Export report as PDF with image and heatmap
            function exportReport() {
                if (!history.length) {
                    alert('No predictions to export.');
                    return;
                }
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF();
                const latest = history[history.length - 1];
                let yOffset = 10;

                // Title
                doc.setFontSize(16);
                doc.text('Groundnut Health Report', 10, yOffset);
                yOffset += 10;

                // Prediction and Confidence
                doc.setFontSize(12);
                doc.text(`Prediction: ${latest.label}`, 10, yOffset);
                yOffset += 10;
                doc.text(`Confidence: ${(latest.confidence * 100).toFixed(2)}%`, 10, yOffset);
                yOffset += 10;

                // Add Input Image
                const imgElement = document.getElementById('imagePreview');
                if (imgElement.src && imgElement.style.display !== 'none') {
                    doc.text('Analyzed Image:', 10, yOffset);
                    yOffset += 10;
                    const imgData = imgElement.src;
                    try {
                        doc.addImage(imgData, 'JPEG', 10, yOffset, 80, 60);
                        yOffset += 70;
                    } catch (e) {
                        doc.text('Unable to include image.', 10, yOffset);
                        yOffset += 10;
                        console.error('PDF image error:', e);
                    }
                }

                // Causes
                doc.text('Causes:', 10, yOffset);
                yOffset += 10;
                latest.causes.forEach((cause, i) => {
                    if (yOffset > 270) {
                        doc.addPage();
                        yOffset = 10;
                    }
                    doc.text(`- ${cause}`, 15, yOffset);
                    yOffset += 10;
                });

                // Cure/Management
                doc.text('Cure/Management:', 10, yOffset);
                yOffset += 10;
                latest.cure.forEach((cure, i) => {
                    if (yOffset > 270) {
                        doc.addPage();
                        yOffset = 10;
                    }
                    doc.text(`- ${cure}`, 15, yOffset);
                    yOffset += 10;
                });

                // Prevention
                doc.text('Prevention:', 10, yOffset);
                yOffset += 10;
                latest.prevention.forEach((prev, i) => {
                    if (yOffset > 270) {
                        doc.addPage();
                        yOffset = 10;
                    }
                    doc.text(`- ${prev}`, 15, yOffset);
                    yOffset += 10;
                });

                // Add Heatmap Snapshot
                const heatmapCanvas = document.getElementById('heatmapCanvas');
                if (heatmapCanvas.width > 0 && heatmapCanvas.height > 0) {
                    if (yOffset > 200) {
                        doc.addPage();
                        yOffset = 10;
                    }
                    doc.text('Confidence Heatmap:', 10, yOffset);
                    yOffset += 10;
                    const heatmapData = heatmapCanvas.toDataURL('image/png');
                    try {
                        doc.addImage(heatmapData, 'PNG', 10, yOffset, 80, 40);
                        yOffset += 50;
                    } catch (e) {
                        doc.text('Unable to include heatmap.', 10, yOffset);
                        yOffset += 10;
                        console.error('PDF heatmap error:', e);
                    }
                }

                doc.save('groundnut_health_report.pdf');
            }

            // Upload and analyze image
            async function uploadImage() {
                if (!currentImageFile) {
                    alert('Please capture or select an image first.');
                    document.getElementById('result').innerHTML = `<div class="alert alert-danger">No image selected</div>`;
                    return;
                }
                const formData = new FormData();
                formData.append('image', currentImageFile);

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    if (data.error) {
                        document.getElementById('result').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        console.error('Prediction error:', data.error);
                    } else {
                        const city = document.getElementById('locationInput').value || 'eMpangeni';
                        const weatherRes = await fetch(`/weather?city=${encodeURIComponent(city)}`);
                        const weather = await weatherRes.json();
                        let weatherAlert = '';
                        if (weather.humidity > 80 && data.weather_alerts.high_humidity) {
                            weatherAlert = `<div class="alert alert-warning">${data.weather_alerts.high_humidity}</div>`;
                        } else if (weather.rain && data.weather_alerts.high_rain) {
                            weatherAlert = `<div class="alert alert-warning">${data.weather_alerts.high_rain}</div>`;
                        }
                        history.push({
                            label: data.predicted_class,
                            confidence: data.confidence,
                            all_confidences: data.all_confidences,
                            causes: data.causes,
                            cure: data.cure,
                            prevention: data.prevention
                        });
                        saveHistory();

                        let html = `
                            <div class="card p-4 text-start">
                                <h5 class="mb-3">✅ Prediction: ${data.predicted_class}</h5>
                                <p><strong>📊 Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                                ${weatherAlert}
                                <h6 class="mt-3">⚠️ Causes</h6>
                                <ul>${data.causes.map(c => `<li>${c}</li>`).join('')}</ul>
                                <h6>💊 Cure / Management</h6>
                                <ul>${data.cure.map(c => `<li>${c}</li>`).join('')}</ul>
                                <h6>🛡️ Prevention</h6>
                                <ul>${data.prevention.map(c => `<li>${c}</li>`).join('')}</ul>
                            </div>`;
                        document.getElementById('result').innerHTML = html;
                        updateCharts();
                        createHeatmap(data.all_confidences, [
                            'Early Rust', 'Late Leaf Spot', 'Nutritional Deficiency',
                            'Healthy Leaf', 'Early Leaf Spot', 'Advanced Rust'
                        ]);
                    }
                } catch (err) {
                    document.getElementById('result').innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
                    console.error('Upload error:', err);
                }
            }

            // Update charts with animations
            function updateCharts() {
                // Destroy existing charts
                Object.keys(chartInstances).forEach(key => {
                    if (chartInstances[key]) {
                        chartInstances[key].destroy();
                        chartInstances[key] = null;
                    }
                });

                const chartBorderColor = '#ffffff';
                const chartTextColor = '#ffffff';
                const confidenceColor = '#34c759';
                const healthyColor = '#34c759';
                const diseasedColor = '#ff6b6b';
                const barHighlightColor = '#34c759';
                const barDefaultColor = '#6c757d';

                const labels = history.map((h, i) => `Sample ${i + 1}`);
                const confidences = history.map(h => (h.confidence * 100).toFixed(2));
                const healthyCount = history.filter(h => h.label.includes("Healthy")).length;
                const diseasedCount = history.length - healthyCount;
                const latestConfidences = history.length > 0 ? history[history.length - 1].all_confidences.map(c => (c * 100).toFixed(2)) : [];

                // Confidence Area Chart
                const areaData = confidences.length > 0 ? confidences.map((conf, i) => {
                    const peak = parseFloat(conf);
                    return [0, peak * 0.5, peak, peak * 0.5, 0];
                }).flat() : [];
                const areaLabels = confidences.length > 0 ? confidences.map((_, i) => {
                    return [`${i + 1} Start`, `${i + 1} Rise`, `${i + 1} Peak`, `${i + 1} Fall`, `${i + 1} End`];
                }).flat() : [];

                chartInstances.confidenceChart = new Chart(document.getElementById("confidenceChart"), {
                    type: 'line',
                    data: {
                        labels: areaLabels,
                        datasets: [{
                            label: 'Confidence (%)',
                            data: areaData,
                            borderColor: confidenceColor,
                            backgroundColor: 'rgba(52, 199, 89, 0.3)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { 
                                beginAtZero: true, 
                                max: 100,
                                ticks: { color: chartTextColor }
                            },
                            x: { 
                                display: true,
                                ticks: { color: chartTextColor }
                            }
                        },
                        plugins: {
                            legend: { 
                                display: true,
                                labels: { color: chartTextColor }
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `Confidence: ${context.raw}%`;
                                    }
                                }
                            }
                        },
                        animation: {
                            duration: 1000,
                            easing: 'easeOutQuart'
                        }
                    }
                });

                // Health Distribution Pie Chart
                chartInstances.healthPie = new Chart(document.getElementById("healthPie"), {
                    type: 'pie',
                    data: {
                        labels: ['Healthy', 'Diseased'],
                        datasets: [{
                            data: [healthyCount, diseasedCount],
                            backgroundColor: [healthyColor, diseasedColor],
                            borderColor: chartBorderColor,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { 
                                position: 'top',
                                labels: { color: chartTextColor }
                            }
                        },
                        animation: {
                            duration: 1000,
                            easing: 'easeOutQuart'
                        }
                    }
                });

                // Category Bar Chart
                const maxIndex = history.length > 0 ? history[history.length - 1].all_confidences.indexOf(Math.max(...history[history.length - 1].all_confidences)) : -1;
                chartInstances.categoryBar = new Chart(document.getElementById("categoryBar"), {
                    type: 'bar',
                    data: {
                        labels: ['Early Rust', 'Late Leaf Spot', 'Nutritional Deficiency', 'Healthy Leaf', 'Early Leaf Spot', 'Advanced Rust'],
                        datasets: [{
                            label: 'Confidence (%)',
                            data: latestConfidences,
                            backgroundColor: latestConfidences.map((_, i) => i === maxIndex ? barHighlightColor : barDefaultColor),
                            borderColor: chartBorderColor,
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { 
                                beginAtZero: true, 
                                max: 100,
                                ticks: { color: chartTextColor }
                            },
                            x: {
                                ticks: { color: chartTextColor }
                            }
                        },
                        plugins: {
                            legend: { 
                                display: false,
                                labels: { color: chartTextColor }
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `${context.label}: ${context.raw}%`;
                                    }
                                }
                            }
                        },
                        animation: {
                            duration: 1000,
                            easing: 'easeOutQuart'
                        }
                    }
                });
            }

            // Initialize on page load
            window.onload = function() {
                const canvas = document.getElementById('heatmapCanvas');
                canvas.width = canvas.offsetWidth;
                canvas.height = canvas.offsetHeight;
                loadHistory();
                fetchWeather();
                document.getElementById('locationInput').addEventListener('change', fetchWeather);
                console.log('Page loaded with background color #343a40');
            };
        </script>
    </body>
    </html>
    """)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or WebP files.'}), 400

    image_array = preprocess_image(file)
    if image_array is None:
        return jsonify({'error': 'Error processing image'}), 500

    prediction_result = predict_class(image_array)
    if prediction_result is None:
        return jsonify({'error': 'Error making prediction'}), 500

    return jsonify(prediction_result)


@app.route('/weather', methods=['GET'])
def weather():
    city = request.args.get('city', 'eMpangeni')
    weather_data = get_weather(city)
    if weather_data is None or 'error' in weather_data:
        return jsonify({'error': weather_data.get('error', 'Unable to fetch weather data')}), 400
    return jsonify(weather_data)


if __name__ == '__main__':
    print("🌱 Groundnut Health Model Server Starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)