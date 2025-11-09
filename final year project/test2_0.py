import os
import torch
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
import json

# ----------------------- Flask Setup -----------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ----------------------- Device ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Class Names -----------------------
CLASS_NAMES = [
    'ALTERNARIA LEAF SPOT', 'ROSETTE', 'early_leaf_spot_1', 'early_rust_1',
    'healthy_leaf_1', 'late_leaf_spot_1', 'nutrition_deficiency_1', 'rust_1'
]

# ----------------------- Load Model ------------------------
MODEL_PATH = "vit_groundnut_cpu_friendly.pth"
model = ViTForImageClassification.from_pretrained(
    "facebook/deit-tiny-patch16-224",
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------- Image Transform -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

# ----------------------- Disease Database ------------------
DISEASE_INFO = {
    "ALTERNARIA LEAF SPOT": {
        "name": "Alternaria Leaf Spot",
        "causes": [
            "Caused by Alternaria spp. (e.g., A. alternata)",
            "Favored by warm, humid conditions",
            "Spread by wind and rain splash",
            "Infected debris sustains inoculum"
        ],
        "cure": [
            "Apply Mancozeb, Chlorothalonil, or Copper oxychloride",
            "Remove and destroy infected leaves",
            "Rotate fungicides to avoid resistance"
        ],
        "prevention": [
            "Use resistant varieties",
            "Crop rotation and field sanitation",
            "Avoid overhead irrigation",
            "Ensure good airflow"
        ],
        "weather_alerts": {
            "high_humidity": "High humidity increases Alternaria risk. Monitor closely."
        }
    },
    "ROSETTE": {
        "name": "Groundnut Rosette Disease",
        "causes": [
            "Caused by Groundnut Rosette Virus (GRV) + satellite RNA",
            "Transmitted by aphids (Aphis craccivora)",
            "Stressed plants more susceptible"
        ],
        "cure": [
            "No cure — remove and destroy infected plants",
            "Control aphids with Imidacloprid or botanical sprays"
        ],
        "prevention": [
            "Use certified virus-free seeds",
            "Plant resistant varieties",
            "Early planting to avoid peak aphid season",
            "Integrated pest management (IPM)"
        ],
        "weather_alerts": {
            "aphid_risk": "Warm, dry conditions favor aphid migration — monitor vectors."
        }
    },
    "early_leaf_spot_1": {
        "name": "Early Leaf Spot",
        "causes": ["High humidity", "Warm temperatures", "Poor airflow", "Infected residues"],
        "cure": ["Copper fungicides", "Propiconazole 25% EC", "Remove infected leaves"],
        "prevention": ["Resistant varieties", "Drip irrigation", "Wider spacing"],
        "weather_alerts": {"high_humidity": "High humidity promotes leaf spot — use drip irrigation."}
    },
    "early_rust_1": {
        "name": "Early Rust",
        "causes": ["20–25°C + high humidity", "Dew/rain", "Wind-borne spores"],
        "cure": ["Triadimefon 25% WP", "Hexaconazole 5% SC", "Remove infected leaves"],
        "prevention": ["Resistant varieties", "Balanced nutrition", "Early monitoring"],
        "weather_alerts": {"high_humidity": "Rust risk rising — increase scouting."}
    },
    "healthy_leaf_1": {
        "name": "Healthy Leaf",
        "causes": ["Optimal conditions", "Balanced nutrition", "Good management"],
        "cure": ["Continue best practices", "Regular monitoring"],
        "prevention": ["High-quality seeds", "Crop rotation", "Avoid over-irrigation"],
        "weather_alerts": {"high_humidity": "Healthy now, but watch for fungal risks in humidity."}
    },
    "late_leaf_spot_1": {
        "name": "Late Leaf Spot",
        "causes": [">85% humidity", "20–25°C", "Prolonged leaf wetness"],
        "cure": ["Chlorothalonil 75% WP", "Tebuconazole + Sulphur", "Alternate fungicides"],
        "prevention": ["Resistant cultivars", "Morning irrigation", "Remove debris"],
        "weather_alerts": {"high_humidity": "Late leaf spot risk high — avoid overhead watering."}
    },
    "nutrition_deficiency_1": {
        "name": "Nutritional Deficiency",
        "causes": ["Imbalanced fertilization", "Soil pH issues", "Poor organic matter"],
        "cure": ["Soil test", "Correct fertilizers", "Foliar sprays"],
        "prevention": ["Maintain soil pH", "Add compost", "Follow fertilizer schedule"],
        "weather_alerts": {"high_rain": "Heavy rain may leach nutrients — test soil."}
    },
    "rust_1": {
        "name": "Advanced Rust",
        "causes": ["Untreated early rust", "Persistent humidity", "Delayed action"],
        "cure": ["Propiconazole 25% EC", "Systemic + contact mix", "Frequent sprays"],
        "prevention": ["Early detection", "Monitor weather", "Maintain plant health"],
        "weather_alerts": {"high_humidity": "Advanced rust detected — urgent action required!"}
    }
}


# ----------------------- Helper Functions ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'}


def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor).logits
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        class_id = int(outputs.argmax(dim=1).item())
        predicted_class = CLASS_NAMES[class_id]
        info = DISEASE_INFO.get(predicted_class, {})
        return {
            "predicted_class": info.get("name", predicted_class),
            "confidence": float(probs[class_id]),
            "all_confidences": [float(p) for p in probs],
            "causes": info.get("causes", []),
            "cure": info.get("cure", []),
            "prevention": info.get("prevention", []),
            "weather_alerts": info.get("weather_alerts", {})
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def get_weather(city="eMpangeni"):
    try:
        api_key = "f3ac3e90c50e44c29cb141812252509"
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
        res = requests.get(url, timeout=10)
        data = res.json()
        if res.status_code == 200:
            return {
                "temp": data["current"]["temp_c"],
                "humidity": data["current"]["humidity"],
                "weather": data["current"]["condition"]["text"],
                "rain": "rain" in data["current"]["condition"]["text"].lower()
            }
        return {"error": data.get("error", {}).get("message", "Weather fetch failed")}
    except Exception as e:
        return {"error": str(e)}


# ----------------------- Routes ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No files selected"}), 400

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            pred = predict_image(filepath)
            if pred:
                results.append(pred)
            else:
                results.append({"error": "Failed to process image"})
        else:
            results.append({"error": f"Invalid file: {file.filename}"})

    return jsonify(results)


@app.route("/weather")
def weather():
    city = request.args.get("city", "eMpangeni")
    return jsonify(get_weather(city))


# ----------------------- Full HTML Template ---------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groundnut Disease ViT Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <style>
        body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; }
        .container { max-width: 1300px; }
        .card { background: #16213e; border: none; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.5); }
        .btn-primary { background: #00d4aa; border: none; border-radius: 12px; padding: 12px 24px; font-weight: 600; }
        .btn-primary:hover { background: #00b894; transform: translateY(-2px); }
        .btn-secondary { background: #4b5e6a; border-radius: 12px; }
        .btn-info { background: #1e90ff; border-radius: 12px; }
        .form-control, .form-select { background: #0f3460; color: #fff; border: 1px solid #1e90ff; }
        .form-control::placeholder { color: #aaa; }
        .result-card { background: #0f3460; border-radius: 12px; padding: 15px; margin-bottom: 15px; }
        .alert-warning { background: #8B8000; color: #fff; }
        .heatmap-label { 
            position: absolute; 
            color: #fff; 
            font-size: 11px; 
            font-weight: bold; 
            text-align: center; 
            pointer-events: none; 
            transform: translate(-50%, -50%); 
            text-shadow: 1px 1px 2px #000;
        }
        .canvas-container { height: 280px; position: relative; }
        #heatmapCanvas { width: 100%; height: 200px; border-radius: 10px; border: 1px solid #1e90ff; }
        .chart-container { 
            background: #0f3460; 
            border-radius: 12px; 
            padding: 15px; 
            height: 300px; 
            margin-bottom: 20px;
        }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .fade-in { animation: fadeIn 0.6s ease-in; }
    </style>
</head>
<body>
<div class="container py-4">
    <h1 class="text-center mb-4 text-primary">Groundnut Disease Classifier (ViT)</h1>

    <!-- Upload & Controls -->
    <div class="card p-4 mb-4">
        <div class="row g-3 align-items-center">
            <div class="col-md-3">
                <input type="text" id="locationInput" class="form-control" placeholder="City (e.g., eMpangeni)" value="eMpangeni">
            </div>
            <div class="col-md-3">
                <input type="file" id="fileInput" class="form-control" accept="image/*" multiple>
            </div>
            <div class="col-md-3">
                <input type="file" id="cameraInput" accept="image/*" capture="environment" class="form-control" style="display:none;">
                <button class="btn btn-secondary w-100" onclick="document.getElementById('cameraInput').click()">Camera</button>
            </div>
            <div class="col-md-3">
                <button class="btn btn-primary w-100" onclick="analyzeImages()">Analyze</button>
            </div>
        </div>
        <div id="weatherInfo" class="mt-3"></div>
    </div>

    <!-- Results -->
    <div id="results" class="row"></div>

    <!-- Dashboard -->
    <div class="mt-5 card p-4">
        <h4 class="mb-4">Analysis Dashboard</h4>
        <div class="row">
            <div class="col-md-6 mb-4">
                <div class="chart-container">
                    <canvas id="confidenceChart"></canvas>
                </div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="chart-container">
                    <canvas id="healthPie"></canvas>
                </div>
            </div>
            <div class="col-12 mb-4">
                <div class="chart-container">
                    <canvas id="categoryBar"></canvas>
                </div>
            </div>
            <div class="col-12">
                <div class="chart-container" style="position:relative; height: 220px;">
                    <canvas id="heatmapCanvas"></canvas>
                    <div id="heatmapLabels"></div>
                </div>
            </div>
        </div>
        <div class="mt-3 text-center">
            <button class="btn btn-info me-2" onclick="exportPDF()">Export PDF Report</button>
            <button class="btn btn-secondary" onclick="clearAll()">Clear All</button>
        </div>
    </div>
</div>

<script>
    let history = [];
    let currentFiles = [];
    let charts = {};

    const CLASS_LABELS = [
        'Alternaria', 'Rosette', 'Early LS', 'Early Rust', 
        'Healthy', 'Late LS', 'Nutrition', 'Advanced Rust'
    ];

    // File handling
    document.getElementById('fileInput').addEventListener('change', (e) => {
        currentFiles = Array.from(e.target.files);
    });
    document.getElementById('cameraInput').addEventListener('change', (e) => {
        if (e.target.files.length) currentFiles = Array.from(e.target.files);
    });

    // Weather
    async function fetchWeather() {
        const city = document.getElementById('locationInput').value || 'eMpangeni';
        try {
            const res = await fetch(`/weather?city=${encodeURIComponent(city)}`);
            const data = await res.json();
            const info = document.getElementById('weatherInfo');
            if (data.error) {
                info.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            } else {
                info.innerHTML = `<p><strong>Weather in ${city}:</strong> ${data.weather} | ${data.temp}°C | ${data.humidity}% humidity</p>`;
            }
        } catch(e) {
            console.error('Weather fetch error:', e);
        }
    }

    // Analyze
    async function analyzeImages() {
        if (!currentFiles.length) {
            alert('Select or capture images first.');
            return;
        }

        const form = new FormData();
        currentFiles.forEach(f => form.append('images', f));

        try {
            const res = await fetch('/predict', { method: 'POST', body: form });
            const results = await res.json();

            const weatherRes = await fetch(`/weather?city=${document.getElementById('locationInput').value || 'eMpangeni'}`);
            const weather = await weatherRes.json();

            const container = document.getElementById('results');
            container.innerHTML = '';

            results.forEach((data, i) => {
                if (data.error) {
                    container.innerHTML += `<div class="col-12"><div class="alert alert-danger">${data.error}</div></div>`;
                    return;
                }

                // Generate alert based on weather
                let alert = '';
                if (weather.humidity > 80 && data.weather_alerts && data.weather_alerts.high_humidity) {
                    alert = `<div class="alert alert-warning mt-2">${data.weather_alerts.high_humidity}</div>`;
                } else if (weather.rain && data.weather_alerts && data.weather_alerts.high_rain) {
                    alert = `<div class="alert alert-warning mt-2">${data.weather_alerts.high_rain}</div>`;
                }

                // Create image URL
                const imgUrl = URL.createObjectURL(currentFiles[i]);

                // Add to history with timestamp
                const historyEntry = {
                    predicted_class: data.predicted_class,
                    confidence: data.confidence,
                    all_confidences: data.all_confidences,
                    timestamp: new Date().toISOString(),
                    imgUrl: imgUrl
                };
                history.push(historyEntry);

                // Display result
                container.innerHTML += `
                <div class="col-md-6 mb-3 fade-in">
                    <div class="result-card">
                        <img src="${imgUrl}" class="img-fluid rounded mb-2" style="max-height:200px; width:100%; object-fit:cover;">
                        <h5>Prediction: ${data.predicted_class}</h5>
                        <p><strong>Confidence:</strong> ${(data.confidence*100).toFixed(1)}%</p>
                        ${alert}
                        <details><summary><strong>Causes</strong></summary><ul>${data.causes.map(c=>`<li>${c}</li>`).join('')}</ul></details>
                        <details><summary><strong>Cure</strong></summary><ul>${data.cure.map(c=>`<li>${c}</li>`).join('')}</ul></details>
                        <details><summary><strong>Prevention</strong></summary><ul>${data.prevention.map(c=>`<li>${c}</li>`).join('')}</ul></details>
                    </div>
                </div>`;
            });

            // Update all visualizations
            updateCharts();

            // Create heatmap for the latest prediction
            if (history.length > 0 && history[history.length - 1].all_confidences) {
                setTimeout(() => {
                    createHeatmap(history[history.length - 1].all_confidences);
                }, 100);
            }

            await fetchWeather();

        } catch(e) {
            console.error('Analysis error:', e);
            alert('Error analyzing images. Please try again.');
        }
    }

    // Charts
    function updateCharts() {
        // Destroy old charts
        Object.keys(charts).forEach(key => {
            if (charts[key]) {
                charts[key].destroy();
                charts[key] = null;
            }
        });

        if (!history.length) return;

        // Prepare data
        const confidences = history.map(h => (h.confidence * 100));
        const labels = history.map((_, i) => `Sample ${i + 1}`);

        // Count healthy vs diseased
        const healthy = history.filter(h => 
            h.predicted_class && h.predicted_class.toLowerCase().includes('healthy')
        ).length;
        const diseased = history.length - healthy;

        // Get latest prediction's all confidences
        const latest = history[history.length - 1].all_confidences.map(c => c * 100);

        // 1. Confidence Trend Line Chart
        const confCtx = document.getElementById('confidenceChart');
        if (confCtx) {
            charts.confidence = new Chart(confCtx, {
                type: 'line',
                data: { 
                    labels: labels, 
                    datasets: [{
                        label: 'Confidence (%)',
                        data: confidences,
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0,212,170,0.2)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { 
                        legend: { 
                            labels: { color: '#fff', font: { size: 12 } } 
                        },
                        tooltip: { 
                            mode: 'index', 
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    return `Confidence: ${context.parsed.y.toFixed(1)}%`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: { 
                            beginAtZero: true,
                            max: 100,
                            ticks: { color: '#fff', callback: (val) => val + '%' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        x: { 
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    }
                }
            });
        }

        // 2. Health Distribution Pie Chart
        const pieCtx = document.getElementById('healthPie');
        if (pieCtx) {
            charts.pie = new Chart(pieCtx, {
                type: 'doughnut',
                data: { 
                    labels: ['Healthy', 'Diseased'], 
                    datasets: [{
                        data: [healthy, diseased],
                        backgroundColor: ['#00d4aa', '#ff6b6b'],
                        borderColor: '#16213e',
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { 
                        legend: { 
                            position: 'bottom', 
                            labels: { color: '#fff', font: { size: 12 } } 
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percent = ((context.parsed / total) * 100).toFixed(1);
                                    return `${context.label}: ${context.parsed} (${percent}%)`;
                                }
                            }
                        }
                    }
                }
            });
        }

        // 3. Category Confidence Bar Chart (Latest Prediction)
        const barCtx = document.getElementById('categoryBar');
        if (barCtx) {
            const maxIdx = latest.indexOf(Math.max(...latest));
            const colors = latest.map((_, i) => i === maxIdx ? '#00d4aa' : '#555');

            charts.bar = new Chart(barCtx, {
                type: 'bar',
                data: { 
                    labels: CLASS_LABELS, 
                    datasets: [{
                        label: 'Confidence (%)',
                        data: latest,
                        backgroundColor: colors,
                        borderColor: colors.map(c => c === '#00d4aa' ? '#00b894' : '#444'),
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { 
                            beginAtZero: true,
                            max: 100,
                            ticks: { color: '#fff', callback: (val) => val + '%' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        x: { 
                            ticks: { 
                                color: '#fff', 
                                maxRotation: 45, 
                                minRotation: 45,
                                font: { size: 10 }
                            },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    },
                    plugins: { 
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => `${context.parsed.y.toFixed(1)}%`
                            }
                        }
                    }
                }
            });
        }
    }

    // Heatmap
    function createHeatmap(confidences) {
        const canvas = document.getElementById('heatmapCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const labelsDiv = document.getElementById('heatmapLabels');

        // Set canvas size
        const parent = canvas.parentElement;
        canvas.width = parent.offsetWidth;
        canvas.height = 200;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (labelsDiv) labelsDiv.innerHTML = '';

        const cols = 4, rows = 2;
        const cellW = canvas.width / cols;
        const cellH = canvas.height / rows;

        confidences.forEach((conf, i) => {
            if (i >= 8) return; // Only 8 classes

            const col = i % cols;
            const row = Math.floor(i / cols);
            const cx = col * cellW + cellW / 2;
            const cy = row * cellH + cellH / 2;
            const radius = Math.min(cellW, cellH) * 0.35;

            // Create gradient based on confidence
            const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
            const hue = 120 * (1 - conf); // Green (120) to Red (0)
            grad.addColorStop(0, `hsla(${hue}, 100%, 60%, 0.9)`);
            grad.addColorStop(0.7, `hsla(${hue}, 100%, 40%, 0.6)`);
            grad.addColorStop(1, 'rgba(0,0,0,0)');

            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();

            // Add label
            if (labelsDiv) {
                const label = document.createElement('div');
                label.className = 'heatmap-label';
                label.style.left = cx + 'px';
                label.style.top = cy + 'px';
                label.innerHTML = `${CLASS_LABELS[i]}<br><strong>${(conf * 100).toFixed(0)}%</strong>`;
                labelsDiv.appendChild(label);
            }
        });
    }

    // Export PDF
    async function exportPDF() {
        if (!history.length) {
            alert('No data to export.');
            return;
        }

        const { jsPDF } = window.jspdf;
        const doc = new jsPDF('p', 'mm', 'a4');
        let y = 20;

        // Title
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(18);
        doc.text('Groundnut Disease Analysis Report', 105, y, { align: 'center' });
        y += 10;

        doc.setFontSize(10);
        doc.setFont('helvetica', 'normal');
        doc.text(`Generated: ${new Date().toLocaleString()}`, 105, y, { align: 'center' });
        y += 15;

        // Summary
        const healthy = history.filter(h => h.predicted_class.toLowerCase().includes('healthy')).length;
        const diseased = history.length - healthy;
        doc.setFontSize(12);
        doc.setFont('helvetica', 'bold');
        doc.text('Summary:', 15, y);
        y += 7;
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        doc.text(`Total Samples Analyzed: ${history.length}`, 15, y);
        y += 6;
        doc.text(`Healthy Samples: ${healthy} (${((healthy/history.length)*100).toFixed(1)}%)`, 15, y);
        y += 6;
        doc.text(`Diseased Samples: ${diseased} (${((diseased/history.length)*100).toFixed(1)}%)`, 15, y);
        y += 12;

        // Add charts if available
        try {
            // Add confidence chart
            const confChart = document.getElementById('confidenceChart');
            if (confChart && charts.confidence) {
                if (y > 200) { doc.addPage(); y = 20; }
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(11);
                doc.text('Confidence Trend:', 15, y);
                y += 5;
                const confImg = confChart.toDataURL('image/png');
                doc.addImage(confImg, 'PNG', 15, y, 180, 60);
                y += 65;
            }

            // Add pie chart
            const pieChart = document.getElementById('healthPie');
            if (pieChart && charts.pie) {
                if (y > 200) { doc.addPage(); y = 20; }
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(11);
                doc.text('Health Distribution:', 15, y);
                y += 5;
                const pieImg = pieChart.toDataURL('image/png');
                doc.addImage(pieImg, 'PNG', 15, y, 90, 60);
                y += 65;
            }

            // Add category bar chart
            const barChart = document.getElementById('categoryBar');
            if (barChart && charts.bar) {
                if (y > 200) { doc.addPage(); y = 20; }
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(11);
                doc.text('Latest Prediction - Class Confidences:', 15, y);
                y += 5;
                const barImg = barChart.toDataURL('image/png');
                doc.addImage(barImg, 'PNG', 15, y, 180, 60);
                y += 65;
            }

        } catch(e) {
            console.error('Error adding charts to PDF:', e);
        }

        // Individual results with images
        doc.addPage();
        y = 20;
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(14);
        doc.text('Detailed Analysis Results', 105, y, { align: 'center' });
        y += 10;

        for (let i = 0; i < history.length; i++) {
            const h = history[i];

            // Check if we need a new page
            if (y > 220) { 
                doc.addPage(); 
                y = 20; 
            }

            // Sample header
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(12);
            doc.text(`Sample ${i + 1}`, 15, y);
            y += 7;

            // Add image if available
            if (h.imgUrl) {
                try {
                    // Create temporary image element to get base64
                    const img = await loadImage(h.imgUrl);
                    const aspectRatio = img.width / img.height;
                    const imgWidth = 60;
                    const imgHeight = imgWidth / aspectRatio;

                    doc.addImage(img, 'JPEG', 15, y, imgWidth, imgHeight);

                    // Add prediction details next to image
                    const textX = 80;
                    let textY = y + 5;

                    doc.setFont('helvetica', 'bold');
                    doc.setFontSize(10);
                    doc.text('Prediction:', textX, textY);
                    textY += 5;
                    doc.setFont('helvetica', 'normal');
                    doc.text(h.predicted_class, textX, textY);
                    textY += 7;

                    doc.setFont('helvetica', 'bold');
                    doc.text('Confidence:', textX, textY);
                    textY += 5;
                    doc.setFont('helvetica', 'normal');
                    doc.text(`${(h.confidence * 100).toFixed(1)}%`, textX, textY);
                    textY += 7;

                    doc.setFont('helvetica', 'bold');
                    doc.text('Analysis Time:', textX, textY);
                    textY += 5;
                    doc.setFont('helvetica', 'normal');
                    const timestamp = new Date(h.timestamp).toLocaleString();
                    doc.text(timestamp, textX, textY);

                    y += Math.max(imgHeight, 35) + 5;

                } catch(e) {
                    console.error('Error adding image to PDF:', e);
                    // Add text details without image
                    doc.setFont('helvetica', 'normal');
                    doc.setFontSize(10);
                    doc.text(`Prediction: ${h.predicted_class}`, 20, y);
                    y += 5;
                    doc.text(`Confidence: ${(h.confidence * 100).toFixed(1)}%`, 20, y);
                    y += 8;
                }
            } else {
                // No image available
                doc.setFont('helvetica', 'normal');
                doc.setFontSize(10);
                doc.text(`Prediction: ${h.predicted_class}`, 20, y);
                y += 5;
                doc.text(`Confidence: ${(h.confidence * 100).toFixed(1)}%`, 20, y);
                y += 8;
            }

            // Add separator line
            doc.setDrawColor(200, 200, 200);
            doc.line(15, y, 195, y);
            y += 5;
        }

        doc.save('groundnut_analysis_report.pdf');
    }

    // Helper function to load image
    function loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'Anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    }

    function clearAll() {
        if (!confirm('Clear all analysis data?')) return;

        history = [];
        currentFiles = [];
        document.getElementById('results').innerHTML = '';
        document.getElementById('fileInput').value = '';

        // Destroy charts
        Object.keys(charts).forEach(k => {
            if (charts[k]) {
                charts[k].destroy();
                charts[k] = null;
            }
        });
        charts = {};

        // Clear heatmap
        const canvas = document.getElementById('heatmapCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        const labelsDiv = document.getElementById('heatmapLabels');
        if (labelsDiv) labelsDiv.innerHTML = '';
    }

    // Initialize on page load
    window.addEventListener('load', () => {
        fetchWeather();
        document.getElementById('locationInput').addEventListener('change', fetchWeather);

        // Handle canvas resize
        window.addEventListener('resize', () => {
            if (history.length > 0 && history[history.length - 1].all_confidences) {
                setTimeout(() => {
                    createHeatmap(history[history.length - 1].all_confidences);
                }, 100);
            }
        });
    });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Groundnut ViT Classifier Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)