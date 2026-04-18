<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Inter&weight=700&size=40&pause=1000&color=A3E635&center=true&vCenter=true&width=800&lines=Smart+Credit+Risk+XAI;AI-Powered+Lending+Intelligence;Dark+Dashboard+%26+Explainable+AI" alt="Typing SVG" />
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3">
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
</p>

# 🏦 Finexcore: AI Lending Intelligence

An enterprise-grade, explainable AI (XAI) dashboard for deep credit risk assessment. It features a hyper-modern, deep slate dashboard with high-visibility neon accents, processing both individual applications and batch CSV portfolio scoring.

<div align="center">
  <img src="https://user-images.githubusercontent.com/placeholder-gif.gif" alt="Dashboard Preview" width="100%" />
</div>

## ✨ Key Features

- **⚡ Real-Time Scoring Engine:** Instantly compute default probabilities using an active ML backend.
- **📈 Portfolio Batch Processing:** Drag-and-drop CSV upload for scoring entire loan portfolios with aggregated KPIs.
- **🔍 Full Explainability (XAI):** Built-in SHAP feature-importance visualizations detailing *exactly* why a decision was made.
- **🎨 Premium Dark Dashboard:** A meticulously crafted, completely local UI featuring glassmorphism, responsive grids, and fluid tab navigation.
- **📊 Interactive Analytics:** Powered by `Chart.js` tailored for optimal dark-mode viewing.

## 🚀 Quick Start

### 1. Launch the AI Backend
The ML engine relies on Python APIs returning SHAP arrays and tier decisions.

```bash
cd backend
pip install -r requirements.txt
python main.py
# Server will default to localhost:7860
```

### 2. Open the Interface
The application requires absolutely **no build step**. Simply open the `index.html` file in your favorite web browser!

```bash
start index.html # On Windows
open index.html  # On Mac
```

## 🛠️ Architecture

- **Frontend:** Pure HTML5/CSS3/Vanilla JS (Single-File App), `Chart.js` via CDN.
- **Backend Architecture:** Python API endpoints (`/predict`, `/predict/csv`, `/model/info`).
- **Design Tokens:** Deep slate `#09090b` heavily accented with vivid neon lime `#a3e635`.

## 📜 Notice
This dashboard was built as part of the Finexcore AI Risk Infrastructure.
