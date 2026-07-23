# 🌾 Dharaveda — AI-Powered Crop Management System

Dharaveda is an end-to-end, AI-driven crop management platform built to give farmers a complete, data-backed blueprint of their fields — from choosing the right crop, to nourishing it correctly, to catching disease early, to tracking its growth against real-time weather conditions.

The platform combines classical machine learning, deep learning, and explainable AI to move farming decisions away from guesswork and toward evidence-based guidance.

🔗 **Live App:** [dhara-veda.vercel.app](https://dhara-veda.vercel.app)

---
<p align="center">

  <table>
    <tr>
      <td align="center">
        <div style="width:150px; height:150px; overflow:hidden;">
          <img src="https://github.com/user-attachments/assets/24ccdca1-df51-4dea-9db6-f076d521a0a4" style="width:150px; height:150px; object-fit:cover;" />
          </div>
      </td>
      <td align="center">
        <div style="width:150px; height:150px; overflow:hidden;">
          <img src="https://github.com/user-attachments/assets/7701a8b5-98b3-4991-9be3-4fef9ca53e86" style="width:150px; height:150px; object-fit:cover;" />
        </div>
      </td>
      <td align="center">
        <div style="width:150px; height:150px; overflow:hidden;">
          <img src="https://github.com/user-attachments/assets/578347da-6bc2-4adf-aec5-c19b39a0b526"  style="width:150px; height:150px; object-fit:cover;" />
        </div>
      </td>
      <td align="center">
        <div style="width:150px; height:150px; overflow:hidden;">
          <img src="https://github.com/user-attachments/assets/a8b10222-ebe5-469f-b755-7b83ce339c5b" style="width:150px; height:150px; object-fit:cover;" />
        </div>
      </td>
      <td align="center">
        <div style="width:150px; height:150px; overflow:hidden;">
          <img src="https://github.com/user-attachments/assets/5a0ca04d-3054-4282-8932-85f80b56d986" style="width:150px; height:150px; object-fit:cover;" />
        </div>
      </td>
    </tr>
  </table>
</p>

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Modules](#-modules)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Database Schema (Supabase)](#-database-schema-supabase)
- [Security](#-security)
- [Installation & Setup](#-installation--setup)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Background Jobs](#-background-jobs)
- [Screenshots](#-screenshots)
- [Future Enhancements](#-future-enhancements)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)
- [Contact](#-contact)

---

## 🔎 Overview

India's agricultural economy supports over 140 million farmers, yet critical decisions — what to grow, how to fertilize, how to detect disease, and how to track a crop's progress — are still largely made on guesswork and experience alone.

**Dharaveda** was built to change that. It integrates multiple machine learning and deep learning models into a single, secure, real-time platform — built entirely with vanilla **HTML, CSS, and JavaScript** on the frontend and a **Python Flask** backend, with **Supabase** as the database layer — giving farmers actionable, explainable, and location-aware recommendations at every stage of the crop lifecycle.

---

## ✨ Key Features

- 🌱 AI-based crop recommendation covering **22+ crop varieties**
- 🧪 AI-based fertilizer recommendation tailored to soil and crop type
- 🔬 Deep learning–based plant disease detection with Grad-CAM explainability
- 📊 Real-time crop tracking with a full growth timeline
- 🌦️ Live, location-based weather integration via Open-Meteo & OpenWeatherMap
- 🚨 AI-generated daily crop health alerts (priority, stress level, growth score)
- 🤖 Conversational AI assistant with streaming responses (multi-model fallback)
- 🔐 Secure authentication with JWT and hashed passwords
- ⚡ Real-time database synchronization via Supabase REST API
- 📱 Clean, responsive, farmer-friendly UI — pure HTML/CSS/JS, no framework overhead

---

## 🧩 Modules

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img width="220" alt="Crop_rmd" src="https://github.com/user-attachments/assets/aa471aa5-6604-49d5-a2f3-14070feee2fc" /><br />
        <sub><b>Crop Recommendation</b></sub>
      </td>
      <td align="center">
        <img width="220" alt="Fertilizer_recomender" src="https://github.com/user-attachments/assets/9f997bff-8b8f-494b-8c05-fa6c0d00a67d" /><br />
        <sub><b>Fertilizer Recommender</b></sub>
      </td>
      <td align="center">
        <img width="220" alt="Disease_Detection" src="https://github.com/user-attachments/assets/210d272a-d5d8-482b-b1d1-8f813bc413fd" /><br />
        <sub><b>Disease Detection</b></sub>
      </td>
      <td align="center">
        <img width="220" alt="lady_bot" src="https://github.com/user-attachments/assets/c957bc48-9f1f-4bf8-946d-fb32fb314eda" /><br />
        <sub><b>Chatbot Assistant</b></sub>
      </td>
    </tr>
  </table>
</p>

### 1. Ksetrajna — Crop Recommendation System
> *"One who knows the field."*

Recommends the most suitable crop from **22+ options** based on core soil and climate parameters.

- **Model:** Random Forest Classifier (`crop_model.pkl` + `label_encoder.pkl`)
- **Inputs:** Nitrogen (N), Phosphorus (P), Potassium (K), Soil pH, Temperature, Humidity, Rainfall
- **Endpoint:** `POST /api/crop_recommend`
- **Output:** Predicted crop + confidence score

---

### 2. Urvara — Fertilizer Recommendation System
> *"Fertile, that which nourishes."*

Recommends the optimal fertilizer based on soil composition and crop requirements, balancing nutrients precisely.

- **Model:** XGBoost Classifier (`fertilizer_model.pkl` + encoders)
- **Inputs:** Soil Type, Soil pH, Soil Moisture, Organic Carbon, Nitrogen, Phosphorus, Potassium levels, Temperature, Humidity, Rainfall, Crop Type, Growth Stage, Season, Previous Crop
- **Endpoint:** `POST /api/fertilizer_recommend`
- **Output:** Recommended fertilizer + confidence score

---

### 3. Krishi Vaidya — Disease Detection System
> *"Physician of the field."*

Detects crop diseases instantly from a leaf image and explains *why* it made that prediction.

- **Model:** EfficientNet-B0 (CNN-based Deep Learning architecture, PyTorch — `image_classification.pth`)
- **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the exact regions of the image influencing the prediction, returned as a base64 heatmap
- **Safeguards:** Image quality check, low-confidence rejection, and a vegetation-pixel heuristic to reject non-leaf images
- **Endpoint:** `POST /api/disease_detect`
- **AI Enrichment:** `POST /api/disease_info_stream` streams a structured breakdown (Description, Symptoms, Cause, Treatment, Prevention) generated via LLM, with fallbacks if the AI call fails
- **Output:** Disease classification, confidence, top-k predictions, Grad-CAM heatmap, AI-generated details

---

### 4. Crop Tracker & Krishi Patal

A complete lifecycle tracking system for every registered crop.

- **Crop Register:** `POST /api/add_crop` — adds a crop to the database and begins tracking it from day one, auto-calculating crop age
- **Harvest Estimation:** `POST /api/estimate_harvest` — LLM-based expected harvest date, with a fallback of sowing date + 120 days
- **Krishi Patal (Dashboard):** Displays records of all tracked crops — growth stage, history, and current status
- **Field Conditions:** `POST /api/add_crop_condition` — logs soil moisture/pH plus a live weather snapshot, auto-generating an AI timeline entry
- **Timeline Events:** `POST /api/add_timeline_event` — manually log custom events per crop
- **Weather Integration:** Live, location-based weather updates via external APIs
- **Daily AI Alerts:** background cron job analyzes each crop's condition, weather, and requirements to generate prioritized alerts and update health/stress/growth scores

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript (no framework) |
| Backend | Python, Flask, Flask-CORS |
| Database | Supabase (PostgreSQL via REST API) |
| Authentication | JWT (PyJWT) |
| Password Security | Werkzeug `generate_password_hash` / `check_password_hash` |
| Crop Recommendation | Random Forest Classifier (Scikit-learn, via `joblib`) |
| Fertilizer Recommendation | XGBoost Classifier (via `joblib`) |
| Disease Detection | EfficientNet-B0, CNN, PyTorch + Grad-CAM |
| Conversational AI / Text Generation | OpenRouter API — Llama 3.3 70B, Gemma 3 27B, Qwen3 235B, Hermes 3 405B, Mistral Small (multi-model fallback chain) |
| Weather Data | Open-Meteo API, OpenWeatherMap (Pro forecast) |
| Streaming | Server-Sent Events (SSE) for chat & disease info |
| Background Jobs | Python `threading` — triggered cron endpoints |
| Deployment | Vercel (frontend) |

---

## 🏗️ System Architecture

<img width="12751" height="9429" alt="System_Design_and_Architecture" src="https://github.com/user-attachments/assets/307cd79b-405c-43ea-a941-6a61c798ea7d" />


<details>
<summary><b>Text-based architecture overview</b></summary>

```
┌──────────────────────────┐
│   Frontend (HTML/CSS/JS)  │
│  home · login · register  │
│  crop_recommendation.html │
│  fertilizer_recommendation│
│  disease_detection.html   │
│  crop_register/tracker    │
└─────────────┬──────────────┘
              │  REST calls (fetch/axios) + JWT
┌─────────────▼──────────────┐
│      Flask Backend (app.py) │
│  Auth · Validation · Routing │
└───┬───────┬────────┬────────┘
    │       │        │
    │       │        └────────────────────┐
    │       │                             │
┌───▼───┐ ┌─▼─────────────┐   ┌───────────▼───────────┐
│Supabase│ │  ML/DL Models  │   │   External APIs        │
│(Postgres│ │ - Random Forest│   │ - OpenRouter (LLM)      │
│  REST)  │ │ - XGBoost      │   │ - Open-Meteo (Weather)  │
│         │ │ - EfficientNet-│   │ - OpenWeatherMap        │
│         │ │   B0 + GradCAM │   │                         │
└─────────┘ └────────────────┘   └─────────────────────────┘
    │
┌───▼─────────────────────────┐
│   Background Cron Threads    │
│ - Daily weather updates       │
│ - Daily AI-generated alerts   │
└────────────────────────────┘
```

</details>

---

## 🗄️ Database Schema (Supabase)

| Table | Purpose |
|---|---|
| `User_database` | Stores farmer accounts — name, contact, age, gender, state, district, hashed password |
| `crop_system` | Core registered-crop record — crop type, area, sowing/harvest dates, current stage, health score, growth progress, location |
| `crop_condition_snapshot` | Point-in-time soil & weather readings per crop (moisture, pH, temp, humidity, rainfall, health/stress scores) |
| `crop_timeline` | Chronological event log per crop (sowing, updates, AI-generated milestones) |
| `crop_weather` | Daily weather snapshot + AI-generated farming advice per crop |
| `crop_alerts` | AI-generated daily alerts with priority, category, title, and description |
| `crop_requirement` | Reference data on ideal conditions per crop type, used to evaluate current crop health |

---

## 🔐 Security

- **JWT-based Authentication & Authorization** — every session is verified through signed tokens (valid 7 days), with strict access control
- **Password Hashing & Encryption** — via Werkzeug (`generate_password_hash` / `check_password_hash`); credentials are never stored in plain text
- **Input Validation** — all user inputs are validated and type-checked before processing, preventing malformed or malicious data
- **Environment-based Secrets** — Supabase keys, JWT secret, and third-party API keys loaded via `.env`, never hardcoded in source
- **Secure API Communication** — RESTful endpoints with Flask-CORS for controlled cross-origin access

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Rajat6125/DharaVeda.git
cd DharaVeda

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install flask flask-cors requests python-dotenv werkzeug pyjwt joblib numpy pandas torch

# 3. Add your environment variables (see below)
# Create a .env file inside backend/

# 4. Ensure these model files are present in backend/
#    crop_model.pkl, label_encoder.pkl
#    fertilizer_model.pkl, fertilizer_encoder.pkl, fertilizer_target_encoder.pkl
#    image_classification.pth

# 5. Run the backend
python app.py
# Server runs on http://localhost:5000

# 6. Frontend
# Simply open index.html in a browser, or serve the root folder
# with any static file server (e.g., Live Server extension in VS Code)
```

---

## 🔑 Environment Variables

Create a `.env` file inside the `backend/` directory:

```env
SUPABASE_KEY=your_supabase_service_or_anon_key
JWT_SECRET=your_jwt_secret_key
OPENROUTER_API_KEY=your_openrouter_api_key
OPENWEATHER_API_KEY=your_openweathermap_api_key
```

> `SUPABASE_URL` is currently set directly in `app.py` — consider moving it to `.env` as well for better portability across environments.

---

## 📡 API Reference

### Authentication
| Endpoint | Method | Description |
|---|---|---|
| `/api/register` | POST | Register a new farmer account |
| `/api/verify` | POST | Check if a contact/email already exists |
| `/api/login` | POST | Authenticate and receive a JWT |

### Crop & Fertilizer Recommendation
| Endpoint | Method | Description |
|---|---|---|
| `/api/crop_recommend` | POST | Ksetrajna — predict best crop from soil/climate data |
| `/api/fertilizer_recommend` | POST | Urvara — predict best fertilizer from soil/crop data |

### Disease Detection
| Endpoint | Method | Description |
|---|---|---|
| `/api/disease_detect` | POST | Krishi Vaidya — detect disease from an uploaded leaf image, returns Grad-CAM heatmap |
| `/api/disease_info_stream` | POST | Streams structured AI-generated disease info (SSE) |

### Crop Tracking
| Endpoint | Method | Description |
|---|---|---|
| `/api/add_crop` | POST | Register a new crop for tracking |
| `/api/estimate_harvest` | POST | AI-estimated expected harvest date |
| `/api/add_crop_condition` | POST | Log soil + live weather snapshot for a crop |
| `/api/add_timeline_event` | POST | Add a custom timeline entry for a crop |

### AI Assistant
| Endpoint | Method | Description |
|---|---|---|
| `/api/ai_chat` | POST | Non-streaming AI chat (multi-model fallback) |
| `/api/ai_chat_stream` | POST | Streaming AI chat via SSE |

### Background / Cron
| Endpoint | Method | Description |
|---|---|---|
| `/api/cron/update_crop_weather` | GET/POST | Triggers background weather refresh for all tracked crops |
| `/api/cron/process_daily_crop_alerts` | GET/POST | Triggers background AI analysis + alert generation for all tracked crops |

---

## 📁 Project Structure

```
DharaVeda/
├── .vscode/
├── Design/                          # System design & architecture docs
├── Extra/                           # Disease detection model integration
├── Pic/                             # Images & static assets
├── backend/                         # Flask backend (app.py, ML models, .env)
├── index.html                       # Entry point
├── home.html                        # Main dashboard
├── login.html
├── register.html
├── crop_recommendation.html         # Ksetrajna UI
├── fertilizer_recommendation.html   # Urvara UI
├── disease_detection.html           # Krishi Vaidya UI
├── crop_register.html               # Add crop to tracker
├── crop_tracker.html                # Crop tracker dashboard (Krishi Patal)
├── dashboard.html
├── error.html
└── README.md
```

---

## ⏱️ Background Jobs

Two long-running processes keep the platform's data fresh, triggered via HTTP (suitable for external cron services like cron-job.org or a scheduled GitHub Action):

1. **`process_weather_cron`** — Fetches live weather (Open-Meteo + OpenWeatherMap) for every tracked crop once per day and stores it with AI-generated farming advice.
2. **`process_daily_crop_alerts_cron`** — Aggregates each crop's latest condition, timeline, weather, and requirements, sends it to an LLM, and stores a structured daily alert while updating the crop's health score, stress level, and growth progress.

Both run in background threads so the triggering HTTP request returns immediately (`202 Accepted`).

---

## 📸 Screenshots

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img width="300" alt="image" src="https://github.com/user-attachments/assets/a2d47d77-5dd6-48fa-af2d-3e68edafcac5" /><br />
        <sub><b>Cover page of the application</b></sub>
      </td>
      <td align="center">
        <img width="300" alt="image" src="https://github.com/user-attachments/assets/78f91d8b-b95f-4b5a-9681-bdbb62905883" /><br />
        <sub><b>Home Page of the application</b></sub>
      </td>
      <td align="center">
        <img width="300" alt="image" src="https://github.com/user-attachments/assets/fef6f119-813a-4be5-96a0-dd0d13eed1ea" /><br />
        <sub><b>Crop Recommendation Page of the application</b></sub>
      </td>
      <td align="center">
        <img width="300" alt="image" src="https://github.com/user-attachments/assets/26626069-624d-46c0-8710-171dda425249" /><br />
        <sub><b>Disease Detection page of the application</b></sub>
      </td>
    </tr>
  </table>
</p>

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img width="300" alt="image" src="https://github.com/user-attachments/assets/8000139d-1df5-4122-8b83-4353a84ea96d" /><br />
        <sub><b>Crop tracker page of the application</b></sub>
      </td>
      <td align="center">
        <img width="300" alt="image" src="https://github.com/user-attachments/assets/3803f7c0-6808-499c-8f3d-212c1d687da3" /><br />
        <sub><b>Crop Dashboard</b></sub>
      </td>
    </tr>
  </table>
</p>

---

## 🚀 Future Enhancements

- Multilingual support for regional farmers
- Offline-first mode for low-connectivity areas
- Market price prediction integration
- Voice-assisted interaction for accessibility
- Migrate `SUPABASE_URL` and table URLs fully into environment configuration
- Expanded disease dataset covering more crop varieties

---

## 🙏 Acknowledgements

This project would not have been possible without:

- The **farmers** whose real, everyday struggles gave this project its purpose and direction
- My **professors and mentors**, whose guidance helped navigate every technical challenge
- The **Futurense internship experience** that shaped how I approach building real-world, production-ready systems rather than purely academic ones

Dharaveda is a small step toward a larger goal — making AI genuinely useful for the people who feed a nation.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Author:** Rajat Thakur
**Email:** rajatthakur6125@gmail.com
**GitHub:** [Rajat6125](https://github.com/Rajat6125)
**LinkedIn:** [Rajat Thakur](https://www.linkedin.com/in/rajat-thakur-b79103321/)
**Live Project:** [dhara-veda.vercel.app](https://dhara-veda.vercel.app)

If you find this project useful or have feedback, feel free to open an issue or reach out directly.
