from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests
import os
import json
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta, timezone
import threading
import time
import joblib
import numpy as np
import pandas as pd

load_dotenv()

app = Flask(__name__)
CORS(app)

# Supabase Configuration
SUPABASE_URL = "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/User_database"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "dharaveda-secret-key-2026")

print("SUPABASE_KEY Loaded:", "YES" if SUPABASE_KEY else "NO")

try:
    model = joblib.load("crop_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    print(f"Warning: Could not load crop models: {e}")

try:
    fert_model = joblib.load("fertilizer_model.pkl")
    fert_encoder = joblib.load("fertilizer_encoder.pkl")
    fert_target_encoder = joblib.load("fertilizer_target_encoder.pkl")
except Exception as e:
    print(f"Warning: Could not load fertilizer models: {e}")

@app.route("/")
def home():
    return "Crop Recommendation API is running"

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        full_name = data.get("full_name")
        contact = data.get("contact")
        age = data.get("age")
        gender = data.get("gender")
        state = data.get("state")
        district = data.get("district")
        password = data.get("password")

        # Validation
        if not all([
            full_name,
            contact,
            age,
            gender,
            state,
            district,
            password
        ]):
            return jsonify({
                "error": "All fields are required"
            }), 400

        try:
            age = int(age)
        except ValueError:
            return jsonify({
                "error": "Age must be a number"
            }), 400

        hashed_password = generate_password_hash(password)

        payload = {
            "Name": full_name,
            "Email_Phone": contact,
            "Age": age,
            "Gender": gender,
            "State": state,
            "District": district,
            "Password": hashed_password
        }

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

        print("\n===== REQUEST DATA =====")
        print(payload)

        response = requests.post(
            SUPABASE_URL,
            json=payload,
            headers=headers
        )

        print("\n===== SUPABASE RESPONSE =====")
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        if response.status_code not in [200, 201]:
            return jsonify({
                "error": "Supabase Error",
                "status": response.status_code,
                "details": response.text
            }), response.status_code

        return jsonify({
            "message": "User registered successfully!",
            "data": response.json()
        }), 201

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json()
        if not data or not data.get("email_phone"):
            return jsonify({"error": "email_phone is required"}), 400
        
        email_phone = data.get("email_phone")
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        # Check if user exists
        response = requests.get(
            SUPABASE_URL,
            headers=headers,
            params={
                "select": "Email_Phone",
                "Email_Phone": f"eq.{email_phone}"
            }
        )
        
        if response.status_code == 200:
            users = response.json()
            if len(users) > 0:
                return jsonify({"exists": True}), 200
            else:
                return jsonify({"exists": False}), 200
        else:
             return jsonify({"error": "Supabase Error", "details": response.text}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data or not data.get("email_phone") or not data.get("password"):
            return jsonify({"error": "email_phone and password are required"}), 400
        
        email_phone = data.get("email_phone")
        password = data.get("password")
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        response = requests.get(
            SUPABASE_URL,
            headers=headers,
            params={
                "select": "Name,Email_Phone,Password,District,State",
                "Email_Phone": f"eq.{email_phone}"
            }
        )
        
        if response.status_code == 200:
            users = response.json()
            if len(users) > 0:
                user = users[0]
                db_password = user.get("Password")
                
                # Check hash first, fallback to plaintext check for backward compatibility
                is_valid = False
                try:
                    is_valid = check_password_hash(db_password, password)
                except ValueError:
                    # In case the hash string is completely invalid format
                    pass
                
                if is_valid or db_password == password:
                    # Generate JWT
                    token = jwt.encode({
                        "name": user.get("Name", "User"),
                        "email_phone": email_phone,
                        "district": user.get("District", ""),
                        "state": user.get("State", ""),
                        "exp": datetime.now(timezone.utc) + timedelta(days=7)
                    }, JWT_SECRET, algorithm="HS256")
                    
                    if isinstance(token, bytes):
                        token = token.decode('utf-8')
                    
                    return jsonify({
                        "success": True, 
                        "message": "Login successful", 
                        "token": token, 
                        "user": {
                            "name": user.get("Name", "User"),
                            "district": user.get("District", ""),
                            "state": user.get("State", "")
                        }
                    }), 200
                else:
                    return jsonify({"success": False, "message": "Invalid password"}), 401
            else:
                return jsonify({"success": False, "message": "User not found"}), 404
        else:
             return jsonify({"error": "Supabase Error", "details": response.text}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/estimate_harvest', methods=['POST'])
def estimate_harvest():
    try:
        data = request.get_json()
        crop = data.get("crop", "")
        area = data.get("area", "")
        sowing_date = data.get("sowing_date", "")
        
        if not crop or not sowing_date:
            return jsonify({"success": False, "error": "crop and sowing_date are required"}), 400
            
        prompt = f"Given the crop '{crop}', an area of {area} acres, and a sowing date of {sowing_date}, what is the expected harvest date? Please reply ONLY with the exact date string in YYYY-MM-DD format, no introductory text."
        
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use a fast model specifically for this quick extraction
        payload = {
            "model": "google/gemma-3-27b-it:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 15,
            "stream": False
        }
        
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=15)
        
        expected_harvest = None
        if resp.ok:
            result = resp.json()
            reply = result["choices"][0]["message"]["content"]
            # Extract YYYY-MM-DD
            import re
            match = re.search(r'\d{4}-\d{2}-\d{2}', reply)
            if match:
                expected_harvest = match.group(0)
                
        if not expected_harvest:
            # Fallback to +120 days if AI fails or returns weird output
            from datetime import datetime, timedelta
            try:
                sowing = datetime.strptime(sowing_date, "%Y-%m-%d")
                expected_harvest = (sowing + timedelta(days=120)).strftime("%Y-%m-%d")
            except:
                expected_harvest = sowing_date
                
        return jsonify({"success": True, "expected_harvest": expected_harvest})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/add_crop', methods=['POST'])
def add_crop():
    try:
        data = request.get_json()
        email_phone = data.get("email_phone")
        crop = data.get("crop")
        area = data.get("area")
        sowing_date = data.get("sowing_date")
        expected_harvest = data.get("expected_harvest")
        latt = data.get("latt")
        long = data.get("long")
        district = data.get("district")
        state = data.get("state")
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        # 1. Get farmer_id
        user_resp = requests.get(
            SUPABASE_URL,
            headers=headers,
            params={"select": "*", "Email_Phone": f"eq.{email_phone}"}
        )
        
        if user_resp.status_code != 200:
            return jsonify({"error": f"Database error (status {user_resp.status_code}): {user_resp.text}"}), 400
            
        users = user_resp.json()
        if len(users) == 0:
            return jsonify({"error": f"User not found for email/phone: '{email_phone}'"}), 404
            
        user = users[0]
        # Try to find the ID column regardless of casing
        farmer_id = user.get("id") or user.get("ID") or user.get("Id") or user.get("farmer_id") or user.get("Farmer_id")
        
        if not farmer_id:
            return jsonify({"error": f"Could not find an 'id' column for user. Found columns: {list(user.keys())}"}), 400
            
        # Calculate age
        try:
            s_date = datetime.strptime(sowing_date, "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()
            age_days = (today - s_date).days
            if age_days < 0: age_days = 0
        except Exception:
            age_days = 0

        # 2. Insert into crop_system
        crop_system_url = "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_system"
        
        payload = {
            "farmer_id": farmer_id,
            "crop": crop,
            "Area": int(area),
            "sowing_date": sowing_date,
            "expected_harvest": expected_harvest,
            "current_stage": "Sowed",
            "age": age_days,
            "health_score": 0,
            "growth_progress": 0,
            "latt": float(latt) if latt else 0.0,
            "long": float(long) if long else 0.0,
            "district": district,
            "state": state
        }
        
        headers["Content-Type"] = "application/json"
        headers["Prefer"] = "return=representation"
        
        insert_resp = requests.post(
            crop_system_url,
            json=payload,
            headers=headers
        )
        
        if insert_resp.status_code in [200, 201]:
            return jsonify({"success": True, "data": insert_resp.json()}), 201
        else:
            return jsonify({"error": f"Failed to add crop: {insert_resp.text}"}), insert_resp.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai_chat", methods=["POST"])
def ai_chat():
    """
    Proxy endpoint for OpenRouter LLM calls.
    Accepts: { "messages": [...] }
    Returns: { "success": True, "reply": "..." }
    Tries multiple free models in order until one succeeds.
    """
    FREE_MODELS = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "qwen/qwen3-235b-a22b:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "mistralai/mistral-small-24b-instruct-2501",
    ]

    try:
        data = request.get_json()
        if not data or "messages" not in data:
            return jsonify({"success": False, "error": "messages array is required"}), 400

        messages = data["messages"]
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://dharaveda.app",
            "X-Title": "DharaVeda"
        }

        last_error = "Unknown error"
        for model_id in FREE_MODELS:
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 800,
                "stream": False
            }
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                if resp.status_code == 429:
                    last_error = f"Rate limited on {model_id}"
                    continue  # try next model
                if not resp.ok:
                    last_error = f"Error {resp.status_code} on {model_id}: {resp.text[:200]}"
                    continue
                result = resp.json()
                reply = result["choices"][0]["message"]["content"]
                return jsonify({"success": True, "reply": reply, "model": model_id})
            except Exception as model_err:
                last_error = str(model_err)
                continue

        return jsonify({"success": False, "error": f"All models failed. Last error: {last_error}"}), 503

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/ai_chat_stream", methods=["POST"])
def ai_chat_stream():
    """
    SSE streaming endpoint for OpenRouter LLM calls.
    Accepts: { "messages": [...] }
    Streams tokens as SSE: data: <token>\n\n
    Always ends with: data: [DONE]\n\n
    Tries multiple free models in order, falling back on 429 or errors.
    """
    FREE_MODELS = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "qwen/qwen3-235b-a22b:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "mistralai/mistral-small-24b-instruct-2501",
    ]

    data = request.get_json()
    if not data or "messages" not in data:
        def error_gen():
            yield b"data: [ERROR] messages array is required\n\n"
            yield b"data: [DONE]\n\n"
        return Response(stream_with_context(error_gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    messages = data["messages"]
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://dharaveda.app",
        "X-Title": "DharaVeda"
    }

    def generate():
        last_error = "Unknown error"
        for model_id in FREE_MODELS:
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 700,
                "stream": True
            }
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60,
                    stream=True
                )
                if resp.status_code == 429:
                    last_error = f"Rate limited on {model_id}"
                    continue
                if not resp.ok:
                    last_error = f"Error {resp.status_code} on {model_id}: {resp.text[:200]}"
                    continue

                # Stream the response tokens
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line.startswith("data: "):
                        chunk_str = line[6:]
                        if chunk_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield f"data: {token}\n\n".encode("utf-8")
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue

                # Successfully streamed — done
                yield b"data: [DONE]\n\n"
                return

            except Exception as model_err:
                last_error = str(model_err)
                continue

        # All models failed
        yield f"data: [ERROR] All models failed. Last error: {last_error}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/crop_recommend", methods=["POST"])
def predict():
    try:
        # Accept both JSON (from fetch/axios) and Form Data (from standard HTML forms)
        data = request.get_json(silent=True) or request.form

        # Safely extract values and convert them to float (handling form string inputs)
        # Ensure order matches standard Kaggle crop dataset: N, P, K, temperature, humidity, ph, rainfall
        features = np.array([[
            float(data.get("N") or 0),
            float(data.get("P") or 0),
            float(data.get("K") or 0),
            float(data.get("temperature") or 0),
            float(data.get("humidity") or 0),
            float(data.get("ph") or 0),
            float(data.get("rainfall") or 0)
        ]])

        # Predict best crop
        predicted_id = model.predict(features)[0]
        predicted_crop = label_encoder.inverse_transform([predicted_id])[0]

        # Confidence
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[predicted_id])

        return jsonify({
            "success": True,
            "crop": predicted_crop,
            "confidence": round(confidence * 100, 2)-10
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route("/api/fertilizer_recommend", methods=["POST"])
def predict_fertilizer():
    try:
        data = request.get_json(silent=True) or request.form

        # Create a DataFrame for the incoming data to match model expectations
        sample = pd.DataFrame({
            "Soil_Type": [data.get("Soil_Type")],
            "Soil_pH": [float(data.get("Soil_pH") or 0)],
            "Soil_Moisture": [float(data.get("Soil_Moisture") or 0)],
            "Organic_Carbon": [float(data.get("Organic_Carbon") or 0)],
            "Nitrogen_Level": [float(data.get("Nitrogen_Level") or 0)],
            "Phosphorus_Level": [float(data.get("Phosphorus_Level") or 0)],
            "Potassium_Level": [float(data.get("Potassium_Level") or 0)],
            "Temperature": [float(data.get("Temperature") or 0)],
            "Humidity": [float(data.get("Humidity") or 0)],
            "Rainfall": [float(data.get("Rainfall") or 0)],
            "Crop_Type": [data.get("Crop_Type")],
            "Crop_Growth_Stage": [data.get("Crop_Growth_Stage")],
            "Season": [data.get("Season")],
            "Previous_Crop": [data.get("Previous_Crop")]
        })

        # Encode categorical features
        cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Previous_Crop"]
        sample[cat_cols] = fert_encoder.transform(sample[cat_cols])

        # Predict fertilizer
        predicted_id = fert_model.predict(sample)[0]
        predicted_fertilizer = fert_target_encoder.inverse_transform([predicted_id])[0]

        # Get confidence probability
        probabilities = fert_model.predict_proba(sample)[0]
        confidence = float(probabilities[predicted_id])

        return jsonify({
            "success": True,
            "fertilizer": predicted_fertilizer,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/api/add_crop_condition', methods=['POST'])
def add_crop_condition():
    try:
        data = request.get_json()
        crop_id = data.get("crop_id")
        crop_name = data.get("crop_name", "Crop")
        soil_moisture = float(data.get("soil_moisture", 0))
        ph = float(data.get("ph", 0))
        latt = float(data.get("latt", 0))
        long = float(data.get("long", 0))

        temp, hum, rain = 0.0, 0.0, 0.0
        if latt != 0 and long != 0:
            try:
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latt}&longitude={long}&current=temperature_2m,relative_humidity_2m,precipitation"
                w_resp = requests.get(weather_url, timeout=5)
                if w_resp.status_code == 200:
                    w_data = w_resp.json().get("current", {})
                    temp = w_data.get("temperature_2m", 0.0)
                    hum = w_data.get("relative_humidity_2m", 0.0)
                    rain = w_data.get("precipitation", 0.0)
            except Exception as e:
                print("Weather API error:", e)

        health_score = 10 if rain > 0 else 0
        stress_level = 0 if rain > 0 else 5
        now_str = datetime.now(timezone.utc).isoformat()

        payload = {
            "crop_id": crop_id,
            "date": now_str,
            "soil_moisture": soil_moisture,
            "ph": ph,
            "temperature": temp,
            "humidity": hum,
            "rainfall": rain,
            "health_score": health_score,
            "stress_level": stress_level
        }

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        insert_resp = requests.post(
            "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_condition_snapshot",
            json=payload,
            headers=headers
        )

        if insert_resp.status_code in [200, 201]:
            # Generate AI description for timeline
            ai_description = f"Crop of {crop_name} sown on {now_str[:10]}. Initial field conditions recorded: Soil Moisture {soil_moisture}%, pH {ph}, Temp {temp}°C, Humidity {hum}%, Rainfall {rain}mm."
            try:
                prompt = f"Write a short, professional 2 sentence timeline entry for a crop registration. Details: Crop: {crop_name}, Date: {now_str[:10]}, Moisture: {soil_moisture}%, pH: {ph}, Temp: {temp}C, Humidity: {hum}%, Rainfall: {rain}mm. Start exactly with: 'Crop of {crop_name} sown on {now_str[:10]}.'"
                api_key = os.getenv("OPENROUTER_API_KEY", "")
                if api_key:
                    ai_resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json={
                            "model": "meta-llama/llama-3.3-70b-instruct:free",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 150
                        },
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=5
                    )
                    if ai_resp.ok:
                        ai_description = ai_resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print("Timeline AI Error:", e)

            # Insert into crop_timeline
            timeline_payload = {
                "crop_id": crop_id,
                "date": now_str,
                "event_type": "Sowing",
                "description": ai_description
            }
            requests.post(
                "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_timeline",
                json=timeline_payload,
                headers=headers
            )
            
            return jsonify({"success": True, "data": insert_resp.json()}), 201
        else:
            return jsonify({"error": f"Failed to add condition: {insert_resp.text}"}), insert_resp.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_timeline_event', methods=['POST'])
def add_timeline_event():
    try:
        data = request.get_json()
        crop_id = data.get("crop_id")
        event_type = data.get("event_type", "Update")
        description = data.get("description", "")
        event_date = data.get("date")
        
        if not crop_id:
            return jsonify({"success": False, "error": "crop_id is required"}), 400
            
        if not event_date:
            event_date = datetime.now(timezone.utc).isoformat()
            
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        payload = {
            "crop_id": crop_id,
            "date": event_date,
            "event_type": event_type,
            "description": description
        }
        
        resp = requests.post(
            "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_timeline",
            json=payload,
            headers=headers
        )
        
        if resp.status_code in [200, 201]:
            return jsonify({"success": True, "data": resp.json()}), 201
        else:
            return jsonify({"success": False, "error": f"Failed to add timeline event: {resp.text}"}), resp.status_code
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def process_weather_cron():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    try:
        resp = requests.get("https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_system?select=crop_id,latt,long,crop", headers=headers)
        if not resp.ok: return
        crops = resp.json()
        
        openweather_key = os.getenv("OPENWEATHER_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        
        for crop in crops:
            crop_id = crop.get("crop_id")
            latt = crop.get("latt")
            long = crop.get("long")
            crop_name = crop.get("crop", "Unknown")
            
            if not latt or not long: continue
            
            temp, rain, humidity, wind = 0, 0, 0, 0
            
            try:
                w_url = f"https://api.open-meteo.com/v1/forecast?latitude={latt}&longitude={long}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
                w_resp = requests.get(w_url, timeout=5)
                if w_resp.ok:
                    curr = w_resp.json().get("current", {})
                    temp = int(curr.get("temperature_2m", 0))
                    humidity = int(curr.get("relative_humidity_2m", 0))
                    rain = int(curr.get("precipitation", 0))
                    wind = int(curr.get("wind_speed_10m", 0))
            except: pass
            
            forecast_text = ""
            if openweather_key:
                try:
                    fw_url = f"https://pro.openweathermap.org/data/2.5/forecast/hourly?lat={latt}&lon={long}&appid={openweather_key}&units=metric"
                    fw_resp = requests.get(fw_url, timeout=5)
                    if fw_resp.ok:
                        fw_data = fw_resp.json()
                        forecast_text = fw_data["list"][0]["weather"][0]["description"].capitalize()
                except: pass
            
            if not forecast_text:
                forecast_text = f"Temp {temp}C, Hum {humidity}%, Rain {rain}mm expected."

            advice_text = "Monitor field conditions."
            if openrouter_key:
                try:
                    prompt = f"Given weather for {crop_name}: Temp={temp}C, Rain={rain}mm, Humidity={humidity}%, Wind={wind}km/h, Forecast: {forecast_text}. Provide 1 concise sentence of farming advice."
                    llm_resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json={
                            "model": "meta-llama/llama-3.3-70b-instruct:free",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 100
                        },
                        headers={"Authorization": f"Bearer {openrouter_key}"},
                        timeout=10
                    )
                    if llm_resp.ok:
                        advice_text = llm_resp.json()["choices"][0]["message"]["content"].strip()
                except: pass
                
            w_payload = {
                "crop_id": crop_id,
                "date": datetime.now(timezone.utc).isoformat(),
                "temp": temp,
                "rain": rain,
                "humidity": humidity,
                "wind": wind,
                "forecast": forecast_text,
                "advice": advice_text,
                "latt": int(latt) if latt else 0,
                "long": int(long) if long else 0
            }
            
            requests.post("https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_weather", json=w_payload, headers=headers)
            time.sleep(1.5)
            
    except Exception as e:
        print("Cron Weather Error:", e)

def process_daily_crop_alerts_cron():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    try:
        resp = requests.get("https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_system?select=crop_id,crop,current_stage", headers=headers)
        if not resp.ok: return
        crops = resp.json()
        
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if not openrouter_key: return
        
        for crop in crops:
            crop_id = crop.get("crop_id")
            crop_name = crop.get("crop")
            
            if not crop_id: continue
            
            # Fetch latest data
            cond_resp = requests.get(
                "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_condition_snapshot", 
                headers=headers, 
                params={"crop_id": f"eq.{crop_id}", "order": "date.desc", "limit": "1"}
            )
            cond_data = cond_resp.json()[0] if cond_resp.ok and cond_resp.json() else {}
            
            timeline_resp = requests.get(
                "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_timeline", 
                headers=headers, 
                params={"crop_id": f"eq.{crop_id}", "order": "date.desc", "limit": "1"}
            )
            timeline_data = timeline_resp.json()[0] if timeline_resp.ok and timeline_resp.json() else {}
            
            weather_resp = requests.get(
                "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_weather", 
                headers=headers, 
                params={"crop_id": f"eq.{crop_id}", "order": "date.desc", "limit": "1"}
            )
            weather_data = weather_resp.json()[0] if weather_resp.ok and weather_resp.json() else {}
            
            # Try matching 'crop_name' or 'crop' in crop_requirement
            req_resp = requests.get(
                "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_requirement", 
                headers=headers, 
                params={"crop_name": f"ilike.{crop_name}"}
            )
            if not req_resp.ok or not req_resp.json():
                req_resp = requests.get(
                    "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_requirement", 
                    headers=headers, 
                    params={"crop": f"ilike.{crop_name}"}
                )
            req_data = req_resp.json()[0] if req_resp.ok and req_resp.json() else {}
            
            prompt = f"""
            Analyze the following crop data for {crop_name} and compare with its requirements to generate a daily alert and condition scores.
            Current Condition: {cond_data}
            Latest Event: {timeline_data}
            Weather Forecast: {weather_data}
            Crop Requirements: {req_data}
            
            Return a JSON object ONLY, with NO extra text or markdown formatting. Use the following exact keys and types:
            "priority": string ("High" / "Medium" / "Low"),
            "category": string ("Weather" / "Health" / "General"),
            "title": string (Short title),
            "description": string (Detailed observation),
            "stress_level": integer (1-10),
            "health_score": integer (1-10),
            "growth_progress": integer (1-100)
            """
            
            llm_resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "messages": [{"role": "user", "content": prompt}]
                },
                headers={"Authorization": f"Bearer {openrouter_key}"},
                timeout=30
            )
            
            if llm_resp.ok:
                try:
                    res_content = llm_resp.json()["choices"][0]["message"]["content"].strip()
                    import re
                    match = re.search(r'```(?:json)?\s*(.*?)\s*```', res_content, re.DOTALL)
                    if match:
                        res_content = match.group(1).strip()
                    
                    parsed = {}
                    try:
                        parsed = json.loads(res_content)
                    except Exception as parse_e:
                        print("JSON Parse error, falling back to defaults:", parse_e)
                        parsed = {
                            "priority": "Medium",
                            "category": "General",
                            "title": "Daily Status Update",
                            "description": "System generated daily check. Unable to parse AI response.",
                            "stress_level": 5,
                            "health_score": 8,
                            "growth_progress": 50
                        }
                    
                    # 1. Insert into crop_alerts
                    alert_payload = {
                        "crop_id": crop_id,
                        "crop": crop_name,
                        "priority": parsed.get("priority", "Low"),
                        "category": parsed.get("category", "General"),
                        "title": parsed.get("title", "Daily Update"),
                        "description": parsed.get("description", "Condition normal"),
                        "status": "Open",
                        "time": datetime.now(timezone.utc).isoformat()
                    }
                    requests.post("https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_alerts", json=alert_payload, headers=headers)
                    
                    # 2. Update crop_condition_snapshot
                    cond_id = cond_data.get("Record_Number") or cond_data.get("id")
                    if cond_id:
                        patch_headers = headers.copy()
                        patch_headers["Prefer"] = "return=minimal"
                        requests.patch(
                            "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_condition_snapshot",
                            params={"Record_Number": f"eq.{cond_id}"},
                            json={"stress_level": parsed.get("stress_level", 5), "health_score": parsed.get("health_score", 8)},
                            headers=patch_headers
                        )
                    
                    # 3. Update crop_system
                    sys_patch_headers = headers.copy()
                    sys_patch_headers["Prefer"] = "return=minimal"
                    requests.patch(
                        "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_system",
                        params={"crop_id": f"eq.{crop_id}"},
                        json={"health_score": parsed.get("health_score", 8), "growth_progress": parsed.get("growth_progress", 10)},
                        headers=sys_patch_headers
                    )
                except Exception as e:
                    print("Alert AI Parsing error:", e, "Response:", res_content)
            else:
                print(f"LLM request failed: {llm_resp.status_code} {llm_resp.text}")
                parsed = {
                    "priority": "Medium",
                    "category": "General",
                    "title": "Daily Status Update",
                    "description": "System generated daily check. AI service unavailable.",
                    "stress_level": 5,
                    "health_score": 8,
                    "growth_progress": 50
                }
                # 1. Insert into crop_alerts
                alert_payload = {
                    "crop_id": crop_id,
                    "crop": crop_name,
                    "priority": parsed.get("priority", "Low"),
                    "category": parsed.get("category", "General"),
                    "title": parsed.get("title", "Daily Update"),
                    "description": parsed.get("description", "Condition normal"),
                    "status": "Open",
                    "time": datetime.now(timezone.utc).isoformat()
                }
                requests.post("https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_alerts", json=alert_payload, headers=headers)
                
                # 2. Update crop_condition_snapshot
                cond_id = cond_data.get("Record_Number") or cond_data.get("id")
                if cond_id:
                    patch_headers = headers.copy()
                    patch_headers["Prefer"] = "return=minimal"
                    requests.patch(
                        "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_condition_snapshot",
                        params={"Record_Number": f"eq.{cond_id}"},
                        json={"stress_level": parsed.get("stress_level", 5), "health_score": parsed.get("health_score", 8)},
                        headers=patch_headers
                    )
                
                # 3. Update crop_system
                sys_patch_headers = headers.copy()
                sys_patch_headers["Prefer"] = "return=minimal"
                requests.patch(
                    "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/crop_system",
                    params={"crop_id": f"eq.{crop_id}"},
                    json={"health_score": parsed.get("health_score", 8), "growth_progress": parsed.get("growth_progress", 10)},
                    headers=sys_patch_headers
                )
            
            time.sleep(2)
            
    except Exception as e:
        print("Cron Alert Error:", e)

@app.route('/api/cron/update_crop_weather', methods=['GET', 'POST'])
def trigger_update_crop_weather():
    thread = threading.Thread(target=process_weather_cron)
    thread.start()
    return jsonify({"success": True, "message": "Weather update triggered in background"}), 202

@app.route('/api/cron/process_daily_crop_alerts', methods=['GET', 'POST'])
def trigger_daily_crop_alerts():
    thread = threading.Thread(target=process_daily_crop_alerts_cron)
    thread.start()
    return jsonify({"success": True, "message": "Daily crop alerts processing triggered in background"}), 202

if __name__ == "__main__":
    app.run(debug=True, port=5000)