from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta, timezone
import joblib
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)

# Supabase Configuration
SUPABASE_URL = "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/User_database"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "dharaveda-secret-key-2026")

print("SUPABASE_KEY Loaded:", "YES" if SUPABASE_KEY else "NO")

model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

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

@app.route("/api/crop_recommend", methods=["POST"])
def predict():
    try:
        # Accept both JSON (from fetch/axios) and Form Data (from standard HTML forms)
        data = request.get_json(silent=True) or request.form

        # Safely extract values and convert them to float (handling form string inputs)
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)