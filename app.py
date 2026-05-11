import os
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.secret_key = "helen_agritech_2026_secure"

# --- 1. AI MODEL SETUP ---
MODEL_PATH = 'helen_agritech_model.h5'
try:
    # Load model with compile=False to avoid local configuration errors
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# CRITICAL: This list MUST match the alphabetical order of your training folders
CLASS_NAMES = [
    'Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy Apple',
    'Healthy Blueberry', 'Healthy Cherry', 'Cherry Powdery Mildew', 
    'Corn Gray Leaf Spot', 'Corn Common Rust', 'Healthy Corn', 
    'Corn Northern Leaf Blight', 'Grape Black Rot', 'Grape Esca', 
    'Healthy Grape', 'Grape Leaf Blight', 'Orange Haunglongbing', 
    'Peach Bacterial Spot', 'Healthy Peach', 'Bell Pepper Bacterial Spot', 
    'Healthy Bell Pepper', 'Potato Early Blight', 'Healthy Potato', 
    'Potato Late Blight', 'Healthy Raspberry', 'Healthy Soybean', 
    'Squash Powdery Mildew', 'Healthy Strawberry', 'Strawberry Leaf Scorch', 
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Healthy Tomato', 
    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 
    'Tomato Spider Mites', 'Tomato Target Spot', 'Tomato Mosaic Virus', 
    'Tomato Yellow Leaf Curl Virus'
]

TREATMENTS = {
    'Healthy Corn': 'No disease detected. Continue regular weeding and moisture monitoring.',
    'Corn Common Rust': 'Apply mancozeb-based fungicide. Ensure good spacing between plants.',
    'Corn Gray Leaf Spot': 'Use resistant hybrids. Apply fungicides like pyraclostrobin if symptoms persist.',
    'Corn Northern Leaf Blight': 'Rotate crops next season. Clear crop debris from the field.',
    'Tomato Early Blight': 'Remove lower infected leaves. Apply copper-based fungicide spray.',
    'Tomato Late Blight': 'CRITICAL: Destroy infected plants. Avoid overhead irrigation.',
    'Tomato Bacterial Spot': 'Use copper-based bactericides. Avoid working in wet fields.',
    'Healthy Tomato': 'Continue balanced fertilization and consistent watering.',
    'Potato Early Blight': 'Maintain plant vigor. Apply preventative fungicides in humid weather.',
    'Potato Late Blight': 'Monitor daily. Remove infected plants immediately.',
    'Healthy Potato': 'Hilling is recommended to protect tubers.',
    'Apple Scab': 'Prune trees for air flow. Apply sulfur-based fungicides early.',
    'Healthy Apple': 'Trees look healthy. Maintain seasonal pruning.',
    'Cedar Apple Rust': 'Remove nearby juniper bushes. Apply preventative fungicide in spring.'
}

# --- 2. DATABASE CONFIG ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'helen_agritech.db')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    phone = db.Column(db.String(20), unique=True)
    password = db.Column(db.String(100))
    is_admin = db.Column(db.Boolean, default=False)

class MarketPrice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crop_name = db.Column(db.String(50), unique=True)
    price = db.Column(db.String(50))

class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text)
    date_posted = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- 3. ROUTES ---

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/about')
def about(): 
    return render_template('about.html')

@app.route('/market')
def market_view():
    prices = MarketPrice.query.all()
    return render_template('market.html', prices=prices)

@app.route('/weather')
def weather():
    API_KEY = "f7c723101f594c533f85eff5f872fa81"
    CITY = "Gondar"
    URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(URL, timeout=5)
        data = response.json()
        forecast = {
            'location': data.get('name', 'Gondar'),
            'temp': f"{round(data['main']['temp'])}°C",
            'condition': data['weather'][0]['main'],
            'humidity': f"{data['main']['humidity']}%",
            'wind': f"{data['wind']['speed']} km/h",
            'advice': 'Conditions in Gondar are ideal for field work.',
            'weekly': [{'day': 'MON', 'icon': '☀️', 'high': '26°', 'low': '14°'}]
        }
    except Exception:
        forecast = {'location': 'Gondar', 'temp': '23°C', 'condition': 'Partly Cloudy', 'humidity': '45%', 'wind': '10 km/h', 'advice': 'Local data active.'}
    return render_template('weather.html', forecast=forecast)

@app.route('/news')
def news():
    items = News.query.order_by(News.date_posted.desc()).all()
    return render_template('news.html', news_items=items)

@app.route('/ai-detection')
@login_required
def ai_detection():
    return render_template('ai_detection.html')

@app.route('/detect', methods=['POST'])
@login_required
def detect():
    file = request.files.get('file')
    if file and model:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        # 1. LOAD & CONVERT
        img = Image.open(path).convert('RGB').resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. THE PREPROCESSING FIX (Scale to -1 to 1)
        # This is standard for MobileNetV2/PlantVillage
        img_array = (img_array / 127.5) - 1.0 
        
        # 3. PREDICT
        preds = model.predict(img_array)
        result_index = np.argmax(preds)
        confidence = round(100 * np.max(preds), 1)
        
        # TERMINAL LOGGING (Watch your console!)
        print(f"--- AI DIAGNOSIS ---")
        print(f"Predicted Index: {result_index}")
        print(f"Confidence: {confidence}%")
        
        diagnosis = CLASS_NAMES[result_index] if result_index < len(CLASS_NAMES) else "Unknown"
        treatment = TREATMENTS.get(diagnosis, 'Consult the Gondar Agricultural Bureau for detailed treatment.')
        
        return render_template('ai_detection.html', 
                               result={'diagnosis': diagnosis, 
                                       'treatment': treatment, 
                                       'confidence': f"{confidence}%",
                                       'icon': '🌱'}, 
                               user_image=filename)
    return redirect(url_for('ai_detection'))

@app.route('/login')
def login(): return render_template('login.html', mode='login')

@app.route('/signup')
def signup(): return render_template('login.html', mode='signup')

@app.route('/auth', methods=['POST'])
def auth():
    f_type = request.form.get('form_type')
    phone = request.form.get('phone', '').strip()
    pwd = request.form.get('password', '').strip()
    user = User.query.filter_by(phone=phone).first()
    if f_type == 'signup' and not user:
        new_u = User(name=request.form.get('name'), phone=phone, password=pwd, is_admin=False)
        db.session.add(new_u); db.session.commit()
        login_user(new_u); return redirect(url_for('home'))
    if user and user.password == pwd:
        login_user(user)
        return redirect(url_for('admin_dashboard' if user.is_admin else 'home'))
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin: return redirect(url_for('home'))
    stats = {'total_users': User.query.count(), 'active_cases': 14, 'market_trend': "+8%", 'system_health': "Optimal"}
    return render_template('admin_dashboard.html', stats=stats)

@app.route('/admin/post-news', methods=['POST'])
@login_required
def post_news():
    content = request.form.get('headline')
    if content:
        db.session.add(News(content=content)); db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(phone="0954799790").first():
            db.session.add(User(name="Helen", phone="0954799790", password="1234", is_admin=True))
        db.session.commit()
    app.run(debug=True)