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
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

TREATMENTS = {
    'Apple___Apple_scab': 'Prune trees for air flow. Apply sulfur-based fungicides early.',
    'Apple___Black_rot': 'Remove all dead wood and mummified fruit. Apply fungicides during the bloom period.',
    'Apple___Cedar_apple_rust': 'Remove nearby juniper bushes. Apply preventative fungicide in spring.',
    'Apple___healthy': 'Your apple tree looks healthy! Maintain seasonal pruning.',
    'Corn___Common_rust': 'Apply mancozeb-based fungicide. Ensure good spacing between plants.',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant hybrids and apply fungicides like pyraclostrobin.',
    'Corn___Northern_Leaf_Blight': 'Rotate crops next season. Clear crop debris from the field.',
    'Corn___healthy': 'Corn is thriving. Continue regular weeding and moisture monitoring.',
    'Potato___Early_blight': 'Maintain plant vigor. Apply preventative fungicides in humid weather.',
    'Potato___Late_blight': 'CRITICAL: Monitor daily. Remove and destroy infected plants immediately.',
    'Potato___healthy': 'Potatoes are healthy. Hilling is recommended to protect tubers.',
    'Tomato___Early_blight': 'Remove lower infected leaves. Apply copper-based fungicide spray.',
    'Tomato___Late_blight': 'CRITICAL: Destroy infected plants. Avoid overhead irrigation.',
    'Tomato___Bacterial_spot': 'Use copper-based bactericides. Avoid working in wet fields.',
    'Tomato___healthy': 'Tomatoes look great! Continue balanced fertilization.',
    'Background_without_leaves': 'The AI does not detect a leaf in this image. Please take a clearer photo.'
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

# FIXED MARKET ROUTE (Matches Dashboard Form)
@app.route('/admin/update-market', methods=['POST'])
@login_required
def update_market_dashboard():
    if not current_user.is_admin:
        return redirect(url_for('home'))
    
    # Check for generic inputs (crop_name and price)
    name = request.form.get('crop_name')
    price = request.form.get('price')
    
    # Also check for specific inputs (teff/wheat) if you haven't updated HTML yet
    teff_val = request.form.get('teff')
    wheat_val = request.form.get('wheat')

    if name and price:
        item = MarketPrice.query.filter_by(crop_name=name).first()
        if item: item.price = price
        else: db.session.add(MarketPrice(crop_name=name, price=price))
    
    if teff_val:
        t_item = MarketPrice.query.filter_by(crop_name='Teff').first()
        if t_item: t_item.price = teff_val
        else: db.session.add(MarketPrice(crop_name='Teff', price=teff_val))

    if wheat_val:
        w_item = MarketPrice.query.filter_by(crop_name='Wheat').first()
        if w_item: w_item.price = wheat_val
        else: db.session.add(MarketPrice(crop_name='Wheat', price=wheat_val))
    
    db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/market', methods=['GET', 'POST'])
@login_required
def admin_market():
    if not current_user.is_admin: return redirect(url_for('home'))
    if request.method == 'POST':
        crop = request.form.get('crop_name')
        new_price = request.form.get('price')
        item = MarketPrice.query.filter_by(crop_name=crop).first()
        if item: item.price = new_price
        else: db.session.add(MarketPrice(crop_name=crop, price=new_price))
        db.session.commit()
        return redirect(url_for('admin_market'))
    prices = MarketPrice.query.all()
    return render_template('admin_market.html', prices=prices)

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
        img = Image.open(path).convert('RGB').resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        result_index = np.argmax(preds)
        raw_name = CLASS_NAMES[result_index]
        diagnosis = raw_name.replace("___", " ").replace("_", " ")
        treatment = TREATMENTS.get(raw_name, 'Consult the Gondar Agricultural Bureau for detailed treatment.')
        return render_template('ai_detection.html', 
                               result={'diagnosis': diagnosis, 'treatment': treatment, 'confidence': f"{round(100 * np.max(preds), 1)}%", 'icon': '🌱'}, 
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