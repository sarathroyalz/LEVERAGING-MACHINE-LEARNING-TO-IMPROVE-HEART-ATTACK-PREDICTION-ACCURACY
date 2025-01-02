import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import pickle
import pymysql

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key for production

# Configure the database connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/heart_disease_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Initialize the database object
db = SQLAlchemy(app)

# Load the model and scaler
try:
    sc = pickle.load(open('sc.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or scaler: {e}")

# User model for registration and login
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# LoginActivity model to log login attempts
class LoginActivity(db.Model):
    __tablename__ = 'login_activity'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    login_time = db.Column(db.DateTime, server_default=db.func.now())
    status = db.Column(db.String(50), default='Success')
    user = db.relationship('User', backref=db.backref('login', lazy=True))

# Default route to redirect to signup page
@app.route('/')
def home():
    return redirect('/signup')

# Route for user registration (signup)
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hash the password for security
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Check if username exists
        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please try a different one.", "danger")
            return redirect('/signup')

        # Add user to the database
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! You can now log in.", "success") 
        return redirect('/login')

    return render_template('signup.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check user credentials
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username

            # Log successful login
            login_attempt = LoginActivity(user_id=user.id, status='Success')
            db.session.add(login_attempt)
            db.session.commit()

            return redirect('/predict')
        else:
            # Log failed login
            if user:
                login_attempt = LoginActivity(user_id=user.id, status='Failed')
                db.session.add(login_attempt)
                db.session.commit()

            flash("Invalid credentials. Please try again.", "danger")
            return redirect('/login')

    return render_template('login.html')

# Route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Ensure user is logged in
    if 'user_id' not in session:
        flash("You need to log in to access this page.", "danger")
        return redirect('/login')

    if request.method == 'POST':
        # Define features and collect data
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features = [
            int(request.form.get('age', 0)),
            int(request.form.get('sex', 0)),
            int(request.form.get('cp', 0)),
            float(request.form.get('trestbps', 0)),
            float(request.form.get('chol', 0)),
            int(request.form.get('fbs', 0)),
            int(request.form.get('restecg', 0)),
            float(request.form.get('thalach', 0)),
            int(request.form.get('exang', 0)),
            float(request.form.get('oldpeak', 0)),
            int(request.form.get('slope', 0)),
            int(request.form.get('ca', 0)),
            int(request.form.get('thal', 0))
        ]

        # Scale and predict
        scaled_features = sc.transform(pd.DataFrame([features], columns=feature_names))
        prediction_proba = model.predict_proba(scaled_features)[0]
        prediction = np.argmax(prediction_proba)
        confidence = prediction_proba[prediction] * 100  # Convert to percentage

        if prediction == 1:
            if confidence >= 90:
                result = "Danger: High chances of heart attack. Immediate consultation recommended!"
            elif confidence >= 80:
                result = "High risk of heart disease. Please consult a doctor soon."
            elif confidence >= 70:
                result = "Moderate risk of heart disease. Monitoring and lifestyle changes advised."
            elif confidence >= 60:
                result = "Moderate risk of heart disease. Consider lifestyle changes and regular check-ups."
            elif confidence >= 50:
                result = "Low to moderate risk of heart disease. Monitor and consult a healthcare provider if necessary."
            elif confidence >= 40:
                result = "Low risk of heart disease. Maintain healthy habits."
            elif confidence >= 30:
                result = "Very low risk of heart disease. Keep up the healthy lifestyle."
            elif confidence >= 20:
                result = "Minimal risk of heart disease. Stay active and healthy."
            elif confidence >= 10:
                result = "Negligible risk of heart disease. Continue with a balanced lifestyle."
            else:
                result = "Extremely low risk of heart disease. Excellent health!"
        else:
            result = "No significant risk detected. Maintain healthy habits."

        # Pass both the result text and confidence percentage to the template
        return render_template('result.html', result=result, confidence=round(confidence, 2))

    # Render the prediction form for GET requests
    return render_template('predict.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


# Route for logging out
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# Run the app
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)
