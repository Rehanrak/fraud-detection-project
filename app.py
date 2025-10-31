from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import re
import json

app = Flask(__name__)
CORS(app)

# In-memory data storage (in production, use a database)
transactions_db = []
community_reports = []
user_behavior_db = {}

# ML Model Configuration
class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
    def train_model(self):
        """Train the fraud detection model with synthetic data"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Legitimate transactions
        legitimate = pd.DataFrame({
            'amount': np.random.lognormal(4, 1, n_samples),
            'hour': np.random.randint(8, 22, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'transaction_frequency': np.random.randint(1, 10, n_samples),
            'amount_deviation': np.random.normal(0, 1, n_samples),
            'is_fraud': 0
        })
        
        # Fraudulent transactions (smaller amount, higher)
        fraud = pd.DataFrame({
            'amount': np.random.lognormal(6, 2, 200),  # Higher amounts
            'hour': np.random.randint(0, 24, 200),  # Any time
            'day_of_week': np.random.randint(0, 7, 200),
            'transaction_frequency': np.random.randint(10, 50, 200),  # High frequency
            'amount_deviation': np.random.normal(3, 2, 200),  # High deviation
            'is_fraud': 1
        })
        
        # Combine data
        df = pd.concat([legitimate, fraud], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        # Train model
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Train isolation forest for anomaly detection
        self.isolation_forest.fit(X_scaled)
        
        self.is_trained = True
        print("Model trained successfully!")
        
    def extract_features(self, transaction_data, user_history):
        """Extract features from transaction data"""
        dt = datetime.fromisoformat(transaction_data['time'].replace('Z', ''))
        
        # Calculate transaction frequency
        recent_transactions = [t for t in user_history 
                              if datetime.fromisoformat(t['time'].replace('Z', '')) > dt - timedelta(hours=24)]
        
        # Calculate amount deviation
        if len(user_history) > 0:
            avg_amount = np.mean([t['amount'] for t in user_history])
            amount_deviation = (transaction_data['amount'] - avg_amount) / (avg_amount + 1)
        else:
            amount_deviation = 0
        
        features = {
            'amount': transaction_data['amount'],
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'transaction_frequency': len(recent_transactions),
            'amount_deviation': amount_deviation
        }
        
        return features
    
    def predict(self, features_dict):
        """Predict if transaction is fraudulent"""
        if not self.is_trained:
            self.train_model()
        
        features_df = pd.DataFrame([features_dict])
        features_scaled = self.scaler.transform(features_df)
        
        # Get prediction probability
        fraud_probability = self.model.predict_proba(features_scaled)[0][1]
        
        # Get anomaly score
        anomaly_score = self.isolation_forest.score_samples(features_scaled)[0]
        
        # Combine scores
        combined_score = (fraud_probability * 0.7) + ((-anomaly_score) * 0.3)
        
        is_fraud = combined_score > 0.5
        
        return {
            'is_fraud': bool(is_fraud),
            'risk_score': float(combined_score),
            'confidence': float(max(fraud_probability, 1 - fraud_probability))
        }

# Initialize model
fraud_model = FraudDetectionModel()

def analyze_transaction_patterns(transaction_data, user_history):
    """Analyze transaction for suspicious patterns"""
    reasons = []
    risk_factors = 0
    
    amount = transaction_data['amount']
    dt = datetime.fromisoformat(transaction_data['time'].replace('Z', ''))
    
    # Check for high amount
    if amount > 10000:
        reasons.append("High transaction amount")
        risk_factors += 1
    
    # Check for unusual time
    if dt.hour < 6 or dt.hour > 23:
        reasons.append("Transaction at unusual hour")
        risk_factors += 1
    
    # Check for rapid transactions
    recent_transactions = [t for t in user_history 
                          if datetime.fromisoformat(t['time'].replace('Z', '')) > dt - timedelta(minutes=30)]
    if len(recent_transactions) > 5:
        reasons.append("Multiple transactions in short time")
        risk_factors += 2
    
    # Check for amount deviation
    if len(user_history) > 0:
        amounts = [t['amount'] for t in user_history]
        avg_amount = np.mean(amounts)
        if amount > avg_amount * 3:
            reasons.append("Amount significantly higher than usual")
            risk_factors += 1
    
    # Check for duplicate transactions
    duplicate_check = [t for t in user_history 
                      if t['amount'] == amount and t['merchant'] == transaction_data['merchant']
                      and datetime.fromisoformat(t['time'].replace('Z', '')) > dt - timedelta(minutes=10)]
    if len(duplicate_check) > 0:
        reasons.append("Potential duplicate transaction")
        risk_factors += 2
    
    return reasons, risk_factors

def check_merchant_legitimacy(merchant_name, qr_data=None):
    """Check if merchant is legitimate"""
    trust_score = 0.5  # Default neutral
    warnings = []
    
    # Check for suspicious patterns in merchant name
    suspicious_keywords = ['test', 'fake', 'temp', 'unknown', '123', 'xxx']
    if any(keyword in merchant_name.lower() for keyword in suspicious_keywords):
        trust_score -= 0.3
        warnings.append("Merchant name contains suspicious keywords")
    
    # Check UPI ID format if provided
    if qr_data:
        upi_pattern = r'^[\w.-]+@[\w.-]+$'
        if not re.match(upi_pattern, qr_data):
            trust_score -= 0.2
            warnings.append("Invalid UPI ID format")
        else:
            trust_score += 0.2
    
    # Check against community reports
    reported_merchants = [r['merchant'] for r in community_reports]
    report_count = reported_merchants.count(merchant_name)
    if report_count > 0:
        trust_score -= (report_count * 0.1)
        warnings.append(f"Merchant has {report_count} community report(s)")
    
    # Bonus for known good patterns
    if '@' in merchant_name and any(bank in merchant_name.lower() 
                                   for bank in ['paytm', 'phonepe', 'gpay', 'bank']):
        trust_score += 0.3
    
    trust_score = max(0, min(1, trust_score))  # Clamp between 0 and 1
    
    return trust_score, warnings

# API Endpoints

@app.route('/api/check-transaction', methods=['POST'])
def check_transaction():
    """Check if a transaction is fraudulent"""
    data = request.json
    user_id = data.get('userId', 'anonymous')
    
    # Get user transaction history
    user_history = user_behavior_db.get(user_id, [])
    
    # Extract features
    features = fraud_model.extract_features(data, user_history)
    
    # Get ML prediction
    prediction = fraud_model.predict(features)
    
    # Analyze patterns
    reasons, risk_factors = analyze_transaction_patterns(data, user_history)
    
    # Adjust risk score based on pattern analysis
    adjusted_risk = min(1.0, prediction['risk_score'] + (risk_factors * 0.1))
    
    is_fraud = adjusted_risk > 0.6
    
    # Generate recommendations
    recommendations = []
    if is_fraud:
        recommendations.append("Do not proceed with this transaction")
        recommendations.append("Verify merchant identity")
        recommendations.append("Contact your bank if unauthorized")
    else:
        recommendations.append("Transaction appears safe")
        if adjusted_risk > 0.4:
            recommendations.append("Monitor your account for any unusual activity")
    
    # Store transaction
    transaction_record = {
        **data,
        'risk_score': adjusted_risk,
        'status': 'fraud' if is_fraud else 'legitimate',
        'timestamp': datetime.now().isoformat()
    }
    transactions_db.append(transaction_record)
    
    # Update user history
    if user_id not in user_behavior_db:
        user_behavior_db[user_id] = []
    user_behavior_db[user_id].append(data)
    
    return jsonify({
        'is_fraud': is_fraud,
        'risk_score': adjusted_risk,
        'confidence': prediction['confidence'],
        'reason': ', '.join(reasons) if reasons else 'Normal transaction pattern',
        'recommendations': ', '.join(recommendations)
    })

@app.route('/api/scan-qr', methods=['POST'])
def scan_qr():
    """Scan and verify QR code legitimacy"""
    data = request.json
    qr_data = data.get('qr_data', '')
    
    # Extract merchant name from QR data (assuming UPI format)
    merchant_name = qr_data.split('@')[0] if '@' in qr_data else qr_data
    
    # Check legitimacy
    trust_score, warnings = check_merchant_legitimacy(merchant_name, qr_data)
    
    is_legitimate = trust_score > 0.5
    
    analysis = []
    if trust_score > 0.7:
        analysis.append("Verified merchant with good reputation")
    elif trust_score > 0.5:
        analysis.append("Merchant appears legitimate but verify before payment")
    elif trust_score > 0.3:
        analysis.append("Suspicious merchant - proceed with caution")
    else:
        analysis.append("High risk merchant - do not proceed")
    
    return jsonify({
        'is_legitimate': is_legitimate,
        'merchant_name': merchant_name,
        'trust_score': trust_score,
        'analysis': ', '.join(analysis),
        'warnings': ', '.join(warnings) if warnings else None
    })

@app.route('/api/community-report', methods=['POST'])
def community_report():
    """Submit a community fraud report"""
    data = request.json
    
    report = {
        **data,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending'
    }
    
    community_reports.append(report)
    
    return jsonify({
        'success': True,
        'message': 'Report submitted successfully. Thank you for helping the community!'
    })

@app.route('/api/community-reports', methods=['GET'])
def get_community_reports():
    """Get recent community reports"""
    # Return last 10 reports
    recent_reports = sorted(community_reports, 
                          key=lambda x: x['timestamp'], 
                          reverse=True)[:10]
    return jsonify(recent_reports)

@app.route('/api/analyze-behavior', methods=['POST'])
def analyze_behavior():
    """Analyze user behavior patterns"""
    data = request.json
    user_id = data.get('user_id')
    time_period = data.get('time_period', '7d')
    
    # Get user transactions
    user_transactions = user_behavior_db.get(user_id, [])
    
    if not user_transactions:
        return jsonify({
            'risk_level': 'unknown',
            'transaction_count': 0,
            'avg_amount': 0,
            'unusual_patterns': 'No transaction history found',
            'analysis': 'User has no transaction history'
        })
    
    # Calculate time window
    time_map = {'24h': 1, '7d': 7, '30d': 30}
    days = time_map.get(time_period, 7)
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Filter transactions
    recent_transactions = [
        t for t in user_transactions 
        if datetime.fromisoformat(t['time'].replace('Z', '')) > cutoff_date
    ]
    
    if not recent_transactions:
        return jsonify({
            'risk_level': 'low',
            'transaction_count': 0,
            'avg_amount': 0,
            'unusual_patterns': 'No recent transactions',
            'analysis': f'No transactions in the last {time_period}'
        })
    
    # Calculate statistics
    amounts = [t['amount'] for t in recent_transactions]
    avg_amount = np.mean(amounts)
    std_amount = np.std(amounts)
    transaction_count = len(recent_transactions)
    
    # Detect unusual patterns
    unusual_patterns = []
    risk_level = 'low'
    
    if transaction_count > 20:
        unusual_patterns.append("High transaction frequency")
        risk_level = 'medium'
    
    if transaction_count > 50:
        unusual_patterns.append("Very high transaction frequency")
        risk_level = 'high'
    
    if avg_amount > 5000:
        unusual_patterns.append("High average transaction amount")
        risk_level = 'medium' if risk_level == 'low' else 'high'
    
    # Check for large deviations
    large_transactions = [a for a in amounts if a > avg_amount + (2 * std_amount)]
    if len(large_transactions) > 3:
        unusual_patterns.append("Multiple large transactions detected")
        risk_level = 'high'
    
    # Check for rapid succession
    times = [datetime.fromisoformat(t['time'].replace('Z', '')) for t in recent_transactions]
    times.sort()
    rapid_transactions = 0
    for i in range(1, len(times)):
        if (times[i] - times[i-1]).total_seconds() < 300:  # 5 minutes
            rapid_transactions += 1
    
    if rapid_transactions > 5:
        unusual_patterns.append("Many rapid transactions detected")
        risk_level = 'high'
    
    analysis = f"User has made {transaction_count} transactions with an average of ₹{avg_amount:.2f}. "
    if risk_level == 'high':
        analysis += "Multiple suspicious patterns detected. Recommend account review."
    elif risk_level == 'medium':
        analysis += "Some unusual activity detected. Monitor closely."
    else:
        analysis += "Normal transaction behavior."
    
    return jsonify({
        'risk_level': risk_level,
        'transaction_count': transaction_count,
        'avg_amount': avg_amount,
        'unusual_patterns': ', '.join(unusual_patterns) if unusual_patterns else 'None detected',
        'analysis': analysis
    })

@app.route('/api/dashboard-stats', methods=['GET'])
def dashboard_stats():
    """Get dashboard statistics"""
    total = len(transactions_db)
    fraud = len([t for t in transactions_db if t['status'] == 'fraud'])
    legitimate = total - fraud
    fraud_rate = (fraud / total * 100) if total > 0 else 0
    
    # Get recent transactions
    recent = sorted(transactions_db, 
                   key=lambda x: x.get('timestamp', ''), 
                   reverse=True)[:10]
    
    return jsonify({
        'total_transactions': total,
        'fraud_detected': fraud,
        'legitimate': legitimate,
        'fraud_rate': round(fraud_rate, 2),
        'recent_transactions': recent
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': fraud_model.is_trained,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Train model on startup
    print("Training fraud detection model...")
    fraud_model.train_model()
    print("Server starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)