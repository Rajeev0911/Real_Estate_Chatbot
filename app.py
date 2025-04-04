from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import logging

# Import your RealEstateAgent class
from chatbot import RealEstateAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Replace with your actual Gemini API key and CSV file path
GEMINI_API_KEY = "AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY"
CSV_PATH = "realestatedata.csv"

# Initialize the real estate agent
try:
    agent = RealEstateAgent(GEMINI_API_KEY, CSV_PATH)
    logger.info("Real Estate Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Real Estate Agent: {e}")
    sys.exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Process the query using your real estate agent
        logger.info(f"Processing query: {user_query}")
        result = agent.process_real_estate_query(user_query)
        
        return jsonify({'response': result})
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
