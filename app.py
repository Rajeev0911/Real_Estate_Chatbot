# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os
# import sys
# import logging

# # Import your RealEstateAgent class
# from chatbot import RealEstateAgent

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logger = logging.getLogger(__name__)

# app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS(app)  # Enable CORS for all routes

# # Replace with your actual Gemini API key and CSV file path
# GEMINI_API_KEY = "AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY"
# CSV_PATH = "realestatedata.csv"

# # Initialize the real estate agent
# try:
#     agent = RealEstateAgent(GEMINI_API_KEY, CSV_PATH)
#     logger.info("Real Estate Agent initialized successfully")
# except Exception as e:
#     logger.error(f"Failed to initialize Real Estate Agent: {e}")
#     sys.exit(1)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/api/query', methods=['POST'])
# def process_query():
#     try:
#         data = request.json
#         user_query = data.get('query', '')
        
#         if not user_query:
#             return jsonify({'error': 'No query provided'}), 400
        
#         # Process the query using your real estate agent
#         logger.info(f"Processing query: {user_query}")
#         result = agent.process_real_estate_query(user_query)
        
#         return jsonify({'response': result})
    
#     except Exception as e:
#         logger.error(f"Error processing query: {e}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)





from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from chatbot import RealEstateAgent
import logging
import os

# Enhanced logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)  # This allows cross-origin requests

# Global instance of the agent
real_estate_agent = None

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def initialize_agent():
    global real_estate_agent
    data = request.json
    api_key = data.get('api_key')
    csv_path = data.get('csv_path')
    
    try:
        real_estate_agent = RealEstateAgent(api_key, csv_path)
        return jsonify({"status": "success", "message": "Agent initialized successfully"})
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    global real_estate_agent
    
    # Auto-initialize with default values if not already initialized
    if not real_estate_agent:
        try:
            # Use environment variables or hardcoded values for testing
            api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY')
            csv_path = os.environ.get('CSV_PATH', 'realestatedata.csv')
            real_estate_agent = RealEstateAgent(api_key, csv_path)
            logger.info("Agent initialized with default values")
        except Exception as e:
            logger.error(f"Error auto-initializing agent: {e}")
            return jsonify({"status": "error", "message": "Agent initialization failed"}), 500
    
    data = request.json
    query = data.get('query')
    
    try:
        # Get the response data
        response_data = real_estate_agent.respond_with_typing(query)
        
        # Extract the full response content
        if isinstance(response_data, dict) and 'full_response' in response_data:
            response_text = response_data['full_response']
        else:
            response_text = str(response_data)
        
        return jsonify({"status": "success", "response": response_text})
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"status": "error", "message": "Error processing your request"}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Make sure the app is accessible from other devices on the network
    app.run(debug=True, host='0.0.0.0', port=5000)