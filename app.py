# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv
from real_estate_agent import RealEstateAgent

# Load environment variables, set up logging etc.
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY')
CSV_PATH = os.getenv('CSV_PATH', 'realestatedata.csv')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

try:
    agent = RealEstateAgent(GEMINI_API_KEY, CSV_PATH)
    logger.info("Real Estate Agent initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Real Estate Agent: {e}")
    agent = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if not agent:
        return jsonify({
            'success': False,
            'message': 'Agent initialization failed. Check server logs.',
            'acknowledgment': 'Sorry, I encountered an error.'
        }), 500

    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({
            'success': False,
            'message': 'No message provided',
            'acknowledgment': 'I need a message to respond to.'
        }), 400

    try:
        response_data = agent.respond_with_typing(user_message)
        return jsonify({
            'success': True,
            'acknowledgment': response_data['acknowledgment'],
            'message': response_data['full_response']
        })
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'acknowledgment': 'Sorry, I encountered an error processing your request.'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    if not agent:
        return jsonify({'success': False, 'message': 'Agent initialization failed'}), 500
    try:
        return jsonify({'success': True, 'stats': agent.stats})
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
