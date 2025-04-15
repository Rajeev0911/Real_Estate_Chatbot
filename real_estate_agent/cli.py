# import random
# import logging

# from .agent import RealEstateAgent

# logger = logging.getLogger(__name__)

# def run_cli_interface():
#     """Run a simple CLI loop for the Real Estate Agent chatbot."""
#     # Set your Gemini API key and CSV path here; in production use environment variables.
#     gemini_api_key = "AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY"  # Replace with your actual API key
#     csv_path = "realestatedata.csv"
    
#     agent = RealEstateAgent(gemini_api_key, csv_path)
#     print("Real Estate Assistant initialized. Type 'exit' to quit.")
#     print(random.choice(agent.greetings))
    
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in ('exit', 'quit', 'bye'):
#             print("Assistant: Thank you for using the Real Estate Assistant. Goodbye!")
#             break
#         # Simulate the typing effect in the response
#         agent.demo_typing_response(user_input)

# if __name__ == "__main__":
#     run_cli_interface()





import random
import logging
import os

from .agent import RealEstateAgent

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging for the CLI application."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_cli_interface():
    """Run a simple CLI loop for the Real Estate Agent chatbot."""
    # Set up logging
    setup_logging()
    
    # Get API key from environment variable or use default for demo
    gemini_api_key = os.environ.get(
        "GEMINI_API_KEY", 
        "AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY"  # Default key for demo
    )
    
    # Get CSV path from environment variable or use default for demo
    csv_path = os.environ.get("REAL_ESTATE_CSV_PATH", "realestatedata.csv")
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found at {csv_path}. Using sample data instead.")
        # Create a sample CSV with minimal data for demo if file doesn't exist
        # create_sample_data(csv_path)
    
    # Initialize the agent
    agent = RealEstateAgent(gemini_api_key, csv_path)
    print("Real Estate Assistant initialized. Type 'exit' to quit.")
    print(random.choice(agent.greetings))
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ('exit', 'quit', 'bye'):
                print("Assistant: Thank you for using the Real Estate Assistant. Goodbye!")
                break
            
            # Simulate the typing effect in the response
            agent.demo_typing_response(user_input)
        except KeyboardInterrupt:
            print("\nAssistant: Session terminated. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            print("\nAssistant: I encountered an error. Let's try again.")

# def create_sample_data(csv_path):
#     """Create a sample CSV file with minimal property data for demo purposes."""
#     import pandas as pd
    
#     # Create sample data
#     data = {
#         'Price': [750000, 1200000, 500000, 300000, 1500000, 25000, 40000, 800000, 600000, 1800000],
#         'Location': ['Downtown', 'Beachfront', 'Suburb', 'City Center', 'Countryside', 
#                      'Riverside', 'Mountain View', 'Business District', 'Historic Area', 'Seaside'],
#         'Type': ['Apartment', 'Villa', 'House', 'Studio', 'Townhouse', 
#                  'Apartment', 'Studio', 'Villa', 'House', 'Penthouse'],
#         'Bedrooms': [2, 4, 3, 1, 5, 1, 1, 3, 2, 4],
#         'Area': [1200, 3000, 1800, 650, 3500, 700, 500, 2200, 1600, 3800],
#         'Description': [
#             'Modern apartment with great city views and amenities.',
#             'Luxury beachfront villa with private pool and garden.',
#             'Spacious family home in quiet neighborhood with good schools.',
#             'Cozy studio apartment, perfect for young professionals.',
#             'Elegant townhouse with modern design and smart home features.',
#             'Affordable apartment with basic amenities and good location.',
#             'Compact studio suitable for students or single occupants.',
#             'Executive villa with high-end finishes and security system.',
#             'Charming house with character features and established garden.',
#             'Spectacular penthouse with panoramic views and luxury finishes.'
#         ],
#         'Title': [
#             'Modern Downtown Apartment',
#             'Luxurious Beachfront Villa',
#             'Family Home in Quiet Suburb',
#             'Cozy City Center Studio',
#             'Elegant Countryside Townhouse',
#             'Affordable Riverside Apartment',
#             'Compact Mountain View Studio',
#             'Executive Business District Villa',
#             'Charming House in Historic Area',
#             'Spectacular Seaside Penthouse'
#         ],
#         'Purpose': ['Sale', 'Sale', 'Sale', 'Rent', 'Sale', 'Rent', 'Rent', 'Sale', 'Sale', 'Sale'],
#         'Furnishing': ['Unfurnished', 'Furnished', 'Unfurnished', 'Furnished', 'Partially Furnished',
#                        'Unfurnished', 'Furnished', 'Furnished', 'Unfurnished', 'Furnished']
#     }
    
#     # Create DataFrame and save to CSV
#     df = pd.DataFrame(data)
#     df.to_csv(csv_path, index=False)
#     logger.info(f"Created sample data file at {csv_path}")

if __name__ == "__main__":
    run_cli_interface()