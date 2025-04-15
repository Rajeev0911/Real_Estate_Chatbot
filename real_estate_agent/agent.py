import os
import re
import sys
import time
import random
import logging
import pandas as pd
import numpy as np
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import helper functions from our submodules
from .filtering import semantic_search
from .insights import get_property_insights, format_response_with_typing_effect, get_dataset_overview
from .message_processor import process_user_message_with_typing, respond_with_typing, demo_typing_response

logger = logging.getLogger(__name__)

class RealEstateAgent:
    def __init__(self, gemini_api_key, csv_path):
        """Initialize the Real Estate Agent with API key, load CSV, and set up conversation state."""
        self.gemini_api_key = gemini_api_key
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        self.conversation_state = "initial"
        self.user_preferences = {}
        self.conversation_history = []
        self.current_query = ""
        self.questions_asked = set()  # for tracking asked questions

        # Load property data from CSV
        self.knowledge_base = self.load_properties_from_csv(csv_path)
        # Preprocess the data for better searching
        self.knowledge_base = self._preprocess_data(self.knowledge_base)
        # Analyze the preprocessed knowledge base
        self.stats = self.analyze_knowledge_base()

        # Define greetings and follow-up templates (if not already defined)
        self.greetings = [
            "Hello! I'm your personal real estate assistant. How can I help you find your perfect property today?",
            "Hi there! I'm here to help you find the ideal property. What are you looking for?",
            "Welcome! I'm your real estate expert. Tell me what you're looking for in your next home.",
            "Good day! I'm ready to assist with your property search. What kind of property are you interested in?"
        ]

    def _preprocess_data(self, df):
        """Preprocess the data for better search and matching."""
        try:
            # Convert column names to standard format
            df.columns = [col.strip() for col in df.columns]
            
            # Ensure essential columns exist
            essential_columns = ['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing']
            for col in essential_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Create processed columns for better text matching
            if 'Location' in df.columns:
                df['Processed_Location'] = df['Location'].astype(str).apply(lambda x: x.lower())
                
            if 'Type' in df.columns:
                df['Processed_Type'] = df['Type'].astype(str).apply(lambda x: x.lower())
                
            if 'Description' in df.columns:
                df['Processed_Description'] = df['Description'].astype(str).apply(lambda x: x.lower())
            
            # Convert price to numeric
            if 'Price' in df.columns:
                df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
                
            # Convert bedrooms to numeric
            if 'Bedrooms' in df.columns:
                # If bedrooms is string, extract numbers
                if df['Bedrooms'].dtype == 'object':
                    df['Bedrooms'] = df['Bedrooms'].astype(str).apply(
                        lambda x: re.search(r'(\d+)', x).group(1) if re.search(r'(\d+)', x) else None
                    )
                df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
            
            logger.info(f"Preprocessed data: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return df

    def load_properties_from_csv(self, csv_path):
        """Load property data from a CSV file into a pandas DataFrame."""
        try:
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return pd.DataFrame(columns=['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing'])
            logger.info(f"Loading CSV file from: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded {len(df)} properties")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return pd.DataFrame(columns=['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing'])

    def analyze_knowledge_base(self):
        """Perform analysis on the knowledge base and store basic statistics."""
        df = self.knowledge_base
        stats = {}
        if 'Purpose' in df.columns:
            stats['purposes'] = df['Purpose'].value_counts().to_dict()
        if 'Type' in df.columns:
            stats['types'] = df['Type'].value_counts().to_dict()
        if 'Price' in df.columns:
            stats['price_min'] = df['Price'].min()
            stats['price_max'] = df['Price'].max()
        if 'Location' in df.columns:
            stats['top_locations'] = df['Location'].value_counts().head(10).to_dict()
        if 'Bedrooms' in df.columns:
            stats['bedrooms'] = df['Bedrooms'].value_counts().to_dict()
        logger.info("Knowledge base analysis complete")
        return stats

    # ----- Methods that use helper functions -----
    def _is_greeting(self, text: str) -> bool:
        """Check if the text is a greeting."""
        greeting_keywords = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        text_lower = text.lower().strip()
        return any(text_lower.startswith(keyword) for keyword in greeting_keywords)

    def _is_basic_question(self, text: str) -> bool:
        """Check if the text contains a basic question about the assistant."""
        basic_questions = [
            'who are you', 'what can you do', 'how can you help', 'what do you do',
            'assist me', 'tell me about yourself'
        ]
        text_lower = text.lower()
        return any(question in text_lower for question in basic_questions)

    def _is_data_question(self, text: str) -> bool:
        """Check if the text is asking for data or statistics about the properties."""
        data_questions = [
            'property types', 'available properties', 'property data', 'show me the data',
            'data overview', 'price ranges', 'top locations', 'property statistics'
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in data_questions)

    def _semantic_search(self, filtered_df, user_query, top_k=5):
        """Perform semantic search on filtered properties using the helper."""
        return semantic_search(filtered_df, user_query, top_k=top_k)

    def get_property_insights(self, properties):
        """Return formatted property details using the helper."""
        return get_property_insights(properties)

    def _format_response_with_typing_effect(self, properties, user_query, query_details):
        """Format response text using typing effect style using the helper."""
        return format_response_with_typing_effect(properties, user_query, query_details)

    def _get_dataset_overview(self):
        """Get an overview of the dataset using the helper, passing in the stored stats."""
        return get_dataset_overview(self.stats)

    def _process_user_message_with_typing(self, message):
        """Process a user message. Delegates to the helper function."""
        # This helper may use methods such as _is_greeting, _is_basic_question, etc.
        # For simplicity, assume these helper checks remain methods of this class.
        return process_user_message_with_typing(self, message)

    def respond_with_typing(self, user_message):
        """Return full assistant response using helper function."""
        return respond_with_typing(self, user_message)

    def demo_typing_response(self, user_message):
        """Demonstrate the full response with a simulated typing effect using helper."""
        return demo_typing_response(self, user_message)

    def _filter_properties(self, query_details):
        """Filter properties based on extracted query details."""
        try:
            # Start with all properties
            df = self.knowledge_base.copy()
            
            # Filter by purpose (rent/sale)
            if 'purpose' in query_details and query_details['purpose']:
                # Case-insensitive match for purpose
                purpose = query_details['purpose'].lower()
                if purpose in ['rent', 'rental', 'lease']:
                    df = df[df['Purpose'].str.lower() == 'rent']
                elif purpose in ['buy', 'purchase', 'sale']:
                    df = df[df['Purpose'].str.lower() == 'sale']
            
            # Filter by location
            if 'location' in query_details and query_details['location']:
                location = query_details['location'].lower()
                if 'Processed_Location' in df.columns:
                    df = df[df['Processed_Location'].str.contains(location, case=False, na=False)]
            
            # Filter by property type
            if 'property_type' in query_details and query_details['property_type']:
                prop_type = query_details['property_type'].lower()
                if 'Processed_Type' in df.columns:
                    df = df[df['Processed_Type'].str.contains(prop_type, case=False, na=False)]
            
            # Filter by number of bedrooms
            if 'bedrooms' in query_details and query_details['bedrooms'] is not None:
                bedrooms = query_details['bedrooms']
                df = df[df['Bedrooms'] == bedrooms]
            
            # Filter by price range
            if 'max_price' in query_details and query_details['max_price'] is not None:
                max_price = query_details['max_price']
                df = df[df['Price'] <= max_price]
                
            if 'min_price' in query_details and query_details['min_price'] is not None:
                min_price = query_details['min_price']
                df = df[df['Price'] >= min_price]
            
            # Handle price qualifiers
            if 'price_qualifier' in query_details and query_details['price_qualifier']:
                qual = query_details['price_qualifier'].lower()
                if qual in ['cheap', 'affordable', 'budget']:
                    # Get properties in the lower 30% of the price range
                    threshold = df['Price'].quantile(0.3)
                    df = df[df['Price'] <= threshold]
                elif qual in ['luxury', 'expensive', 'high-end']:
                    # Get properties in the upper 30% of the price range
                    threshold = df['Price'].quantile(0.7)
                    df = df[df['Price'] >= threshold]
            
            # Filter by furnishing status
            if 'furnishing' in query_details and query_details['furnishing']:
                furnishing = query_details['furnishing'].lower()
                df = df[df['Furnishing'].str.lower() == furnishing]
            
            # Log filtering results
            logger.info(f"Filtered properties: {len(df)} remain after applying filters")
            
            # If no properties remain, return at least the original first 5 for a fallback
            if df.empty:
                logger.warning("No properties match the filters, using top 5 from original dataset")
                return self.knowledge_base.head(5)
                
            return df
            
        except Exception as e:
            logger.error(f"Error filtering properties: {e}")
            # In case of error, return first 5 properties as fallback
            return self.knowledge_base.head(5)

    def _call_gemini_api(self, prompt: str) -> dict:
        """
        Call the Gemini API to extract structured details from the query.
        The prompt instructs the model to return raw JSON without any markdown formatting.
        """
        try:
            logger.info(f"Calling Gemini API with prompt: {prompt[:100]}...")
            # Append additional instruction to get raw JSON output
            refined_prompt = prompt + "\n\nPlease return only raw JSON without any markdown formatting, code fences, or additional text."
            payload = {
                "contents": [{"parts": [{"text": refined_prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024}
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.gemini_url, headers=headers, data=json.dumps(payload))
        
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}, {response.text}")
                return self._fallback_query_parser(prompt)
        
            response_json = response.json()
            candidate = response_json.get("candidates", [{}])[0]
        
            # Extract text from candidate; try both possible keys
            if "content" in candidate and "parts" in candidate["content"]:
                generated_text = candidate["content"]["parts"][0].get("text", "")
            elif "output" in candidate:
                generated_text = candidate["output"]
            else:
                logger.error(f"Unexpected Gemini API response format: {candidate}")
                return self._fallback_query_parser(prompt)
        
            # Remove markdown code fences if present
            if "```" in generated_text:
                json_text = re.search(r'```(?:json)?(.*?)```', generated_text, re.DOTALL)
                generated_text = json_text.group(1) if json_text else generated_text.strip('`')
        
            # Extra cleaning: try to extract a JSON substring if extra text is included
            try:
                query_details = json.loads(generated_text)
                return query_details
            except json.JSONDecodeError:
                # Attempt a rough extraction of JSON by finding the first '{' and the last '}'
                start = generated_text.find('{')
                end = generated_text.rfind('}') + 1
                if start != -1 and end != -1:
                    json_text = generated_text[start:end]
                    try:
                        query_details = json.loads(json_text)
                        return query_details
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Extraction attempt failed: {json_err}")
                logger.error(f"Error parsing JSON from Gemini API response. Raw text: {generated_text}")
                return self._fallback_query_parser(prompt)
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return self._fallback_query_parser(prompt)

    def _fallback_query_parser(self, prompt: str) -> dict:
        """
        Fallback query parser when Gemini API fails.
        Uses simple keyword matching to extract information.
        """
        query = prompt.lower()
        result = {
            "location": None,
            "bedrooms": None,
            "property_type": None,
            "max_price": None,
            "min_price": None,
            "price_qualifier": None,
            "amenities": [],
            "purpose": None,
            "furnishing": None
        }
        
        # Extract location (simplified)
        location_match = re.search(r'in\s+([a-zA-Z\s]+?)(?:with|\.|$)', query)
        if location_match:
            result["location"] = location_match.group(1).strip()
            
        # Extract bedrooms
        bedroom_match = re.search(r'(\d+)\s+bed', query)
        if bedroom_match:
            result["bedrooms"] = int(bedroom_match.group(1))
            
        # Extract property type
        property_types = ['house', 'apartment', 'villa', 'studio', 'townhouse']
        for p_type in property_types:
            if p_type in query:
                result["property_type"] = p_type
                break
                
        # Extract price
        price_match = re.search(r'(\d+)[k]?\s*(?:to|-)?\s*(\d+)[k]?', query)
        if price_match:
            result["min_price"] = float(price_match.group(1))
            result["max_price"] = float(price_match.group(2))
        else:
            under_match = re.search(r'under\s*(\d+)[k]?', query)
            if under_match:
                result["max_price"] = float(under_match.group(1))
                
        # Extract price qualifier
        price_qualifiers = ['cheap', 'affordable', 'luxury', 'budget', 'expensive']
        for qualifier in price_qualifiers:
            if qualifier in query:
                result["price_qualifier"] = qualifier
                break
                
        # Extract purpose
        if any(word in query for word in ['rent', 'lease']):
            result["purpose"] = "Rent"
        elif any(word in query for word in ['buy', 'purchase', 'sale']):
            result["purpose"] = "Sale"
            
        # Extract furnishing
        if 'furnished' in query:
            result["furnishing"] = "Furnished"
        elif 'unfurnished' in query:
            result["furnishing"] = "Unfurnished"
            
        logger.info(f"Fallback parser extracted: {result}")
        return result

    def _process_user_message(self, message):
        """Internal method to process and respond to a user message."""
        return self._process_user_message_with_typing(message)
    
    def _create_gemini_query_prompt(self, user_query: str) -> str:
        """
        Create a prompt for the Gemini API that extracts structured details from the query.
        The prompt instructs Gemini to output a JSON with keys like location, bedrooms, property_type, max_price, etc.
        """
        prompt = f"""
    Analyze the following real estate query and extract structured details:

    Query: "{user_query}"

    Please provide a JSON response with the following keys:
    - location: Extracted city or region (string or null)
    - bedrooms: Number of bedrooms (integer or null)
    - property_type: Type of property (string or null, options: 'house', 'apartment', 'villa', 'studio', 'townhouse')
    - max_price: Maximum budget (float or null)
    - min_price: Minimum budget (float or null)
    - price_qualifier: If a qualitative price descriptor is used (string or null, examples: 'cheap', 'affordable', 'luxury', 'budget', 'expensive')
    - amenities: List of desired amenities (list of strings or empty list)
    - purpose: Whether the user wants to rent or buy (string or null, options: 'Rent', 'Sale')
    - furnishing: Furnishing preferences (string or null, options: 'Furnished', 'Unfurnished')

    If the query mentions "cheap" or "affordable", set price_qualifier to that value.
    Look for keywords related to renting (e.g., "rent", "lease") or buying (e.g., "buy", "purchase") to determine purpose.

    Provide null or empty values if not specified in the query.
    """
        return prompt.strip()

# If running in command-line mode, use the CLI helper
if __name__ == "__main__":
    from .cli import run_cli_interface
    run_cli_interface()