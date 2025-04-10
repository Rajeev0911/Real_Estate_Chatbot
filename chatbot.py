import requests
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import os
import random
import time
import sys

# Enhanced logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_properties_from_csv(csv_path):
    """Load property data from a CSV file into a pandas DataFrame."""
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            # Return an empty DataFrame with expected columns instead of raising exception
            return pd.DataFrame(columns=['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing'])
        
        # Print file details for debugging
        logger.info(f"Loading CSV file from: {csv_path}")
        
        # Try to load the file and print a preview
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} properties")
        
        # Generate dataset statistics for better understanding
        if 'Purpose' in df.columns:
            rent_count = df[df['Purpose'] == 'Rent'].shape[0]
            sale_count = df[df['Purpose'] == 'Sale'].shape[0]
            logger.info(f"Dataset contains {rent_count} properties for rent and {sale_count} properties for sale")
        
        if 'Type' in df.columns:
            property_types = df['Type'].value_counts().to_dict()
            logger.info(f"Property types in dataset: {property_types}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        # Return an empty DataFrame with expected columns instead of raising exception
        return pd.DataFrame(columns=['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing'])

class RealEstateAgent:
    def __init__(self, gemini_api_key, csv_path):
        """Initialize the Real Estate Agent with API key and knowledge base."""
        self.gemini_api_key = gemini_api_key
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        
        # Load and preprocess the knowledge base
        logger.info("Initializing Real Estate Agent")
        self.knowledge_base = load_properties_from_csv(csv_path)
        self._preprocess_knowledge_base()
        self._initialize_search_capabilities()
        self._analyze_knowledge_base()
        
        # Add conversation state
        self.conversation_state = "initial"  # Start with initial state
        self.user_preferences = {}
        self.conversation_history = []
        self.questions_asked = set()  # Track which questions have been asked
        
        # Greeting templates
        self.greetings = [
            "Hello! I'm your personal real estate assistant. How can I help you find your perfect property today?",
            "Hi there! I'm here to help you find the ideal property. What are you looking for?",
            "Welcome! I'm your real estate expert. Tell me what you're looking for in your next home.",
            "Good day! I'm ready to assist with your property search. What kind of property are you interested in?"
        ]
        
        # Follow-up question templates
        self.follow_up_questions = {
            "location": "Which area or neighborhood are you interested in?",
            "budget": "What's your budget range for this property?",
            "bedrooms": "How many bedrooms are you looking for?",
            "property_type": "What type of property are you interested in? (apartment, villa, townhouse, etc.)",
            "amenities": "Are there any specific amenities you'd like, such as a pool, gym, or garden?",
            "furnishing": "Would you prefer a furnished or unfurnished property?"
        }
    
    def _analyze_knowledge_base(self):
        """Analyze the knowledge base to extract key statistics and insights."""
        df = self.knowledge_base
        
        # Store key statistics
        self.stats = {}
        
        # Property purpose statistics (Rent vs Sale)
        if 'Purpose' in df.columns:
            purpose_counts = df['Purpose'].value_counts()
            self.stats['purposes'] = purpose_counts.to_dict()
            logger.info(f"Purpose stats: {self.stats['purposes']}")
        
        # Property type statistics
        if 'Type' in df.columns:
            type_counts = df['Type'].value_counts()
            self.stats['types'] = type_counts.to_dict()
            logger.info(f"Type stats: {self.stats['types']}")
        
        # Price ranges
        if 'Price' in df.columns:
            self.stats['price_min'] = df['Price'].min()
            self.stats['price_max'] = df['Price'].max()
            self.stats['price_median'] = df['Price'].median()
            
            # Price ranges by purpose
            if 'Purpose' in df.columns:
                rent_prices = df[df['Purpose'] == 'Rent']['Price']
                sale_prices = df[df['Purpose'] == 'Sale']['Price']
                
                if not rent_prices.empty:
                    self.stats['rent_price_min'] = rent_prices.min()
                    self.stats['rent_price_max'] = rent_prices.max()
                    self.stats['rent_price_median'] = rent_prices.median()
                
                if not sale_prices.empty:
                    self.stats['sale_price_min'] = sale_prices.min()
                    self.stats['sale_price_max'] = sale_prices.max()
                    self.stats['sale_price_median'] = sale_prices.median()
        
        # Location statistics
        if 'Location' in df.columns:
            location_counts = df['Location'].value_counts().head(10)
            self.stats['top_locations'] = location_counts.to_dict()
        
        # Bedroom statistics
        if 'Bedrooms' in df.columns:
            bedroom_counts = df['Bedrooms'].value_counts()
            self.stats['bedrooms'] = bedroom_counts.to_dict()
        
        logger.info("Knowledge base analysis complete")
    
    def _preprocess_knowledge_base(self):
        """Preprocess the knowledge base by converting and normalizing columns."""
        logger.info("Preprocessing knowledge base")
        
        # Make column names consistent (case-insensitive)
        self.knowledge_base.columns = [col.strip() for col in self.knowledge_base.columns]
        column_map = {col: col.capitalize() for col in self.knowledge_base.columns}
        self.knowledge_base = self.knowledge_base.rename(columns=column_map)
        
        # Ensure critical columns exist
        essential_columns = ['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing']
        for col in essential_columns:
            if col not in self.knowledge_base.columns:
                logger.warning(f"Essential column '{col}' not found in CSV. Available columns: {self.knowledge_base.columns.tolist()}")
                self.knowledge_base[col] = np.nan
        
        # Handle numeric columns
        if 'Price' in self.knowledge_base.columns:
            self.knowledge_base['Price'] = self.knowledge_base['Price'].astype(str)
            self.knowledge_base['Price'] = self.knowledge_base['Price'].str.replace('$', '', regex=False)
            self.knowledge_base['Price'] = self.knowledge_base['Price'].str.replace(',', '', regex=False)
            self.knowledge_base['Price'] = pd.to_numeric(self.knowledge_base['Price'], errors='coerce')
            logger.info(f"Price range: {self.knowledge_base['Price'].min()} to {self.knowledge_base['Price'].max()}")
        
        if 'Area' in self.knowledge_base.columns:
            # Retain the original format for display
            self.knowledge_base['Original_Area'] = self.knowledge_base['Area']
        
            # Extract valid numeric values (e.g., "12,002 sqft" -> "12002")
            self.knowledge_base['Area'] = self.knowledge_base['Area'].astype(str)
            self.knowledge_base['Area'] = self.knowledge_base['Area'].str.extract(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)')[0]
            # Remove commas and convert to numeric
            self.knowledge_base['Area'] = self.knowledge_base['Area'].str.replace(',', '', regex=False)
            self.knowledge_base['Area'] = pd.to_numeric(self.knowledge_base['Area'], errors='coerce')
            logger.info(f"Sample Area values after preprocessing: {self.knowledge_base['Area'].head()}")
    
        # Normalize text columns
        text_columns = ['Location', 'City', 'Country', 'Type', 'Title', 'Description', 'Purpose', 'Furnishing']
        for col in text_columns:
            if col in self.knowledge_base.columns:
                self.knowledge_base[f'Processed_{col}'] = self.knowledge_base[col].fillna('').astype(str).str.lower()
        
        logger.info(f"Preprocessing complete. Knowledge base has {len(self.knowledge_base)} properties.")
    
    def _initialize_search_capabilities(self):
        """Initialize TF-IDF-based semantic search."""
        logger.info("Initializing search capabilities")
        # Determine available columns for search
        search_columns = []
        for field in ['Processed_Location', 'Processed_City', 'Processed_Country', 'Processed_Type', 'Processed_Description', 'Processed_Purpose', 'Processed_Furnishing']:
            if field in self.knowledge_base.columns:
                search_columns.append(field)
        
        # Create search texts
        self.search_texts = self.knowledge_base.apply(
            lambda row: ' '.join(str(row.get(col, '')) for col in search_columns), 
            axis=1
        )
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
        logger.info("Search capabilities initialized")
    
    def _is_greeting(self, text):
        """Check if the user input is a greeting."""
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy']
        text_lower = text.lower().strip()
        return any(text_lower == greeting or text_lower.startswith(greeting) for greeting in greetings)
    
    def _is_basic_question(self, text):
        """Check if the user is asking a basic question about the agent."""
        text_lower = text.lower()
        basic_questions = [
            'who are you', 'what can you do', 'how can you help', 'what do you do',
            'help me', 'assist me', 'your services', 'tell me about yourself'
        ]
        return any(question in text_lower for question in basic_questions)
    
    def _is_data_question(self, text):
        """Check if the user is asking about available data or statistics."""
        text_lower = text.lower()
        data_questions = [
            'what types of properties', 'property types', 'what do you have',
            'property statistics', 'available properties', 'property data',
            'rental properties', 'properties for sale', 'what areas',
            'price ranges', 'average price', 'typical price', 'dataset',
            'data overview', 'show me the data', 'what locations'
        ]
        return any(phrase in text_lower for phrase in data_questions)
    
    def _call_gemini_api(self, prompt):
        """Call the Google Gemini API to extract structured query details."""
        try:
            logger.info(f"Calling Gemini API with prompt: {prompt[:100]}...")
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024
                }
            }   
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.gemini_url, headers=headers, data=json.dumps(payload))
        
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}, {response.text}")
                return self._fallback_query_parser(prompt)
        
            response_json = response.json()
        
            # Check for the expected structure in the response
            if "candidates" not in response_json or not response_json["candidates"]:
                logger.error(f"Unexpected Gemini API response format: {response_json}")
                return self._fallback_query_parser(prompt)
            
            candidate = response_json["candidates"][0]
        
            # Handle different possible response structures
            if "content" in candidate and "parts" in candidate["content"]:
                generated_text = candidate["content"]["parts"][0].get("text", "")
            elif "output" in candidate:
                generated_text = candidate["output"]
            else:
                logger.error(f"Unable to extract text from Gemini response: {candidate}")
                return self._fallback_query_parser(prompt)
        
            # Remove markdown code fences if present
            if generated_text.startswith("```"):
                generated_text = re.sub(r'^```(?:json)?\s*', '', generated_text)
                generated_text = re.sub(r'\s*```$', '', generated_text)
        
            try:
                query_details = json.loads(generated_text)
                return query_details
            except json.JSONDecodeError as json_err:
                logger.error(f"Error parsing JSON from Gemini response: {json_err}")
                logger.error(f"Raw response text: {generated_text}")
                return self._fallback_query_parser(prompt)
    
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            # Return a fallback query structure based on basic keyword matching
            return self._fallback_query_parser(prompt)
    
    def _fallback_query_parser(self, query):
        """Fallback query parser for when the API call fails."""
        query = query.lower()
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
        
        # Basic location extraction
        locations = ["dubai", "downtown", "waterfront", "marina", "palm jumeirah", "arabian ranches"]
        for loc in locations:
            if loc in query:
                result["location"] = loc
                break
        
        # Basic bedroom extraction
        bedroom_match = re.search(r'(\d+)[- ]?bed', query)
        if bedroom_match:
            result["bedrooms"] = int(bedroom_match.group(1))
        
        # Basic property type extraction
        property_types = {
            "apartment": ["apartment", "flat", "condo"],
            "villa": ["villa", "mansion"],
            "townhouse": ["townhouse", "town house"],
            "house": ["house", "home"],
            "studio": ["studio"]
        }
        
        for prop_type, keywords in property_types.items():
            if any(keyword in query for keyword in keywords):
                result["property_type"] = prop_type
                break
        
        # Basic price extraction
        price_match = re.search(r'(less than|under|below) (\d+)', query)
        if price_match:
            result["max_price"] = float(price_match.group(2))
        
        # Basic price qualifier
        if any(word in query for word in ["cheap", "affordable", "budget"]):
            result["price_qualifier"] = "cheap"
        elif any(word in query for word in ["luxury", "expensive", "high-end"]):
            result["price_qualifier"] = "luxury"
        
        # Purpose extraction (rent or sale)
        if any(word in query for word in ["rent", "rental", "lease", "renting"]):
            result["purpose"] = "Rent"
        elif any(word in query for word in ["buy", "purchase", "sale", "buying"]):
            result["purpose"] = "Sale"
        
        # Furnishing status
        if "furnished" in query:
            result["furnishing"] = "Furnished"
        elif "unfurnished" in query:
            result["furnishing"] = "Unfurnished"
        
        # Basic amenities
        amenities = ["pool", "gym", "balcony", "view", "parking", "furnished", "garden", "terrace"]
        for amenity in amenities:
            if amenity in query:
                result["amenities"].append(amenity)
        
        logger.info(f"Fallback query parser result: {result}")
        return result
    
    def _create_gemini_query_prompt(self, user_query):
        """Create a prompt for Gemini to extract query details."""
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
    
    def _filter_properties(self, query_details):
        """Filter properties based on query details."""
        filtered_df = self.knowledge_base.copy()
        logger.info(f"Starting with {len(filtered_df)} properties before filtering")
        
        # Start with less restrictive filtering for small datasets
        filters_applied = False
        
        # Purpose filtering (rent or sale)
        if query_details.get('purpose'):
            purpose = query_details['purpose']
            if 'Purpose' in filtered_df.columns:
                temp_filtered = filtered_df[filtered_df['Purpose'] == purpose]
                if not temp_filtered.empty:
                    filtered_df = temp_filtered
                    filters_applied = True
                    logger.info(f"After purpose filtering for '{purpose}': {len(filtered_df)} properties")
                else:
                    logger.warning(f"Purpose filter for '{purpose}' would remove all properties, skipping this filter")
        
        # Furnishing filtering
        if query_details.get('furnishing'):
            furnishing = query_details['furnishing']
            if 'Furnishing' in filtered_df.columns:
                temp_filtered = filtered_df[filtered_df['Furnishing'] == furnishing]
                if not temp_filtered.empty:
                    filtered_df = temp_filtered
                    filters_applied = True
                    logger.info(f"After furnishing filtering for '{furnishing}': {len(filtered_df)} properties")
                else:
                    logger.warning(f"Furnishing filter for '{furnishing}' would remove all properties, skipping this filter")
        
        # Price qualifier handling
        if query_details.get('price_qualifier') in ['cheap', 'affordable', 'budget']:
            # Set a reasonable max price if none was explicitly provided
            if query_details.get('max_price') is None and 'Price' in filtered_df.columns:
                # Calculate the 30th percentile of prices for cheap properties
                if len(filtered_df) >= 10:  # Only apply percentile for larger datasets
                    price_threshold = filtered_df['Price'].quantile(0.3)
                else:
                    # Use median for small datasets
                    price_threshold = filtered_df['Price'].median() * 0.8
                
                logger.info(f"Setting max price threshold to {price_threshold} based on 'cheap' qualifier")
                filtered_df = filtered_df[filtered_df['Price'] <= price_threshold]
                filters_applied = True
                logger.info(f"After price qualifier filtering: {len(filtered_df)} properties")
        
        elif query_details.get('price_qualifier') in ['luxury', 'expensive', 'high-end']:
            # For luxury properties, look at the top 30%
            if query_details.get('min_price') is None and 'Price' in filtered_df.columns:
                if len(filtered_df) >= 10:
                    price_threshold = filtered_df['Price'].quantile(0.7)
                else:
                    price_threshold = filtered_df['Price'].median() * 1.2
                
                logger.info(f"Setting min price threshold to {price_threshold} based on 'luxury' qualifier")
                filtered_df = filtered_df[filtered_df['Price'] >= price_threshold]
                filters_applied = True
                logger.info(f"After luxury price filtering: {len(filtered_df)} properties")
        
        # Location filtering
        if query_details.get('location'):
            loc = query_details['location'].lower()
            location_mask = pd.Series(False, index=filtered_df.index)
            
            for col in ['Processed_Location', 'Processed_City', 'Processed_Country']:
                if col in filtered_df.columns:
                    location_mask = location_mask | filtered_df[col].str.contains(loc, na=False, regex=True, case=False)
            
            # If we have a 'downtown' query, also look for 'center' and 'central'
            if loc.lower() == 'downtown':
                for col in ['Processed_Location', 'Processed_City', 'Processed_Country']:
                    if col in filtered_df.columns:
                        location_mask = location_mask | filtered_df[col].str.contains('center', na=False, regex=True, case=False)
                        location_mask = location_mask | filtered_df[col].str.contains('central', na=False, regex=True, case=False)
            
            temp_filtered = filtered_df[location_mask]
            # Only apply filter if it doesn't eliminate all properties
            if not temp_filtered.empty:
                filtered_df = temp_filtered
                filters_applied = True
                logger.info(f"After location filtering for '{loc}': {len(filtered_df)} properties")
            else:
                logger.warning(f"Location filter for '{loc}' would remove all properties, skipping this filter")
        
        # Bedrooms filtering - with threshold
        if query_details.get('bedrooms') is not None:
            if 'Bedrooms' in filtered_df.columns:
                # First try exact match
                exact_match = filtered_df[filtered_df['Bedrooms'] == query_details['bedrooms']]
                if not exact_match.empty:
                    filtered_df = exact_match
                    filters_applied = True
                    logger.info(f"After exact bedroom filtering: {len(filtered_df)} properties")
                else:
                    # If no exact match, try +/- 1 bedroom
                    logger.info("No exact bedroom match, trying with +/- 1 bedroom")
                    bedroom_mask = (
                        (filtered_df['Bedrooms'] >= query_details['bedrooms'] - 1) & 
                        (filtered_df['Bedrooms'] <= query_details['bedrooms'] + 1)
                    )
                    bedroom_filtered = filtered_df[bedroom_mask]
                    if not bedroom_filtered.empty:
                        filtered_df = bedroom_filtered
                        filters_applied = True
                        logger.info(f"After flexible bedroom filtering: {len(filtered_df)} properties")
                    else:
                        logger.warning("Bedroom filter would remove all properties, skipping this filter")
        
        # Price filtering (for explicit prices)
        if query_details.get('max_price') is not None:
            max_price = float(query_details['max_price'])
            if 'Price' in filtered_df.columns:
                temp_filtered = filtered_df[filtered_df['Price'] <= max_price]
                # Only apply filter if it doesn't eliminate all properties
                if not temp_filtered.empty:
                    filtered_df = temp_filtered
                    filters_applied = True
                    logger.info(f"After max price filtering: {len(filtered_df)} properties")
                else:
                    # If the filter would remove all properties, try with a higher threshold
                    logger.warning(f"Max price filter would remove all properties, trying with 20% higher threshold")
                    temp_filtered = filtered_df[filtered_df['Price'] <= max_price * 1.2]
                    if not temp_filtered.empty:
                        filtered_df = temp_filtered
                        filters_applied = True
                        logger.info(f"After adjusted max price filtering: {len(filtered_df)} properties")
                    else:
                        logger.warning("Price filter would remove all properties, skipping this filter")
        
        if query_details.get('min_price') is not None:
            min_price = float(query_details['min_price'])
            if 'Price' in filtered_df.columns:
                temp_filtered = filtered_df[filtered_df['Price'] >= min_price]
                if not temp_filtered.empty:
                    filtered_df = temp_filtered
                    filters_applied = True
                    logger.info(f"After min price filtering: {len(filtered_df)} properties")
                else:
                    logger.warning("Min price filter would remove all properties, skipping this filter")
        
        # Property type filtering
        if query_details.get('property_type'):
            req_type = query_details['property_type'].lower()
            
            # Define type variants
            type_variants = {
                'house': ['house', 'villa', 'townhouse', 'home', 'bungalow', 'mansion'],
                'apartment': ['apartment', 'flat', 'unit', 'condo', 'condominium', 'penthouse'],
                'villa': ['villa', 'luxury villa', 'mansion'],
                'studio': ['studio', 'studio apartment'],
                'townhouse': ['townhouse', 'town house', 'row house']
            }
            
            type_list = type_variants.get(req_type, [req_type])
            
            # Flexible type matching
            type_mask = pd.Series(False, index=filtered_df.index)
            for property_type in type_list:
                if 'Processed_Type' in filtered_df.columns:
                    type_mask = type_mask | filtered_df['Processed_Type'].str.contains(property_type, na=False, regex=True, case=False)
                if 'Type' in filtered_df.columns:
                    type_mask = type_mask | filtered_df['Type'].str.lower().str.contains(property_type, na=False, regex=True)
            
            # Try to match in description too
            if 'Description' in filtered_df.columns:
                for property_type in type_list:
                    type_mask = type_mask | filtered_df['Description'].str.lower().str.contains(property_type, na=False, regex=True)
            
            temp_filtered = filtered_df[type_mask]
            if not temp_filtered.empty:
                filtered_df = temp_filtered
                filters_applied = True
                logger.info(f"After property type filtering: {len(filtered_df)} properties")
            else:
                logger.warning(f"Property type filter for '{req_type}' would remove all properties, skipping this filter")
        
        # Special handling for "waterfront" or "ocean view"
        if hasattr(self, 'current_query') and ("waterfront" in self.current_query.lower() or "ocean view" in self.current_query.lower()):
            water_keywords = ["waterfront", "ocean", "sea", "beach", "marine", "water view", "waterside"]
            water_mask = pd.Series(False, index=filtered_df.index)
            
            for col in ['Description', 'Processed_Description', 'Title', 'Processed_Title']:
                if col in filtered_df.columns:
                    for keyword in water_keywords:
                        water_mask = water_mask | filtered_df[col].str.contains(keyword, na=False, regex=True, case=False)
            
            temp_filtered = filtered_df[water_mask]
            if not temp_filtered.empty:
                filtered_df = temp_filtered
                filters_applied = True
                logger.info(f"After waterfront/ocean view filtering: {len(filtered_df)} properties")
            else:
                logger.warning("Waterfront filter would remove all properties, skipping this filter")
        
        # Fall back to original data if no filters could be applied or if we filtered too much
        if filtered_df.empty or not filters_applied:
            logger.warning("All filters were too restrictive or no filters were applied. Using original dataset.")
            return self.knowledge_base
        
        return filtered_df
    
    def _semantic_search(self, filtered_df, user_query, top_k=5):
        """Perform semantic search on filtered properties."""
        if filtered_df.empty:
            logger.warning("Cannot perform semantic search on empty dataset")
            return filtered_df
        
        # If we only have a few properties, return them all
        if len(filtered_df) <= top_k:
            logger.info(f"Only {len(filtered_df)} properties in filtered set, returning all without semantic search")
            return filtered_df
        
        # Limit results to available data
        top_k = min(top_k, len(filtered_df))
        
        # Create search texts
        search_columns = []
        for field in ['Processed_Location', 'Processed_City', 'Processed_Country', 'Processed_Type', 'Processed_Description', 'Title']:
            if field in filtered_df.columns:
                search_columns.append(field)
        
        if not search_columns:
            logger.warning("No text columns available for semantic search")
            return filtered_df.head(top_k)
        
        search_texts = filtered_df.apply(
            lambda row: ' '.join(str(row.get(col, '')) for col in search_columns),
            axis=1
        )
        
        # Calculate similarities
        try:
            corpus = list(search_texts) + [user_query.lower()]
            tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(corpus)
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            return filtered_df.iloc[top_indices]
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            # Fallback to simple ranking if semantic search fails
            return filtered_df.head(top_k)
    
    def get_property_insights(self, properties):
        """Format property details for presentation."""
        insights = []
        for _, prop in properties.iterrows():
            try:
                price = float(prop['Price']) if pd.notna(prop.get('Price')) else np.nan
            
                price_formatted = f"AED {price:,.2f}" if pd.notna(price) else "N/A"
            
                # Extract numeric value from Bedrooms column (e.g., "5 beds" -> 5)
                bedrooms = prop.get('Bedrooms', 'N/A')
                if isinstance(bedrooms, str):
                    match = re.search(r'(\d+)', bedrooms)
                    bedrooms = int(match.group(1)) if match else "N/A"
                elif pd.isna(bedrooms):
                    bedrooms = "N/A"
            
                # Format the area
                area = prop.get('Area(Sqft)', prop.get('Area', 'N/A'))
                if isinstance(area, str):
                    area_clean = re.sub(r'[^\d.]', '', area)
                    area_formatted = f"{float(area_clean):,.2f} sqft" if area_clean else "N/A"
                elif pd.notna(area) and isinstance(area, (int, float)):
                    area_formatted = f"{area:,.2f} sqft"
                else:
                    area_formatted = "N/A"
            
                # Format location
                location = prop.get('Location', 'N/A')
                city = prop.get('city', '')
                country = prop.get('country', '')
            
                full_location = f"{location}"
                if city and pd.notna(city):
                    full_location += f", {city}"
                if country and pd.notna(country):
                    full_location += f", {country}"
            
                # Get property details
                property_insight = {
                    'title': prop.get('Title', 'Property'),
                    'price': price_formatted,
                    'purpose': prop.get('Purpose', 'N/A'),
                    'type': prop.get('Type', 'N/A'),
                    'bedrooms': bedrooms,
                    'bathrooms': prop.get('Bathrooms', 'N/A'),
                    'area': area_formatted,
                    'location': full_location,
                    'furnishing': prop.get('Furnishing', 'N/A'),
                    'description': prop.get('Description', 'No description available'),
                    'amenities': prop.get('Amenities', 'N/A')
                }
            
                insights.append(property_insight)
            except Exception as e:
               logger.error(f"Error processing property: {e}")
    
        return insights

    def _format_response_with_typing_effect(self, properties, user_query, query_details):
        """Format response for a more conversational, typing-like experience."""
        if not properties.empty:
            insights = self.get_property_insights(properties)
        
            # Create intro message for the response
            purpose_str = f" for {query_details.get('purpose', 'purchase or rent')}" if query_details.get('purpose') else ""
            location_str = f" in {query_details.get('location')}" if query_details.get('location') else ""
            bedrooms_str = f" with {query_details.get('bedrooms')} bedrooms" if query_details.get('bedrooms') is not None else ""
            price_str = ""
            if query_details.get('max_price'):
                price_str = f" under AED {query_details.get('max_price'):,.2f}"
        
            intro = f"📋 I've found {len(insights)} properties{purpose_str}{location_str}{bedrooms_str}{price_str} that might interest you.\n\nLet me share the details one by one:"
        
            # Format each property in a more conversational way
            property_responses = []
            for i, insight in enumerate(insights[:3]):  # Limit to 3 properties for brevity
                response = f"\n\n🏡 **Property {i+1}: {insight['title']}**\n"
                response += f"💰 **Price**: {insight['price']} ({insight['purpose']})\n"
                response += f"🏢 **Type**: {insight['type']}\n"
                response += f"🛏️ **Bedrooms**: {insight['bedrooms']}\n"
                response += f"📐 **Area**: {insight['area']}\n"
                response += f"📍 **Location**: {insight['location']}\n"
                response += f"🏠 **Furnishing**: {insight['furnishing']}\n"
            
                # Add a brief description
                description = insight['description']
                if isinstance(description, str) and len(description) > 150:
                    description = description[:147] + "..."
                response += f"\n*{description}*"
            
                property_responses.append(response)
        
            # Add more results info if applicable
            more_results = ""
            if len(insights) > 3:
                more_results = f"\n\nI have {len(insights) - 3} more properties that match your criteria. Would you like to see more options?"
        
            # Add follow-up question
            follow_up = "\n\nIs there anything specific about these properties you'd like to know more about? Or would you like to refine your search?"
        
            full_response = intro + "".join(property_responses) + more_results + follow_up
            return full_response
        else:
            return "I couldn't find any properties matching your criteria. Would you like to try a different search?"

    def _get_dataset_overview(self):
        """Generate an overview of the available data in the knowledge base."""
        try:
            stats = self.stats
            overview = "📊 **Real Estate Dataset Overview:**\n\n"
        
            # Property purpose statistics
            if 'purposes' in stats:
                overview += "**Properties by Purpose:**\n"
                for purpose, count in stats['purposes'].items():
                    overview += f"- {purpose}: {count} properties\n"
                overview += "\n"
        
            # Property types
            if 'types' in stats:
                overview += "**Property Types Available:**\n"
                for prop_type, count in stats['types'].items():
                    overview += f"- {prop_type}: {count} properties\n"
                overview += "\n"
        
            # Price ranges
            if 'price_min' in stats and 'price_max' in stats:
                overview += "**Price Ranges:**\n"
                overview += f"- Overall: AED {stats['price_min']:,.2f} to AED {stats['price_max']:,.2f}\n"
            
                if 'rent_price_min' in stats and 'rent_price_max' in stats:
                    overview += f"- Rental properties: AED {stats['rent_price_min']:,.2f} to AED {stats['rent_price_max']:,.2f}\n"
            
                if 'sale_price_min' in stats and 'sale_price_max' in stats:
                    overview += f"- Properties for sale: AED {stats['sale_price_min']:,.2f} to AED {stats['sale_price_max']:,.2f}\n"
            
                overview += "\n"
        
            # Top locations
            if 'top_locations' in stats:
                overview += "**Top Locations:**\n"
                for location, count in list(stats['top_locations'].items())[:5]:
                    overview += f"- {location}: {count} properties\n"
                overview += "\n"
        
            # Bedroom distribution
            if 'bedrooms' in stats:
                overview += "**Bedroom Distribution:**\n"
                for bedrooms, count in stats['bedrooms'].items():
                    if pd.notna(bedrooms):
                        overview += f"- {bedrooms} bedrooms: {count} properties\n"
                overview += "\n"
        
            overview += "Would you like to search for a specific type of property or learn more about properties in a particular area?"
        
            return overview
    
        except Exception as e:
            logger.error(f"Error generating dataset overview: {e}")
            return "I have information on various properties for rent and sale. What type of property are you looking for?"

    def _process_user_message_with_typing(self, message):
        """Process user message and generate a response with simulated typing."""
        self.current_query = message
        self.conversation_history.append({"role": "user", "content": message})
    
        # Check for special cases
        if self._is_greeting(message):
            self.conversation_state = "introduced"
            greeting = random.choice(self.greetings)
            self.conversation_history.append({"role": "assistant", "content": greeting})
            return greeting
    
        if self._is_basic_question(message):
            response = "I'm your real estate assistant powered by AI. I can help you find properties based on your preferences such as location, price range, number of bedrooms, and more. I can provide details about available properties, including prices, amenities, and locations. What kind of property are you looking for today?"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
    
        if self._is_data_question(message):
            overview = self._get_dataset_overview()
            self.conversation_history.append({"role": "assistant", "content": overview})
            return overview
    
        # For property searches, use Gemini API to extract structured data
        gemini_prompt = self._create_gemini_query_prompt(message)
        query_details = self._call_gemini_api(gemini_prompt)
    
        if query_details:
            # Update user preferences
            for key, value in query_details.items():
                if value is not None and value != []:
                    self.user_preferences[key] = value
        
            # Log extracted details
            logger.info(f"Extracted query details: {query_details}")
        
            # Filter properties
            filtered_properties = self._filter_properties(query_details)
        
            # Semantic search for relevance
            relevant_properties = self._semantic_search(filtered_properties, message, top_k=5)
        
            # Format response
            response = self._format_response_with_typing_effect(relevant_properties, message, query_details)
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        else:
            fallback = "I'm not quite sure what you're looking for. Could you provide more details about the type of property, location, or price range you're interested in?"
            self.conversation_history.append({"role": "assistant", "content": fallback})
            return fallback

    def respond_with_typing(self, user_message):
        """Main method to respond to user messages with a typing effect."""
        # Start with an immediate acknowledgment
        acknowledgment = None
    
        # Choose an acknowledgment based on message content
        if self._is_greeting(user_message):
            acknowledgment = "Hi there! 👋"
        elif "help" in user_message.lower() or "looking for" in user_message.lower():
            acknowledgment = "I'll help you find that! 🔍"
        elif "available" in user_message.lower() or "show me" in user_message.lower():
            acknowledgment = "Let me check what's available..."
        elif "price" in user_message.lower() or "cost" in user_message.lower():
            acknowledgment = "Checking prices for you..."
        elif "location" in user_message.lower() or "area" in user_message.lower():
            acknowledgment = "Looking at locations for you..."
        else:
            acknowledgment = "Let me look that up for you..."
    
        # Process the message to get the full response
        full_response = self._process_user_message_with_typing(user_message)
    
        # Return both the acknowledgment and full response
        return {"acknowledgment": acknowledgment, "full_response": full_response}

    def demo_typing_response(self, user_message):
        """Demonstrate the typing response functionality."""
        response_data = self.respond_with_typing(user_message)
    
        # Print acknowledgment immediately
        print(f"\nAssistant: {response_data['acknowledgment']}")
    
        # Simulate typing for the full response
        full_response = response_data['full_response']
    
        # Divide the response into words and "type" them with variable delay
        words = full_response.split()
        typed_text = ""
    
        for i, word in enumerate(words):
            typed_text += word + " "
        
            # Print the text so far after every few words
            if i % 5 == 0 or i == len(words) - 1:
                # Clear previous line and print updated text
                sys.stdout.write("\rAssistant: " + typed_text)
                sys.stdout.flush()
            
                # Add varied typing delay for realism
                time.sleep(random.uniform(0.1, 0.3))
    
        # Ensure we end with a newline
        print()
        return full_response

    # Example usage in a command-line interface
    def run_cli_interface():
        """Run a simple command-line interface for the Real Estate Agent."""
        # Set up API key - replace with your actual Gemini API key
        gemini_api_key = "AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY"  # Replace with your actual API key

        # Path to the CSV file with property data
        csv_path = "realestatedata.csv"

        # Initialize the agent
        agent = RealEstateAgent(gemini_api_key, csv_path)

        print("Real Estate Assistant initialized. Type 'exit' to quit.")
        print(random.choice(agent.greetings))

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ('exit', 'quit', 'bye'):
                print("Assistant: Thank you for using the Real Estate Assistant. Goodbye!")
                break
    
            # Use the typing effect response
            agent.demo_typing_response(user_input)

# Call the function outside the class
    if __name__ == "__main__":
        run_cli_interface()