# import requests
# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# import re

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def load_markdown_table(md_path):
#     """
#     Load a markdown table from a .md file and convert it into a pandas DataFrame.
#     Assumes the table is defined using pipe-delimited rows.
#     """
#     with open(md_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
    
#     # Filter out lines that start with a pipe
#     table_lines = [line.strip() for line in lines if line.strip().startswith("|")]
#     if not table_lines:
#         raise ValueError("No markdown table found in the file.")
    
#     # The first line is header, second line is separator, then data rows
#     header_line = table_lines[0]
#     headers = [h.strip() for h in header_line.strip("|").split("|")]
    
#     data_rows = []
#     for line in table_lines[2:]:
#         row = [cell.strip() for cell in line.strip("|").split("|")]
#         if len(row) == len(headers):
#             data_rows.append(row)
    
#     df = pd.DataFrame(data_rows, columns=headers)
#     return df

# class DebugRealEstateAgent:
#     def __init__(self, gemini_api_key, md_path):
#         """
#         Initialize the Debug Real Estate Agent.
        
#         :param gemini_api_key: Google Gemini API key for natural language processing.
#         :param md_path: Path to the Markdown knowledge base file.
#         """
#         self.gemini_api_key = gemini_api_key
#         self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        
#         # Load and preprocess the knowledge base from the Markdown file
#         try:
#             self.knowledge_base = load_markdown_table(md_path)
#             logger.debug(f"Loaded knowledge base columns: {self.knowledge_base.columns}")
#             logger.debug(f"Total properties loaded: {len(self.knowledge_base)}")
#             self._preprocess_knowledge_base()
#         except Exception as e:
#             logger.error(f"Error loading knowledge base: {e}")
#             raise
        
#         # Initialize semantic search capabilities
#         self._initialize_search_capabilities()
    
#     def _preprocess_knowledge_base(self):
#         """
#         Preprocess the knowledge base:
#           - Convert numeric columns to numbers.
#           - Normalize text columns.
#         """
#         # Logging the raw data before preprocessing
#         logger.debug("Raw knowledge base data:")
#         logger.debug(self.knowledge_base.head())
        
#         # Properly clean and convert numeric columns
#         if 'Price' in self.knowledge_base.columns:
#             # Fix: Handle different price formats with more robust conversion
#             self.knowledge_base['Price'] = self.knowledge_base['Price'].astype(str)
#             self.knowledge_base['Price'] = self.knowledge_base['Price'].str.replace('$', '', regex=False)
#             self.knowledge_base['Price'] = self.knowledge_base['Price'].str.replace(',', '', regex=False)
#             self.knowledge_base['Price'] = pd.to_numeric(self.knowledge_base['Price'], errors='coerce')
#             logger.debug(f"Price column after conversion: {self.knowledge_base['Price'].head()}")
        
#         if 'Bedrooms' in self.knowledge_base.columns:
#             self.knowledge_base['Bedrooms'] = pd.to_numeric(self.knowledge_base['Bedrooms'], errors='coerce')
        
#         if 'Area' in self.knowledge_base.columns:
#             # Handle potential text in Area column
#             self.knowledge_base['Area'] = self.knowledge_base['Area'].astype(str)
#             self.knowledge_base['Area'] = self.knowledge_base['Area'].str.replace('sqft', '', regex=False)
#             self.knowledge_base['Area'] = self.knowledge_base['Area'].str.replace(',', '', regex=False)
#             self.knowledge_base['Area'] = pd.to_numeric(self.knowledge_base['Area'], errors='coerce')
        
#         # Normalize text columns - ensure we're checking if column exists first
#         text_columns = ['Location', 'city', 'country', 'Type', 'Title', 'Description']
#         for col in text_columns:
#             if col in self.knowledge_base.columns:
#                 self.knowledge_base[f'Processed_{col}'] = self.knowledge_base[col].fillna('').astype(str).str.lower()
        
#         # Logging after preprocessing
#         logger.debug("Preprocessed knowledge base:")
#         logger.debug(f"Data types: {self.knowledge_base.dtypes}")
#         logger.debug(self.knowledge_base.head())
    
#     def _initialize_search_capabilities(self):
#         """
#         Initialize TF-IDF-based semantic search using selected fields.
#         """
#         # Create combined search text from all relevant fields
#         search_columns = []
        
#         # Add fields that actually exist in the DataFrame
#         for field in ['Processed_Location', 'Processed_city', 'Processed_country', 'Processed_Type', 'Processed_Description']:
#             if field in self.knowledge_base.columns:
#                 search_columns.append(field)
        
#         # Add numeric columns as strings if they exist
#         for field in ['Bedrooms', 'Price']:
#             if field in self.knowledge_base.columns:
#                 search_columns.append(field)
        
#         # Create the search texts by combining available columns
#         self.search_texts = self.knowledge_base.apply(
#             lambda row: ' '.join(str(row.get(col, '')) for col in search_columns), 
#             axis=1
#         )
        
#         logger.debug(f"First few search texts: {list(self.search_texts[:2])}")
        
#         self.vectorizer = TfidfVectorizer(stop_words='english')
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
#         logger.debug(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
#     def _call_gemini_api(self, prompt):
#         """
#         Call the Google Gemini API to process and structure the user query.
        
#         :param prompt: Prompt string.
#         :return: Parsed query details as a dictionary, or None on failure.
#         """
#         try:
#             payload = {
#                 "contents": [{"parts": [{"text": prompt}]}],
#                 "generationConfig": {
#                     "temperature": 0.3,
#                     "maxOutputTokens": 1024
#                 }
#             }
#             headers = {'Content-Type': 'application/json'}
#             response = requests.post(self.gemini_url, headers=headers, data=json.dumps(payload))
            
#             logger.debug(f"Gemini raw response: {response.text}")
#             if response.status_code != 200:
#                 logger.error(f"Gemini API error {response.status_code}: {response.text}")
#                 return None
            
#             response_json = response.json()
#             candidate = response_json.get("candidates", [{}])[0]
#             generated_text = candidate.get("output")
#             if not generated_text:
#                 generated_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            
#             logger.debug(f"Gemini generated text: {generated_text}")
            
#             # Remove markdown code fences if present
#             if generated_text.startswith("```"):
#                 generated_text = re.sub(r'^```(?:json)?\s*', '', generated_text)
#                 generated_text = re.sub(r'\s*```$', '', generated_text)
            
#             query_details = json.loads(generated_text)
#             return query_details
        
#         except Exception as e:
#             logger.error(f"Gemini API call failed: {e}")
#             return None
    
#     def _create_gemini_query_prompt(self, user_query):
#         """
#         Create a prompt for Gemini to extract structured query details.
        
#         :param user_query: The user's original query.
#         :return: A formatted prompt string.
#         """
#         prompt = f"""
# Analyze the following real estate query and extract structured details:

# Query: "{user_query}"

# Please provide a JSON response with the following keys:
# - location: Extracted city or region (string or null)
# - bedrooms: Number of bedrooms (integer or null)
# - property_type: Type of property (string or null, options: 'house', 'apartment', 'villa', 'studio')
# - max_price: Maximum budget (float or null)
# - min_price: Minimum budget (float or null)
# - amenities: List of desired amenities (list of strings or empty list)

# Provide null or empty values if not specified in the query.

# Example output:
# {{
#     "location": "Dubai",
#     "bedrooms": 3,
#     "property_type": "house",
#     "max_price": 3000000.0,
#     "min_price": null,
#     "amenities": ["parking", "gym"]
# }}
# """
#         return prompt.strip()
    
#     def _filter_properties(self, query_details):
#         """
#         Filter the knowledge base DataFrame based on the extracted query details.
        
#         :param query_details: Dictionary of query details.
#         :return: A filtered DataFrame.
#         """
#         filtered_df = self.knowledge_base.copy()
        
#         # Log initial dataframe size
#         logger.debug(f"Initial dataframe size: {len(filtered_df)}")
        
#         # Print sample data for debugging
#         logger.debug(f"Sample data before filtering:\n{filtered_df[['Price', 'Bedrooms', 'Type']].head().to_dict()}")
        
#         # Enhanced location filtering
#         if query_details.get('location'):
#             loc = query_details['location'].lower()
#             logger.debug(f"Filtering by location: {loc}")
            
#             location_mask = False
#             for col in ['Processed_Location', 'Processed_city', 'Processed_country']:
#                 if col in filtered_df.columns:
#                     location_mask = location_mask | filtered_df[col].str.contains(loc, na=False, regex=True)
            
#             filtered_df = filtered_df[location_mask]
#             logger.debug(f"Properties after location filter: {len(filtered_df)}")
        
#         # Bedrooms filtering - only if specified
#         if query_details.get('bedrooms') is not None:
#             logger.debug(f"Filtering by bedrooms: {query_details['bedrooms']}")
#             # Use approximate matching for bedrooms to be more flexible
#             filtered_df = filtered_df[filtered_df['Bedrooms'] == query_details['bedrooms']]
#             logger.debug(f"Properties after bedrooms filter: {len(filtered_df)}")
        
#         # Price filtering - important fix
#         if query_details.get('max_price') is not None:
#             max_price = float(query_details['max_price'])
#             logger.debug(f"Filtering by max price: {max_price}")
#             logger.debug(f"Price column type: {filtered_df['Price'].dtype}")
#             logger.debug(f"Price sample values: {filtered_df['Price'].head()}")
            
#             # Make sure Price column is numeric before comparison
#             if 'Price' in filtered_df.columns:
#                 filtered_df = filtered_df[filtered_df['Price'] <= max_price]
#                 logger.debug(f"Properties after max price filter: {len(filtered_df)}")
#                 logger.debug(f"Sample properties after max price filter: {filtered_df[['Price', 'Bedrooms', 'Type']].head()}")
        
#         if query_details.get('min_price') is not None:
#             min_price = float(query_details['min_price'])
#             logger.debug(f"Filtering by min price: {min_price}")
#             if 'Price' in filtered_df.columns:
#                 filtered_df = filtered_df[filtered_df['Price'] >= min_price]
#                 logger.debug(f"Properties after min price filter: {len(filtered_df)}")
        
#         # Enhanced property type filtering
#         if query_details.get('property_type'):
#             req_type = query_details['property_type'].lower()
#             logger.debug(f"Filtering by property type: {req_type}")
            
#             # Define type variants and synonyms
#             type_variants = {
#                 'house': ['house', 'villa', 'townhouse', 'home', 'bungalow'],
#                 'apartment': ['apartment', 'flat', 'unit', 'condo', 'condominium'],
#                 'villa': ['villa', 'luxury villa', 'mansion'],
#                 'studio': ['studio', 'studio apartment']
#             }
            
#             type_list = type_variants.get(req_type, [req_type])
#             logger.debug(f"Looking for property types: {type_list}")
#             logger.debug(f"Unique property types in data: {filtered_df['Type'].unique()}")
            
#             # More flexible type matching using string contains
#             type_mask = False
#             for property_type in type_list:
#                 type_mask = type_mask | filtered_df['Processed_Type'].str.contains(property_type, na=False, regex=True)
            
#             filtered_with_type = filtered_df[type_mask]
            
#             if not filtered_with_type.empty:
#                 filtered_df = filtered_with_type
            
#             logger.debug(f"Properties after type filter: {len(filtered_df)}")
        
#         # Safety check for empty results
#         if filtered_df.empty:
#             logger.warning("No properties matched all criteria. Returning partial matches.")
#             # Fall back to just using the most critical filter: price or location
#             if query_details.get('max_price') is not None:
#                 filtered_df = self.knowledge_base[self.knowledge_base['Price'] <= float(query_details['max_price'])]
#             elif query_details.get('location'):
#                 loc = query_details['location'].lower()
#                 location_mask = False
#                 for col in ['Processed_Location', 'Processed_city', 'Processed_country']:
#                     if col in self.knowledge_base.columns:
#                         location_mask = location_mask | self.knowledge_base[col].str.contains(loc, na=False, regex=True)
#                 filtered_df = self.knowledge_base[location_mask]
            
#             logger.debug(f"Properties after fallback filtering: {len(filtered_df)}")
        
#         return filtered_df
    
#     def _semantic_search(self, filtered_df, user_query, top_k=5):
#         """
#         Perform a semantic search using cosine similarity on filtered properties.
        
#         :param filtered_df: DataFrame after filtering.
#         :param user_query: The original query.
#         :param top_k: Number of top results to return.
#         :return: A DataFrame of top matching properties.
#         """
#         if filtered_df.empty:
#             return filtered_df
        
#         # Ensure we don't try to return more properties than we have
#         top_k = min(top_k, len(filtered_df))
        
#         # Create search texts for the filtered properties
#         search_columns = []
#         for field in ['Processed_Location', 'Processed_city', 'Processed_country', 'Processed_Type', 'Processed_Description']:
#             if field in filtered_df.columns:
#                 search_columns.append(field)
        
#         for field in ['Bedrooms', 'Price']:
#             if field in filtered_df.columns:
#                 search_columns.append(field)
        
#         search_texts = filtered_df.apply(
#             lambda row: ' '.join(str(row.get(col, '')) for col in search_columns), 
#             axis=1
#         )
        
#         # Create corpus with user query and calculate similarities
#         corpus = list(search_texts) + [user_query]
#         try:
#             tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(corpus)
#             similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
            
#             # Get top indices and return corresponding properties
#             top_indices = similarities.argsort()[-top_k:][::-1]
#             top_properties = filtered_df.iloc[top_indices]
            
#             logger.debug(f"Top property similarity scores: {similarities[top_indices]}")
#             return top_properties
            
#         except Exception as e:
#             logger.error(f"Error in semantic search: {e}")
#             # Return the filtered DataFrame if semantic search fails
#             return filtered_df.head(top_k)
    
#     def get_property_insights(self, properties):
#         """
#         Generate formatted insights for each matched property.
        
#         :param properties: DataFrame of properties.
#         :return: List of property insight dictionaries.
#         """
#         insights = []
#         for _, prop in properties.iterrows():
#             try:
#                 # Safely handle numeric values
#                 price = float(prop['Price']) if pd.notna(prop['Price']) else np.nan
#                 area = float(prop['Area']) if pd.notna(prop['Area']) and float(prop['Area']) > 0 else np.nan
                
#                 price_per_sqft = f"${price / area:,.2f}" if pd.notna(price) and pd.notna(area) else "N/A"
                
#                 # Format price with proper currency
#                 price_formatted = f"${price:,.2f}" if pd.notna(price) else "N/A"
                
#             except Exception as e:
#                 logger.error(f"Error formatting property insights: {e}")
#                 price_formatted = "N/A"
#                 price_per_sqft = "N/A"
            
#             # Create property insight
#             insight = {
#                 'Title': str(prop.get('Title', 'No Title')),
#                 'Location': str(prop.get('Location', 'Unknown')),
#                 'Price': price_formatted,
#                 'Bedrooms': str(prop.get('Bedrooms', 'N/A')),
#                 'Type': str(prop.get('Type', 'N/A')),
#                 'Area(sqft)': str(prop.get('Area', 'N/A')),
#                 'Price per sqft': price_per_sqft,
#                 'Description': str(prop.get('Description', 'No description provided'))
#             }
#             insights.append(insight)
        
#         return insights
    
#     def process_real_estate_query(self, user_query):
#         """
#         Process the user query: use Gemini to extract structured details,
#         filter properties, perform semantic search, and return insights.
        
#         :param user_query: The user's query string.
#         :return: Dictionary with query details and matching property insights.
#         """
#         logger.info(f"Processing query: {user_query}")
        
#         prompt = self._create_gemini_query_prompt(user_query)
#         query_details = self._call_gemini_api(prompt)
        
#         if not query_details:
#             return {
#                 'error': 'Unable to process query. Please rephrase.',
#                 'raw_query': user_query
#             }
        
#         logger.debug(f"Extracted Query Details: {query_details}")
        
#         filtered = self._filter_properties(query_details)
#         logger.info(f"Found {len(filtered)} properties after filtering")
        
#         if filtered.empty:
#             return {
#                 'status': 'No matching properties found',
#                 'query_details': query_details,
#                 'suggestions': 'Try broadening your search criteria'
#             }
        
#         matching = self._semantic_search(filtered, user_query)
#         logger.info(f"Selected {len(matching)} top matching properties")
        
#         insights = self.get_property_insights(matching)
#         return {
#             'status': 'Success',
#             'query_details': query_details,
#             'property_count': len(matching),
#             'properties': insights
#         }

# def main():
#     # Replace with your actual Gemini API key and Markdown file path
#     GEMINI_API_KEY = "AIzaSyAQJQK4fROJzHS0e1n_hrzDIP7Ed3Bz3dI"
#     MD_PATH = "output.md"
    
#     try:
#         agent = DebugRealEstateAgent(GEMINI_API_KEY, MD_PATH)
#         # Example queries
#         queries = [
#             "I want a 3-bedroom house in Dubai having cost less than 3000000",
#             "Looking for a cheap apartment near downtown",
#             "Waterfront properties with ocean view in dubai"
#         ]
        
#         for query in queries:
#             print(f"\nQuery: {query}")
#             result = agent.process_real_estate_query(query)
#             print(json.dumps(result, indent=2))
    
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
#         import traceback
#         logger.error(traceback.format_exc())

# if __name__ == "__main__":
#     main()


















import requests
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import os

# Enhanced logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_properties_from_csv(csv_path):
    """Load property data from a CSV file into a pandas DataFrame."""
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Print file details for debugging
        logger.info(f"Loading CSV file from: {csv_path}")
        logger.info(f"File size: {os.path.getsize(csv_path)} bytes")
        
        # Try to load the file and print a preview
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} properties")
        logger.info(f"CSV columns: {df.columns.tolist()}")
        logger.info(f"First few rows:\n{df.head(2)}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise ValueError(f"Error loading CSV file: {e}")

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
    
    def _preprocess_knowledge_base(self):
        """Preprocess the knowledge base by converting and normalizing columns."""
        logger.info("Preprocessing knowledge base")
        
        # Make column names consistent (case-insensitive)
        self.knowledge_base.columns = [col.strip() for col in self.knowledge_base.columns]
        column_map = {col: col.capitalize() for col in self.knowledge_base.columns}
        self.knowledge_base = self.knowledge_base.rename(columns=column_map)
        
        # Ensure critical columns exist
        essential_columns = ['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title']
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
        
        if 'Bedrooms' in self.knowledge_base.columns:
            self.knowledge_base['Bedrooms'] = pd.to_numeric(self.knowledge_base['Bedrooms'], errors='coerce')
            logger.info(f"Bedroom range: {self.knowledge_base['Bedrooms'].min()} to {self.knowledge_base['Bedrooms'].max()}")
        
        if 'Area' in self.knowledge_base.columns:
            self.knowledge_base['Area'] = self.knowledge_base['Area'].astype(str)
            self.knowledge_base['Area'] = self.knowledge_base['Area'].str.replace('sqft', '', regex=False)
            self.knowledge_base['Area'] = self.knowledge_base['Area'].str.replace(',', '', regex=False)
            self.knowledge_base['Area'] = pd.to_numeric(self.knowledge_base['Area'], errors='coerce')
        
        # Normalize text columns
        text_columns = ['Location', 'City', 'Country', 'Type', 'Title', 'Description']
        for col in text_columns:
            if col in self.knowledge_base.columns:
                self.knowledge_base[f'Processed_{col}'] = self.knowledge_base[col].fillna('').astype(str).str.lower()
        
        logger.info(f"Preprocessing complete. Knowledge base has {len(self.knowledge_base)} properties.")
    
    def _initialize_search_capabilities(self):
        """Initialize TF-IDF-based semantic search."""
        logger.info("Initializing search capabilities")
        # Determine available columns for search
        search_columns = []
        for field in ['Processed_Location', 'Processed_City', 'Processed_Country', 'Processed_Type', 'Processed_Description']:
            if field in self.knowledge_base.columns:
                search_columns.append(field)
        
        logger.info(f"Using following columns for search: {search_columns}")
        
        # Create search texts
        self.search_texts = self.knowledge_base.apply(
            lambda row: ' '.join(str(row.get(col, '')) for col in search_columns), 
            axis=1
        )
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
        logger.info("Search capabilities initialized")
    
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
                return None
            
            response_json = response.json()
            candidate = response_json.get("candidates", [{}])[0]
            generated_text = candidate.get("output")
            if not generated_text:
                generated_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Remove markdown code fences if present
            if generated_text.startswith("```"):
                generated_text = re.sub(r'^```(?:json)?\s*', '', generated_text)
                generated_text = re.sub(r'\s*```$', '', generated_text)
            
            query_details = json.loads(generated_text)
            logger.info(f"Extracted query details: {query_details}")
            return query_details
        
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
            "amenities": []
        }
        
        # Basic location extraction
        locations = ["dubai", "downtown", "waterfront"]
        for loc in locations:
            if loc in query:
                result["location"] = loc
                break
        
        # Basic bedroom extraction
        bedroom_match = re.search(r'(\d+)[- ]?bed', query)
        if bedroom_match:
            result["bedrooms"] = int(bedroom_match.group(1))
        
        # Basic property type extraction
        if "house" in query:
            result["property_type"] = "house"
        elif "apartment" in query:
            result["property_type"] = "apartment"
        elif "villa" in query:
            result["property_type"] = "villa"
        
        # Basic price extraction
        price_match = re.search(r'(less than|under|below) (\d+)', query)
        if price_match:
            result["max_price"] = float(price_match.group(2))
        
        # Basic price qualifier
        if any(word in query for word in ["cheap", "affordable", "budget"]):
            result["price_qualifier"] = "cheap"
        elif any(word in query for word in ["luxury", "expensive", "high-end"]):
            result["price_qualifier"] = "luxury"
        
        # Basic amenities
        amenities = ["pool", "gym", "balcony", "view", "parking", "furnished"]
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
- property_type: Type of property (string or null, options: 'house', 'apartment', 'villa', 'studio')
- max_price: Maximum budget (float or null)
- min_price: Minimum budget (float or null)
- price_qualifier: If a qualitative price descriptor is used (string or null, examples: 'cheap', 'affordable', 'luxury', 'budget', 'expensive')
- amenities: List of desired amenities (list of strings or empty list)

If the query mentions "cheap" or "affordable", set price_qualifier to that value.

Provide null or empty values if not specified in the query.
"""
        return prompt.strip()
    
    def _filter_properties(self, query_details):
        """Filter properties based on query details."""
        filtered_df = self.knowledge_base.copy()
        logger.info(f"Starting with {len(filtered_df)} properties before filtering")
        
        # Start with less restrictive filtering for small datasets
        filters_applied = False
        
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
                'studio': ['studio', 'studio apartment']
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
        
        logger.info(f"Using columns for semantic search: {search_columns}")
        
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
            
            logger.info(f"Semantic search similarities range: {similarities.min()} to {similarities.max()}")
            
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
                area = float(prop['Area']) if pd.notna(prop.get('Area')) and float(prop['Area']) > 0 else np.nan
                
                price_per_sqft = f"${price / area:,.2f}" if pd.notna(price) and pd.notna(area) else "N/A"
                price_formatted = f"${price:,.2f}" if pd.notna(price) else "N/A"
                
            except Exception as e:
                logger.error(f"Error formatting price or area: {e}")
                price_formatted = "N/A"
                price_per_sqft = "N/A"
            
            insight = {
                'Title': str(prop.get('Title', 'No Title')),
                'Location': str(prop.get('Location', 'Unknown')),
                'Price': price_formatted,
                'Bedrooms': str(prop.get('Bedrooms', 'N/A')),
                'Type': str(prop.get('Type', 'N/A')),
                'Area(sqft)': str(prop.get('Area', 'N/A')),
                'Price per sqft': price_per_sqft,
                'Description': str(prop.get('Description', 'No description provided'))
            }
            insights.append(insight)
        
        return insights
    
    def format_results_as_points(self, user_query, matching_properties):
        """Format results as readable points without JSON structure."""
        if not matching_properties:
            return f"Query: {user_query}\n\nNo matching properties found. Try broadening your search criteria."
        
        result = f"Query: {user_query}\n\n"
        result += f"Found {len(matching_properties)} matching properties:\n\n"
        
        for i, prop in enumerate(matching_properties, 1):
            result += f"Property {i}:\n"
            result += f"- Title: {prop['Title']}\n"
            result += f"- Location: {prop['Location']}\n"
            result += f"- Price: {prop['Price']}\n"
            result += f"- Type: {prop['Type']}\n"
            result += f"- Bedrooms: {prop['Bedrooms']}\n"
            result += f"- Area: {prop['Area(sqft)']} sqft\n"
            result += f"- Price per sqft: {prop['Price per sqft']}\n"
            
            # Truncate description if too long
            desc = prop['Description']
            if len(desc) > 150:
                desc = desc[:147] + "..."
            result += f"- Description: {desc}\n\n"
        
        return result
    
    def process_real_estate_query(self, user_query):
        """Process user query and return matching properties in a simple format."""
        # Extract query details
        prompt = self._create_gemini_query_prompt(user_query)
        query_details = self._call_gemini_api(prompt)
    
        if not query_details:
            return f"Query: {user_query}\n\nUnable to process query. Please rephrase."
    
        # Store the query in a class variable instead of using global
        self.current_query = user_query
    
        # Filter and search with more flexible criteria
        filtered = self._filter_properties(query_details)
    
        if filtered.empty:
            return f"Query: {user_query}\n\nNo matching properties found. Try broadening your search criteria."
    
        logger.info(f"Performing semantic search on {len(filtered)} filtered properties")
        matching = self._semantic_search(filtered, user_query)
    
        if matching.empty:
            return f"Query: {user_query}\n\nNo matching properties found after semantic search."
    
        insights = self.get_property_insights(matching)
    
        # Format results as plain text points
        return self.format_results_as_points(user_query, insights)

def main():
    # Replace with your actual Gemini API key and CSV file path
    GEMINI_API_KEY = "AIzaSyAQJQK4fROJzHS0e1n_hrzDIP7Ed3Bz3dI"
    CSV_PATH = "realestatedata.csv"
    
    try:
        # First, validate the CSV file exists
        if not os.path.exists(CSV_PATH):
            logger.error(f"CSV file not found: {CSV_PATH}")
            print(f"Error: CSV file not found at {CSV_PATH}")
            return
        
        logger.info(f"Starting Real Estate Agent with data from {CSV_PATH}")
        agent = RealEstateAgent(GEMINI_API_KEY, CSV_PATH)
        
        # Example queries to test
        queries = [
            # "I want a 3-bedroom house in Dubai having cost less than 3000000",
            # "Looking for a cheap apartment near downtown",
            # "Waterfront properties with ocean view in dubai"
            "Give me 2 ready to move-in properties"
        ]
        
        # Process each query
        for query in queries:
            logger.info(f"Processing query: {query}")
            result = agent.process_real_estate_query(query)
            print(result)
            print("=" * 80)
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()