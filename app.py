# import requests
# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class RealEstateBot:
#     def __init__(self, api_key, csv_path):
#         """
#         Initialize the Real Estate Bot
        
#         :param api_key: Google Gemini API key
#         :param csv_path: Path to the CSV knowledge base
#         """
#         self.api_key = api_key
#         self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
        
#         # Load knowledge base
#         try:
#             self.knowledge_base = pd.read_csv(csv_path)
            
#             # Preprocess columns for better matching
#             self.knowledge_base['Processed_Location'] = self.knowledge_base['Location'].fillna('').str.lower()
#             self.knowledge_base['Processed_City'] = self.knowledge_base['city'].fillna('').str.lower()
#             self.knowledge_base['Processed_Country'] = self.knowledge_base['country'].fillna('').str.lower()
#         except Exception as e:
#             logger.error(f"Error loading CSV: {e}")
#             raise
        
#         # Prepare text for search
#         self.prepare_search_index()
    
#     def prepare_search_index(self):
#         """
#         Prepare TF-IDF vectorizer for semantic search
#         """
#         # Combine relevant columns for searching
#         self.search_texts = self.knowledge_base.apply(
#             lambda row: f"{row['Processed_Location']} {row['Processed_City']} {row['Processed_Country']} {row['Bedrooms']} {row['Type']} {row['Price']}", 
#             axis=1
#         )
        
#         # Create TF-IDF vectorizer
#         self.vectorizer = TfidfVectorizer(stop_words='english')
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
    
#     def extract_query_details(self, user_query):
#         """
#         Extract key details from user query
        
#         :param user_query: Original user query
#         :return: Dictionary of extracted query details
#         """
#         # Convert query to lowercase for easier parsing
#         query_lower = user_query.lower()
        
#         # Define patterns for extraction
#         location_patterns = [
#             r'in\s+([a-zA-Z\s]+)',  # "in San Francisco"
#             r'([a-zA-Z\s]+)\s+property',  # "San Francisco property"
#             r'([a-zA-Z\s]+)\s+house',  # "San Francisco house"
#             r'([a-zA-Z\s]+)\s+apartment'  # "San Francisco apartment"
#         ]
        
#         # Extract location
#         location = None
#         for pattern in location_patterns:
#             match = re.search(pattern, query_lower)
#             if match:
#                 location = match.group(1).strip()
#                 break
        
#         # Extract bedrooms
#         bedroom_patterns = [
#             r'(\d+)\s*bedroom',
#             r'(\d+)\s*bed'
#         ]
        
#         bedrooms = None
#         for pattern in bedroom_patterns:
#             match = re.search(pattern, query_lower)
#             if match:
#                 bedrooms = match.group(1)
#                 break
        
#         # Extract property type
#         property_types = {
#             'house': ['house', 'villa'],
#             'apartment': ['apartment', 'flat'],
#             'studio': ['studio']
#         }
        
#         property_type = None
#         for type_key, type_variations in property_types.items():
#             if any(variation in query_lower for variation in type_variations):
#                 property_type = type_key
#                 break
        
#         return {
#             'original_query': user_query,
#             'location': location,
#             'bedrooms': bedrooms,
#             'property_type': property_type
#         }
    
#     def filter_properties(self, query_details):
#         """
#         Filter properties based on extracted query details
        
#         :param query_details: Dictionary of query details
#         :return: Filtered DataFrame
#         """
#         filtered_df = self.knowledge_base.copy()
        
#         # Filter by location (if specified)
#         if query_details['location']:
#             location_lower = query_details['location'].lower()
#             filtered_df = filtered_df[
#                 (filtered_df['Processed_Location'].str.contains(location_lower, case=False, na=False)) |
#                 (filtered_df['Processed_City'].str.contains(location_lower, case=False, na=False)) |
#                 (filtered_df['Processed_Country'].str.contains(location_lower, case=False, na=False))
#             ]
        
#         # Filter by bedrooms (if specified)
#         if query_details['bedrooms']:
#             try:
#                 bedrooms = int(query_details['bedrooms'])
#                 filtered_df = filtered_df[filtered_df['Bedrooms'] == bedrooms]
#             except ValueError:
#                 pass
        
#         # Filter by property type (if specified)
#         if query_details['property_type']:
#             type_mapping = {
#                 'house': ['house', 'villa'],
#                 'apartment': ['apartment', 'flat'],
#                 'studio': ['studio']
#             }
            
#             type_variants = type_mapping.get(query_details['property_type'], [])
#             if type_variants:
#                 filtered_df = filtered_df[
#                     filtered_df['Type'].str.lower().isin(type_variants)
#                 ]
        
#         return filtered_df
    
#     def semantic_search(self, filtered_df, query, top_k=3):
#         """
#         Perform semantic search on filtered properties
        
#         :param filtered_df: Filtered DataFrame
#         :param query: Original query
#         :param top_k: Number of top results to return
#         :return: Top matching property results
#         """
#         # If filtered DataFrame is empty, return empty
#         if filtered_df.empty:
#             return filtered_df
        
#         # Prepare search texts for filtered properties
#         search_texts = filtered_df.apply(
#             lambda row: f"{row['Location']} {row['Bedrooms']} {row['Price']} {row['Type']}", 
#             axis=1
#         )
        
#         # Vectorize the query
#         vectorizer = TfidfVectorizer(stop_words='english')
#         corpus = list(search_texts) + [query]
#         tfidf_matrix = vectorizer.fit_transform(corpus)
        
#         # Compute cosine similarity
#         similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
#         # Get top K indices
#         top_indices = similarities.argsort()[-top_k:][::-1]
        
#         # Return top matching properties
#         return filtered_df.iloc[top_indices]
    
#     def process_query(self, user_query):
#         """
#         Main method to process user query
        
#         :param user_query: Original user query
#         :return: Matching property results
#         """
#         # Step 1: Extract query details
#         query_details = self.extract_query_details(user_query)
#         logger.info(f"Extracted Query Details: {query_details}")
        
#         # Step 2: Filter properties
#         filtered_properties = self.filter_properties(query_details)
        
#         # Step 3: Semantic search on filtered properties
#         matching_properties = self.semantic_search(filtered_properties, user_query)
        
#         return matching_properties

# def main():
#     # Replace with your actual API key and CSV path
#     API_KEY = "AIzaSyAQJQK4fROJzHS0e1n_hrzDIP7Ed3Bz3dI"
#     CSV_PATH = "real_estate_properties.csv"
    
#     try:
#         # Initialize bot
#         bot = RealEstateBot(API_KEY, CSV_PATH)
        
#         # Example queries
#         queries = [
#             "I want a 3-bedroom house in Dubai having cost less than 3000000",
#             "Looking for a cheap apartment near downtown",
#             "Waterfront properties with ocean view in dubai"
#         ]
        
#         for query in queries:
#             print(f"\nQuery: {query}")
#             results = bot.process_query(query)
            
#             if results.empty:
#                 print("No matching properties found.")
#             else:
#                 print("Matching Properties:")
#                 # Select and display relevant columns
#                 display_columns = [
#                     'Serial No.', 'Title', 'Location', 'city', 
#                     'country', 'Bedrooms', 'Type', 'Price'
#                 ]
#                 print(results[display_columns])
    
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()








# import requests
# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# import re
# import json

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class AdvancedRealEstateAgent:
#     def __init__(self, gemini_api_key, csv_path):
#         """
#         Initialize the Advanced Real Estate Agent
        
#         :param gemini_api_key: Google Gemini API key for natural language processing
#         :param csv_path: Path to the CSV knowledge base
#         """
#         self.gemini_api_key = gemini_api_key
#         self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        
#         # Load and preprocess knowledge base
#         try:
#             self.knowledge_base = pd.read_csv(csv_path)
#             self._preprocess_knowledge_base()
#         except Exception as e:
#             logger.error(f"Error loading knowledge base: {e}")
#             raise
        
#         # Advanced search and analysis initialization
#         self._initialize_search_capabilities()
    
#     def _preprocess_knowledge_base(self):
#         """
#         Preprocess the knowledge base for efficient searching and filtering
#         """
#         # Convert numeric columns
#         numeric_columns = ['Price', 'Bedrooms', 'Area']
#         for col in numeric_columns:
#             self.knowledge_base[col] = pd.to_numeric(self.knowledge_base[col], errors='coerce')
        
#         # Normalize text columns
#         text_columns = ['Location', 'city', 'country', 'Type']
#         for col in text_columns:
#             self.knowledge_base[f'Processed_{col}'] = self.knowledge_base[col].fillna('').str.lower()
    
#     def _initialize_search_capabilities(self):
#         """
#         Initialize advanced search capabilities
#         """
#         # Prepare text for semantic search
#         self.search_texts = self.knowledge_base.apply(
#             lambda row: f"{row['Processed_Location']} {row['Processed_city']} {row['Processed_country']} {row['Bedrooms']} {row['Type']} {row['Price']}", 
#             axis=1
#         )
        
#         # Create TF-IDF vectorizer
#         self.vectorizer = TfidfVectorizer(stop_words='english')
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
    
#     def _call_gemini_api(self, prompt):
#         """
#         Call Google Gemini API for advanced query understanding
        
#         :param prompt: Input prompt for query understanding
#         :return: Parsed query details
#         """
#         try:
#             payload = {
#                 "contents": [{"parts": [{"text": prompt}]}],
#                 "generationConfig": {
#                     "temperature": 0.3,
#                     "maxOutputTokens": 1024
#                 }
#             }
            
#             headers = {
#                 'Content-Type': 'application/json'
#             }
            
#             response = requests.post(self.gemini_url, headers=headers, data=json.dumps(payload))
#             response_json = response.json()
            
#             # Extract text from response
#             generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
            
#             # Parse JSON from generated text
#             query_details = json.loads(generated_text)
#             return query_details
        
#         except Exception as e:
#             logger.error(f"Gemini API call failed: {e}")
#             return None
    
#     def _create_gemini_query_prompt(self, user_query):
#         """
#         Create a prompt for Gemini to understand query details
        
#         :param user_query: Original user query
#         :return: Structured prompt for query understanding
#         """
#         prompt = f"""
#         Analyze the following real estate query and extract structured details:
        
#         Query: "{user_query}"
        
#         Please provide a JSON response with the following keys:
#         - location: Extracted city or region (string or null)
#         - bedrooms: Number of bedrooms (integer or null)
#         - property_type: Type of property (string or null, options: 'house', 'apartment', 'villa', 'studio')
#         - max_price: Maximum budget (float or null)
#         - min_price: Minimum budget (float or null)
#         - amenities: List of desired amenities (list of strings or empty list)
        
#         Provide null or empty values if not specified in the query.
        
#         Example output format:
#         {{
#             "location": "Dubai",
#             "bedrooms": 3,
#             "property_type": "house",
#             "max_price": 3000000.0,
#             "min_price": null,
#             "amenities": ["parking", "gym"]
#         }}
#         """
#         return prompt
    
#     def _semantic_search(self, filtered_df, user_query, top_k=5):
#         """
#         Advanced semantic search with cosine similarity
        
#         :param filtered_df: Filtered DataFrame
#         :param user_query: Original query
#         :param top_k: Number of top results
#         :return: Top matching properties
#         """
#         if filtered_df.empty:
#             return filtered_df
        
#         # Prepare search texts
#         search_texts = filtered_df.apply(
#             lambda row: f"{row['Location']} {row['Bedrooms']} {row['Type']} {row['Price']}", 
#             axis=1
#         )
        
#         # Vectorize corpus including query
#         corpus = list(search_texts) + [user_query]
#         tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(corpus)
        
#         # Compute similarities
#         similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
#         # Get top indices
#         top_indices = similarities.argsort()[-top_k:][::-1]
        
#         return filtered_df.iloc[top_indices]
    
#     def _filter_properties(self, query_details):
#         """
#         Filter properties based on query details
        
#         :param query_details: Parsed query details
#         :return: Filtered DataFrame
#         """
#         filtered_df = self.knowledge_base.copy()
        
#         # Location filtering
#         if query_details.get('location'):
#             location_lower = query_details['location'].lower()
#             filtered_df = filtered_df[
#                 (filtered_df['Processed_Location'].str.contains(location_lower, case=False, na=False)) |
#                 (filtered_df['Processed_city'].str.contains(location_lower, case=False, na=False)) |
#                 (filtered_df['Processed_country'].str.contains(location_lower, case=False, na=False))
#             ]
        
#         # Bedrooms filtering
#         if query_details.get('bedrooms'):
#             filtered_df = filtered_df[filtered_df['Bedrooms'] == query_details['bedrooms']]
        
#         # Property type filtering
#         if query_details.get('property_type'):
#             type_variants = {
#                 'house': ['house', 'villa'],
#                 'apartment': ['apartment', 'flat'],
#                 'studio': ['studio']
#             }
#             type_list = type_variants.get(query_details['property_type'], [])
#             filtered_df = filtered_df[filtered_df['Type'].str.lower().isin(type_list)]
        
#         # Price filtering
#         if query_details.get('max_price'):
#             filtered_df = filtered_df[filtered_df['Price'] <= query_details['max_price']]
        
#         if query_details.get('min_price'):
#             filtered_df = filtered_df[filtered_df['Price'] >= query_details['min_price']]
        
#         # Amenities filtering (if implemented)
#         if query_details.get('amenities'):
#             # Future enhancement for amenities filtering
#             pass
        
#         return filtered_df
    
#     def get_property_insights(self, properties):
#         """
#         Generate comprehensive insights for matched properties
        
#         :param properties: DataFrame of matched properties
#         :return: Detailed property insights
#         """
#         insights = []
#         for _, prop in properties.iterrows():
#             insight = {
#                 'title': prop['Title'],
#                 'location_summary': f"{prop['Location']}, {prop['city']}, {prop['country']}",
#                 'price_details': {
#                     'total_price': f"${prop['Price']:,.2f}",
#                     'price_per_sqft': f"${prop['Price'] / prop['Area']:,.2f}" if prop['Area'] > 0 else "N/A"
#                 },
#                 'property_specs': {
#                     'bedrooms': prop['Bedrooms'],
#                     'type': prop['Type'],
#                     'area': f"{prop['Area']} sq ft"
#                 },
#                 'additional_features': prop.get('Features', 'No additional features listed')
#             }
#             insights.append(insight)
        
#         return insights
    
#     def process_real_estate_query(self, user_query):
#         """
#         Comprehensive query processing method
        
#         :param user_query: Original user query
#         :return: Comprehensive query results
#         """
#         # Step 1: Use Gemini to understand query
#         gemini_prompt = self._create_gemini_query_prompt(user_query)
#         query_details = self._call_gemini_api(gemini_prompt)
        
#         if not query_details:
#             return {
#                 'error': 'Unable to process query. Please rephrase.',
#                 'raw_query': user_query
#             }
        
#         logger.info(f"Extracted Query Details: {query_details}")
        
#         # Step 2: Filter properties
#         filtered_properties = self._filter_properties(query_details)
        
#         # Step 3: Semantic search
#         matching_properties = self._semantic_search(filtered_properties, user_query)
        
#         # Step 4: Generate insights
#         if matching_properties.empty:
#             return {
#                 'status': 'No matching properties found',
#                 'query_details': query_details,
#                 'suggestions': 'Try broadening your search criteria'
#             }
        
#         property_insights = self.get_property_insights(matching_properties)
        
#         return {
#             'status': 'Success',
#             'query_details': query_details,
#             'property_count': len(matching_properties),
#             'properties': property_insights
#         }

# def main():
#     # Replace with your actual API key and CSV path
#     GEMINI_API_KEY = "AIzaSyAQJQK4fROJzHS0e1n_hrzDIP7Ed3Bz3dI"
#     CSV_PATH = "real_estate_properties.csv"
    
#     try:
#         # Initialize advanced real estate agent
#         agent = AdvancedRealEstateAgent(GEMINI_API_KEY, CSV_PATH)
        
#         # Example queries
#         queries = [
#             "I want a 3-bedroom house in Dubai having cost less than 3000000",
#             "Looking for a cheap apartment near downtown",
#             "Waterfront properties with ocean view in dubai"
#         ]
        
#         for query in queries:
#             print(f"\nQuery: {query}")
#             results = agent.process_real_estate_query(query)
            
#             # Pretty print results
#             print(json.dumps(results, indent=2))
    
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()





# import requests
# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class AdvancedRealEstateAgent:
#     def __init__(self, gemini_api_key, csv_path):
#         """
#         Initialize the Advanced Real Estate Agent
        
#         :param gemini_api_key: Google Gemini API key for natural language processing
#         :param csv_path: Path to the CSV knowledge base
#         """
#         self.gemini_api_key = gemini_api_key
#         self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        
#         # Load and preprocess knowledge base
#         try:
#             self.knowledge_base = pd.read_csv(csv_path)
#             self._preprocess_knowledge_base()
#         except Exception as e:
#             logger.error(f"Error loading knowledge base: {e}")
#             raise
        
#         # Advanced search and analysis initialization
#         self._initialize_search_capabilities()
    
#     def _preprocess_knowledge_base(self):
#         """
#         Preprocess the knowledge base for efficient searching and filtering
#         """
#         # Convert numeric columns (adjust column names as needed)
#         numeric_columns = ['Price', 'Bedrooms', 'Area']
#         for col in numeric_columns:
#             if col in self.knowledge_base.columns:
#                 self.knowledge_base[col] = pd.to_numeric(self.knowledge_base[col], errors='coerce')
        
#         # Normalize text columns (adjust column names as needed)
#         text_columns = ['Location', 'city', 'country', 'Type']
#         for col in text_columns:
#             if col in self.knowledge_base.columns:
#                 self.knowledge_base[f'Processed_{col}'] = self.knowledge_base[col].fillna('').str.lower()
    
#     def _initialize_search_capabilities(self):
#         """
#         Initialize advanced search capabilities using TF-IDF
#         """
#         # Prepare text for semantic search using a combination of fields
#         self.search_texts = self.knowledge_base.apply(
#             lambda row: f"{row.get('Processed_Location', '')} {row.get('Processed_city', '')} {row.get('Processed_country', '')} {row.get('Bedrooms', '')} {row.get('Type', '')} {row.get('Price', '')}", 
#             axis=1
#         )
        
#         # Create TF-IDF vectorizer
#         self.vectorizer = TfidfVectorizer(stop_words='english')
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
    
#     def _call_gemini_api(self, prompt):
#         """
#         Call Google Gemini API for advanced query understanding
        
#         :param prompt: Input prompt for query understanding
#         :return: Parsed query details (dictionary) or None if fails
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
            
#             # Log raw response for debugging
#             logger.info(f"Gemini raw response: {response.text}")
            
#             if response.status_code != 200:
#                 logger.error(f"Gemini API error {response.status_code}: {response.text}")
#                 return None
            
#             response_json = response.json()
            
#             # Attempt to extract the generated text from common response structures
#             candidate = response_json.get("candidates", [{}])[0]
#             generated_text = candidate.get("output")
#             if not generated_text:
#                 generated_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            
#             logger.info(f"Gemini generated text: {generated_text}")
            
#             # Remove markdown code fences if present
#             if generated_text.startswith("```"):
#                 generated_text = re.sub(r'^```(?:json)?\s*', '', generated_text)
#                 generated_text = re.sub(r'\s*```$', '', generated_text)
            
#             # Parse the cleaned generated text as JSON
#             query_details = json.loads(generated_text)
#             return query_details
        
#         except Exception as e:
#             logger.error(f"Gemini API call failed: {e}")
#             return None
    
#     def _create_gemini_query_prompt(self, user_query):
#         """
#         Create a prompt for Gemini to understand query details
        
#         :param user_query: Original user query
#         :return: Structured prompt for query understanding
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

# Example output format:
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
    
#     def _semantic_search(self, filtered_df, user_query, top_k=5):
#         """
#         Advanced semantic search with cosine similarity
        
#         :param filtered_df: Filtered DataFrame
#         :param user_query: Original query
#         :param top_k: Number of top results
#         :return: Top matching properties DataFrame
#         """
#         if filtered_df.empty:
#             return filtered_df
        
#         # Prepare search texts from filtered DataFrame
#         search_texts = filtered_df.apply(
#             lambda row: f"{row.get('Location', '')} {row.get('Bedrooms', '')} {row.get('Type', '')} {row.get('Price', '')}", 
#             axis=1
#         )
        
#         # Create a new TF-IDF matrix including the query
#         corpus = list(search_texts) + [user_query]
#         tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(corpus)
        
#         # Compute cosine similarities between the query and all documents
#         similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
#         # Get indices of top k matches
#         top_indices = similarities.argsort()[-top_k:][::-1]
#         return filtered_df.iloc[top_indices]
    
#     def _filter_properties(self, query_details):
#         """
#         Filter properties based on query details
        
#         :param query_details: Parsed query details dictionary
#         :return: Filtered DataFrame
#         """
#         filtered_df = self.knowledge_base.copy()
        
#         # Filter by location if specified
#         if query_details.get('location'):
#             location_lower = query_details['location'].lower()
#             filtered_df = filtered_df[
#                 (filtered_df['Processed_Location'].str.contains(location_lower, case=False, na=False)) |
#                 (filtered_df['Processed_city'].str.contains(location_lower, case=False, na=False)) |
#                 (filtered_df['Processed_country'].str.contains(location_lower, case=False, na=False))
#             ]
        
#         # Filter by bedrooms if specified
#         if query_details.get('bedrooms') is not None:
#             filtered_df = filtered_df[filtered_df['Bedrooms'] == query_details['bedrooms']]
        
#         # Filter by property type if specified
#         if query_details.get('property_type'):
#             type_variants = {
#                 'house': ['house', 'villa'],
#                 'apartment': ['apartment', 'flat'],
#                 'studio': ['studio']
#             }
#             type_list = type_variants.get(query_details['property_type'].lower(), [])
#             filtered_df = filtered_df[filtered_df['Type'].str.lower().isin(type_list)]
        
#         # Filter by price if specified
#         if query_details.get('max_price') is not None:
#             filtered_df = filtered_df[filtered_df['Price'] <= query_details['max_price']]
#         if query_details.get('min_price') is not None:
#             filtered_df = filtered_df[filtered_df['Price'] >= query_details['min_price']]
        
#         # Amenities filtering can be added if needed
        
#         return filtered_df
    
#     def get_property_insights(self, properties):
#         """
#         Generate comprehensive insights for matched properties
        
#         :param properties: DataFrame of matched properties
#         :return: List of property insights dictionaries
#         """
#         insights = []
#         for _, prop in properties.iterrows():
#             try:
#                 price = prop['Price']
#                 area = prop['Area'] if prop['Area'] and prop['Area'] > 0 else np.nan
#                 price_per_sqft = f"${price / area:,.2f}" if pd.notna(area) else "N/A"
#             except Exception:
#                 price_per_sqft = "N/A"
#             insight = {
#                 'title': prop.get('Title', 'No Title'),
#                 'location_summary': f"{prop.get('Location', '')}, {prop.get('city', '')}, {prop.get('country', '')}",
#                 'price_details': {
#                     'total_price': f"${price:,.2f}" if pd.notna(price) else "N/A",
#                     'price_per_sqft': price_per_sqft
#                 },
#                 'property_specs': {
#                     'bedrooms': prop.get('Bedrooms', 'N/A'),
#                     'type': prop.get('Type', 'N/A'),
#                     'area': f"{prop.get('Area', 'N/A')} sq ft"
#                 },
#                 'additional_features': prop.get('Description', 'No additional features listed')
#             }
#             insights.append(insight)
#         return insights
    
#     def process_real_estate_query(self, user_query):
#         """
#         Comprehensive query processing method
        
#         :param user_query: Original user query
#         :return: Comprehensive query results as a dictionary
#         """
#         # Step 1: Use Gemini to understand and structure the query
#         gemini_prompt = self._create_gemini_query_prompt(user_query)
#         query_details = self._call_gemini_api(gemini_prompt)
        
#         if not query_details:
#             return {
#                 'error': 'Unable to process query. Please rephrase.',
#                 'raw_query': user_query
#             }
        
#         logger.info(f"Extracted Query Details: {query_details}")
        
#         # Step 2: Filter properties based on query details
#         filtered_properties = self._filter_properties(query_details)
        
#         # Step 3: Use semantic search to rank the filtered results
#         matching_properties = self._semantic_search(filtered_properties, user_query)
        
#         # Step 4: Generate property insights from matching properties
#         if matching_properties.empty:
#             return {
#                 'status': 'No matching properties found',
#                 'query_details': query_details,
#                 'suggestions': 'Try broadening your search criteria'
#             }
        
#         property_insights = self.get_property_insights(matching_properties)
        
#         return {
#             'status': 'Success',
#             'query_details': query_details,
#             'property_count': len(matching_properties),
#             'properties': property_insights
#         }

# def main():
#     # Replace with your actual Gemini API key and CSV path
#     GEMINI_API_KEY = "AIzaSyAQJQK4fROJzHS0e1n_hrzDIP7Ed3Bz3dI"
#     CSV_PATH = "real_estate_properties.csv"
    
#     try:
#         # Initialize the advanced real estate agent
#         agent = AdvancedRealEstateAgent(GEMINI_API_KEY, CSV_PATH)
        
#         # Example queries
#         queries = [
#             "I want a 3-bedroom house in Dubai having cost less than 3000000",
#             "Looking for a cheap apartment near downtown",
#             "Waterfront properties with ocean view in dubai"
#         ]
        
#         for query in queries:
#             print(f"\nQuery: {query}")
#             results = agent.process_real_estate_query(query)
#             print(json.dumps(results, indent=2))
    
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()






# import requests
# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class AdvancedRealEstateAgent:
#     def __init__(self, gemini_api_key, csv_path):
#         """
#         Initialize the Advanced Real Estate Agent
        
#         :param gemini_api_key: Google Gemini API key for natural language processing
#         :param csv_path: Path to the CSV knowledge base
#         """
#         self.gemini_api_key = gemini_api_key
#         self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        
#         # Load and preprocess knowledge base
#         try:
#             self.knowledge_base = pd.read_csv(csv_path)
#             self._preprocess_knowledge_base()
#         except Exception as e:
#             logger.error(f"Error loading knowledge base: {e}")
#             raise
        
#         # Advanced search and analysis initialization
#         self._initialize_search_capabilities()
    
#     def _preprocess_knowledge_base(self):
#         """
#         Preprocess the knowledge base for efficient searching and filtering.
#         Converts numeric columns and normalizes text columns.
#         """
#         # Convert numeric columns (adjust column names as needed)
#         numeric_columns = ['Price', 'Bedrooms', 'Area']
#         for col in numeric_columns:
#             if col in self.knowledge_base.columns:
#                 self.knowledge_base[col] = pd.to_numeric(self.knowledge_base[col], errors='coerce')
        
#         # Normalize text columns (adjust column names as needed)
#         text_columns = ['Location', 'city', 'country', 'Type']
#         for col in text_columns:
#             if col in self.knowledge_base.columns:
#                 self.knowledge_base[f'Processed_{col}'] = self.knowledge_base[col].fillna('').str.lower()
    
#     def _initialize_search_capabilities(self):
#         """
#         Initialize advanced search capabilities using TF-IDF.
#         """
#         # Prepare text for semantic search using a combination of fields
#         self.search_texts = self.knowledge_base.apply(
#             lambda row: f"{row.get('Processed_Location', '')} {row.get('Processed_city', '')} {row.get('Processed_country', '')} {row.get('Bedrooms', '')} {row.get('Type', '')} {row.get('Price', '')}", 
#             axis=1
#         )
        
#         # Create TF-IDF vectorizer
#         self.vectorizer = TfidfVectorizer(stop_words='english')
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
    
#     def _call_gemini_api(self, prompt):
#         """
#         Call Google Gemini API for advanced query understanding.
        
#         :param prompt: Input prompt for query understanding.
#         :return: Parsed query details (dictionary) or None if fails.
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
            
#             # Log raw response for debugging
#             logger.info(f"Gemini raw response: {response.text}")
            
#             if response.status_code != 200:
#                 logger.error(f"Gemini API error {response.status_code}: {response.text}")
#                 return None
            
#             response_json = response.json()
            
#             # Attempt to extract the generated text from common response structures
#             candidate = response_json.get("candidates", [{}])[0]
#             generated_text = candidate.get("output")
#             if not generated_text:
#                 generated_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            
#             logger.info(f"Gemini generated text: {generated_text}")
            
#             # Remove markdown code fences if present
#             if generated_text.startswith("```"):
#                 generated_text = re.sub(r'^```(?:json)?\s*', '', generated_text)
#                 generated_text = re.sub(r'\s*```$', '', generated_text)
            
#             # Parse the cleaned generated text as JSON
#             query_details = json.loads(generated_text)
#             return query_details
        
#         except Exception as e:
#             logger.error(f"Gemini API call failed: {e}")
#             return None
    
#     def _create_gemini_query_prompt(self, user_query):
#         """
#         Create a prompt for Gemini to understand query details.
        
#         :param user_query: Original user query.
#         :return: Structured prompt for query understanding.
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

# Example output format:
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
    
#     def _semantic_search(self, filtered_df, user_query, top_k=5):
#         """
#         Advanced semantic search with cosine similarity.
        
#         :param filtered_df: Filtered DataFrame.
#         :param user_query: Original query.
#         :param top_k: Number of top results.
#         :return: Top matching properties DataFrame.
#         """
#         if filtered_df.empty:
#             return filtered_df
        
#         # Prepare search texts from filtered DataFrame
#         search_texts = filtered_df.apply(
#             lambda row: f"{row.get('Location', '')} {row.get('Bedrooms', '')} {row.get('Type', '')} {row.get('Price', '')}", 
#             axis=1
#         )
        
#         # Create a new TF-IDF matrix including the query
#         corpus = list(search_texts) + [user_query]
#         tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(corpus)
        
#         # Compute cosine similarities between the query and all documents
#         similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
#         # Get indices of top k matches
#         top_indices = similarities.argsort()[-top_k:][::-1]
#         return filtered_df.iloc[top_indices]
    
#     def _filter_properties(self, query_details):
#         """
#         Filter properties based on query details.
        
#         :param query_details: Parsed query details dictionary.
#         :return: Filtered DataFrame.
#         """
#         filtered_df = self.knowledge_base.copy()
        
#         # Filter by location if specified
#         if query_details.get('location'):
#             location_lower = query_details['location'].lower()
#             filtered_df = filtered_df[
#                 (filtered_df['Processed_Location'].str.contains(location_lower, case=False, na=False)) |
#                 (filtered_df['Processed_city'].str.contains(location_lower, case=False, na=False)) |
#                 (filtered_df['Processed_country'].str.contains(location_lower, case=False, na=False))
#             ]
        
#         # Filter by bedrooms if specified
#         if query_details.get('bedrooms') is not None:
#             filtered_df = filtered_df[filtered_df['Bedrooms'] == query_details['bedrooms']]
        
#         # Filter by price if specified
#         if query_details.get('max_price') is not None:
#             filtered_df = filtered_df[filtered_df['Price'] <= query_details['max_price']]
#         if query_details.get('min_price') is not None:
#             filtered_df = filtered_df[filtered_df['Price'] >= query_details['min_price']]
        
#         # Filter by property type if specified; if this filter results in no data, relax it.
#         if query_details.get('property_type'):
#             type_variants = {
#                 'house': ['house', 'villa'],
#                 'apartment': ['apartment', 'flat'],
#                 'studio': ['studio']
#             }
#             requested_type = query_details['property_type'].lower()
#             type_list = type_variants.get(requested_type, [requested_type])
            
#             df_type_filtered = filtered_df[filtered_df['Type'].str.lower().isin(type_list)]
#             if not df_type_filtered.empty:
#                 filtered_df = df_type_filtered
#             else:
#                 logger.warning("No properties matched the specified property type; relaxing this filter.")
        
#         # (Amenities filtering can be added here if needed)
        
#         return filtered_df
    
#     def get_property_insights(self, properties):
#         """
#         Generate comprehensive insights for matched properties.
        
#         :param properties: DataFrame of matched properties.
#         :return: List of property insights dictionaries.
#         """
#         insights = []
#         for _, prop in properties.iterrows():
#             try:
#                 price = prop['Price']
#                 area = prop['Area'] if prop['Area'] and prop['Area'] > 0 else np.nan
#                 price_per_sqft = f"${price / area:,.2f}" if pd.notna(area) else "N/A"
#             except Exception:
#                 price_per_sqft = "N/A"
#             insight = {
#                 'title': prop.get('Title', 'No Title'),
#                 'location_summary': f"{prop.get('Location', '')}, {prop.get('city', '')}, {prop.get('country', '')}",
#                 'price_details': {
#                     'total_price': f"${price:,.2f}" if pd.notna(price) else "N/A",
#                     'price_per_sqft': price_per_sqft
#                 },
#                 'property_specs': {
#                     'bedrooms': prop.get('Bedrooms', 'N/A'),
#                     'type': prop.get('Type', 'N/A'),
#                     'area': f"{prop.get('Area', 'N/A')} sq ft"
#                 },
#                 'additional_features': prop.get('Description', 'No additional features listed')
#             }
#             insights.append(insight)
#         return insights
    
#     def process_real_estate_query(self, user_query):
#         """
#         Comprehensive query processing method.
        
#         :param user_query: Original user query.
#         :return: Comprehensive query results as a dictionary.
#         """
#         # Step 1: Use Gemini to understand and structure the query
#         gemini_prompt = self._create_gemini_query_prompt(user_query)
#         query_details = self._call_gemini_api(gemini_prompt)
        
#         if not query_details:
#             return {
#                 'error': 'Unable to process query. Please rephrase.',
#                 'raw_query': user_query
#             }
        
#         logger.info(f"Extracted Query Details: {query_details}")
        
#         # Step 2: Filter properties based on query details
#         filtered_properties = self._filter_properties(query_details)
        
#         # Step 3: Use semantic search to rank the filtered results
#         matching_properties = self._semantic_search(filtered_properties, user_query)
        
#         # Step 4: Generate property insights from matching properties
#         if matching_properties.empty:
#             return {
#                 'status': 'No matching properties found',
#                 'query_details': query_details,
#                 'suggestions': 'Try broadening your search criteria'
#             }
        
#         property_insights = self.get_property_insights(matching_properties)
        
#         return {
#             'status': 'Success',
#             'query_details': query_details,
#             'property_count': len(matching_properties),
#             'properties': property_insights
#         }

# def main():
#     # Replace with your actual Gemini API key and CSV path
#     GEMINI_API_KEY = "AIzaSyAQJQK4fROJzHS0e1n_hrzDIP7Ed3Bz3dI"
#     CSV_PATH = "real_estate_properties.csv"
    
#     try:
#         # Initialize the advanced real estate agent
#         agent = AdvancedRealEstateAgent(GEMINI_API_KEY, CSV_PATH)
        
#         # Example queries
#         queries = [
#             "I want a 3-bedroom house in Dubai having cost less than 3000000",
#             "Looking for a cheap apartment near downtown",
#             "Waterfront properties with ocean view in dubai"
#         ]
        
#         for query in queries:
#             print(f"\nQuery: {query}")
#             results = agent.process_real_estate_query(query)
#             print(json.dumps(results, indent=2))
    
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()





# import requests
# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

# class AdvancedRealEstateAgent:
#     def __init__(self, gemini_api_key, md_path):
#         """
#         Initialize the Advanced Real Estate Agent.
        
#         :param gemini_api_key: Google Gemini API key for natural language processing.
#         :param md_path: Path to the Markdown knowledge base file.
#         """
#         self.gemini_api_key = gemini_api_key
#         self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        
#         # Load and preprocess the knowledge base from the Markdown file
#         try:
#             self.knowledge_base = load_markdown_table(md_path)
#             self._preprocess_knowledge_base()
#         except Exception as e:
#             logger.error(f"Error loading knowledge base: {e}")
#             raise
        
#         # Initialize semantic search capabilities
#         self._initialize_search_capabilities()
    
#     def _preprocess_knowledge_base(self):
#         """
#         Preprocess the knowledge base:
#           - Convert numeric columns ("Price", "Bedrooms", "Area") to numbers.
#           - Normalize text columns ("Location", "city", "country", "Type", "Title").
#         """
#         numeric_columns = ['Price', 'Bedrooms', 'Area']
#         for col in numeric_columns:
#             if col in self.knowledge_base.columns:
#                 self.knowledge_base[col] = pd.to_numeric(self.knowledge_base[col].str.replace(',', ''), errors='coerce')
        
#         text_columns = ['Location', 'city', 'country', 'Type', 'Title', 'Description']
#         for col in text_columns:
#             if col in self.knowledge_base.columns:
#                 self.knowledge_base[f'Processed_{col}'] = self.knowledge_base[col].fillna('').str.lower()
    
#     def _initialize_search_capabilities(self):
#         """
#         Initialize TF-IDF-based semantic search using selected fields.
#         """
#         self.search_texts = self.knowledge_base.apply(
#             lambda row: f"{row.get('Processed_Location', '')} {row.get('Processed_city', '')} {row.get('Processed_country', '')} {row.get('Bedrooms', '')} {row.get('Processed_Type', '')} {row.get('Price', '')}", 
#             axis=1
#         )
#         self.vectorizer = TfidfVectorizer(stop_words='english')
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
    
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
            
#             logger.info(f"Gemini raw response: {response.text}")
#             if response.status_code != 200:
#                 logger.error(f"Gemini API error {response.status_code}: {response.text}")
#                 return None
            
#             response_json = response.json()
#             candidate = response_json.get("candidates", [{}])[0]
#             generated_text = candidate.get("output")
#             if not generated_text:
#                 generated_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            
#             logger.info(f"Gemini generated text: {generated_text}")
            
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
        
#         search_texts = filtered_df.apply(
#             lambda row: f"{row.get('Location', '')} {row.get('Bedrooms', '')} {row.get('Type', '')} {row.get('Price', '')}", 
#             axis=1
#         )
#         corpus = list(search_texts) + [user_query]
#         tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(corpus)
#         similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
#         top_indices = similarities.argsort()[-top_k:][::-1]
#         return filtered_df.iloc[top_indices]
    
#     def _filter_properties(self, query_details):
#         """
#         Filter the knowledge base DataFrame based on the extracted query details.
        
#         :param query_details: Dictionary of query details.
#         :return: A filtered DataFrame.
#         """
#         filtered_df = self.knowledge_base.copy()
        
#         # Filter by location
#         if query_details.get('location'):
#             loc = query_details['location'].lower()
#             filtered_df = filtered_df[
#                 (filtered_df['Processed_Location'].str.contains(loc, na=False)) |
#                 (filtered_df['Processed_city'].str.contains(loc, na=False)) |
#                 (filtered_df['Processed_country'].str.contains(loc, na=False))
#             ]
        
#         # Filter by bedrooms
#         if query_details.get('bedrooms') is not None:
#             filtered_df = filtered_df[filtered_df['Bedrooms'] == query_details['bedrooms']]
        
#         # Filter by price
#         if query_details.get('max_price') is not None:
#             filtered_df = filtered_df[filtered_df['Price'] <= query_details['max_price']]
#         if query_details.get('min_price') is not None:
#             filtered_df = filtered_df[filtered_df['Price'] >= query_details['min_price']]
        
#         # Filter by property type (if filter results in empty set, relax it)
#         if query_details.get('property_type'):
#             type_variants = {
#                 'house': ['house', 'villa'],
#                 'apartment': ['apartment', 'flat'],
#                 'studio': ['studio']
#             }
#             req_type = query_details['property_type'].lower()
#             type_list = type_variants.get(req_type, [req_type])
#             df_type = filtered_df[filtered_df['Type'].str.lower().isin(type_list)]
#             if not df_type.empty:
#                 filtered_df = df_type
#             else:
#                 logger.warning("No properties matched property type; relaxing filter.")
        
#         return filtered_df
    
#     def get_property_insights(self, properties):
#         """
#         Generate formatted insights for each matched property.
        
#         :param properties: DataFrame of properties.
#         :return: List of property insight dictionaries.
#         """
#         insights = []
#         for _, prop in properties.iterrows():
#             try:
#                 price = prop['Price']
#                 area = prop['Area'] if prop['Area'] and prop['Area'] > 0 else np.nan
#                 price_per_sqft = f"${price / area:,.2f}" if pd.notna(area) else "N/A"
#             except Exception:
#                 price_per_sqft = "N/A"
#             insight = {
#                 'Title': prop.get('Title', 'No Title'),
#                 'Location': prop.get('Location', 'Unknown'),
#                 'Price': f"${price:,.2f}" if pd.notna(price) else "N/A",
#                 'Bedrooms': prop.get('Bedrooms', 'N/A'),
#                 'Type': prop.get('Type', 'N/A'),
#                 'Area(sqft)': prop.get('Area', 'N/A'),
#                 'Price per sqft': price_per_sqft,
#                 'Description': prop.get('Description', 'No description provided')
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
#         prompt = self._create_gemini_query_prompt(user_query)
#         query_details = self._call_gemini_api(prompt)
        
#         if not query_details:
#             return {
#                 'error': 'Unable to process query. Please rephrase.',
#                 'raw_query': user_query
#             }
        
#         logger.info(f"Extracted Query Details: {query_details}")
        
#         filtered = self._filter_properties(query_details)
#         matching = self._semantic_search(filtered, user_query)
        
#         if matching.empty:
#             return {
#                 'status': 'No matching properties found',
#                 'query_details': query_details,
#                 'suggestions': 'Try broadening your search criteria'
#             }
        
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
#         agent = AdvancedRealEstateAgent(GEMINI_API_KEY, MD_PATH)
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

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_markdown_table(md_path):
    """
    Load a markdown table from a .md file and convert it into a pandas DataFrame.
    Assumes the table is defined using pipe-delimited rows.
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Filter out lines that start with a pipe
    table_lines = [line.strip() for line in lines if line.strip().startswith("|")]
    if not table_lines:
        raise ValueError("No markdown table found in the file.")
    
    # The first line is header, second line is separator, then data rows
    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    
    data_rows = []
    for line in table_lines[2:]:
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) == len(headers):
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows, columns=headers)
    return df

class DebugRealEstateAgent:
    def __init__(self, gemini_api_key, md_path):
        """
        Initialize the Debug Real Estate Agent.
        
        :param gemini_api_key: Google Gemini API key for natural language processing.
        :param md_path: Path to the Markdown knowledge base file.
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        
        # Load and preprocess the knowledge base from the Markdown file
        try:
            self.knowledge_base = load_markdown_table(md_path)
            logger.debug(f"Loaded knowledge base columns: {self.knowledge_base.columns}")
            logger.debug(f"Total properties loaded: {len(self.knowledge_base)}")
            self._preprocess_knowledge_base()
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise
        
        # Initialize semantic search capabilities
        self._initialize_search_capabilities()
    
    def _preprocess_knowledge_base(self):
        """
        Preprocess the knowledge base:
          - Convert numeric columns to numbers.
          - Normalize text columns.
        """
        # Logging the raw data before preprocessing
        logger.debug("Raw knowledge base data:")
        logger.debug(self.knowledge_base.head())
        
        numeric_columns = ['Price', 'Bedrooms', 'Area']
        for col in numeric_columns:
            if col in self.knowledge_base.columns:
                self.knowledge_base[col] = pd.to_numeric(
                    self.knowledge_base[col].str.replace(',', '').str.replace('$', ''), 
                    errors='coerce'
                )
        
        text_columns = ['Location', 'city', 'country', 'Type', 'Title', 'Description']
        for col in text_columns:
            if col in self.knowledge_base.columns:
                self.knowledge_base[f'Processed_{col}'] = self.knowledge_base[col].fillna('').str.lower()
        
        # Logging after preprocessing
        logger.debug("Preprocessed knowledge base:")
        logger.debug(self.knowledge_base.head())
    
    def _initialize_search_capabilities(self):
        """
        Initialize TF-IDF-based semantic search using selected fields.
        """
        self.search_texts = self.knowledge_base.apply(
            lambda row: f"{row.get('Processed_Location', '')} {row.get('Processed_city', '')} {row.get('Processed_country', '')} {row.get('Bedrooms', '')} {row.get('Processed_Type', '')} {row.get('Price', '')}", 
            axis=1
        )
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
    
    def _call_gemini_api(self, prompt):
        """
        Call the Google Gemini API to process and structure the user query.
        
        :param prompt: Prompt string.
        :return: Parsed query details as a dictionary, or None on failure.
        """
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024
                }
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.gemini_url, headers=headers, data=json.dumps(payload))
            
            logger.debug(f"Gemini raw response: {response.text}")
            if response.status_code != 200:
                logger.error(f"Gemini API error {response.status_code}: {response.text}")
                return None
            
            response_json = response.json()
            candidate = response_json.get("candidates", [{}])[0]
            generated_text = candidate.get("output")
            if not generated_text:
                generated_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            
            logger.debug(f"Gemini generated text: {generated_text}")
            
            # Remove markdown code fences if present
            if generated_text.startswith("```"):
                generated_text = re.sub(r'^```(?:json)?\s*', '', generated_text)
                generated_text = re.sub(r'\s*```$', '', generated_text)
            
            query_details = json.loads(generated_text)
            return query_details
        
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None
    
    def _create_gemini_query_prompt(self, user_query):
        """
        Create a prompt for Gemini to extract structured query details.
        
        :param user_query: The user's original query.
        :return: A formatted prompt string.
        """
        prompt = f"""
Analyze the following real estate query and extract structured details:

Query: "{user_query}"

Please provide a JSON response with the following keys:
- location: Extracted city or region (string or null)
- bedrooms: Number of bedrooms (integer or null)
- property_type: Type of property (string or null, options: 'house', 'apartment', 'villa', 'studio')
- max_price: Maximum budget (float or null)
- min_price: Minimum budget (float or null)
- amenities: List of desired amenities (list of strings or empty list)

Provide null or empty values if not specified in the query.

Example output:
{{
    "location": "Dubai",
    "bedrooms": 3,
    "property_type": "house",
    "max_price": 3000000.0,
    "min_price": null,
    "amenities": ["parking", "gym"]
}}
"""
        return prompt.strip()
    
    def _filter_properties(self, query_details):
        """
        Filter the knowledge base DataFrame based on the extracted query details.
        
        :param query_details: Dictionary of query details.
        :return: A filtered DataFrame.
        """
        filtered_df = self.knowledge_base.copy()
        
        # Log initial dataframe size
        logger.debug(f"Initial dataframe size: {len(filtered_df)}")
        
        # Detailed logging for each filtering step
        # Location filtering
        if query_details.get('location'):
            loc = query_details['location'].lower()
            logger.debug(f"Filtering by location: {loc}")
            location_mask = (
                filtered_df['Processed_Location'].str.contains(loc, na=False) |
                filtered_df.get('Processed_city', pd.Series()).str.contains(loc, na=False) |
                filtered_df.get('Processed_country', pd.Series()).str.contains(loc, na=False)
            )
            filtered_df = filtered_df[location_mask]
            logger.debug(f"Properties after location filter: {len(filtered_df)}")
        
        # Bedrooms filtering
        if query_details.get('bedrooms') is not None:
            logger.debug(f"Filtering by bedrooms: {query_details['bedrooms']}")
            filtered_df = filtered_df[filtered_df['Bedrooms'] == query_details['bedrooms']]
            logger.debug(f"Properties after bedrooms filter: {len(filtered_df)}")
        
        # Price filtering
        if query_details.get('max_price') is not None:
            logger.debug(f"Filtering by max price: {query_details['max_price']}")
            filtered_df = filtered_df[filtered_df['Price'] <= query_details['max_price']]
            logger.debug(f"Properties after max price filter: {len(filtered_df)}")
        
        if query_details.get('min_price') is not None:
            logger.debug(f"Filtering by min price: {query_details['min_price']}")
            filtered_df = filtered_df[filtered_df['Price'] >= query_details['min_price']]
            logger.debug(f"Properties after min price filter: {len(filtered_df)}")
        
        # Enhanced property type filtering
        if query_details.get('property_type'):
            type_variants = {
                'house': ['house', 'villa', 'townhouse'],
                'apartment': ['apartment', 'flat', 'unit'],
                'studio': ['studio']
            }
            req_type = query_details['property_type'].lower()
            type_list = type_variants.get(req_type, [req_type])
            
            # Log all unique property types in the dataframe
            logger.debug(f"Unique property types in dataframe: {filtered_df['Type'].unique()}")
            logger.debug(f"Searching for property types: {type_list}")
            
            # Case-insensitive type matching
            df_type = filtered_df[filtered_df['Type'].str.lower().isin(type_list)]
            
            logger.debug(f"Properties matching type {type_list}: {len(df_type)}")
            
            if not df_type.empty:
                filtered_df = df_type
            else:
                logger.warning(f"No properties matched property type {req_type}. Available types: {filtered_df['Type'].unique()}")
        
        return filtered_df
    
    def _semantic_search(self, filtered_df, user_query, top_k=5):
        """
        Perform a semantic search using cosine similarity on filtered properties.
        
        :param filtered_df: DataFrame after filtering.
        :param user_query: The original query.
        :param top_k: Number of top results to return.
        :return: A DataFrame of top matching properties.
        """
        if filtered_df.empty:
            return filtered_df
        
        search_texts = filtered_df.apply(
            lambda row: f"{row.get('Location', '')} {row.get('Bedrooms', '')} {row.get('Type', '')} {row.get('Price', '')}", 
            axis=1
        )
        corpus = list(search_texts) + [user_query]
        tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return filtered_df.iloc[top_indices]
    
    def get_property_insights(self, properties):
        """
        Generate formatted insights for each matched property.
        
        :param properties: DataFrame of properties.
        :return: List of property insight dictionaries.
        """
        insights = []
        for _, prop in properties.iterrows():
            try:
                price = prop['Price']
                area = prop['Area'] if prop['Area'] and prop['Area'] > 0 else np.nan
                price_per_sqft = f"${price / area:,.2f}" if pd.notna(area) else "N/A"
            except Exception:
                price_per_sqft = "N/A"
            insight = {
                'Title': prop.get('Title', 'No Title'),
                'Location': prop.get('Location', 'Unknown'),
                'Price': f"${price:,.2f}" if pd.notna(price) else "N/A",
                'Bedrooms': prop.get('Bedrooms', 'N/A'),
                'Type': prop.get('Type', 'N/A'),
                'Area(sqft)': prop.get('Area', 'N/A'),
                'Price per sqft': price_per_sqft,
                'Description': prop.get('Description', 'No description provided')
            }
            insights.append(insight)
        return insights
    
    def process_real_estate_query(self, user_query):
        """
        Process the user query: use Gemini to extract structured details,
        filter properties, perform semantic search, and return insights.
        
        :param user_query: The user's query string.
        :return: Dictionary with query details and matching property insights.
        """
        prompt = self._create_gemini_query_prompt(user_query)
        query_details = self._call_gemini_api(prompt)
        
        if not query_details:
            return {
                'error': 'Unable to process query. Please rephrase.',
                'raw_query': user_query
            }
        
        logger.debug(f"Extracted Query Details: {query_details}")
        
        filtered = self._filter_properties(query_details)
        matching = self._semantic_search(filtered, user_query)
        
        if matching.empty:
            return {
                'status': 'No matching properties found',
                'query_details': query_details,
                'suggestions': 'Try broadening your search criteria'
            }
        
        insights = self.get_property_insights(matching)
        return {
            'status': 'Success',
            'query_details': query_details,
            'property_count': len(matching),
            'properties': insights
        }

def main():
    # Replace with your actual Gemini API key and Markdown file path
    GEMINI_API_KEY = "AIzaSyAQJQK4fROJzHS0e1n_hrzDIP7Ed3Bz3dI"
    MD_PATH = "output.md"
    
    try:
        agent = DebugRealEstateAgent(GEMINI_API_KEY, MD_PATH)
        # Example queries
        queries = [
            "I want a 3-bedroom house in Dubai having cost less than 3000000",
            "Looking for a cheap apartment near downtown",
            "Waterfront properties with ocean view in dubai"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            result = agent.process_real_estate_query(query)
            print(json.dumps(result, indent=2))
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()