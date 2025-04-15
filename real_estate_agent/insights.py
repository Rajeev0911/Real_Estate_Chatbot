import re
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_property_insights(properties: pd.DataFrame) -> list:
    """Format property details for presentation.
    
    Extracts and formats key fields (e.g., price, bedrooms, area, location) and returns
    a list of dictionaries with details.
    """
    insights = []
    for _, prop in properties.iterrows():
        try:
            price = float(prop['Price']) if pd.notna(prop.get('Price')) else np.nan
            price_formatted = f"AED {price:,.2f}" if pd.notna(price) else "N/A"
            
            # Extract number of bedrooms from string
            bedrooms = prop.get('Bedrooms', 'N/A')
            if isinstance(bedrooms, str):
                match = re.search(r'(\d+)', bedrooms)
                bedrooms = int(match.group(1)) if match else "N/A"
            elif pd.isna(bedrooms):
                bedrooms = "N/A"
                
            # Format area (try column "Area(Sqft)" first, fallback to "Area")
            area = prop.get('Area(Sqft)', prop.get('Area', 'N/A'))
            if isinstance(area, str):
                area_clean = re.sub(r'[^\d.]', '', area)
                area_formatted = f"{float(area_clean):,.2f} sqft" if area_clean else "N/A"
            elif pd.notna(area) and isinstance(area, (int, float)):
                area_formatted = f"{area:,.2f} sqft"
            else:
                area_formatted = "N/A"
            
            # Format location (concatenate location, city, country if available)
            location = prop.get('Location', 'N/A')
            city = prop.get('city', '')
            country = prop.get('country', '')
            full_location = location
            if city and pd.notna(city):
                full_location += f", {city}"
            if country and pd.notna(country):
                full_location += f", {country}"
            
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

def format_response_with_typing_effect(properties: pd.DataFrame, user_query: str, query_details: dict) -> str:
    """Format a conversational response string with property insights.
    
    Uses a brief introduction, then details for up to 3 properties, and a follow-up question.
    """
    if properties.empty:
        return ("I couldn't find any properties exactly matching your criteria. "
                "Would you like to try a broader search?")
    
    insights = get_property_insights(properties)
    
    purpose_str = f" for {query_details.get('purpose', 'purchase or rent')}" if query_details.get('purpose') else ""
    location_str = f" in {query_details.get('location')}" if query_details.get('location') else ""
    bedrooms_str = f" with {query_details.get('bedrooms')} bedrooms" if query_details.get('bedrooms') is not None else ""
    
    # Handle price range clarity
    price_str = ""
    if query_details.get('min_price') and query_details.get('max_price'):
        price_str = f" between AED {query_details.get('min_price'):,.2f} and AED {query_details.get('max_price'):,.2f}"
    elif query_details.get('min_price'):
        price_str = f" above AED {query_details.get('min_price'):,.2f}"
    elif query_details.get('max_price'):
        price_str = f" under AED {query_details.get('max_price'):,.2f}"
    
    intro = (f"📋 I've found {len(insights)} properties{purpose_str}{location_str}{bedrooms_str}{price_str} "
             "that match your criteria.\n\nHere are the details:")
    
    property_responses = []
    for i, insight in enumerate(insights[:3]):  # Limit to three properties for brevity
        response = f"\n\n🏡 **Property {i+1}: {insight['title']}**\n"
        response += f"💰 **Price**: {insight['price']} ({insight['purpose']})\n"
        response += f"🏢 **Type**: {insight['type']}\n"
        response += f"🛏️ **Bedrooms**: {insight['bedrooms']}\n"
        response += f"📐 **Area**: {insight['area']}\n"
        response += f"📍 **Location**: {insight['location']}\n"
        response += f"🏠 **Furnishing**: {insight['furnishing']}\n"
        desc = insight['description']
        if isinstance(desc, str) and len(desc) > 150:
            desc = desc[:147] + "..."
        response += f"\n*{desc}*"
        property_responses.append(response)
    
    more_results = ""
    if len(insights) > 3:
        more_results = (f"\n\nI have {len(insights) - 3} more properties that match your criteria. "
                        "Would you like to see more options?")
    
    follow_up = ("\n\nIs there anything specific about these properties you'd like to know more about, "
                 "or would you like to refine your search?")
    
    return intro + "".join(property_responses) + more_results + follow_up

def get_dataset_overview(stats: dict) -> str:
    """Generate an overview of the dataset based on provided statistics."""
    try:
        overview = "📊 **Real Estate Dataset Overview:**\n\n"
        if 'purposes' in stats:
            overview += "**Properties by Purpose:**\n"
            for purpose, count in stats['purposes'].items():
                overview += f"- {purpose}: {count} properties\n"
            overview += "\n"
        if 'types' in stats:
            overview += "**Property Types Available:**\n"
            for prop_type, count in stats['types'].items():
                overview += f"- {prop_type}: {count} properties\n"
            overview += "\n"
        if 'price_min' in stats and 'price_max' in stats:
            overview += "**Price Ranges:**\n"
            overview += f"- Overall: AED {stats['price_min']:,.2f} to AED {stats['price_max']:,.2f}\n"
            overview += "\n"
        if 'top_locations' in stats:
            overview += "**Top Locations:**\n"
            for location, count in list(stats['top_locations'].items())[:5]:
                overview += f"- {location}: {count} properties\n"
            overview += "\n"
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
        return ("I have information on various properties for rent and sale. "
                "What type of property are you looking for?")
