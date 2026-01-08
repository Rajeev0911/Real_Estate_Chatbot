"""
Simple sequential message processor without AI/Gemini API.
Asks questions one by one, stores answers, filters data, returns results.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def process_user_message_with_typing(agent, message: str) -> dict:
    """Process user message with simple sequential flow - NO GEMINI API."""
    agent.current_query = message
    agent.conversation_history.append({"role": "user", "content": message})
    
    # Handle greeting
    if agent._is_greeting(message) and agent.conversation_state == "initial":
        greeting = "Hi! 👋 Let me help you find your perfect property.\n\nWhat do you want to do? (Buy/Rent)"
        agent.conversation_history.append({"role": "assistant", "content": greeting})
        agent.conversation_state = "asking_purpose"
        return {"query_details": None, "response": greeting}
    
    # Sequential question flow
    if agent.conversation_state == "asking_purpose":
        # Save purpose
        purpose = None
        if "buy" in message.lower() or "sale" in message.lower():
            purpose = "Sale"
        elif "rent" in message.lower():
            purpose = "Rent"
        
        if purpose:
            agent.user_requirements['transaction_type'] = purpose
            agent.conversation_state = "asking_type"
            response = "What type of property are you looking for?\n(Villa, Apartment, Townhouse, etc.)"
            agent.conversation_history.append({"role": "assistant", "content": response})
            return {"query_details": None, "response": response}
        else:
            response = "Please specify if you want to Buy or Rent."
            agent.conversation_history.append({"role": "assistant", "content": response})
            return {"query_details": None, "response": response}
    
    elif agent.conversation_state == "asking_type":
        # Save property type
        agent.user_requirements['property_type'] = message
        agent.conversation_state = "asking_furnishing"
        response = "Would you prefer it Furnished or Unfurnished?"
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    elif agent.conversation_state == "asking_furnishing":
        # Save furnishing
        agent.user_requirements['furnishing'] = message
        agent.conversation_state = "asking_bedrooms"
        response = "How many Bedrooms do you need?"
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    elif agent.conversation_state == "asking_bedrooms":
        # Save bedrooms
        try:
            bedrooms = int(message.strip())
            agent.user_requirements['bedrooms'] = bedrooms
            agent.conversation_state = "asking_bathrooms"
            response = "How many Bathrooms do you prefer?"
            agent.conversation_history.append({"role": "assistant", "content": response})
            return {"query_details": None, "response": response}
        except:
            response = "Please enter a valid number for bedrooms."
            agent.conversation_history.append({"role": "assistant", "content": response})
            return {"query_details": None, "response": response}
    
    elif agent.conversation_state == "asking_bathrooms":
        # Save bathrooms
        try:
            bathrooms = int(message.strip())
            agent.user_requirements['bathrooms'] = bathrooms
            agent.conversation_state = "asking_location"
            response = "Which location or area would you prefer?"
            agent.conversation_history.append({"role": "assistant", "content": response})
            return {"query_details": None, "response": response}
        except:
            response = "Please enter a valid number for bathrooms."
            agent.conversation_history.append({"role": "assistant", "content": response})
            return {"query_details": None, "response": response}
    
    elif agent.conversation_state == "asking_location":
        # Save location
        agent.user_requirements['location'] = message
        agent.conversation_state = "asking_budget"
        response = "What is your Budget (AED)?"
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    elif agent.conversation_state == "asking_budget":
        # Save budget and SEARCH
        try:
            budget = float(message.replace(',', '').replace('AED', '').strip())
            agent.user_requirements['budget'] = budget
            agent.user_requirements['max_price'] = budget
            agent.conversation_state = "searching"
            
            # Now perform the search
            response = search_matching_properties(agent)
            agent.conversation_history.append({"role": "assistant", "content": response})
            
            # Reset for next query
            agent.conversation_state = "initial"
            return {"query_details": None, "response": response}
        except:
            response = "Please enter a valid budget amount."
            agent.conversation_history.append({"role": "assistant", "content": response})
            return {"query_details": None, "response": response}
    
    # Default fallback
    response = "I didn't understand that. Let's start over. What do you want to do? (Buy/Rent)"
    agent.conversation_history.append({"role": "assistant", "content": response})
    agent.conversation_state = "asking_purpose"
    return {"query_details": None, "response": response}


def search_matching_properties(agent):
    """Search and filter properties based on stored requirements."""
    try:
        # Build query details from user_requirements
        query_details = {
            'purpose': agent.user_requirements.get('transaction_type'),
            'property_type': agent.user_requirements.get('property_type'),
            'furnishing': agent.user_requirements.get('furnishing'),
            'bedrooms': agent.user_requirements.get('bedrooms'),
            'bathrooms': agent.user_requirements.get('bathrooms'),
            'location': agent.user_requirements.get('location'),
            'max_price': agent.user_requirements.get('max_price')
        }
        
        logger.info(f"Searching with filters: {query_details}")
        
        # Filter properties
        filtered_df = agent._filter_properties(query_details)
        
        if filtered_df.empty:
            return "❌ I couldn't find any properties exactly matching your criteria.\n\nWould you like to try with different requirements?"
        
        # Format results
        results_count = len(filtered_df)
        requirements_summary = format_requirements_summary(agent.user_requirements)
        
        response = f"✅ **I found {results_count} propert{'y' if results_count == 1 else 'ies'} matching your criteria!**\n\n"
        response += f"📋 **Your Requirements:**\n{requirements_summary}\n\n"
        response += "🏠 **Property Listings:**\n\n"
        
        # Show top 5 properties
        for idx, (_, prop) in enumerate(filtered_df.head(5).iterrows(), 1):
            response += f"\n**Property {idx}:**\n"
            response += f"📍 **Location:** {prop.get('Location', 'N/A')}\n"
            response += f"🏢 **Type:** {prop.get('Type', 'N/A')}\n"
            response += f"💰 **Price:** AED {int(prop.get('Price', 0)):,}\n"
            response += f"🛏️ **Bedrooms:** {int(prop.get('Bedrooms', 0))}\n"
            response += f"🛁 **Bathrooms:** {int(prop.get('Bathrooms', 0))}\n"
            response += f"📐 **Area:** {prop.get('Area(Sqft)', 'N/A')}\n"
            response += f"🪑 **Furnishing:** {prop.get('Furnishing', 'N/A')}\n"
            response += "---\n"
        
        if results_count > 5:
            response += f"\n💡 *Showing 5 out of {results_count} properties. Would you like to refine your search?*"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return "I encountered an error while searching. Let's try refining your requirements."


def format_requirements_summary(requirements):
    """Format user requirements into a readable summary."""
    summary = ""
    
    if requirements.get('transaction_type'):
        summary += f"- Purpose: {requirements['transaction_type']}\n"
    
    if requirements.get('property_type'):
        summary += f"- Type: {requirements['property_type']}\n"
    
    if requirements.get('furnishing'):
        summary += f"- Furnishing: {requirements['furnishing']}\n"
    
    if requirements.get('bedrooms'):
        summary += f"- Bedrooms: {requirements['bedrooms']}\n"
    
    if requirements.get('bathrooms'):
        summary += f"- Bathrooms: {requirements['bathrooms']}\n"
    
    if requirements.get('location'):
        summary += f"- Location: {requirements['location']}\n"
    
    if requirements.get('budget') or requirements.get('max_price'):
        price = requirements.get('budget') or requirements.get('max_price')
        summary += f"- Budget: AED {price:,}\n"
    
    return summary.strip()



def respond_with_typing(agent, message: str) -> dict:
    """Wrapper function for compatibility with app.py."""
    result = process_user_message_with_typing(agent, message)
    response_text = result.get('response', '')
    
    # Return format expected by app.py
    return {
        'acknowledgment': '...',  # Quick acknowledgment
        'full_response': response_text
    }



def demo_typing_response(text: str, typing_speed: float = 0.05) -> str:
    """Demo function for typing effect."""
    return text
