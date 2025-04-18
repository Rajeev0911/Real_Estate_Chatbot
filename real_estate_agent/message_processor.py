import sys
import time
import random
import logging
import pandas as pd
import re

logger = logging.getLogger(__name__)

def process_user_message_with_typing(agent, message: str) -> dict:
    """Process a user's message following the conversation flow."""
    agent.current_query = message
    agent.conversation_history.append({"role": "user", "content": message})
    
    # Handle initial greeting
    if agent._is_greeting(message):
        agent.conversation_state = "initial"
        greeting = "Hi! 👋 Let me help you find your perfect property.\n\nWhat do you want to do? (Buy/Rent)"
        agent.conversation_history.append({"role": "assistant", "content": greeting})
        return {"query_details": None, "response": greeting}
    
    # Handle property detail requests with more flexible pattern matching
    property_patterns = [
        r'(?:show|give|tell).*(?:details|more|info).*property\s*(\d+)',
        r'property\s*(\d+)\s*(?:details|info)',
        r'more\s*(?:about|on)\s*property\s*(\d+)'
    ]
    
    for pattern in property_patterns:
        match = re.search(pattern, message.lower())
        if match:
            try:
                property_num = int(match.group(1))
                if hasattr(agent, 'last_shown_properties') and agent.last_shown_properties is not None:
                    detailed_info = get_detailed_property_info(agent, property_num)
                    agent.conversation_history.append({"role": "assistant", "content": detailed_info})
                    return {"query_details": None, "response": detailed_info}
                else:
                    return {"query_details": None, "response": "I don't have any properties in context. Let's start your search first."}
            except (AttributeError, ValueError) as e:
                logger.error(f"Error processing property request: {e}")
                return {"query_details": None, "response": "Could you specify which property number you're interested in?"}
    
    # Process based on conversation state
    if agent.conversation_state == "initial":
        if "buy" in message.lower():
            agent.current_transaction = "buy"
            agent.user_requirements['transaction_type'] = 'buy'
            agent.conversation_state = "transaction_type"
            response = "What type of user best describes you? (End User/Investor)"
        elif "rent" in message.lower():
            agent.current_transaction = "rent"
            agent.user_requirements['transaction_type'] = 'rent'
            agent.conversation_state = "transaction_type"
            response = "Are you looking to rent short term or long term?"
        else:
            response = "Please specify if you want to Buy or Rent."
        
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    # Handle other states in the conversation flow
    current_flow = agent.conversation_flow.get(agent.conversation_state)
    if current_flow:
        # Update user requirements based on the response
        agent.user_requirements[agent.conversation_state] = message
        
        # Move to next state
        next_state = current_flow.get('next')
        if next_state:
            agent.conversation_state = next_state
            next_question = agent.conversation_flow[next_state]['question']
            
            # Format the accumulated requirements
            requirements_summary = format_requirements_summary(agent.user_requirements)
            response = f"{requirements_summary}\n\n{next_question}"
        else:
            # We've reached the end of the flow, show matching properties
            response = search_matching_properties(agent)
        
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    # For property searches, use Gemini API to extract structured data
    gemini_prompt = agent._create_gemini_query_prompt(message)
    query_details = agent._call_gemini_api(gemini_prompt)
    
    if query_details:
        # Update user preferences on the agent instance
        for key, value in query_details.items():
            if value is not None and value != []:
                agent.user_preferences[key] = value
        logger.info(f"Extracted query details: {query_details}")
        return {"query_details": query_details, "response": None}
    else:
        fallback = ("I'm not quite sure what you're looking for. Could you provide more details about the type of property, "
                    "location, or price range you're interested in?")
        agent.conversation_history.append({"role": "assistant", "content": fallback})
        return {"query_details": None, "response": fallback}

def respond_with_typing(agent, user_message: str) -> dict:
    """Determine an initial acknowledgment and then process the full message using typing effect."""
    # Choose an acknowledgment based on keywords in the message.
    lower_msg = user_message.lower()
    if agent._is_greeting(user_message):
        acknowledgment = "Hi there! 👋"
    elif "help" in lower_msg or "looking for" in lower_msg:
        acknowledgment = "I'll help you find that! 🔍"
    elif "available" in lower_msg or "show me" in lower_msg:
        acknowledgment = "Let me check what's available..."
    elif "price" in lower_msg or "cost" in lower_msg:
        acknowledgment = "Checking prices for you..."
    elif "location" in lower_msg or "area" in lower_msg:
        acknowledgment = "Looking at locations for you..."
    else:
        acknowledgment = "Let me look that up for you..."
    
    # Process the user's message and retrieve query details
    result = process_user_message_with_typing(agent, user_message)
    query_details = result.get("query_details")
    # If query details were extracted, we then filter properties and produce the full formatted response.
    if query_details is not None:
        filtered_properties = agent._filter_properties(query_details)
        relevant_properties = agent._semantic_search(filtered_properties, user_message, top_k=5)
        full_response = agent._format_response_with_typing_effect(relevant_properties, user_message, query_details)
    else:
        full_response = result.get("response")
    
    # Add to conversation history and return both acknowledgment and full response.
    agent.conversation_history.append({"role": "assistant", "content": full_response})
    return {"acknowledgment": acknowledgment, "full_response": full_response}

def demo_typing_response(agent, user_message: str) -> str:
    """Simulate a typing effect with more natural pauses and brief responses."""
    response_data = respond_with_typing(agent, user_message)
    
    print(f"\nAssistant: ", end='', flush=True)
    
    full_response = response_data['full_response']
    sentences = full_response.split('\n')
    
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            print(word, end=' ', flush=True)
            delay = min(len(word) * 0.05 + random.uniform(0.05, 0.15), 0.3)
            time.sleep(delay)
        print()  # New line after each sentence
        time.sleep(0.5)  # Pause between sentences
    
    return full_response

def format_requirements_summary(requirements):
    """Format the accumulated requirements into a natural sentence."""
    summary = "I am a client looking to "
    
    if requirements.get('transaction_type') == 'buy':
        summary += "buy "
    else:
        summary += "rent "
    
    if requirements.get('ready_status'):
        summary += f"{requirements['ready_status']} "
    
    if requirements.get('property_type'):
        summary += f"{requirements['property_type']} "
    
    if requirements.get('layout'):
        summary += f"with {requirements['layout']} layout "
    
    if requirements.get('budget'):
        summary += f"minimum budget of {requirements['budget']} AED "
    
    if requirements.get('location'):
        summary += f"in {requirements['location']}"
    
    return summary.strip()

def search_matching_properties(agent):
    """Search for properties matching the user's requirements."""
    try:
        # Convert user requirements to query details format
        query_details = {
            'purpose': 'Sale' if agent.current_transaction == 'buy' else 'Rent',
            'property_type': agent.user_requirements.get('property_type'),
            'bedrooms': agent.user_requirements.get('layout'),
            'location': agent.user_requirements.get('location'),
            'min_price': float(agent.user_requirements.get('budget', 0)) if agent.user_requirements.get('budget') else None,
            'max_price': None,
            'furnishing': None
        }

        # Filter properties based on requirements
        filtered_properties = agent._filter_properties(query_details)
        relevant_properties = agent._semantic_search(filtered_properties, str(agent.user_requirements), top_k=3)
        
        # Reset property display numbers
        agent.property_display_numbers = {}
        agent.last_shown_properties = relevant_properties
        
        if relevant_properties.empty:
            return "I couldn't find any properties matching your exact criteria. Would you like to adjust your requirements?"

        # Format response
        response = [
            f"Based on your requirements, I found {len(relevant_properties)} matching properties.",
            "Here's a brief overview:",
            ""
        ]

        # Add property summaries with consistent numbering
        for i, (idx, prop) in enumerate(relevant_properties.iterrows(), 1):
            agent.property_display_numbers[i] = idx
            price = f"{prop.get('Price', 'N/A'):,}" if pd.notna(prop.get('Price')) else 'N/A'
            summary = [
                f"🏠 Property {i}:",
                f"- {prop.get('Type', 'N/A')} in {prop.get('Location', 'N/A')}",
                f"- Price: AED {price}",
                f"- {prop.get('Bedrooms', 'N/A')} bedrooms",
                ""
            ]
            response.extend(summary)

        response.append("To see more details about a specific property, just say 'show me details of property X' (where X is the property number).")
        response.append("Or we can refine your search if these don't match your preferences.")
        
        return "\n".join(response)

    except Exception as e:
        logger.error(f"Error in search_matching_properties: {e}")
        return "I encountered an error while searching. Let's try refining your requirements."

def get_detailed_property_info(agent, property_num):
    """Get detailed information for a specific property."""
    try:
        # Validate property number
        if not hasattr(agent, 'property_display_numbers') or property_num not in agent.property_display_numbers:
            return f"I couldn't find property {property_num}. Please specify a valid property number from the list above."

        # Get the actual property data
        actual_index = agent.property_display_numbers[property_num]
        property_data = agent.last_shown_properties.loc[actual_index]

        # Format detailed response
        details = [
            f"📍 Detailed information for Property {property_num}:",
            "",
            f"Location: {property_data.get('Location', 'N/A')}",
            f"Type: {property_data.get('Type', 'N/A')}",
            f"Price: AED {property_data.get('Price', 'N/A'):,}",
            f"Bedrooms: {property_data.get('Bedrooms', 'N/A')}",
            f"Bathrooms: {property_data.get('Bathrooms', 'N/A')}",
            f"Area: {property_data.get('Area', 'N/A')}",
            "",
            f"Description: {property_data.get('Description', 'No description available')}",
            "",
            f"Amenities: {property_data.get('Amenities', 'N/A')}",
            "",
            "Would you like to:"
            "1. Schedule a viewing"
            "2. Ask more questions about this property"
            "3. See other similar properties"
        ]
        
        return "\n".join(details)
    except Exception as e:
        logger.error(f"Error getting detailed property info: {e}")
        return "I encountered an error while retrieving the property details. Please try again."
