import sys
import time
import random
import logging

logger = logging.getLogger(__name__)

# You can choose to keep the simple keyword checks as methods on the agent;
# here we assume the agent instance already has _is_greeting, _is_basic_question, and _is_data_question methods.
# For this example, we’ll use them from the passed agent instance.

def process_user_message_with_typing(agent, message: str) -> dict:
    """Process a user's message and update conversation history. Returns the query details as extracted via Gemini."""
    agent.current_query = message
    agent.conversation_history.append({"role": "user", "content": message})
    
    # Check for special cases using agent's built-in methods
    if agent._is_greeting(message):
        agent.conversation_state = "introduced"
        greeting = random.choice(agent.greetings)
        agent.conversation_history.append({"role": "assistant", "content": greeting})
        return {"query_details": None, "response": greeting}
    
    if agent._is_basic_question(message):
        response = ("I'm your real estate assistant powered by AI. I can help you find properties based on your preferences "
                    "such as location, price range, number of bedrooms, and more. What kind of property are you looking for today?")
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    if agent._is_data_question(message):
        overview = agent._get_dataset_overview()
        agent.conversation_history.append({"role": "assistant", "content": overview})
        return {"query_details": None, "response": overview}
    
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
    """Simulate a typing effect to reveal the full response gradually."""
    response_data = respond_with_typing(agent, user_message)
    
    # Immediately print the acknowledgment
    print(f"\nAssistant: {response_data['acknowledgment']}")
    
    full_response = response_data['full_response']
    words = full_response.split()
    typed_text = ""
    for i, word in enumerate(words):
        typed_text += word + " "
        if i % 5 == 0 or i == len(words) - 1:
            sys.stdout.write("\rAssistant: " + typed_text)
            sys.stdout.flush()
            time.sleep(random.uniform(0.1, 0.3))
    print()  # End with newline
    return full_response
