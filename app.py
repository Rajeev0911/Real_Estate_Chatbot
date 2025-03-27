class PredefinedQueryBot:
    def __init__(self):
        self.responses = {
            "hello": "Hi there! How can I help you?",
            "how are you": "I'm just a bot, but I'm doing great! How about you?",
            "bye": "Goodbye! Have a great day!",
            "who are you": "I'm a chatbot here to assist you.",
            "help": "I can answer predefined queries. Try asking 'hello' or 'who are you'."
        }

    def get_response(self, query):
        return self.responses.get(query.lower(), "Sorry, I don't understand that.")

# Run the bot
if __name__ == "__main__":
    bot = PredefinedQueryBot()
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Bot: Exiting... Have a nice day!")
            break
        print(f"Bot: {bot.get_response(user_input)}")
