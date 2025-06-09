from transformers import pipeline

# Load the Flan-T5-base model for the "text2text-generation" task
try:
    chatbot = pipeline("text2text-generation", model="google/flan-t5-base")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you are connected to the internet or try again later.")
    exit()

print("Chatbot is ready. Type 'exit' or 'bye' to quit.")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "bye", "خروج", "بای"]:
        print("Chatbot: Goodbye!")
        break

    # Generate response using the model
    response = chatbot(user_input, max_length=50, num_return_sequences=1)

    # Display the response
    if response:
        print(f"Chatbot: {response[0]['generated_text']}")
    else:
        print("Chatbot: I didn't understand. Can you ask again?")
