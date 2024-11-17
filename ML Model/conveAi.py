import openai

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key="03fce9be-b876-4ba0-a2ea-2de406bd44dc",
    base_url="https://api.sambanova.ai/v1",
)

# Initialize chat memory with system prompt
messages = [{"role": "system", "content": "You are a helpful assistant"}]

while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break
    
    # Append user message to memory
    messages.append({"role": "user", "content": user_input})
    
    # Get response from the model
    response = client.chat.completions.create(
        model='Meta-Llama-3.1-8B-Instruct',
        messages=messages,
        temperature=0.1,
        top_p=0.1
    )
    
    # Extract assistant's reply
    assistant_reply = response.choices[0].message.content
    print(f"Assistant: {assistant_reply}")
    
    # Append assistant's reply to memory
    messages.append({"role": "assistant", "content": assistant_reply})
