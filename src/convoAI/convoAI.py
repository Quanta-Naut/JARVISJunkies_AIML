import openai

class ConvoAI:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key="a3766117-6a72-454d-9727-aa8abd0312e2",
            base_url="https://api.sambanova.ai/v1",
        )
        self.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

    def get_response(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=self.conversation_history,
            temperature=0.1,
            top_p=0.1,
        )
        assistant_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response