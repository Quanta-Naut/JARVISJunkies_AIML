import openai
import base64


class EmotionDetect:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key="a3766117-6a72-454d-9727-aa8abd0312e2",
            base_url="https://api.sambanova.ai/v1",
        )
        self.message = []

    def get_emotion(self, img_src):
        with open(img_src, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        self.message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + encoded_image  # Include base64 string
                            },
                        },
                    ],
                }
            ]
        response = self.client.chat.completions.create(
            model="Llama-3.2-11B-Vision-Instruct",
            messages=self.message,
            temperature=0.1,
            top_p=0.1,
        )
        assistant_response = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )
        return assistant_response
