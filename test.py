import os
import openai

client = openai.OpenAI(
    api_key="a3766117-6a72-454d-9727-aa8abd0312e2",
    base_url="https://api.sambanova.ai/v1",
)

response = client.chat.completions.create(
    model='Llama-3.2-11B-Vision-Instruct',
    messages=[{"role":"user","content":[{"type":"text","text":"What do you see in this image"},{"type":"C:\Users\tarun\Desktop\Folders\ConversationalAI\temp.jpg","image_url":{"url":"<image_in_base_64>"}}]}],
    temperature =  0.1,
    top_p = 0.1
)

print(response.choices[0].message.content)
      