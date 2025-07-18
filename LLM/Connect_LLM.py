from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio-key",  # LM Studio không yêu cầu key thực
)

response = client.chat.completions.create(
    model="vinallama-7b-chat", 
    messages=[
    {"role": "system", "content": "Bạn là một trợ lý AI thân thiện, trả lời bằng tiếng Việt thật tự nhiên."},
    {"role": "user", "content": "Hãy nêu khái niệm về môi trường"},
],
    max_tokens=256,
    temperature=0.7
)

# Trích nội dung trả lời 
reply = response.choices[0].message.content

# Ghi ra file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(reply)