import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("YI_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.lingyiwanwu.com/v1")


def openai_chat(text: str):
    response = client.chat.completions.create(
        model="yi-lightning",
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": text}
        ],
        stream=True
    )
    # 用于存储逐步输出的内容
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, flush=True)
    return partial_message


# 调试方法
if __name__ == "__main__":
    user_input = "你好。"
    openai_chat(user_input)
