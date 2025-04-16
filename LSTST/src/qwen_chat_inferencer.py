from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenChatInferencer:
    def __init__(self, model_name="Qwen/QwQ-32B", max_new_tokens=2048):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model and tokenizer ready.")

    def chat(self, prompt: str | list[dict], return_text=True) -> str:
        """
        支持：
        - prompt: 纯字符串，会包装成 user message
        - prompt: 消息列表 [{"role": "user", "content": "..."}]
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        # 模板化聊天输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
            )

        # 去掉输入部分，仅保留回复
        output_ids = generated[0][inputs.input_ids.shape[1]:]

        if return_text:
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_ids

if __name__ == "__main__":
    inferencer = QwenChatInferencer(model_name="Qwen/QwQ-32B", max_new_tokens=1024)

    prompt = 'How many r\'s are in the word "strawberry"?'
    response = inferencer.chat(prompt)
    print("🤖 Qwen says:", response)