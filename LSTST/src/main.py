import torch
from transformers import AutoModelForCausalLM

from src.modules.wireless_optimization_agent.short_evaluation.lstst.patch_embedding_injector import \
    PatchEmbeddingInjector
from src.modules.wireless_optimization_agent.short_evaluation.lstst.prompt_builder import PromptBuilder
from src.modules.wireless_optimization_agent.short_evaluation.lstst.scaffold_tokenizer_wrapper import \
    ScaffoldTokenizerWrapper
from src.modules.wireless_optimization_agent.short_evaluation.lstst.time_series_patch_embedding import \
    TimeSeriesPatchEmbedding

if __name__ == "__main__":
    batch_size = 1
    time_steps = 100000
    num_features = 55
    d_model = 768
    kernel_size = 10000
    stride = 5000

    time_series_patch_embedding = TimeSeriesPatchEmbedding(num_features, d_model, kernel_size=kernel_size, stride=stride)

    # shape is [B,P,D]--[1, 19, 768]
    good_series_data = torch.randn(batch_size, time_steps, num_features)
    good_patch_embeddings = time_series_patch_embedding(good_series_data)

    # shape is [B,P,D]--[1, 19, 768]
    bad_series_data = torch.randn(batch_size, time_steps, num_features)
    bad_patch_embeddings = time_series_patch_embedding(bad_series_data)

    # shape is [B,P,D]--[1, 1, 768]
    target_time_series_patch_embedding = TimeSeriesPatchEmbedding(num_features, d_model, kernel_size=1, stride=stride)
    target_series_data = torch.randn(batch_size, 1, num_features)
    target_patch_embeddings = target_time_series_patch_embedding(target_series_data)

    # 1. 加载模型 & tokenizer wrapper
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer_wrapper = ScaffoldTokenizerWrapper(model_name=model_name)
    model.resize_token_embeddings(len(tokenizer_wrapper.tokenizer))

    # 2. 构造 prompt
    builder = PromptBuilder()
    prompt = builder.build(good_patch_embeddings.shape[1], bad_patch_embeddings.shape[1])
    print(prompt)
    encoding = tokenizer_wrapper.encode_scaffold(prompt)
    input_ids = encoding["input_ids"]  # shape: [1, L]
    attention_mask = encoding["attention_mask"]

    # 3. 模拟 patch embeddings
    batch_size = 1
    d_model = model.config.n_embd

    patch_embeddings = {
        "good": good_patch_embeddings,
        "bad": bad_patch_embeddings,
        "target": target_patch_embeddings
    }

    # 4. 注入器初始化
    injector = PatchEmbeddingInjector(
        model,
        token_type_to_id={
            "good": tokenizer_wrapper.good_patch_token_id,
            "bad": tokenizer_wrapper.bad_patch_token_id,
            "target": tokenizer_wrapper.target_patch_token_id
        }
    )
    # 5. 执行注入
    inputs_embeds = injector.inject(input_ids, patch_embeddings)

    # 6. 使用 generate 方法生成模型回复
    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=30,  # 控制生成长度，可调整
            do_sample=False,  # 不采样，取最大概率
        )

    # 7. 取生成的新 token（去掉 prompt 部分）
    new_token_ids = generated_ids[0, input_ids.shape[1]:]
    output_text = tokenizer_wrapper.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    print("📦 Special Token Vocab Check:")
    for tok in ["<|GOOD_PATCH|>", "<|BAD_PATCH|>", "<|TARGET_PATCH|>"]:
        tok_id = tokenizer_wrapper.tokenizer.convert_tokens_to_ids(tok)
        print(f"{tok}: {tok_id}")

    print("🧱 Raw generated token IDs:", generated_ids.tolist())
    print("🧱 Generated tail IDs (new tokens):", new_token_ids.tolist())
    # 8. 打印最终输出的那句话
    print(f"📢 LLM输出:\n{output_text}")





