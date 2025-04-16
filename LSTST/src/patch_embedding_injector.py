from src.modules.wireless_optimization_agent.short_evaluation.lstst.prompt_builder import PromptBuilder
from src.modules.wireless_optimization_agent.short_evaluation.lstst.scaffold_tokenizer_wrapper import \
    ScaffoldTokenizerWrapper
from transformers import AutoModelForCausalLM
import torch

def verify_injection(input_ids, inputs_embeds, patch_embeddings, token_id, label, tokenizer):
    positions = (input_ids[0] == token_id).nonzero(as_tuple=True)[0]
    print(f"\n Verifying: {label.upper()} PATCH | Token ID: {token_id}")

    for i, pos in enumerate(positions):
        actual = inputs_embeds[0, pos]
        expected = patch_embeddings[label][0, i]
        diff = (actual - expected).abs().max().item()
        print(f" {label.upper()} patch {i:02d} at pos {pos.item():03d} | Max diff: {diff:.6f}")

    print(f"\n Token Embeddings Snapshot ({label.upper()} TOKENs):")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for idx, (tok_id, token_str) in enumerate(zip(input_ids[0], tokens)):
        vec = inputs_embeds[0, idx, :3]  # 取前3维
        vec_str = ", ".join([f"{v.item():.2f}" for v in vec])
        highlight = " <--" if tok_id == token_id else ""
        print(f"[{idx:03d}] {token_str:<15} → [{vec_str}]{highlight}")


class PatchEmbeddingInjector:
    def __init__(self, model, token_type_to_id):
        """
        token_type_to_id: dict like {
            'good': <|GOOD_PATCH|> id,
            'bad': <|BAD_PATCH|> id,
            'target': <|TARGET_PATCH|> id
        }
        """
        self.model = model
        self.embedding_layer = model.get_input_embeddings()
        self.token_type_to_id = token_type_to_id

    def inject(self, input_ids, patch_embeddings):
        """
        input_ids: (B, L)
        patch_embeddings: dict {
            'good': (B, N_good, D),
            'bad': (B, N_bad, D),
            'target': (B, 1, D)
        }
        return: inputs_embeds: (B, L, D)
        """
        input_embeds = self.embedding_layer(input_ids.clone())
        B, L = input_ids.shape

        for b in range(B):
            for token_type, token_id in self.token_type_to_id.items():
                positions = (input_ids[b] == token_id).nonzero(as_tuple=True)[0]
                expected = patch_embeddings[token_type].shape[1]

                if len(positions) != expected:
                    raise ValueError(f"[{token_type}] Batch {b}: Found {len(positions)} tokens, "
                                     f"but got {expected} embeddings.")

                for i, pos in enumerate(positions):
                    input_embeds[b, pos] = patch_embeddings[token_type][b, i]

        return input_embeds


if __name__ == '__main__':
    # 1. 加载模型 & tokenizer wrapper
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer_wrapper = ScaffoldTokenizerWrapper(model_name=model_name)
    model.resize_token_embeddings(len(tokenizer_wrapper.tokenizer))

    # 2. 构造 prompt
    num_good = 10
    num_bad = 10
    builder = PromptBuilder()
    prompt = builder.build(num_good, num_bad)
    encoding = tokenizer_wrapper.encode_scaffold(prompt)
    input_ids = encoding["input_ids"]  # shape: [1, L]
    attention_mask = encoding["attention_mask"]

    # 3. 模拟 patch embeddings
    batch_size = 1
    d_model = model.config.n_embd

    patch_embeddings = {
        "good": torch.ones(batch_size, num_good, d_model) * 111,  # 全 111
        "bad": torch.ones(batch_size, num_bad, d_model) * 222,  # 全 222
        "target": torch.ones(batch_size, 1, d_model) * 999  # 全 999
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

    # 6. 检查是否替换成功
    print("Checking GOOD PATCH positions...")
    verify_injection(
        input_ids,
        inputs_embeds,
        patch_embeddings,
        tokenizer_wrapper.good_patch_token_id,
        "good",
        tokenizer_wrapper.tokenizer
    )

    print("\n✅ Checking BAD PATCH positions...")
    verify_injection(
        input_ids,
        inputs_embeds,
        patch_embeddings,
        tokenizer_wrapper.bad_patch_token_id,
        "bad",
        tokenizer_wrapper.tokenizer
    )

    print("\n✅ Checking TARGET PATCH position...")
    verify_injection(
        input_ids,
        inputs_embeds,
        patch_embeddings,
        tokenizer_wrapper.target_patch_token_id,
        "target",
        tokenizer_wrapper.tokenizer
    )
