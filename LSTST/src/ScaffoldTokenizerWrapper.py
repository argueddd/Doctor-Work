from transformers import AutoTokenizer


class ScaffoldTokenizerWrapper:
    def __init__(self, model_name='gpt2', patch_token='<|PATCH|>'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.patch_token = patch_token

        # 添加特殊 token
        self._add_patch_token()

        # 保存 patch_token 对应的 token_id
        self.patch_token_id = self.tokenizer.convert_tokens_to_ids(self.patch_token)

    def _add_patch_token(self):
        if self.patch_token not in self.tokenizer.get_vocab():
            print(f"🔧 Adding special token: {self.patch_token}")
            self.tokenizer.add_special_tokens({'additional_special_tokens': [self.patch_token]})
        else:
            print(f"✅ Token {self.patch_token} already in vocabulary")

    def encode_scaffold(self, scaffold_text, return_tensors=True):
        encoding = self.tokenizer(
            scaffold_text,
            return_tensors='pt' if return_tensors else None,
            return_attention_mask=True
        )
        return encoding

    def find_patch_positions(self, input_ids):
        """
        返回 <|PATCH|> token 在输入序列中的位置索引（用于替换 embedding）
        input_ids: Tensor (batch_size, sequence_length)
        返回: List[patch_positions_per_sample]
        """
        patch_positions = []
        for sample in input_ids:
            positions = (sample == self.patch_token_id).nonzero(as_tuple=True)[0]
            patch_positions.append(positions.tolist())
        return patch_positions


if __name__ == '__main__':
    scaffold_text = (
        "The patient is 55 years old. The gender is male. "
        "Also, time series data exists with "
        "patch1 is <|PATCH|>, patch2 is <|PATCH|>, patch3 is <|PATCH|>, "
        "What is label? Please choose in [0, 1, 2]. A:"
    )

    tokenizer_wrapper = ScaffoldTokenizerWrapper(model_name='gpt2', patch_token='<|PATCH|>')

    encoding = tokenizer_wrapper.encode_scaffold(scaffold_text)
    input_ids = encoding['input_ids']  # (1, seq_len)

    print("📄 Tokenized Input IDs:\n", input_ids)
    patch_positions = tokenizer_wrapper.find_patch_positions(input_ids)
    print("📌 <|PATCH|> positions:", patch_positions)