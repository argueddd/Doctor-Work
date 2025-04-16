from transformers import AutoTokenizer


from transformers import AutoTokenizer

from src.modules.wireless_optimization_agent.short_evaluation.lstst.prompt_builder import PromptBuilder


class ScaffoldTokenizerWrapper:
    def __init__(self, model_name='gpt2', good_patch_token='<|GOOD_PATCH|>', bad_patch_token='<|BAD_PATCH|>', target_patch_token='<|TARGET_PATCH|>'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.good_patch_token = good_patch_token
        self.bad_patch_token = bad_patch_token
        self.target_patch_token = target_patch_token

        # 添加特殊 token
        self._add_special_tokens()

        # 保存 token ID
        self.good_patch_token_id = self.tokenizer.convert_tokens_to_ids(self.good_patch_token)
        self.bad_patch_token_id = self.tokenizer.convert_tokens_to_ids(self.bad_patch_token)
        self.target_patch_token_id = self.tokenizer.convert_tokens_to_ids(self.target_patch_token)

    def _add_special_tokens(self):
        special_tokens = [self.good_patch_token, self.bad_patch_token, self.target_patch_token]
        new_tokens = [tok for tok in special_tokens if tok not in self.tokenizer.get_vocab()]
        if new_tokens:
            print(f"Adding special tokens: {new_tokens}")
            self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        else:
            print(f"Special tokens already in vocabulary")

    def encode_scaffold(self, scaffold_text, return_tensors=True):
        encoding = self.tokenizer(
            scaffold_text,
            return_tensors='pt' if return_tensors else None,
            return_attention_mask=True,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        return encoding

    def find_token_positions(self, input_ids, token_id):
        """
        查找指定 token_id 在输入序列中的位置
        """
        all_positions = []
        for sample in input_ids:
            positions = (sample == token_id).nonzero(as_tuple=True)[0]
            all_positions.append(positions.tolist())
        return all_positions

    def visualize_token_alignment(self, scaffold_text):
        """
        打印 token 的 index、token ID、token 字符串、原始文本片段（按列对齐）
        """
        encoding = self.tokenizer(
            scaffold_text,
            return_tensors='pt',
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        input_ids = encoding['input_ids'][0]
        offsets = encoding['offset_mapping'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        print("🔍 Token Alignment:\n")
        print(f"{'Idx':<6} {'Token ID':<10} {'Token':<20} {'Text Fragment'}")
        print("-" * 70)
        for i, (tok_id, tok, (start, end)) in enumerate(zip(input_ids, tokens, offsets)):
            fragment = scaffold_text[start:end]
            print(f"{i:<6} {tok_id.item():<10} {tok:<20} {repr(fragment)}")



if __name__ == '__main__':
    prompt = PromptBuilder().build(3,3)
    tokenizer_wrapper = ScaffoldTokenizerWrapper(model_name='gpt2')
    tokenizer_wrapper.visualize_token_alignment(prompt)
