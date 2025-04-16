class PromptBuilder:
    def __init__(self, good_token='<|GOOD_PATCH|>', bad_token='<|BAD_PATCH|>', target_token='<|TARGET_PATCH|>'):
        self.good_token = good_token
        self.bad_token = bad_token
        self.target_token = target_token

    def build(
        self,
        num_good_patches: int,
        num_bad_patches: int,
        instruct: bool = True
    ) -> str:
        """
        构建 prompt 文本

        参数：
        - num_good_patches: 好样本 patch 数量
        - num_bad_patches: 坏样本 patch 数量
        - target_cell_id: 可选，目标小区 ID，用于 prompt 中加入指示
        - instruct: 是否加入评分指令

        返回：
        - 完整 prompt 字符串
        """
        lines = []

        if num_bad_patches > 0:
            lines.append(f"We observed {num_bad_patches} poor-performance cells, represented by the following patches:")
            for i in range(num_bad_patches):
                lines.append(f"bad_patch{i+1} is {self.bad_token}.")
            lines.append("")

        if num_good_patches > 0:
            lines.append(f"We also observed {num_good_patches} high-quality cells, represented by:")
            for i in range(num_good_patches):
                lines.append(f"good_patch{i+1} is {self.good_token}.")
            lines.append("")

        # 目标 patch

        lines.append("Now we want to evaluate a new target cell, represented as:")
        lines.append(f"target_patch is {self.target_token}.\n")

        # 指令部分
        if instruct:
            lines.append("Please give it a quality score between 0 and 100.")
            lines.append("Just answer with the integer score.")
            lines.append("A:")

        return "\n".join(lines)

if __name__ == '__main__':
    builder = PromptBuilder()
    prompt = builder.build(
        num_good_patches=25,
        num_bad_patches=20,
    )
    print(prompt)