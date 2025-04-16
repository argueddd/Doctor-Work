class LanguageScaffoldBuilder:
    def __init__(self, patch_token="<|patch|>"):
        self.patch_token = patch_token

    def gd(self, static_info: dict) -> str:
        """
        Convert static features like age, gender into a sentence.
        Example input: {"age": 45, "gender": "male"}
        """
        parts = [f"The patient is {static_info.get('age', 'unknown')} years old."]
        if 'gender' in static_info:
            parts.append(f"The gender is {static_info['gender']}.")
        return " ".join(parts)

    def gp(self, num_patches: int) -> str:
        """
        Generate positional scaffold for patch embeddings.
        """
        return " ".join([
            f"patch{i+1} is {self.patch_token}," for i in range(num_patches)
        ])

    def gq(self, num_classes: int) -> str:
        """
        Generate the final question prompt.
        """
        class_list = ", ".join([str(i) for i in range(num_classes)])
        return f"What is label? Please choose in [{class_list}]. A:"

    def build_scaffold(self, static_info: dict, num_patches: int, num_classes: int) -> str:
        """
        Full scaffold template builder.
        """
        parts = [
            self.gd(static_info),
            "Also, time series data exists with",
            self.gp(num_patches),
            self.gq(num_classes)
        ]
        return " ".join(parts)


if __name__ == "__main__":
    scaffold_builder = LanguageScaffoldBuilder()

    static_info = {"age": 38, "gender": "female"}
    num_patches = 10
    num_classes = 4

    scaffold_text = scaffold_builder.build_scaffold(
        static_info=static_info,
        num_patches=num_patches,
        num_classes=num_classes
    )

    print("📄 Generated Scaffold:\n", scaffold_text)