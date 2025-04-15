import torch

from LSTST.src.TimeSeriesEncoder import TimeSeriesEncoder


def main():
    # 假数据参数
    batch_size = 2
    time_steps = 100
    num_features = 4
    d_model = 32
    kernel_size = 8
    stride = 4
    padding = 2

    # 随机生成不规则时间戳（升序）
    timestamps = torch.sort(torch.rand(batch_size, time_steps) * 100, dim=1).values

    # 随机生成时间序列数据
    x = torch.randn(batch_size, time_steps, num_features)

    # 初始化编码器
    encoder = TimeSeriesEncoder(
        input_dim=num_features,
        d_model=d_model,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # 编码
    embeddings = encoder(x, timestamps)
    print("✅ Final embedding shape:", embeddings.shape)  # -> (B, num_patches, d_model)


if __name__ == "__main__":
    main()
