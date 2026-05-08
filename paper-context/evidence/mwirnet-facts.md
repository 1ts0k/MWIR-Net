# MWIR-Net Facts

- 模型文件：`net/mwirnet.py`。
- 主体结构：重叠 patch 嵌入、多尺度 Transformer 编码器、latent 特征、解码器、refinement blocks 和残差输出。
- Prompt 模块：`WeatherPromptBlock`，包含 prompt 参数库、线性权重、softmax 加权、3x3 卷积和 `PromptChannelAttention`。
- 消融模式：`full`、`zero_prompt`、`no_channel_attention`。
- 训练损失：L1 或 Charbonnier，加 Sobel 边缘一致性损失。
- 优化器：AdamW。
- 推理：常规推理或 8 路 TTA 自集成。
