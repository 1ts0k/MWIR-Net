# MWIR-Net Facts

- 网络全称：Multi-scale Weather-aware Image Restoration Network。
- 输入输出：三通道退化图像输入，输出三通道复原图像，并采用残差输出 `output + inp_img`。
- 主体结构：overlap patch embedding，多尺度encoder-decoder，level1-level3编码，latent transformer，level3-level1解码和refinement blocks。
- Transformer复原块：LayerNorm + multi-DConv head transposed self-attention + GDFN。
- 天气感知提示：WeatherPromptBlock通过全局平均特征生成prompt权重，融合prompt字典，经3×3卷积和通道注意力后注入解码阶段。
- 消融模式：`full`、`zero prompt`、`no channel attention`。
- 损失函数：L1或Charbonnier像素损失加Sobel边缘一致性损失。
- 推理策略：普通推理和8路旋转/翻转TTA自集成。
