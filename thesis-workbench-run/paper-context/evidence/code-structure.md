# Code Structure Evidence

- `MWIR-Net/net/mwirnet.py`：MWIR-Net模型主体，包含多尺度编码解码、TransformerBlock、WeatherPromptBlock、PromptChannelAttention。
- `MWIR-Net/train.py`：Lightning训练入口，包含SobelEdgeLoss、CharbonnierLoss、AdamW优化器和warmup cosine调度。
- `MWIR-Net/test.py`：测试入口，包含去雨、去雾测试流程和8路TTA自集成推理。
- `MWIR-Net/utils/dataset_utils.py`：训练和测试数据集读取、裁剪、增强、配对和尺寸匹配。
- `MWIR-Net/tools/evaluate_baseline_outputs.py`：保存图像口径的PSNR、SSIM、LPIPS复算。
- `MWIR-Net/tools/prepare_mwir_data.py`：训练数据软链接与列表生成。
