# Experiment Results Evidence

来源：`MWIR-Net/docs/所有模型指标汇总.md`。

关键结果：

- Rain100L：MWIR-Net-final_tta_multisplit PSNR 33.08、SSIM 0.9442、LPIPS 0.087578。
- Test1200：MWIR-Net-final_plain_multisplit PSNR 30.03、SSIM 0.8702、LPIPS 0.090416。
- Test2800：MWIR-Net-final_plain_multisplit PSNR 30.66、SSIM 0.9078、LPIPS 0.057636。
- GT-RAIN-test：MWIR-Net-gtrain_plain PSNR 21.03、SSIM 0.5963、LPIPS 0.293823。
- SOTS outdoor：MWIR-Net-stage2_charb_edge002_tta_dehaze PSNR 32.04、SSIM 0.9804、LPIPS 0.009871。
- nyuhaze500：MWIR-Net-final_plain_multisplit_dehaze PSNR 17.20、SSIM 0.8239、LPIPS 0.101394。
- 公平消融：zero prompt与no channel attention的均值差异小于seed间波动，不能认定通道注意力在当前协议下有稳定独立增益。
