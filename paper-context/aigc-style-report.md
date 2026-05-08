# AIGC Style Governance Report

This is a style-risk report, not an AI-detector score.

- Overall risk: `low`
- Paragraphs: `208`
- Risk counts: `{'clear': 205, 'low': 3, 'medium': 0, 'high': 0}`

## Pattern Counts

- `rigid_sequence`: 3

## Hard Failures

- None.

## Paragraph Findings

### Paragraph 32 - low risk
- Score: `1`
- Patterns: rigid_sequence
- Cliche terms: none
- Preview: 雾图像退化可由大气散射模型描述。设观测图像为 I(x)，无雾图像为 J(x)，大气光为 A，透射率为 t(x)，像素位置为 x，则常用模型可以写成式（2-1）。其中第一项表示场景辐射经过介质衰减后到达相机，第二项表示大气光散射带来的亮度叠加

### Paragraph 72 - low risk
- Score: `1`
- Patterns: rigid_sequence
- Cliche terms: none
- Preview: 图3-2展示了复原块内部的数据流。输入特征经过归一化后进入注意力分支，输出与输入相加；随后再次归一化并进入前馈分支，得到最终特征。该设计使模型同时具备局部纹理建模和较大范围特征交互能力。对于雨纹任务，注意力可以捕捉方向性退化；对于去雾任务，

### Paragraph 123 - low risk
- Score: `1`
- Patterns: rigid_sequence
- Cliche terms: none
- Preview: | 项目 | 口径 | | --- | --- | | PSNR/SSIM | 保存图像复算，预测图与目标图共同尺寸中心裁剪 | | LPIPS | alex backbone，数值越低越好 | | 去雨匹配 | 优先同名匹配，不同后缀按 

## Suggested Revision Order

1. Fix vague attribution with verified sources or remove it.
2. Replace generic conclusions with concrete claims, limits, or next-step questions.
3. Break rigid enumeration and repeated paragraph rhythm.
4. Remove filler phrases and excessive academic cliches.
5. Preserve facts and mark unsupported claims as `needs_source`.
