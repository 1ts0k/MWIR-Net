#!/usr/bin/env python3
"""Write the structured MWIR-Net thesis package in the isolated run workspace."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "thesis-workbench-run"
CTX = RUN / "paper-context"
OUT = RUN / "paper-output"
STD = RUN / "thesis-ai-standard" / "templates"
TITLE = "基于深度学习在雾雨退化场景下的图像复原算法研究"


FIGURES = [
    ("图2-1 雾雨退化图像示例", "figure-2-1-degradation-examples.png"),
    ("图3-1 MWIR-Net总体结构", "figure-3-1-mwirnet-architecture.png"),
    ("图3-2 Transformer复原块结构", "figure-3-2-transformer-block.png"),
    ("图3-3 天气感知提示模块结构", "figure-3-3-prompt-module.png"),
    ("图3-4 TTA自集成推理流程", "figure-3-4-tta-workflow.png"),
    ("图4-1 GT-RAIN真实雨图像样例", "figure-4-2-gtrain-examples.png"),
    ("图4-2 训练损失变化曲线", "figure-4-3-training-loss.png"),
    ("图4-3 实验训练与评价流程", "figure-4-1-experiment-workflow.png"),
    ("图5-1 Rain100L PSNR指标对比", "figure-5-3-rain100l-psnr.png"),
    ("图5-2 SOTS outdoor PSNR指标对比", "figure-5-4-sots-psnr.png"),
    ("图5-3 MWIR-Net多去雨数据集PSNR结果", "figure-5-5-derain-multisplit.png"),
    ("图5-4 Rain100L去雨视觉对比", "figure-5-1-rain-visual.png"),
    ("图5-5 SOTS outdoor去雾视觉对比", "figure-5-2-haze-visual.png"),
    ("图5-6 Rain100L公平消融实验结果", "figure-5-6-ablation.png"),
]


REFERENCES = [
    {
        "id": 1,
        "entry": "NARASIMHAN S G, NAYAR S K. Vision and the atmosphere[J]. International Journal of Computer Vision, 2002, 48(3): 233-254.",
        "url": "https://link.springer.com/article/10.1023/A:1016328200723",
        "status": "verified_by_public_index",
    },
    {
        "id": 2,
        "entry": "HE K, SUN J, TANG X. Single image haze removal using dark channel prior[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2011, 33(12): 2341-2353.",
        "url": "https://ieeexplore.ieee.org/document/5567108",
        "status": "verified_by_public_index",
    },
    {
        "id": 3,
        "entry": "TAN R T. Visibility in bad weather from a single image[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Anchorage: IEEE, 2008: 1-8.",
        "url": "https://ieeexplore.ieee.org/document/4587643",
        "status": "verified_by_public_index",
    },
    {
        "id": 4,
        "entry": "FATTAL R. Single image dehazing[J]. ACM Transactions on Graphics, 2008, 27(3): 1-9.",
        "url": "https://dl.acm.org/doi/10.1145/1360612.1360671",
        "status": "verified_by_public_index",
    },
    {
        "id": 5,
        "entry": "CAI B, XU X, JIA K, et al. DehazeNet: An end-to-end system for single image haze removal[J]. IEEE Transactions on Image Processing, 2016, 25(11): 5187-5198.",
        "url": "https://ieeexplore.ieee.org/document/7539399",
        "status": "verified_by_public_index",
    },
    {
        "id": 6,
        "entry": "REN W, LIU S, ZHANG H, et al. Single image dehazing via multi-scale convolutional neural networks[C]//European Conference on Computer Vision. Cham: Springer, 2016: 154-169.",
        "url": "https://link.springer.com/chapter/10.1007/978-3-319-46475-6_10",
        "status": "verified_by_public_index",
    },
    {
        "id": 7,
        "entry": "LI B, PENG X, WANG Z, et al. AOD-Net: All-in-one dehazing network[C]//Proceedings of the IEEE International Conference on Computer Vision. Venice: IEEE, 2017: 4780-4788.",
        "url": "https://openaccess.thecvf.com/content_iccv_2017/html/Li_AOD-Net_All-In-One_Dehazing_ICCV_2017_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 8,
        "entry": "YANG W, TAN R T, FENG J, et al. Deep joint rain detection and removal from a single image[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Honolulu: IEEE, 2017: 1357-1366.",
        "url": "https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Deep_Joint_Rain_CVPR_2017_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 9,
        "entry": "FU X, HUANG J, DING X, et al. Removing rain from single images via a deep detail network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Honolulu: IEEE, 2017: 3855-3863.",
        "url": "https://openaccess.thecvf.com/content_cvpr_2017/html/Fu_Removing_Rain_From_CVPR_2017_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 10,
        "entry": "ZHANG H, PATEL V M. Density-aware single image de-raining using a multi-stream dense network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Salt Lake City: IEEE, 2018: 695-704.",
        "url": "https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Density-Aware_Single_Image_CVPR_2018_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 11,
        "entry": "LI B, REN W, FU D, et al. Benchmarking single-image dehazing and beyond[J]. IEEE Transactions on Image Processing, 2019, 28(1): 492-505.",
        "url": "https://ieeexplore.ieee.org/document/8451944",
        "status": "verified_by_public_index",
    },
    {
        "id": 12,
        "entry": "WANG Z, BOVIK A C, SHEIKH H R, et al. Image quality assessment: From error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.",
        "url": "https://ieeexplore.ieee.org/document/1284395",
        "status": "verified_by_public_index",
    },
    {
        "id": 13,
        "entry": "ZHANG R, ISOLA P, EFROS A A, et al. The unreasonable effectiveness of deep features as a perceptual metric[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Salt Lake City: IEEE, 2018: 586-595.",
        "url": "https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 14,
        "entry": "VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. Long Beach: Curran Associates, 2017: 5998-6008.",
        "url": "https://papers.nips.cc/paper/7181-attention-is-all-you-need",
        "status": "verified_by_public_index",
    },
    {
        "id": 15,
        "entry": "DOSOVITSKIY A, BEYER L, KOLESNIKOV A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[C]//International Conference on Learning Representations. 2021.",
        "url": "https://openreview.net/forum?id=YicbFdNTTy",
        "status": "verified_by_public_index",
    },
    {
        "id": 16,
        "entry": "HU J, SHEN L, SUN G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Salt Lake City: IEEE, 2018: 7132-7141.",
        "url": "https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 17,
        "entry": "ZAMIR S W, ARORA A, KHAN S, et al. Multi-stage progressive image restoration[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Nashville: IEEE, 2021: 14821-14831.",
        "url": "https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 18,
        "entry": "ZAMIR S W, ARORA A, KHAN S, et al. Restormer: Efficient transformer for high-resolution image restoration[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. New Orleans: IEEE, 2022: 5728-5739.",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 19,
        "entry": "LI B, LIU X, HU P, et al. All-in-one image restoration for unknown corruption[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. New Orleans: IEEE, 2022: 17452-17462.",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 20,
        "entry": "VALANARASU J M J, YASAR A, PATEL V M. TransWeather: Transformer-based restoration of images degraded by adverse weather conditions[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. New Orleans: IEEE, 2022: 2353-2363.",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 21,
        "entry": "POTLAPALLI V, ZAMIR S W, KHAN S, et al. PromptIR: Prompting for all-in-one blind image restoration[C]//Advances in Neural Information Processing Systems. New Orleans: Curran Associates, 2023, 36.",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/0c51d8b5edc6f8d4b8c1f8f0b7a81d77-Abstract-Conference.html",
        "status": "verified_by_public_index",
    },
    {
        "id": 22,
        "entry": "LOSHCHILOV I, HUTTER F. Decoupled weight decay regularization[C]//International Conference on Learning Representations. 2019.",
        "url": "https://openreview.net/forum?id=Bkg6RiCqY7",
        "status": "verified_by_public_index",
    },
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def image_map() -> dict[str, str]:
    return {label: str((OUT / "figures" / filename).resolve()) for label, filename in FIGURES}


def standards_yaml() -> str:
    return f"""
schema_version: "1.0"
language: "zh-CN"
encoding: "utf-8"

profile:
  name: "武汉理工大学本科生毕业设计论文撰写规范-隔离工作区"
  school: "武汉理工大学"
  college: "待填写"
  major: "待填写"
  updated_at: "2026-05-08"
  status: "resolved_with_limitations"

source_priority:
  - level: 1
    name: "武汉理工大学本科生毕业设计（论文）撰写规范.pdf"
    file_or_url: "MWIR-Net/docs/武汉理工大学本科生毕业设计（论文）撰写规范.pdf"
    enforce: true
    confirmation_status: "confirmed"
    notes: "采用其中的结构、字数、页边距、字体字号、图表和参考文献要求。未提供学校统一封面 DOCX 模板，因此 DOCX 采用规则化生成而非模板复刻。"
  - level: 2
    name: "任务书"
    file_or_url: "MWIR-Net/docs/任务书.md"
    enforce: true
    confirmation_status: "confirmed"
    notes: "任务书要求正文字数不少于12000字、参考文献不少于20篇、正文不少于12幅图，并包含英译汉翻译任务。"
  - level: 3
    name: "参考论文样文"
    file_or_url: "MWIR-Net/docs/参考论文.docx"
    enforce: false
    confirmation_status: "parsed"
    notes: "仅用于观察章节层级、图表节奏和版式倾向，不覆盖学校规范。"
  - level: 4
    name: "教育部抽检和学术规范底线"
    file_or_url: "https://www.moe.gov.cn/srcsite/A11/s7057/202101/t20210107_509019.html"
    enforce: true
    confirmation_status: "baseline"
    notes: "用于真实性、规范性和证据链检查。"

conflict_resolution:
  rule: "学校PDF规范 > 任务书 > 用户明确限制 > 样文观察 > 项目源码和实验记录"
  unresolved:
    - item: "未提供学校统一封面DOCX模板、作者、学院、专业班级、指导教师等封面字段"
      candidate_sources:
        - "学校PDF规范只给出示例"
        - "用户未提供个人字段"
      chosen_source: "limited_continue"
      notes: "主论文保留待填写字段；正文、图表、参考文献和实验部分可完成。"
    - item: "用户禁止读取、使用或更改MWIR-Net/paper-output与MWIR-Net/paper-context"
      candidate_sources:
        - "技能默认交付到paper-output/paper-context"
        - "用户明确禁止"
      chosen_source: "用户明确限制"
      notes: "所有治理文件和交付物重定向至MWIR-Net/thesis-workbench-run。"

standard_versions:
  thesis_writing:
    name: "武汉理工大学本科生毕业设计（论文）撰写规范"
    version: "2024-07-25"
    use_policy: "主控格式标准"
  references:
    name: "GB/T 7714"
    version: "2005（学校PDF明示）"
    use_policy: "按学校PDF示例著录，正文先出现先引用"
  punctuation:
    name: "GB/T 15834"
    version: "2011"
    use_policy: "中文标点一致性检查"

format_defaults:
  paper_size: "A4"
  page_margin: "上2.5cm，下2cm，左2.5cm，右2cm；页眉2.6cm，页脚2.4cm"
  body_font: "宋体小四号，固定值20磅"
  heading_numbering: "第1章、1.1、1.1.1"
  figure_caption_position: "图题置于图下，按章编号"
  table_caption_position: "表题置于表上，按章编号"
  equation_numbering: "公式独立成行，编号右侧或随公式文本标注，例如（2-1）"
  table_style: "三线表或清晰网格表"
  reference_citation: "正文引用与文末条目一一闭环，先出现先编号"

content_rules:
  abstract:
    required: true
    chinese_length: "300-600字"
    english_length: "约300个英文实词"
  keywords:
    default_count: "3-5"
  body:
    minimum_chars: 12000
    task_required_figures: 12
    delivered_figures: 14
  references:
    task_required_minimum: 20
    delivered: 22
    recent_foreign_minimum: 3
    delivered_recent_foreign: 5

academic_integrity:
  forbid:
    - "编造实验数据"
    - "编造模型结构"
    - "编造数据集或参考文献"
    - "正文暴露AI协作过程"
  require:
    - "实验结果来自MWIR-Net/docs/所有模型指标汇总.md"
    - "模型结构来自MWIR-Net/net/mwirnet.py"
    - "训练和推理方法来自train.py、test.py和tools脚本"
    - "图件全部登记到figure-registry.yaml和image-map.json"
"""


def spec_yaml() -> str:
    return f"""
schema_version: "2.0"
language: "zh-CN"
encoding: "utf-8"

paper:
  title: "{TITLE}"
  type: "本科毕业论文"
  type_profile: "empirical_research"
  school: "武汉理工大学"
  college: "待填写"
  major: "待填写"
  author: "待填写"
  advisor: "待填写"
  submission_date: "2026-05-08"
  template_profile: "templates/standard-profile.yaml"
  delivery_root: "MWIR-Net/thesis-workbench-run/paper-output"

topic:
  background: "雾、雨退化会导致图像对比度下降、细节模糊、遮挡增强和颜色偏移，影响视觉系统的可靠性。"
  problem_statement: "传统物理模型和单一退化复原方法对复杂雾雨场景适应性有限，需要设计兼顾多尺度结构、天气退化提示和可复现实验链路的深度复原模型。"
  objectives:
    - "分析雾雨退化机理与公共数据集特征。"
    - "设计并实现MWIR-Net多尺度天气感知图像复原网络。"
    - "在去雨、去雾数据集上用PSNR、SSIM和LPIPS进行验证，并与经典和前沿算法对比。"
    - "完成不少于12000字、22篇参考文献和14幅图的本科毕业论文交付。"
  significance:
    theoretical: "将Transformer复原块、多尺度编码解码和天气感知prompt结合，用于雾雨退化图像复原。"
    practical: "为交通监控、室外视觉感知和低质图像预处理提供可复现实验参考。"
  scope:
    included:
      - "去雨：Rain100L、Rain100H、Test100、Test1200、Test2800、GT-RAIN-test。"
      - "去雾：SOTS outdoor、nyuhaze500。"
      - "训练协议、模型实现、指标复算、视觉对比和公平消融。"
    excluded:
      - "不承诺真实道路系统部署。"
      - "不承诺超过本地实验记录之外的模型成绩。"
      - "未生成教师指定英文文献的5000汉字翻译正文，需另行补充指定原文。"

research_or_project:
  domain: "图像复原、恶劣天气视觉增强、深度学习"
  object: "MWIR-Net多尺度天气感知图像复原网络"
  methodology:
    - "雾雨退化机理分析"
    - "Transformer多尺度编码解码建模"
    - "Weather-aware prompt与通道注意力设计"
    - "Charbonnier像素损失与Sobel边缘一致性联合训练"
    - "PSNR、SSIM、LPIPS客观评价与主观视觉对比"
  data_sources:
    - name: "RAIN13K"
      status: "provided"
      notes: "用于去雨训练，列表见data_dir/rainy/rainTrain.txt。"
    - name: "RESIDE/OTS/SOTS outdoor"
      status: "provided"
      notes: "用于去雾训练和SOTS outdoor测试。"
    - name: "GT-RAIN"
      status: "provided"
      notes: "用于真实雨场景补充测试。"
    - name: "nyuhaze500"
      status: "provided"
      notes: "用于去雾泛化评价。"
  implementation_sources:
    code_repository: "MWIR-Net"
    database_schema: "not_applicable"
    api_docs: "not_applicable"
    test_reports: "MWIR-Net/docs/所有模型指标汇总.md"

technology_or_method_stack:
  frontend: []
  backend: []
  database: []
  analysis_tools:
    - "Python"
    - "Pillow"
    - "matplotlib"
    - "python-docx"
  experiment_tools:
    - "PyTorch"
    - "Lightning"
    - "torchvision"
    - "scikit-image"
    - "LPIPS"
  external_services: []
  notes: "论文为算法实验研究，不涉及Web系统和数据库。"

chapters:
  - id: "chapter-1"
    title: "绪论"
    word_budget: 2600
    evidence_required:
      - "MWIR-Net/docs/任务书.md"
      - "文献核验清单"
  - id: "chapter-2"
    title: "雾雨退化机理与图像复原理论基础"
    word_budget: 2600
    evidence_required:
      - "学校规范"
      - "参考文献"
      - "评价指标定义"
  - id: "chapter-3"
    title: "MWIR-Net模型设计"
    word_budget: 3000
    evidence_required:
      - "MWIR-Net/net/mwirnet.py"
      - "MWIR-Net/train.py"
      - "MWIR-Net/test.py"
  - id: "chapter-4"
    title: "数据集构建与实验实现"
    word_budget: 2500
    evidence_required:
      - "MWIR-Net/utils/dataset_utils.py"
      - "MWIR-Net/tools/prepare_mwir_data.py"
      - "MWIR-Net/logs"
  - id: "chapter-5"
    title: "实验结果与分析"
    word_budget: 3200
    evidence_required:
      - "MWIR-Net/docs/所有模型指标汇总.md"
      - "MWIR-Net/outputs"
      - "PromptIR/official_output"
  - id: "chapter-6"
    title: "结论与展望"
    word_budget: 1000
    evidence_required:
      - "全文已出现结果"

evidence_index:
  standards_summary: "paper-context/evidence/standards-summary.md"
  code_structure: "paper-context/evidence/code-structure.md"
  tech_stack: "paper-context/evidence/tech-stack.md"
  model_facts: "paper-context/evidence/mwirnet-facts.md"
  experiment_results: "paper-context/evidence/experiment-results.md"
  references: "paper-output/{TITLE}-文献核验清单.json"
"""


def registry_yaml() -> str:
    figure_items = []
    for label, filename in FIGURES:
        chapter = label[1]
        figure_items.append(
            f"""  - id: "{label.split()[0]}"
    title: "{' '.join(label.split()[1:])}"
    chapter: "{chapter}"
    type: "model | flowchart | experiment_screenshot | chart_export | other"
    purpose: "支撑正文对应章节的结构、流程、样例、视觉对比或指标分析。"
    source_kind: "chart_export"
    source_file: "MWIR-Net/thesis-workbench-run/scripts/generate_mwir_assets.py"
    export_file: "paper-output/figures/{filename}"
    evidence:
      - "MWIR-Net/net/mwirnet.py"
      - "MWIR-Net/docs/所有模型指标汇总.md"
      - "MWIR-Net/test and MWIR-Net/outputs"
    first_mentioned_in: "第{chapter}章"
    status: "inserted"
    risk_notes: "图件由本次隔离工作区重新生成，未使用原paper-output内容。\""""
        )
    return f"""
schema_version: "2.0"
language: "zh-CN"
encoding: "utf-8"

rules:
  caption:
    figure: "图题置于图下，例如：图3-1 MWIR-Net总体结构"
    table: "表题置于表上，例如：表4-1 实验环境与工具"
    equation: "公式独立成行，按章编号"
  numbering:
    mode: "按章编号"
    figure_pattern: "图{{chapter}}-{{index}}"
    table_pattern: "表{{chapter}}-{{index}}"
    equation_pattern: "({{chapter}}-{{index}})"
    continuous_check_required: true
  source_policy:
    source_required: true

figures:
{chr(10).join(figure_items)}

tables:
  - id: "表2-1"
    title: "本文主要评价指标"
    chapter: "2"
    type: "metric_definition"
    source_file: "正文公式与参考文献"
    columns: ["指标", "含义", "优化方向"]
    evidence: ["参考文献[12]", "参考文献[13]"]
    first_mentioned_in: "第2章第2.3节"
    status: "inserted"
  - id: "表4-1"
    title: "实验环境与工具"
    chapter: "4"
    type: "environment"
    source_file: "MWIR-Net/env.yml"
    columns: ["项目", "配置"]
    evidence: ["MWIR-Net/env.yml"]
    first_mentioned_in: "第4章第4.3节"
    status: "inserted"
  - id: "表4-2"
    title: "主要训练协议"
    chapter: "4"
    type: "experiment_setting"
    source_file: "MWIR-Net/docs/所有模型指标汇总.md"
    columns: ["阶段", "任务", "关键参数", "用途"]
    evidence: ["MWIR-Net/docs/所有模型指标汇总.md"]
    first_mentioned_in: "第4章第4.3节"
    status: "inserted"
  - id: "表5-1"
    title: "Rain100L去雨结果对比"
    chapter: "5"
    type: "experiment_result"
    source_file: "MWIR-Net/docs/所有模型指标汇总.md"
    columns: ["模型", "PSNR", "SSIM", "LPIPS"]
    evidence: ["MWIR-Net/docs/所有模型指标汇总.md"]
    first_mentioned_in: "第5章第5.1节"
    status: "inserted"
  - id: "表5-2"
    title: "MWIR-Net多数据集结果"
    chapter: "5"
    type: "experiment_result"
    source_file: "MWIR-Net/docs/所有模型指标汇总.md"
    columns: ["任务", "数据集", "图像数", "PSNR", "SSIM", "LPIPS"]
    evidence: ["MWIR-Net/docs/所有模型指标汇总.md"]
    first_mentioned_in: "第5章第5.3节"
    status: "inserted"
  - id: "表5-3"
    title: "Rain100L公平消融结果"
    chapter: "5"
    type: "ablation"
    source_file: "MWIR-Net/docs/所有模型指标汇总.md"
    columns: ["模式", "PSNR mean±std", "SSIM mean±std", "LPIPS mean±std"]
    evidence: ["MWIR-Net/docs/所有模型指标汇总.md"]
    first_mentioned_in: "第5章第5.5节"
    status: "inserted"

equations:
  - id: "(2-1)"
    title: "大气散射模型"
    chapter: "2"
    expression_source: "参考文献[1]、[2]"
    status: "inserted"
  - id: "(2-2)"
    title: "峰值信噪比"
    chapter: "2"
    expression_source: "图像复原通用评价指标"
    status: "inserted"
  - id: "(3-1)"
    title: "联合训练损失"
    chapter: "3"
    expression_source: "MWIR-Net/train.py"
    status: "inserted"
"""


def workflow_files() -> None:
    write(
        CTX / "workflow" / "material-inventory.md",
        """
# 材料盘点

| 材料 | 优先级 | 状态 | 用途 | 缺失影响 |
|---|---|---|---|---|
| MWIR-Net/docs/任务书.md | required | provided | 论文题目、研究内容、字数、图表和参考文献要求 | 无 |
| MWIR-Net/docs/武汉理工大学本科生毕业设计（论文）撰写规范.pdf | required | provided | 格式、结构、字数、图表、参考文献与装订顺序 | 无 |
| MWIR-Net/docs/参考论文.docx | strongly_recommended | provided | 样文章节层级和图表节奏观察 | 无 |
| MWIR-Net/docs/所有模型指标汇总.md | required | provided | 实验结果、训练协议、对比方法和消融结论 | 无 |
| MWIR-Net源码、日志、输出图像 | required | provided | 模型设计、实验实现和图件证据 | 无 |
| 学校统一封面DOCX模板 | strongly_recommended | missing | 严格复刻封面、目录和页眉页脚 | DOCX已按PDF规则生成，但不能承诺与未提供模板完全一致 |
| 作者、学院、专业班级、指导教师 | required_for_cover | missing | 封面和声明页字段 | 以“待填写”保留，正文不受影响 |
| 教师指定英文原文 | strongly_recommended | missing | 5000汉字英译汉附件 | 本次无法生成指定翻译，只能在交付报告中标为待补 |

用户明确要求不读取、使用或更改 `MWIR-Net/paper-output` 与 `MWIR-Net/paper-context`，因此本次工作区为 `MWIR-Net/thesis-workbench-run`。
""",
    )
    write(
        CTX / "workflow" / "user-decisions.md",
        """
# 用户决策记录

- 2026-05-08：用户要求使用 chinese-thesis-workbench 流程生成 MWIR-Net 毕设论文。
- 2026-05-08：用户提供任务书、学校撰写规范PDF、参考论文DOCX、指标汇总与源码路径。
- 2026-05-08：用户明确禁止读取、使用或更改 `MWIR-Net/paper-output` 和 `MWIR-Net/paper-context`，本次交付重定向到 `MWIR-Net/thesis-workbench-run/paper-output`。
- 2026-05-08：由于未提供学校统一DOCX模板，交付方式采用“从零生成并按PDF规范后处理”，不声明模板级完全复刻。
""",
    )
    write(
        CTX / "workflow" / "workflow-status.md",
        """
phase: delivery_done
status: needs_review
blocked_reason:
  - "未提供学校统一封面DOCX模板、作者、学院、专业班级、指导教师字段。"
  - "未提供教师指定英文原文，无法完成任务书中的5000汉字英译汉翻译附件。"
missing_materials:
  - type: "school_template_docx"
    required_for: "封面、目录和页面效果的模板级复刻"
    acceptable_inputs:
      - "学校统一论文模板.docx"
  - type: "cover_fields"
    required_for: "封面和声明页"
    acceptable_inputs:
      - "学院、专业班级、学生姓名、指导教师"
  - type: "translation_source"
    required_for: "5000汉字英译汉翻译"
    acceptable_inputs:
      - "教师指定英文文献原文PDF或DOCX"
next_action:
  - "人工填写封面字段并在Word中更新目录。"
  - "补充教师指定英文原文后生成翻译附件。"
can_continue_with_limitations: true
""",
    )
    write(
        CTX / "workflow" / "blocker-report.md",
        """
# Blocker Report

blocker_type: limited_continue

受影响范围：

- 封面字段和模板级DOCX复刻：缺少学校统一模板与个人字段。
- 任务书第8项英译汉翻译：缺少教师指定英文原文。

不受影响范围：

- 正文研究内容、章节结构、图表、参考文献、实验结果、主论文DOCX和附件DOCX已可生成。

推荐路径：

1. 先使用本次生成的主论文和附件进行内容审阅。
2. 补充学校模板和个人字段后，再进行模板副本填充或人工套版。
3. 补充英文原文后单独生成翻译附件。
""",
    )
    write(
        CTX / "workflow" / "user-dashboard.md",
        f"""
# 用户进度看板

当前阶段：delivery_done

当前状态：needs_review

已完成：

- 初始化隔离工作区：`MWIR-Net/thesis-workbench-run`
- 解析学校规范PDF和参考论文DOCX
- 抽取源码、数据、训练、推理和指标证据
- 生成14幅论文图件
- 生成 `{TITLE}.md`
- 生成 `{TITLE}.docx`
- 生成 `{TITLE}-附件.docx`
- 生成 `{TITLE}-image-map.json`
- 生成 `{TITLE}-文献核验清单.json`

待人工确认：

- 学院、专业班级、学生姓名、指导教师、日期等封面字段。
- Word中更新目录域和页码显示。
- 教师指定英文文献原文，用于生成5000汉字英译汉附件。
""",
    )
    write(
        CTX / "workflow" / "sample-template-analysis.md",
        """
# 样文与规范分析

学校规范PDF要点：

- 中文题目一般不超过25个汉字。
- 中文摘要300-600字，英文摘要约300个英文实词。
- 关键词3-5个。
- 毕业论文正文字数一般不少于12000字。
- 参考文献按正文首次引用顺序编号，学校PDF一般要求不少于15篇，任务书提高为不少于20篇。
- 页面为A4，上2.5cm、下2cm、左2.5cm、右2cm，正文宋体小四号，固定值20磅。
- 图题置于图下，表题置于表上，均按章编号。

样文DOCX观察：

- 样文采用“第1章”“1.1”“1.1.1”层级。
- 图表主要集中在理论模型、实验实现和结果分析章节。
- 样文仅作为结构与节奏参考，不作为格式优先级来源。
""",
    )
    write(CTX / "workflow" / "content-decisions.md", "# 内容取舍记录\n\n- 未发现用户要求排除的内容。\n- 本文聚焦算法实验研究，不写数据库、接口或业务系统模块。\n")
    write(CTX / "workflow" / "chapter-progress.md", "# 章节进度\n\n| 章节 | 状态 |\n|---|---|\n| 摘要/Abstract | done |\n| 第1章 绪论 | done |\n| 第2章 理论基础 | done |\n| 第3章 模型设计 | done |\n| 第4章 实验实现 | done |\n| 第5章 结果分析 | done |\n| 第6章 结论与展望 | done |\n| 参考文献 | done |\n| 致谢 | done |\n")


def evidence_files() -> None:
    write(
        CTX / "evidence" / "standards-summary.md",
        """
# Standards Summary

- 学校：武汉理工大学。
- 规范来源：`MWIR-Net/docs/武汉理工大学本科生毕业设计（论文）撰写规范.pdf`。
- 正文字数：不少于12000字；任务书同样要求不少于12000字。
- 参考文献：任务书要求不少于20篇，其中近5年外文文献不少于3篇；本次交付22篇，近5年外文文献5篇。
- 正文图件：任务书要求不少于12幅；本次交付14幅。
- 页面设置：A4；上2.5cm、下2cm、左2.5cm、右2cm；页眉2.6cm、页脚2.4cm。
- 字体：正文宋体小四号，图题表题宋体小四号，参考文献宋体五号。
""",
    )
    write(
        CTX / "evidence" / "code-structure.md",
        """
# Code Structure Evidence

- `MWIR-Net/net/mwirnet.py`：MWIR-Net模型主体，包含多尺度编码解码、TransformerBlock、WeatherPromptBlock、PromptChannelAttention。
- `MWIR-Net/train.py`：Lightning训练入口，包含SobelEdgeLoss、CharbonnierLoss、AdamW优化器和warmup cosine调度。
- `MWIR-Net/test.py`：测试入口，包含去雨、去雾测试流程和8路TTA自集成推理。
- `MWIR-Net/utils/dataset_utils.py`：训练和测试数据集读取、裁剪、增强、配对和尺寸匹配。
- `MWIR-Net/tools/evaluate_baseline_outputs.py`：保存图像口径的PSNR、SSIM、LPIPS复算。
- `MWIR-Net/tools/prepare_mwir_data.py`：训练数据软链接与列表生成。
""",
    )
    write(
        CTX / "evidence" / "tech-stack.md",
        """
# Technology Stack Evidence

- 主要语言：Python。
- 深度学习框架：PyTorch、Lightning。
- 图像处理：Pillow、torchvision、scikit-image。
- 指标评价：PSNR、SSIM、LPIPS(alex)。
- 可视化与文档生成：matplotlib、python-docx。
- 训练环境文件：`MWIR-Net/env.yml`。
""",
    )
    write(
        CTX / "evidence" / "mwirnet-facts.md",
        """
# MWIR-Net Facts

- 网络全称：Multi-scale Weather-aware Image Restoration Network。
- 输入输出：三通道退化图像输入，输出三通道复原图像，并采用残差输出 `output + inp_img`。
- 主体结构：overlap patch embedding，多尺度encoder-decoder，level1-level3编码，latent transformer，level3-level1解码和refinement blocks。
- Transformer复原块：LayerNorm + multi-DConv head transposed self-attention + GDFN。
- 天气感知提示：WeatherPromptBlock通过全局平均特征生成prompt权重，融合prompt字典，经3×3卷积和通道注意力后注入解码阶段。
- 消融模式：`full`、`zero prompt`、`no channel attention`。
- 损失函数：L1或Charbonnier像素损失加Sobel边缘一致性损失。
- 推理策略：普通推理和8路旋转/翻转TTA自集成。
""",
    )
    write(
        CTX / "evidence" / "experiment-results.md",
        """
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
""",
    )


def diagram_sources() -> str:
    return """
```mermaid
flowchart LR
  A[Input fog/rain image] --> B[Overlap patch embedding]
  B --> C[Encoder L1-L3]
  C --> D[Latent Transformer]
  D --> E[Decoder L3-L1]
  P[Weather-aware prompts] --> E
  E --> F[Refinement blocks]
  F --> G[Restored image]
```
图1-1 MWIR-Net总体结构源码

```mermaid
flowchart LR
  X[Feature] --> N1[LayerNorm]
  N1 --> A[MDTA Attention]
  A --> R1[Residual Add]
  R1 --> N2[LayerNorm]
  N2 --> F[GDFN]
  F --> R2[Residual Add]
```
图1-2 Transformer复原块源码

```mermaid
flowchart LR
  F[Decoder feature] --> GAP[Global pooling]
  GAP --> W[Linear + Softmax]
  W --> D[Prompt dictionary]
  D --> C[3x3 convolution]
  C --> CA[Channel attention]
  CA --> Fuse[Prompt fusion]
```
图1-3 天气感知提示模块源码

```mermaid
flowchart LR
  Datasets[Datasets] --> Prep[prepare_mwir_data.py]
  Prep --> Train[train.py]
  Train --> Ckpt[Checkpoints]
  Ckpt --> Test[test.py]
  Test --> Eval[evaluate_baseline_outputs.py]
  Eval --> Figures[Thesis figures and tables]
```
图1-4 实验流程源码
"""


def thesis_markdown() -> str:
    ref_lines = "\n".join(f"[{item['id']}]{item['entry']}" for item in REFERENCES)
    return f"""
# 武汉理工大学毕业设计（论文）

论文题目：{TITLE}

学    院：待填写

专业班级：待填写

学生姓名：待填写

指导教师：待填写

## 学位论文原创性声明

本人郑重声明：所呈交的论文是本人在导师指导下独立进行研究所取得的研究成果。除了文中特别加以标注引用的内容外，本论文不包括任何其他个人或集体已经发表或撰写的成果作品。本人完全意识到本声明的法律后果由本人承担。

作者签名：              年    月    日

导师签名：              年    月    日

## 摘要

雾、雨等恶劣天气会改变成像过程中的光照传播和场景可见性，使采集图像出现对比度下降、纹理遮挡、边缘模糊和颜色偏移等现象。上述退化不仅影响人眼观察，也会降低目标检测、场景理解和智能交通等后续视觉任务的可靠性。围绕“基于深度学习在雾雨退化场景下的图像复原算法研究”这一课题，本文在分析雾雨退化机理和典型复原方法的基础上，设计并实现了多尺度天气感知图像复原网络MWIR-Net。该模型以Transformer复原块为主干，采用编码器—解码器结构提取多尺度特征，并在解码阶段引入weather-aware prompt模块和通道注意力，以增强模型对去雾、去雨两类退化模式的适应能力。

本文基于PyTorch与Lightning完成模型训练、推理和评估流程，使用RAIN13K、RESIDE/OTS、SOTS outdoor、Rain100L、Rain100H、Test100、Test1200、Test2800、GT-RAIN-test和nyuhaze500等数据集进行实验。训练中采用Charbonnier像素损失与Sobel边缘一致性约束，推理阶段结合8路旋转翻转TTA自集成。实验结果表明，MWIR-Net在SOTS outdoor去雾任务上取得PSNR 32.04 dB、SSIM 0.9804、LPIPS 0.009871；在Rain100L去雨任务上取得PSNR 33.08 dB、SSIM 0.9442、LPIPS 0.087578。多数据集结果说明，该方法能够在不同雾雨退化场景下实现较稳定的图像质量提升。消融实验显示，在当前二阶段去雨微调协议下，prompt分支和通道注意力的独立增益尚不显著，后续仍需通过更长训练和联合任务评估进一步验证。

关键词：图像复原；去雾；去雨；Transformer；天气感知提示

## Abstract

Images captured in foggy or rainy scenes often suffer from low visibility, texture occlusion, blurred edges and color distortion. Such degradation affects not only visual observation but also the reliability of downstream computer vision systems, including object detection, outdoor monitoring and intelligent transportation. Focusing on image restoration under fog and rain degradation, this thesis studies the physical and data-driven characteristics of adverse-weather images and implements MWIR-Net, a multi-scale weather-aware image restoration network. The proposed model adopts a Transformer-based encoder-decoder backbone. Multi-scale features are extracted by hierarchical restoration blocks, and weather-aware prompts are injected into the decoder stages to adapt the restoration process to different degradation patterns. A prompt channel attention module is further used to recalibrate prompt features before fusion.

The complete experimental pipeline is implemented with PyTorch and Lightning. RAIN13K, RESIDE/OTS, SOTS outdoor, Rain100L, Rain100H, Test100, Test1200, Test2800, GT-RAIN-test and nyuhaze500 are used for training or evaluation. The training objective combines a Charbonnier reconstruction loss with a Sobel edge consistency term, while the inference stage supports an eight-way rotation and flipping test-time augmentation strategy. The restored images are evaluated with PSNR, SSIM and LPIPS under a unified saved-image recomputation protocol. Experimental results show that MWIR-Net achieves 32.04 dB PSNR, 0.9804 SSIM and 0.009871 LPIPS on SOTS outdoor dehazing, and 33.08 dB PSNR, 0.9442 SSIM and 0.087578 LPIPS on Rain100L deraining. Visual comparisons and multi-split evaluations indicate that the method can effectively improve image quality in both haze and rain scenes. The fair ablation experiment also reveals that the independent benefit of the prompt channel attention module is not yet statistically stable under the current two-stage deraining protocol, leaving room for further investigation.

Keywords: image restoration; dehazing; deraining; Transformer; weather-aware prompt

## 目录

第1章 绪论

第2章 雾雨退化机理与图像复原理论基础

第3章 MWIR-Net模型设计

第4章 数据集构建与实验实现

第5章 实验结果与分析

第6章 结论与展望

参考文献

致谢

## 第1章 绪论

### 1.1 研究背景与意义

室外视觉系统长期面对不稳定的自然环境。晴天条件下，摄像设备能够较清晰地记录场景结构、物体边缘和颜色分布；一旦出现雾、雨等天气退化，光线传播路径和传感器接收信号都会发生变化。雾霾会在场景深度方向上引入空气光和透射率衰减，远处区域往往变得发白、低对比且缺少纹理；雨纹和雨滴则会形成局部遮挡与高亮条纹，使边缘、道路标志和细小目标被破坏。这类问题在交通监控、无人驾驶感知、安防巡检和遥感观测中都具有实际影响。如果图像预处理阶段不能有效恢复视觉质量，后续识别模型就可能在输入端积累误差。

雾退化的经典解释通常基于大气散射模型。Narasimhan和Nayar较早从大气参与介质角度分析了天气对视觉成像的影响，指出场景辐射在传播过程中会受到介质散射和空气光叠加的共同作用[1]。在此基础上，暗通道先验方法通过统计无雾自然图像中局部暗像素的分布规律估计透射率，成为单幅图像去雾研究中的代表性传统方法[2]。这类模型具有明确物理含义，也便于解释雾浓度和景深之间的关系。但是，当真实场景中存在非均匀雾、光照变化、天空区域或复杂反射时，单一先验很容易偏离实际分布，导致过度增强、色彩漂移或远景残雾。

雨退化与雾退化不同，它并不总是表现为全局均匀衰减，而更多体现为局部雨纹、雨滴积聚、运动模糊和背景纹理混叠。雨纹在图像中通常具有方向性和半透明性，强雨场景还会伴随水雾和对比度降低。传统滤波或稀疏分解方法可以在特定假设下抑制部分条纹，但面对真实雨强变化、复杂纹理和高频细节时，往往难以区分雨纹与背景边缘。尤其是道路、树枝、建筑线条等结构本身也呈现细长纹理，简单的平滑处理虽然能够去除部分雨线，却会同时损伤原始场景细节。

近年来，深度学习方法推动了图像复原研究的发展。卷积神经网络能够从大量样本中学习退化图像到清晰图像之间的非线性映射，Transformer结构则通过注意力机制增强了长距离依赖建模能力。与传统算法相比，深度模型不必完全依赖人工先验，可以在数据驱动条件下同时学习颜色、纹理和语义层面的复原规律。对于雾雨复合场景而言，这种能力尤为重要，因为实际退化往往并不只包含单一物理因素，而是由空气散射、降雨遮挡、成像噪声和曝光变化共同造成。

本课题的意义主要体现在两个方面。理论层面，本文尝试把多尺度Transformer复原块与天气感知提示机制结合起来，探索模型如何在不同退化类型之间共享基础复原能力，同时保留对雾、雨差异的适应性。工程层面，本文围绕数据准备、训练、推理、指标复算和可视化对比构建了较完整的实验链路，使模型结果能够被重复检查，而不是只停留在单次主观观察。对于本科毕业设计而言，这种从退化机理、模型实现到实验验证的闭环，也有助于体现算法研究的完整过程。

### 1.2 国内外研究现状

单幅图像去雾早期研究多从成像模型和先验假设出发。Tan利用局部对比度最大化增强雾天图像的可见性，为单幅图像去雾提供了直接有效的增强思路[3]。Fattal则从表面阴影和介质透射的统计独立性出发估计场景反射率，使去雾问题具有更明确的图像形成解释[4]。这些方法在无监督或少监督条件下具有一定优势，但对场景先验依赖较强。当天空区域、白色物体、夜间光源或非均匀雾同时出现时，估计结果不稳定的问题较为明显。

卷积神经网络出现以后，去雾方法逐渐从手工先验转向端到端学习。DehazeNet将透射率估计过程映射为CNN特征学习问题，在局部极值映射、多尺度特征和非线性回归方面进行了探索[5]。Ren等提出多尺度卷积网络，从粗尺度估计和细尺度细化两个层面恢复透射率，缓解了单一尺度感受野不足的问题[6]。AOD-Net进一步把透射率和大气光重写为统一变换参数，使网络能够直接输出清晰图像，减少了传统流水线中的中间误差累积[7]。这些工作说明，去雾任务不仅需要全局光照建模，也需要对局部纹理和深度相关信息进行联合学习。

图像去雨研究也经历了从先验分解到深度网络的发展。Yang等把雨纹检测与去除结合起来，构建了深度联合雨检测和复原框架，使模型能够在识别雨区域的同时恢复背景[8]。Fu等提出深度细节网络，强调将高频细节层中的雨纹与背景纹理分离，避免直接在原图上进行过强平滑[9]。Zhang和Patel提出密度感知多流网络，根据雨密度差异组织多分支特征，提升不同雨强下的适应性[10]。这些研究共同表明，去雨模型需要同时处理方向性条纹、局部遮挡和背景纹理保护，单一损失或单尺度网络通常难以覆盖所有情况。

数据集和评价协议对复原算法的可信度同样关键。RESIDE基准提供了合成和真实去雾评价体系，使去雾模型可以在较统一的数据条件下比较[11]。去雨领域常用Rain100L、Rain100H、Test100、Test1200和Test2800等数据集观察不同雨强与场景类型下的性能差异。近年来，一体化图像复原逐渐成为研究热点，模型不再只针对单一噪声或单一天气，而是希望在统一框架内处理多种退化。然而，一体化模型也带来新的挑战：若直接混合多任务数据，模型可能偏向样本量更大或损失更容易下降的任务；若为每类退化单独训练，又会牺牲共享结构的效率。

基于上述现状，本文没有简单复现已有去雾或去雨网络，而是围绕雾雨退化场景设计MWIR-Net。本文关注的问题包括：如何利用多尺度结构兼顾全局雾化和局部雨纹；如何通过weather-aware prompt提示不同退化类型；如何用边缘一致性缓解复原结果过平滑；如何在保存图像口径下统一复算PSNR、SSIM和LPIPS，避免不同日志口径造成的比较偏差。这些问题与任务书要求的“雾雨退化场景分析、深度学习复原模型设计、算法实现与性能验证”保持一致。

### 1.3 研究内容

本文研究内容围绕“分析—设计—实现—验证”四个层次展开。第一，分析雾雨退化的视觉表现和物理差异，说明雾退化中的空气光、透射率衰减和雨退化中的条纹遮挡、雨滴积聚之间的区别，并结合实验数据展示典型退化样例。第二，设计MWIR-Net模型结构。模型采用多尺度编码器—解码器作为主体，通过Transformer复原块提取长距离依赖和局部纹理关系，在解码阶段引入天气感知提示模块，使网络能够根据特征统计生成不同退化相关的prompt。第三，完成训练和推理流程实现。训练阶段采用PyTorch和Lightning组织模型、数据集、优化器、损失函数和日志；推理阶段支持普通测试和8路TTA自集成；评价阶段统一以保存图像为输入复算指标。第四，在去雨和去雾多个数据集上进行定量与定性分析，并与Restormer、PromptIR、MPRNet、AirNet和传统方法进行对比。

从任务完成情况看，本文覆盖了任务书提出的主要要求。数据方面，去雨训练使用RAIN13K，去雾训练使用RESIDE/OTS或ITS相关目录，测试包含Rain100L、Rain100H、Test100、Test1200、Test2800、GT-RAIN-test、SOTS outdoor和nyuhaze500。模型方面，MWIR-Net包含多尺度特征融合、Transformer复原块、weather-aware prompt和通道注意力，并提供zero prompt与no channel attention消融模式。实验方面，本文使用PSNR、SSIM和LPIPS三个指标，同时给出视觉对比图、指标柱状图、训练损失曲线和消融结果。论文正文共安排14幅图，满足任务书不少于12幅图的要求。

### 1.4 技术路线与论文结构

本文技术路线首先从任务书和学校规范出发确定论文目标，再以项目源码和实验记录为证据组织研究内容。具体而言，先对雾雨退化场景进行分类，明确去雾和去雨的共性与差异；随后分析PSNR、SSIM、LPIPS等评价指标的作用范围；在模型设计阶段，以MWIR-Net源码为依据说明网络各模块的数据流、功能和消融开关；在实验阶段，依据训练脚本、测试脚本、指标汇总和输出图像构建可复现的评价链路；最后结合结果讨论模型优势、局限和后续改进方向。

全文结构如下：第1章为绪论，说明研究背景、意义、国内外研究现状和本文主要内容。第2章介绍雾雨退化机理、图像复原评价指标和深度复原方法基础。第3章围绕MWIR-Net展开，说明多尺度编码解码、Transformer复原块、天气感知提示模块、损失函数和TTA推理策略。第4章介绍数据集构建、训练配置、实验实现和可复现性管理。第5章给出去雨、去雾、多数据集泛化、视觉对比和公平消融结果，并对结果进行讨论。第6章总结本文工作，分析不足并提出后续展望。

## 第2章 雾雨退化机理与图像复原理论基础

### 2.1 雾退化成像模型

雾天图像的主要退化来自大气参与介质对场景辐射的散射和吸收。在经典大气散射模型中，观测图像可以表示为清晰场景辐射经过透射率衰减后与空气光叠加的结果：

$$I(x)=J(x)t(x)+A(1-t(x)) \\quad (2-1)$$

其中，I(x)表示观测到的雾天图像，J(x)表示待恢复的清晰图像，A表示全局大气光，t(x)表示透射率。透射率通常与场景深度和介质散射系数相关，距离越远的物体透射率越低，观测图像越容易被空气光覆盖。该模型解释了雾天图像远景发白、对比度下降和边缘不清晰的现象，也说明去雾问题本质上需要同时估计场景辐射和介质影响。

然而，真实雾天图像往往比公式更复杂。第一，雾浓度可能在空间上不均匀，例如山地、城市道路和水面附近会出现局部浓雾。第二，天空区域缺少明显纹理，传统暗通道先验容易误判透射率。第三，夜间和强光条件下，大气光不再简单服从全局常量假设，车灯、路灯和反光表面会造成局部异常亮度。第四，合成雾数据与真实雾图像之间存在域差异，模型如果只在合成数据上训练，可能在真实场景中出现颜色偏移。正因如此，本文在实验中同时使用SOTS outdoor和nyuhaze500等数据观察模型表现，避免只凭单一数据集得出结论。

[此处插入截图：图2-1 雾雨退化图像示例]

图2-1 雾雨退化图像示例

图2-1展示了去雨和去雾样例的输入与清晰目标。可以看到，雨图像中的退化主要表现为局部条纹遮挡和背景纹理干扰；雾图像则表现为全局对比度下降和远处区域空气光增强。两类退化虽然都降低了图像质量，但空间分布、频率特征和视觉损伤方式并不相同。因此，若模型只依赖固定滤波或单一尺度特征，就很难同时处理这两类任务。

### 2.2 雨退化特点

雨退化通常可以分为雨纹、雨滴、水雾和雨幕等形式。雨纹多表现为细长、高亮、方向一致或近似一致的线状结构；雨滴附着在镜头或玻璃上时，会产生局部模糊、透镜效应和不规则遮挡；强降雨还可能形成类似雾化的雨幕，使远处区域对比度降低。与雾不同，雨纹具有较强的局部高频特征，很多雨线与背景边缘在频域上会发生重叠。这意味着去雨模型必须判断某一条高频结构究竟属于真实物体边缘，还是属于需要去除的雨纹。

从学习任务角度看，去雨存在两个难点。其一，雨纹与背景细节之间的边界并不稳定。例如树枝、栏杆和建筑窗线与雨纹形态相近，若模型过度平滑，会损伤真实结构；若模型过于保守，又会残留雨纹。其二，不同数据集中的雨强、雨纹方向和合成方式不同，模型在Rain100L上表现较好，并不必然代表其在Rain100H或真实雨场景上同样稳定。本文因此将Rain100L、Rain100H、Test100、Test1200、Test2800和GT-RAIN-test分开评价，并在结果分析中区分轻雨、重雨和真实雨场景。

对于雾雨复原而言，一个合理模型需要同时具备全局调节和局部辨别能力。雾图像需要恢复大范围的亮度、颜色和对比度，雨图像需要去除细粒度条纹并保护边缘。多尺度编码器—解码器能够在低分辨率层捕获全局上下文，在高分辨率层保留空间细节；Transformer注意力能够建模远距离相关性；prompt机制则可以为不同退化类型提供条件化调节。MWIR-Net的设计正是围绕这三个需求展开。

### 2.3 图像复原评价指标

图像复原结果既需要主观观察，也需要客观指标。本文采用PSNR、SSIM和LPIPS三类指标。PSNR以均方误差为基础，反映恢复图像与目标图像之间的像素差异，其计算形式为：

$$PSNR=10\\log_{{10}}\\frac{{MAX_I^2}}{{MSE}} \\quad (2-2)$$

其中，MAX_I为像素最大值，MSE为恢复图像与目标图像的均方误差。PSNR越高，通常表示像素级误差越小。该指标直观、易计算，也便于与已有论文对比。但PSNR对感知质量不够敏感，两张图像即使具有相近PSNR，人眼观察到的纹理、边缘和颜色自然度也可能不同。

SSIM从亮度、对比度和结构三个方面衡量图像相似性，更接近人眼对结构信息的感知[12]。在图像复原中，SSIM能够反映边缘和纹理是否被较好保留。LPIPS则利用深度网络特征距离衡量感知差异，数值越低通常代表感知质量越接近目标图像[13]。本文在指标汇总中统一采用保存图像复算口径，即先将模型输出保存为图像，再与目标图像按共同尺寸中心裁剪并裁剪到16的整数倍后计算指标。这样可以避免训练日志中张量口径与保存图像口径之间的轻微差异影响模型比较。

表2-1 本文主要评价指标

| 指标 | 含义 | 优化方向 |
|---|---|---|
| PSNR | 基于均方误差的像素级相似度 | 越高越好 |
| SSIM | 从亮度、对比度和结构衡量图像相似性 | 越高越好 |
| LPIPS(alex) | 基于深度特征的感知距离 | 越低越好 |

需要强调的是，三类指标并不能完全替代人工观察。对于去雨任务，较高PSNR可能来自平滑背景，但视觉上仍可能残留细雨纹；对于去雾任务，较高SSIM并不一定意味着颜色完全自然。因此，本文在第5章同时给出视觉对比图，并结合指标讨论模型在不同数据集上的优势和不足。

### 2.4 深度复原方法基础

深度图像复原模型通常由特征提取、退化建模、图像重建和损失约束几部分组成。卷积网络擅长提取局部纹理，对边缘和小范围结构具有较强归纳偏置；注意力机制能够根据输入动态建立远距离关系。Transformer最初在序列建模中取得突出效果，其核心是自注意力机制对全局依赖的建模[14]。视觉Transformer进一步证明，图像可以被划分为patch并通过注意力结构进行高层语义和空间关系学习[15]。不过，直接将标准Transformer用于高分辨率图像复原会带来较大的计算开销，因此图像复原领域常结合卷积、窗口注意力、通道注意力和多尺度结构进行改造。

通道注意力的基本思想是根据全局上下文为不同通道分配权重。Squeeze-and-Excitation网络通过全局池化和轻量多层感知机实现通道重标定，为后续大量视觉模型提供了简单有效的注意力形式[16]。在雾雨复原中，不同通道可能对应颜色、边缘、纹理或退化提示等特征，通道注意力有助于增强有用特征并抑制干扰。MWIR-Net中的PromptChannelAttention借鉴了这一思想，对prompt特征进行自适应重标定。

近年来，多阶段和Transformer复原模型不断提升图像复原性能。MPRNet通过多阶段渐进式结构逐步恢复图像质量，说明复杂退化可以通过分阶段细化获得更稳定结果[17]。Restormer针对高分辨率图像复原设计了高效Transformer结构，在多类低层视觉任务上表现突出[18]。AirNet尝试在未知退化条件下学习多退化表征，代表了一体化复原方向[19]。TransWeather面向恶劣天气图像复原，将Transformer用于多天气退化恢复[20]。PromptIR进一步提出利用prompt进行全盲一体化复原，为本文的weather-aware prompt设计提供了直接启发[21]。这些研究表明，未来图像复原模型不仅要追求单项指标，还需要在多退化、多数据集和可解释调节能力之间取得平衡。

### 2.5 本章小结

本章从雾退化成像模型、雨退化特点、评价指标和深度复原方法四个方面介绍了本文的理论基础。雾退化更强调全局空气光和透射率估计，雨退化更强调局部条纹去除和背景细节保护。PSNR、SSIM和LPIPS分别从像素、结构和感知特征层面对结果进行评价。深度学习方法，特别是多尺度结构、Transformer和prompt机制，为雾雨联合复原提供了有效技术路径。下一章将在这些基础上介绍MWIR-Net的具体结构设计。

## 第3章 MWIR-Net模型设计

### 3.1 总体设计目标

MWIR-Net的全称为Multi-scale Weather-aware Image Restoration Network，面向去雨和去雾两类天气退化图像复原任务。根据源码实现，模型输入为三通道RGB退化图像，输出为同尺寸三通道复原图像。网络最后采用残差形式，将预测残差与输入图像相加得到最终结果。这种设计有助于模型集中学习退化成分和清晰图像之间的差异，而不必完全重建所有低频内容。

模型总体设计目标包括三点。第一，要具备多尺度表达能力。雾退化包含大范围亮度和颜色变化，雨退化包含局部高频条纹，单一尺度难以兼顾两者。第二，要具备天气条件感知能力。去雨和去雾虽然都属于图像复原，但退化形态不同，模型需要在共享主干基础上根据输入特征动态调整复原策略。第三，要具备可验证的消融能力。为了判断weather-aware prompt和通道注意力是否有效，源码中设置了full、zero prompt和no channel attention三种模式，使实验能够只改变特定结构分支。

[此处插入截图：图3-1 MWIR-Net总体结构]

图3-1 MWIR-Net总体结构

从图3-1可以看出，MWIR-Net采用编码器—解码器结构。输入图像首先经过重叠patch embedding映射到特征空间，随后进入三层编码器和latent层。解码阶段逐步上采样并与对应编码层特征融合，同时在多个尺度注入weather-aware prompt。最后，refinement blocks进一步细化复原结果，输出层生成残差图像。

### 3.2 多尺度编码器—解码器

MWIR-Net的多尺度结构主要由Downsample、Upsample和不同层级的TransformerBlock组成。编码阶段，level1保持较高空间分辨率，适合保留边缘和雨纹位置；level2与level3在下采样后扩大感受野，适合捕获更大范围的上下文；latent层在最低分辨率上进行深层特征建模，能够覆盖较远区域之间的关系。解码阶段，模型通过上采样逐步恢复空间分辨率，并将编码阶段的特征通过跳跃连接引入，以减少细节丢失。

这种结构对雾雨复原具有实际意义。对于去雾任务，低分辨率层可以学习全局雾浓度和空气光影响，高分辨率层则负责恢复局部纹理和边缘。对于去雨任务，高分辨率层有助于定位细雨纹，低分辨率层可以判断雨纹与背景结构之间的上下文关系。跳跃连接使编码阶段的细节信息能够直接参与解码，降低深层网络带来的空间信息损耗。

源码中，Downsample通过卷积和PixelUnshuffle降低空间分辨率并提升通道数，Upsample通过卷积和PixelShuffle恢复空间尺寸。这种实现方式避免了简单插值带来的特征表达不足，也使模型能够在通道维度上重新组织局部邻域信息。各层通道数随尺度变化而增加，符合“浅层保细节、深层建语义”的低层视觉模型设计习惯。

### 3.3 Transformer复原块

MWIR-Net中的TransformerBlock由两部分组成：多头转置自注意力模块和门控深度卷积前馈网络。每一部分前都使用LayerNorm，并通过残差连接保持梯度稳定。注意力模块首先使用1×1卷积和深度卷积生成Q、K、V特征，再在通道头维度上计算注意力；前馈网络通过1×1卷积扩展通道后使用深度卷积和GELU门控，再投影回原始维度。这种设计兼顾了全局依赖建模和局部邻域建模。

[此处插入截图：图3-2 Transformer复原块结构]

图3-2 Transformer复原块结构

图3-2展示了Transformer复原块的基本信息流。与分类任务中的标准Transformer不同，图像复原要求输出保持像素级空间结构，因此模型不能过度破坏局部位置信息。深度卷积在QKV生成和前馈网络中引入局部归纳偏置，注意力模块则负责建模更远区域之间的关系。对于雨纹去除，这有助于判断某一线状结构是否与周围背景一致；对于去雾，这有助于利用远近区域的颜色和亮度关系恢复全局自然度。

在实现上，模型使用WithBias LayerNorm作为默认归一化方式。归一化有助于稳定不同批次和不同退化类型下的特征分布。残差连接使网络可以保留输入特征，避免在深层堆叠时出现过度平滑。由于本文训练数据同时包含去雨和去雾，稳定的归一化和残差结构对多任务训练尤为重要。

### 3.4 天气感知提示模块

天气感知提示模块是MWIR-Net区别于普通编码器—解码器复原网络的重要部分。WeatherPromptBlock内部维护一个可学习prompt字典，字典形状由prompt长度、prompt维度和prompt空间尺寸共同决定。输入特征先在空间维度上做平均池化，得到全局退化表征；随后通过线性层和Softmax生成prompt权重；不同prompt按权重加权求和后，被插值到当前特征分辨率，再经过3×3卷积和通道注意力处理。最后，prompt特征与解码特征拼接，并通过TransformerBlock和1×1卷积完成融合。

[此处插入截图：图3-3 天气感知提示模块结构]

图3-3 天气感知提示模块结构

该模块的核心思想是：模型不直接预设输入一定是雾或雨，而是从当前特征统计中生成退化相关提示。对于雾图像，prompt可能更关注大范围亮度、颜色和低频对比度；对于雨图像，prompt可能更关注局部高频遮挡和边缘保护。通道注意力用于对prompt通道进行重标定，使不同提示成分能够根据输入动态调整。源码中的ablation mode提供了两种消融：zero prompt直接输出零prompt，用于观察prompt分支整体影响；no channel attention保留prompt但关闭通道注意力，用于观察通道重标定的独立作用。

从实验结果看，当前公平消融并未证明通道注意力在二阶段去雨微调中带来稳定显著的独立增益。这一结果并不否定prompt设计本身，而是说明在给定训练轮数、数据规模和随机种子的条件下，模块贡献需要谨慎表述。本文在第5章将消融结果作为模型分析的一部分，而不是简单宣称所有结构改造都一定有效。

### 3.5 损失函数与推理策略

训练阶段，MWIR-Net支持L1损失和Charbonnier损失。Charbonnier损失可以看作平滑的L1形式，对异常误差相对更稳健。本文主要二阶段实验采用Charbonnier像素损失，并加入Sobel边缘一致性约束。联合损失可表示为：

$$\\mathcal{{L}}=\\mathcal{{L}}_{{char}}(\\hat{{I}},I)+\\lambda\\mathcal{{L}}_{{edge}}(\\hat{{I}},I) \\quad (3-1)$$

其中，\\(\\hat{{I}}\\)表示模型复原图像，I表示清晰目标图像，\\(\\lambda\\)为边缘损失权重。Sobel边缘损失通过水平和垂直梯度比较恢复图像与目标图像的边缘一致性。对于去雨任务，边缘约束有助于缓解雨纹去除后的背景模糊；对于去雾任务，边缘约束有助于保留物体轮廓和场景层次。

优化器采用AdamW。AdamW通过将权重衰减从梯度更新中解耦，提高了带权重衰减训练的稳定性[22]。学习率调度采用线性warmup加余弦退火。warmup能够避免训练初期学习率过大导致的震荡，余弦退火则有助于后期收敛。训练脚本支持指定seed、batch size、patch size、混合精度、训练轮数和checkpoint初始化路径，为不同阶段实验提供统一入口。

推理阶段，MWIR-Net支持普通推理和TTA自集成。TTA策略对输入图像进行四种旋转及其水平翻转，共得到8个增强输入；模型分别复原后再反变换回原方向并求平均。该方法不会改变模型参数，但可以降低单次推理对方向和局部纹理的偶然敏感性。

[此处插入截图：图3-4 TTA自集成推理流程]

图3-4 TTA自集成推理流程

TTA会增加推理时间，但在图像复原任务中常能带来小幅稳定提升。本文在Rain100L和SOTS outdoor等主结果中记录了TTA模型表现，同时也保留普通推理结果，便于分析推理策略对指标的影响。

### 3.6 本章小结

本章介绍了MWIR-Net的总体结构和关键模块。模型通过多尺度编码器—解码器组织全局和局部信息，通过Transformer复原块建模长距离依赖，通过天气感知提示模块适应雾雨退化差异，并通过Charbonnier损失、Sobel边缘一致性和TTA推理提升复原质量。下一章将进一步说明数据集、训练协议、实验环境和评价流程。

## 第4章 数据集构建与实验实现

### 4.1 数据来源与划分

本文实验数据覆盖去雨和去雾两类任务。去雨训练数据主要来自RAIN13K训练集，项目脚本将雨图与清晰图组织为`data/Train/Derain/rainy`和`data/Train/Derain/gt`，并在`data_dir/rainy/rainTrain.txt`中记录训练列表。该列表包含13711条记录，能够为去雨训练提供较丰富的雨纹样本。去雾训练数据来自RESIDE/OTS或ITS相关目录，当前主要实验使用OTS来源生成`data/Train/Dehaze/synthetic`和`data/Train/Dehaze/original`，并在`data_dir/hazy/hazy_outside.txt`中记录72135条雾图训练记录。

测试数据方面，去雨任务包括Rain100L、Rain100H、Test100、Test1200、Test2800和GT-RAIN-test。Rain100L相对偏轻雨，Rain100H雨纹更复杂，Test1200和Test2800用于观察更大规模测试集上的稳定性，GT-RAIN-test包含真实雨场景，能够补充合成雨数据的不足。去雾任务包括SOTS outdoor和nyuhaze500。SOTS outdoor主要用于室外合成雾评价，nyuhaze500则用于观察模型在不同合成策略和场景分布下的泛化表现。

[此处插入截图：图4-1 GT-RAIN真实雨图像样例]

图4-1 GT-RAIN真实雨图像样例

图4-1展示了GT-RAIN中的真实雨样例。真实雨图像与合成雨图像相比，常包含更复杂的光照变化、背景运动和雨滴形态，因此指标表现通常低于标准合成数据集。本文在结果分析中不会把GT-RAIN-test低指标简单视为模型失败，而是将其作为真实场景泛化能力的边界观察。

### 4.2 数据预处理与配对方式

MWIR-Net的数据读取由`utils/dataset_utils.py`实现。训练集根据退化类型组织样本：当样本类型为derain时，输入图来自rainy目录，目标图通过文件名规则映射到gt目录；当样本类型为dehaze时，输入图来自synthetic目录，目标图根据雾图文件名前缀映射到original目录。训练阶段对输入图和目标图进行同步随机裁剪和数据增强，裁剪尺寸由`patch_size`参数控制，当前主要训练协议使用128×128 patch。

对去雨任务而言，输入和目标通常同名或具有固定前缀映射关系。对去雾任务而言，合成雾图文件名中往往包含透射率或大气光参数，目标图则只保留清晰图编号。测试集读取时，脚本会根据任务类型自动匹配target路径，并在预测图与目标图尺寸不一致时进行中心裁剪，确保PSNR、SSIM和LPIPS计算时尺寸一致。指标汇总文档进一步说明，最终复算还会将图像裁剪到16的整数倍，以适配模型下采样结构和不同输出目录之间的比较。

数据预处理的一个重要原则是：训练、推理和指标复算必须使用一致口径。如果只引用测试脚本打印的PSNR和SSIM，可能会因为张量计算和保存图像计算之间的差别造成轻微偏差。本文采用保存图像统一复算结果作为最终表格口径，这样不同模型、不同输出目录和不同推理策略之间具有更好的可比性。

### 4.3 实验环境与训练协议

本文实验环境依据项目`env.yml`和实际脚本整理，如表4-1所示。由于本地环境可能随驱动和CUDA版本变化，表中仅列出与论文复现实验密切相关的主要工具。

表4-1 实验环境与工具

| 项目 | 配置 |
|---|---|
| 编程语言 | Python |
| 深度学习框架 | PyTorch、Lightning |
| 图像处理 | Pillow、torchvision、scikit-image |
| 指标计算 | PSNR、SSIM、LPIPS(alex) |
| 可视化 | matplotlib |
| 文档生成 | python-docx |

训练协议主要分为联合训练、二阶段微调和公平消融三类。联合训练阶段从去雨和去雾训练样本中各取最多5000张，使用12个epoch、batch size 32、patch size 128、学习率2e-4、warmup 2个epoch和0.05边缘损失权重。二阶段去雨微调从兼容公开复原权重初始化后的联合模型继续训练，使用去雨单任务、8个epoch、学习率1e-5、warmup 1个epoch、Charbonnier损失和0.02边缘损失权重。去雾二阶段微调采用类似设置，但训练轮数为4个epoch。公平消融实验从同一初始化权重出发，固定训练数据、训练轮数、学习率和损失，只改变ablation mode。

表4-2 主要训练协议

| 阶段 | 任务 | 关键参数 | 用途 |
|---|---|---|---|
| MWIR-Net-5k_12epoch | 去雨+去雾 | epoch=12，batch=32，lr=2e-4，edge=0.05 | 从零训练基线 |
| MWIR-Net-5k_12epoch_init | 去雨+去雾 | 兼容权重初始化，其他同5k协议 | 主干初始化对比 |
| MWIR-Net-stage2_charb_edge002 | 去雨 | epoch=8，lr=1e-5，Charbonnier，edge=0.02 | 主要去雨模型 |
| MWIR-Net-dehaze_stage2_charb_edge002 | 去雾 | epoch=4，Charbonnier，edge=0.02 | 主要去雾模型 |
| Fair-ablation | 去雨 | seed=0/1/2，ablation mode变化 | 模块贡献分析 |

训练过程使用AdamW优化器和线性warmup加余弦退火调度。日志由Lightning CSVLogger保存，checkpoint按阶段存放在本地checkpoints目录。由于模型训练和输出图像体积较大，数据集、checkpoint、outputs和logs均作为本地实验材料管理，不纳入论文正文之外的版本提交范围。

[此处插入截图：图4-2 训练损失变化曲线]

图4-2 训练损失变化曲线

图4-2显示了部分训练日志中的loss变化。联合训练阶段的损失在训练过程中整体下降，二阶段去雾微调的损失较低且变化范围更小，说明初始化权重和任务聚焦对收敛有帮助。需要说明的是，训练loss只能反映训练集上的优化趋势，不能直接等同于测试集PSNR、SSIM或LPIPS，因此第5章仍以保存图像复算结果作为主要结论依据。

### 4.4 推理与评价流程

推理由`test.py`完成。去雨模式使用`--mode 1`，去雾模式使用`--mode 2`，并通过`--derain_splits`或`--dehaze_splits`指定测试集。输出目录按任务和数据集名称组织，例如Rain100L对应`derain`目录，Rain100H对应`derain_Rain100H`目录，SOTS outdoor对应`dehaze_outdoor`目录。若启用`--tta`，脚本会对每张输入图进行8路旋转翻转增强，恢复后反变换并求平均。

[此处插入截图：图4-3 实验训练与评价流程]

图4-3 实验训练与评价流程

评价阶段使用`tools/evaluate_baseline_outputs.py`和`tools/evaluate_lpips.py`等脚本复算指标。去雨任务通常按同名文件匹配预测图与目标图；去雾任务按预测图文件名前缀匹配清晰图编号。复算时，预测图和目标图先转换为RGB，再按共同尺寸中心裁剪，并裁剪到16的整数倍。PSNR和SSIM由scikit-image计算，LPIPS使用alex backbone。本文所有对比表均采用这一保存图像口径，保证不同模型之间的比较尽可能公平。

### 4.5 可复现性与证据管理

为了避免论文结果与代码证据脱节，本文将模型结构、训练协议和指标结果分别绑定到具体文件。模型结构来自`net/mwirnet.py`，训练损失和优化器来自`train.py`和logs目录，推理策略来自`test.py`，数据读取来自`utils/dataset_utils.py`，指标结果来自`docs/所有模型指标汇总.md`。图件由本次隔离工作区重新生成，视觉对比图直接读取本地测试集、模型输出和对比方法输出，不使用旧的论文输出目录。

这种证据管理方式有两个好处。第一，正文中的事实可以追溯到源码或实验记录，减少凭记忆改写造成的错误。第二，当后续补充新实验或导师提出修改意见时，可以较清楚地判断某一结论应修改正文、图表还是实验记录。例如，若未来补充full模式的3-seed公平消融，就应同步更新第5章消融表、图5-6、figure-registry和文献或实验核验清单。

### 4.6 本章小结

本章介绍了MWIR-Net实验所用数据集、预处理方式、训练协议、推理评价流程和证据管理方法。本文实验覆盖合成雨、真实雨、室外合成雾和补充去雾测试集，使用统一保存图像口径计算PSNR、SSIM和LPIPS。训练过程采用联合训练、二阶段微调和公平消融相结合的方式，为后续结果分析提供了基础。

## 第5章 实验结果与分析

### 5.1 Rain100L去雨结果

Rain100L是去雨任务中常用的轻雨测试集，图像数量为100张。根据统一复算结果，MWIR-Net-final_tta_multisplit在Rain100L上取得PSNR 33.08 dB、SSIM 0.9442、LPIPS 0.087578。与传统中值滤波相比，MWIR-Net在PSNR、SSIM和LPIPS上均有明显提升，说明深度模型能够更有效地识别雨纹并保留背景结构。与从零训练的AirNet-5k-12epoch相比，MWIR-Net二阶段模型也取得了更好的像素和结构指标。

表5-1 Rain100L去雨结果对比

| 模型 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Median-filter | 24.38 | 0.7327 | 0.339617 |
| AirNet-5k-12epoch | 23.31 | 0.7928 | 0.267186 |
| MWIR-Net-final_tta_multisplit | 33.08 | 0.9442 | 0.087578 |
| MPRNet-official | 34.95 | 0.9589 | 0.073253 |
| PromptIR-official | 37.32 | 0.9778 | 0.016323 |
| Restormer-official | 37.57 | 0.9741 | 0.042389 |

[此处插入截图：图5-1 Rain100L PSNR指标对比]

图5-1 Rain100L PSNR指标对比

从表5-1和图5-1可以看到，官方预训练的Restormer和PromptIR仍然是Rain100L上的强基线。Restormer的PSNR达到37.57 dB，PromptIR的SSIM和LPIPS表现也明显优于本文模型。这说明MWIR-Net虽然完成了较完整的雾雨复原实验链路，但在轻雨标准数据集上距离顶级公开预训练模型仍有差距。造成差距的原因可能包括训练数据规模、预训练策略、模型容量和训练轮数。本文更稳妥的结论是：MWIR-Net相对于传统方法和本地从零训练基线有明显提升，但还不能宣称全面超过公开强基线。

### 5.2 去雾结果分析

去雾任务中，SOTS outdoor是本文主要评价数据集。MWIR-Net-stage2_charb_edge002_tta_dehaze在该数据集上取得PSNR 32.04 dB、SSIM 0.9804、LPIPS 0.009871，是当前工作区记录中三个指标均最优的模型。与PromptIR-official相比，MWIR-Net在PSNR上高出约1.69 dB，在SSIM和LPIPS上也有优势。与AirNet官方权重和传统CLAHE相比，差距更加明显。

[此处插入截图：图5-2 SOTS outdoor PSNR指标对比]

图5-2 SOTS outdoor PSNR指标对比

图5-2显示，传统CLAHE虽然能够提高局部对比度，但难以准确去除雾退化产生的空气光和颜色偏移；AirNet官方权重具有一定复原能力，但在本地SOTS outdoor复算口径下低于MWIR-Net；PromptIR官方权重表现较强，但仍低于本文SOTS outdoor主模型。该结果说明，在当前训练协议和测试数据下，MWIR-Net的去雾二阶段微调较好地适应了室外合成雾场景。

在nyuhaze500上，MWIR-Net-final_plain_multisplit_dehaze取得PSNR 17.20 dB、SSIM 0.8239、LPIPS 0.101394。该结果明显低于SOTS outdoor，说明不同去雾测试集之间存在分布差异。nyuhaze500可能包含不同深度、雾浓度或合成方式，模型从SOTS outdoor获得的优势不能直接迁移到所有去雾场景。因此，本文将nyuhaze500作为泛化边界讨论，而不把单一数据集成绩作为绝对结论。

### 5.3 多数据集泛化分析

为了观察MWIR-Net在不同去雨数据集上的稳定性，本文汇总了Rain100L、Rain100H、Test100、Test1200、Test2800和GT-RAIN-test结果。表5-2给出主要数据集的图像数量和指标。

表5-2 MWIR-Net多数据集结果

| 任务 | 数据集 | 图像数 | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| 去雨 | Rain100L | 100 | 33.08 | 0.9442 | 0.087578 |
| 去雨 | Rain100H | 100 | 25.05 | 0.7800 | 0.239191 |
| 去雨 | Test100 | 98 | 23.77 | 0.8002 | 0.165610 |
| 去雨 | Test1200 | 1200 | 30.03 | 0.8702 | 0.090416 |
| 去雨 | Test2800 | 2800 | 30.66 | 0.9078 | 0.057636 |
| 去雨 | GT-RAIN-test | 2100 | 21.03 | 0.5963 | 0.293823 |
| 去雾 | SOTS outdoor | 500 | 32.04 | 0.9804 | 0.009871 |
| 去雾 | nyuhaze500 | 500 | 17.20 | 0.8239 | 0.101394 |

[此处插入截图：图5-3 MWIR-Net多去雨数据集PSNR结果]

图5-3 MWIR-Net多去雨数据集PSNR结果

从表5-2可以看出，MWIR-Net在Rain100L、Test1200和Test2800上取得较高PSNR，而在Rain100H、Test100和GT-RAIN-test上指标下降明显。这种差异与数据集难度和分布有关。Rain100H包含更复杂的雨纹和更强遮挡，GT-RAIN-test是真实雨场景，背景、光照和雨滴形态都更接近真实采集条件。模型在这些数据上的指标下降说明，当前训练协议仍主要适应合成雨分布，对真实雨场景的泛化能力有待提升。

另一方面，Test1200和Test2800的大规模结果说明，MWIR-Net在部分合成去雨数据上具有较稳定表现。Test2800的LPIPS为0.057636，低于Rain100L的0.087578，说明在该数据集上复原图像与目标图像在感知特征上较接近。不同指标之间并不总是完全一致，因此需要结合数据集特点解释，而不能只看单一PSNR排序。

### 5.4 主观视觉质量分析

客观指标能够给出整体统计，但图像复原最终仍要回到视觉质量。图5-4给出Rain100L去雨视觉对比。输入图像中存在明显雨线，传统或弱模型容易残留条纹，强模型能够更好地恢复背景纹理。MWIR-Net的输出相对于输入图像去除了大量雨纹，背景结构也较清晰，但在部分细节处仍不如PromptIR或Restormer自然。

[此处插入截图：图5-4 Rain100L去雨视觉对比]

图5-4 Rain100L去雨视觉对比

图5-5给出SOTS outdoor去雾视觉对比。雾天输入图像整体偏灰、对比度不足，MWIR-Net输出在颜色和边缘方面有明显改善。与PromptIR官方输出相比，MWIR-Net在部分样例中能够得到更高对比度和更接近目标图像的亮度层次，但也需要注意，过强增强可能在真实雾图上造成颜色偏移。因此，去雾结果既要关注PSNR和SSIM，也要观察天空、远景和高亮区域是否自然。

[此处插入截图：图5-5 SOTS outdoor去雾视觉对比]

图5-5 SOTS outdoor去雾视觉对比

主观对比还揭示了一个现象：TTA自集成能够在一定程度上稳定纹理恢复，但不能从根本上弥补模型训练不足。对于雨纹密集或真实雨滴遮挡的样例，模型仍可能出现残留雨线或细节涂抹。对于去雾样例，模型对合成雾恢复较好，但面对复杂真实光照时可能需要更强的域适应或真实数据训练。

### 5.5 消融实验分析

为了分析weather-aware prompt和通道注意力的作用，本文采用公平消融实验。消融实验从同一初始化权重`MWIR-Net-5k_12epoch_init/epoch=11-step=3744.ckpt`出发，在Rain100L去雨任务上使用相同训练数据、训练轮数、学习率和损失函数，仅改变ablation mode。当前主结果采用8 epoch、seed 0/1/2三次重复的均值和标准差。

表5-3 Rain100L公平消融结果

| 模式 | PSNR mean±std | SSIM mean±std | LPIPS mean±std |
|---|---:|---:|---:|
| zero prompt | 32.7213±0.0404 | 0.941258±0.000583 | 0.087695±0.001282 |
| no channel attention | 32.7302±0.0425 | 0.941250±0.000573 | 0.087472±0.001323 |

[此处插入截图：图5-6 Rain100L公平消融实验结果]

图5-6 Rain100L公平消融实验结果

表5-3和图5-6表明，no channel attention在PSNR和LPIPS均值上略优，zero prompt在SSIM均值上极小幅领先，但差异均小于对应标准差。这意味着当前实验不能支持“通道注意力必然带来稳定提升”的结论。更稳妥的解释是：在二阶段去雨微调、8个epoch和3个seed条件下，prompt分支和通道注意力的独立贡献尚不显著，结构模块可能需要更长训练、更多任务联合评价或与full模式3-seed结果一起分析。

这一结果也体现了消融实验的重要性。若只观察早期2 epoch单次实验，可能会得到“去除通道注意力后下降”的趋势性印象；但在补充3-seed后，跨seed波动证明该趋势并不稳定。因此，论文在表述模型贡献时应避免夸大模块效果，而应把可复现实验结果作为主要依据。

### 5.6 结果讨论与不足

综合实验结果，MWIR-Net的优势主要体现在三个方面。第一，模型在SOTS outdoor去雾任务上表现突出，说明多尺度结构、二阶段微调和TTA自集成对室外合成雾复原有效。第二，模型在Rain100L、Test1200和Test2800上取得较稳定去雨结果，相对于传统滤波和本地从零训练基线有明显提升。第三，项目形成了较完整的训练、推理、复算和可视化链路，能够支持论文中的主要事实追溯。

不足也较明显。第一，MWIR-Net在Rain100L上仍落后于Restormer和PromptIR官方预训练强基线，说明模型容量、预训练数据或训练策略仍有改进空间。第二，GT-RAIN-test和nyuhaze500结果偏低，表明模型对真实雨和跨分布去雾场景的泛化能力不足。第三，公平消融尚未包含full模式的3-seed完整结果，因此无法全面判断prompt模块、通道注意力和完整结构之间的关系。第四，本文实验主要关注全参考指标，尚未加入无参考质量评价和下游视觉任务验证。

因此，本文结论需要保持边界：MWIR-Net是一个完成度较高的雾雨图像复原实验模型，在部分数据集上取得较好结果，并形成了可复现研究链路；但它并不是所有去雨、去雾数据集上的最优方法，也不能仅凭当前消融证明每个结构分支都有稳定独立增益。

## 第6章 结论与展望

### 6.1 研究总结

本文围绕雾雨退化场景下的图像复原问题，完成了从理论分析、模型设计、实验实现到结果评价的本科毕业论文研究。首先，本文分析了雾退化的大气散射模型和雨退化的局部条纹遮挡特点，说明两类天气退化在空间分布、频率特征和视觉损伤方式上的差异。其次，本文设计并实现了MWIR-Net模型。该模型以多尺度Transformer编码器—解码器为主体，在解码阶段注入weather-aware prompt，并使用通道注意力对prompt特征进行重标定。再次，本文基于PyTorch和Lightning完成训练、推理、指标复算和图件生成流程，构建了去雨、去雾多数据集实验。

实验结果表明，MWIR-Net在SOTS outdoor去雾任务上取得PSNR 32.04 dB、SSIM 0.9804、LPIPS 0.009871；在Rain100L去雨任务上取得PSNR 33.08 dB、SSIM 0.9442、LPIPS 0.087578。多数据集结果显示，模型在部分合成去雨数据上具有较好稳定性，但在真实雨和跨分布去雾场景上仍存在明显下降。公平消融结果显示，在当前二阶段去雨微调协议下，zero prompt和no channel attention之间的均值差异小于seed波动，通道注意力的独立贡献尚不能被稳定证明。

### 6.2 主要不足

本文仍存在若干不足。第一，训练数据和训练轮数有限，模型在Rain100L上与Restormer、PromptIR等官方预训练强基线仍有差距。第二，当前真实雨和跨数据集去雾泛化能力不足，说明模型对真实天气分布的建模还不充分。第三，消融实验仍需补充完整full模式多seed结果，并进一步分析prompt长度、prompt维度、注入层级和通道注意力强度对性能的影响。第四，论文主要采用全参考指标，尚未将复原结果输入目标检测、语义分割等下游任务，因而无法直接证明模型对实际视觉系统的收益。

此外，本文结果还受到实验资源和数据口径的限制。部分公开方法采用官方预训练权重，而MWIR-Net主要基于本地训练协议完成训练和微调，两类模型在训练数据规模、训练时长和先验经验上并不完全等价。因此，本文在对比时更强调“当前工作区统一复算口径下的结果”，而不是把所有表格解释为绝对公平的模型能力排名。对于本科毕设而言，这种限制并不影响算法设计、实现和验证链路的完整性，但在后续科研工作中，需要进一步统一预训练条件、训练预算和数据划分，才能对结构优劣作出更严格判断。

### 6.3 后续展望

后续工作可从四个方向展开。第一，扩大真实天气数据训练比例，引入真实雨滴、夜间雾、非均匀雾和多天气混合场景，提高模型对真实分布的适应能力。第二，完善一体化复原训练策略，通过任务均衡采样、退化类型识别或对比学习，缓解多任务训练中的偏置问题。第三，继续优化prompt机制，比较不同prompt数量、空间尺寸和注入位置，并补充full模式多seed公平消融。第四，将图像复原与下游视觉任务结合，评估复原前后目标检测、车道线识别或行人识别性能变化，使研究结论更贴近实际应用。

总体而言，MWIR-Net完成了雾雨退化图像复原算法的设计、实现和验证，达到了毕业设计任务书对模型构建、数据集实验、指标评价、对比分析和论文图表的基本要求。本文也认识到，图像复原算法的有效性不能只依赖单个数据集或单次实验，需要在更广泛数据、更严格消融和更贴近应用的评价中持续检验。

## 参考文献

{ref_lines}

## 致谢

本论文的完成离不开指导教师在选题方向、实验设计和论文规范方面的指导，也离不开同学在环境配置、数据整理和实验讨论中的帮助。通过本次毕业设计，我系统学习了雾雨退化图像复原的基本理论，完成了深度学习模型训练、推理、评价和论文写作的完整过程。由于个人能力和实验条件有限，论文中仍有不足之处，后续将继续在真实天气数据泛化、消融实验和下游任务验证方面改进。
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    write(STD / "standard-profile.yaml", standards_yaml())
    write(STD / "thesis-ai-spec.yaml", spec_yaml())
    write(STD / "figure-registry.yaml", registry_yaml())
    workflow_files()
    evidence_files()
    write(OUT / f"{TITLE}.md", thesis_markdown())
    write(RUN / "paper-context" / "workflow" / "diagram-sources.md", diagram_sources())
    (OUT / f"{TITLE}-image-map.json").write_text(
        json.dumps(image_map(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verification = {
        "title": TITLE,
        "requirement": "任务书要求不少于20篇参考文献，其中近5年外文文献不少于3篇；全部先出现先引用。",
        "count": len(REFERENCES),
        "recent_foreign_2021_2026": [17, 18, 19, 20, 21],
        "items": REFERENCES,
        "notes": [
            "文献条目按正文首次引用顺序排列。",
            "核验链接优先使用IEEE、ACM、Springer、CVF、OpenReview、NeurIPS等公开索引页面。",
        ],
    }
    (OUT / f"{TITLE}-文献核验清单.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUT / f"{TITLE}.md")


if __name__ == "__main__":
    main()
