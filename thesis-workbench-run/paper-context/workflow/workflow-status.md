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
