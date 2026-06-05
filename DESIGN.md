# 智能文档问答 Agent 原型 — 设计文档

> 输入：财报 PDF 文档（agent开发作业样本.pdf）
> 目标：构建可解释、可验证、可迁移的文档问答系统
> 当前状态：已完成核心功能，可运行、可测试

---

## 一、项目要求

围绕 PDF 文档完成以下闭环：

1. ✅ 判断 PDF 类型并选择解析策略
2. ✅ 提取/识别正文、条款编号、表格信息
3. ✅ 构建可检索知识库
4. ✅ 接收用户问题，检索相关证据
5. ✅ 生成答案，返回来源页码/片段
6. ✅ 对答案做基本自检：是否有依据、是否可能幻觉、是否需要拒答
7. ✅ 给出测试方法和不同业务场景下的保障方案

---

## 二、整体架构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  PDF 输入    │───▶│  文档解析层   │───▶│  知识构建层   │───▶│  Agent 推理   │───▶│  输出 + 自检  │
│  (财报 PDF)  │    │ PyMuPDF 解析 │    │ ChromaDB+BM25│    │ 检索+ LLM生成│    │ 答案+来源+验证│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## 三、技术栈

| 组件 | 方案 | 说明 |
|------|------|------|
| **LLM** | 阿里百炼 qwen-plus | 通过 OpenAI 兼容 API 接入，配置化切换 |
| **Embedding** | 阿里百炼 text-embedding-v3 | 与 LLM 同厂商，统一 API Key |
| **向量库** | ChromaDB | 轻量，pip 安装即用，持久化 |
| **PDF 解析** | PyMuPDF (fitz) | 文本提取 + 表格检测 |
| **表格提取** | PyMuPDF find_tables + Markdown 转换 | 自动识别表格区域 |
| **关键词检索** | rank-bm25 + jieba 分词 | 混合检索，语义+关键词 |
| **配置** | python-dotenv + .env | 环境变量管理 API Key |
| **测试** | pytest | 30 个测试用例 |
| **日志** | Python logging | 文件 + 控制台双输出 |

### 为什么这样选

- **全部在线 API**：LLM、Embedding 走阿里百炼，通过 `.env` 配置 `base_url` + `model` + `api_key`，可切换任意厂商
- **裸写 Python**：核心逻辑（切分策略、检索、自检）手写，LangChain 仅用于 LLM 调用（OpenAI 兼容 client）
- **ChromaDB**：轻量级，pip 安装即用，面试官复现成本最低
- **配置化**：所有模型配置通过 `.env` 环境变量管理，敏感信息不提交 Git
- **混合检索**：语义检索（向量相似度）+ 关键词检索（BM25），加权融合提升召回率

---

## 四、核心模块设计

### 模块 1：文档解析层 (pdf_parser.py)

**职责：** 使用 PyMuPDF 解析 PDF，自动判断类型并选择策略

- 文本型 PDF → 直接提取文本块（block 级别）
- 扫描件 PDF → OCR 引擎（Tesseract / PaddleOCR 备选）
- 表格提取 → `page.find_tables()` 自动检测表格区域 → Markdown 表格输出

**输出格式：**
```python
{
    "text": "段落或表格内容",
    "page": 页码,
    "source": "文件名",
    "chunk_type": "text" | "table"
}
```

### 模块 2：知识库构建 (knowledge_base.py)

**核心功能：**

1. **文本分块**：滑动窗口（800 字符，重叠 150 字符）
2. **向量存储**：ChromaDB 持久化存储，Embedding 分批（每批 10 条避免 API 限流）
3. **BM25 索引**：jieba 分词 + rank-bm25 构建关键词索引
4. **混合检索**：语义分数 + BM25 分数加权融合（默认 BM25 权重 30%）

### 模块 3：Agent 问答引擎 (agent.py)

**System Prompt 核心约束：**

```
1. 只能根据提供的文档内容回答，不要使用外部知识
2. 如果文档中没有足够信息，明确说"文档中未找到相关信息"
3. 回答时引用来源编号，例如 [来源1]
4. 如果涉及表格数据，请保持原始数值不变
5. 回答要简洁、准确
```

### 模块 4：自检模块 (agent.py 内嵌)

**检查维度：**

| 维度 | 检查方法 | 说明 |
|------|---------|------|
| 是否有依据 | LLM 对比答案与证据 | has_evidence |
| 可信度 | LLM 评分 0-1 | confidence |
| 是否幻觉 | 检查是否捏造不存在的信息 | is_hallucination |
| 是否拒答 | 知识库无相关信息时 | should_refuse |

**自检结果封装：**
```python
{
    "pass": True/False,           # 综合判断
    "confidence": 0.0-1.0,
    "has_evidence": True/False,
    "is_hallucination": True/False,
    "should_refuse": True/False,
    "issues": ["问题描述"],
    "explanation": "审核说明"
}
```

---

## 五、配置方式

本项目使用 `.env` 文件配置 API Key（通过 python-dotenv 加载），使用 `config.py` 中的 `Config` dataclass 管理所有参数。

### .env 文件

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 配置参数（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| llm_model | qwen-plus | LLM 模型名 |
| embedding_model | text-embedding-v3 | Embedding 模型 |
| chunk_size | 800 | 文本分块大小 |
| chunk_overlap | 150 | 重叠大小 |
| retrieval_top_k | 5 | 检索返回数 |
| bm25_weight | 0.3 | BM25 权重 |
| self_check_enabled | True | 启用自检 |

---

## 六、开发环境

- **Python 环境**：Anaconda 虚拟环境 `zuoye`（Python 3.13.13）
- **环境路径**：`C:\Users\ruipeng\.conda\envs\zuoye`
- **激活命令**：`call C:\ProgramData\anaconda3\Scripts\activate.bat zuoye`
- **Python 路径**：`C:\Users\ruipeng\.conda\envs\zuoye\python.exe`

---

## 七、实际项目结构

```
rag_test/
├── README.md
├── DESIGN.md                       # 本文档
├── requirements.txt
├── .env.example                    # 环境变量模板
├── .gitignore
├── agent开发作业样本.pdf            # 财报 PDF 输入
├── src/
│   ├── __init__.py
│   ├── config.py                   # 配置管理
│   ├── pdf_parser.py               # PDF 解析 + 表格提取
│   ├── knowledge_base.py           # ChromaDB + BM25 构建+检索
│   ├── agent.py                    # 问答引擎 + 自检
│   └── main.py                     # CLI 入口
├── tests/
│   ├── __init__.py
│   └── test_rag.py                 # 30 个测试用例
├── data/
├── outputs/
│   ├── chroma_db/                  # ChromaDB 持久化
│   └── agent开发作业样本_parsed.json # 解析结果缓存
└── logs/
```

---

## 八、测试策略

### 8.1 测试用例覆盖（30 个测试用例）

**PDF 解析测试（7 个）：**
| # | 测试方法 | 说明 |
|---|---------|------|
| 1 | test_pdf_exists | 验证 PDF 文件存在 |
| 2 | test_pdf_type_detection | 自动判断扫描件/文本型 |
| 3 | test_parse_returns_records | 解析返回非空记录 |
| 4 | test_parse_has_text_chunks | 包含正文块 |
| 5 | test_parse_has_table_chunks | 包含表格块 |
| 6 | test_page_numbers_valid | 页码范围有效 |
| 7 | test_ocr_error_handling | OCR 引擎不可用时不崩溃 |

**知识库测试（5 个）：**
| # | 测试方法 | 说明 |
|---|---------|------|
| 8 | test_kb_build_success | 知识库构建成功 |
| 9 | test_kb_search_returns_results | 检索返回结果 |
| 10 | test_bm25_index_built | BM25 索引构建 |
| 11 | test_top_k_limit | top_k 限制结果数量 |
| 12 | test_search_scores_valid | 分数在 0-1 范围 |

**Agent 问答测试（7 个）：**
| # | 测试方法 | 说明 |
|---|---------|------|
| 13 | test_agent_answers_body_question | 正文问题 |
| 14 | test_agent_answers_technical_question | 技术细节问题 |
| 15 | test_agent_answers_table_question | 表格问题 |
| 16 | test_agent_no_answer_question | 无答案问题（天气） |
| 17 | test_agent_vague_question | 模糊问题 |
| 18 | test_answer_has_sources | 答案包含来源 |
| 19 | test_evidence_count | 检索证据数量 |

**自检测试（3 个）：**
| # | 测试方法 | 说明 |
|---|---------|------|
| 20 | test_self_check_present | 自检结果存在 |
| 21 | test_self_check_structure | 自检结构完整 |
| 22 | test_self_check_confidence_range | 可信度范围检查 |

**OCR 鲁棒性测试（3 个）：**
| # | 测试方法 | 说明 |
|---|---------|------|
| 23 | test_ocr_char_confusion | 字符混淆不崩溃 |
| 24 | test_empty_text_not_crashing | 空文本处理 |
| 25 | test_very_short_text | 极短文本处理 |

**配置测试（3 个）：**
| # | 测试方法 | 说明 |
|---|---------|------|
| 26 | test_config_defaults | 默认配置可用 |
| 27 | test_get_config_singleton | 单例模式 |
| 28 | test_directories_created | 目录自动创建 |

**回归测试（2 个）：**
| # | 测试方法 | 说明 |
|---|---------|------|
| 29 | test_agent_always_returns_required_fields | 返回结构完整 |
| 30 | test_multiple_questions_no_crash | 连续问答不崩溃 |

### 8.3 自检模块验证

- 有依据的答案 → pass=True, confidence 高
- 域外问题 → pass=False, should_refuse=True
- 无答案问题 → 正确识别为 evidence 不足

### 8.4 自检模块测试

- 构造 3 个故意偏离原文的答案 → 检查自检能否发现
- 构造 3 个完全正确的答案 → 检查自检是否误报
- 指标：幻觉检测召回率 > 90%，误报率 < 20%

---

## 九、业务场景迁移方案

| 场景 | 差异点 | 适配策略 |
|------|--------|---------|
| 金融合同 | 术语密集、金额精确 | 自定义金融词典 + 金额正则校验 + 条款交叉引用 |
| 合规文档 | 法律条文引用 | 降低 LLM 改写自由度，只用原文片段回答 |
| 产品手册 | 图文混排 | 增强布局分析，区分正文/图注/表格 |
| 医疗报告 | 手写体、隐私 | 接入专病种 OCR + PII 自动脱敏 |
| 国标文件 | 条款编号 | 正则匹配编号格式，BM25 精确检索 |

**核心设计原则：** 所有模块通过配置/插件化接口设计，更换业务场景只需替换 OCR 模型 + 调整词典 + 修改 Prompt 模板。

---

## 十、实现完成情况

| 模块 | 状态 | 说明 |
|------|------|------|
| PDF 解析（文本型） | ✅ 完成 | PyMuPDF 提取文本块 + 表格 |
| PDF 解析（扫描件） | ⚠️ 未支持 | 建议使用成熟产品，如mineru |
| 表格提取 | ✅ 完成 | PyMuPDF find_tables → Markdown |
| ChromaDB 向量库 | ✅ 完成 | 持久化 + 分批 embedding |
| BM25 关键词索引 | ✅ 完成 | jieba 分词 + rank-bm25 |
| 混合检索 | ✅ 完成 | 语义+关键词加权融合 |
| LLM 问答 | ✅ 完成 | 百炼 qwen-plus + 来源标注 |
| 自检模块 | ✅ 完成 | 依据/幻觉/拒答/可信度 |
| CLI 交互 | ✅ 完成 | build + ask 子命令 |
| 测试脚本 | ✅ 完成 | 30 个测试用例 |
| README | ✅ 完成 | 本文档 + README.md |
| OCR 扫描件支持 | ⚠️ 未支持 | 建议使用成熟产品，如mineru |
| 多轮对话 | ❌ 未实现 | 当前仅单轮问答 |
| 增量更新 | ❌ 未实现 | 需 force_rebuild |

---

## 十一、已知问题与限制

1. **表格跨页**：当前 PyMuPDF `find_tables()` 不处理跨页表格，需后续增强
2. **多轮对话**：未实现，当前仅单轮问答
3. **Rerank**：未接入在线 Rerank API，当前使用纯向量+BM25 融合

## 十二、AI 工具使用说明

本项目在开发过程中使用 Cline（Claude）作为 AI 编程助手：

- **方案设计阶段**：AI 辅助分析 PDF 结构、讨论切分策略、修正边界问题
- **编码阶段**：AI 辅助生成代码框架，人工审查和修正
- **测试阶段**：AI 辅助生成测试用例，人工验证结果
- **所有 AI 输出均经过人工校验**，确保代码正确性和设计合理性