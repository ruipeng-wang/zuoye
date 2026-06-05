# 智能文档问答 Agent 原型

> 岗位：Agent 开发工程师（大模型方向）技术笔试作业
> 输入：财报 PDF 文档
> 目标：构建可解释、可验证、可迁移的文档问答系统

---

## 快速开始

### 环境要求

- Python 3.10+（推荐 3.13）
- Anaconda 虚拟环境 `zuoye`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> 默认使用阿里百炼（DashScope）的 OpenAI 兼容 API，可切换为 DeepSeek、OpenAI 等任意兼容厂商。

### 放置 PDF 文件

将 PDF 文件放在项目根目录，默认名称为 `agent开发作业样本.pdf`。

---

## 使用方法

### 1. 构建知识库

```bash
python -m src.main build "agent开发作业样本.pdf"
```

输出示例：
```
正在解析 PDF: agent开发作业样本.pdf
解析完成: 126 条记录
  - 正文块: 78
  - 表格块: 48
解析结果已保存: outputs\agent开发作业样本_parsed.json
正在构建知识库...
已添加批次 1: 10 条 chunks
...
知识库构建完成!
```

### 2. 交互问答

```bash
python -m src.main ask
```

```
智能文档问答 Agent (输入 'quit' 退出, 'verbose' 切换详细模式)

请输入问题: 联营企业最大的投资对象是哪家？

============================================================
问题: 联营企业最大的投资对象是哪家？
============================================================
答案:
根据提供的文档内容，联营企业中投资金额最大的是**中信建投证券股份有限公司**，账面价值为 4,053,770,084.70 元。

[来源1, 第5页]
------------------------------------------------------------
自检: ✓ 通过
可信度: 0.95
```

### 3. 单次问答

```bash
python -m src.main ask -q "短期借款的即期偿还金额是多少？" -v
```

---

## 运行测试

```bash
pytest tests/test_rag.py -v -s
```

测试覆盖：PDF 解析、知识库构建、Agent 问答、自检、OCR 鲁棒性、配置、回归测试，共 30 个测试用例。

```bash
# 仅运行 PDF 解析相关测试
pytest tests/test_rag.py -v -k "test_pdf"

# 仅运行 Agent 问答测试
pytest tests/test_rag.py -v -k "test_agent"
```

---

## 项目结构

```
rag_test/
├── README.md                         # 本文档
├── DESIGN.md                         # 设计文档
├── requirements.txt                  # Python 依赖
├── .env.example                      # 环境变量模板
├── .gitignore
├── agent开发作业样本.pdf              # 输入 PDF
├── src/
│   ├── __init__.py
│   ├── config.py                     # 配置管理
│   ├── pdf_parser.py                 # PDF 解析 + 表格提取
│   ├── knowledge_base.py             # ChromaDB + BM25
│   ├── agent.py                      # 问答引擎 + 自检
│   └── main.py                       # CLI 入口
├── tests/
│   ├── __init__.py
│   └── test_rag.py                   # 30 个测试用例
├── outputs/
│   ├── chroma_db/                    # ChromaDB 持久化存储
│   └── *.json                        # 解析结果缓存
└── logs/
    └── rag_agent.log
```

---

## 核心设计

### 架构概览

```
PDF 输入 → 文档解析(PyMuPDF) → 知识库(ChromaDB+BM25) → Agent 问答(LLM) → 自检验证
```

### 关键特性

- **混合检索**：语义向量检索 + BM25 关键词检索，加权融合
- **表格提取**：PyMuPDF `find_tables()` 自动检测表格 → Markdown 格式
- **来源标注**：每个答案附带页码和原文片段
- **自检模块**：自动检测答案是否有依据、是否幻觉、是否需要拒答
- **配置化**：通过 `.env` 切换 LLM 厂商，无需改代码

### 技术栈

| 组件 | 方案 |
|------|------|
| LLM | 阿里百炼 qwen-plus（OpenAI 兼容 API） |
| Embedding | 阿里百炼 text-embedding-v3 |
| 向量库 | ChromaDB |
| PDF 解析 | PyMuPDF (fitz) |
| 分词 | jieba |
| 关键词检索 | rank-bm25 |
| 测试 | pytest |

---

## 问答效果示例

| 问题 | 答案 | 来源 | 自检 |
|------|------|------|------|
| 联营企业最大投资对象 | 中信建投证券，4,053,770,084.70 元 | 第5页 | ✅ pass |
| 短期借款即期偿还金额 | 390,209,470.57 元 | 第5页 | ✅ pass |
| 员工人数 | 文档中未找到相关信息 | 多页检索 | ✅ pass |
| 今天天气怎么样 | 文档中未找到相关信息 | 无 | ✅ should_refuse |

---

## 业务场景迁移

| 场景 | 适配策略 |
|------|---------|
| 金融合同 | 自定义金融词典 + 金额正则校验 |
| 合规文档 | 降低 LLM 改写自由度，原文引用 |
| 产品手册 | 增强布局分析，图文混排处理 |
| 医疗报告 | 专病种 OCR + PII 脱敏 |
| 国标文件 | 条款编号正则匹配 + BM25 |

---

## AI 工具使用说明

本项目使用 Cline（Claude）作为 AI 编程助手：

- **方案设计**：AI 辅助分析 PDF 结构、讨论检索策略
- **编码**：AI 辅助生成代码框架，人工审查修正
- **测试**：AI 辅助生成测试用例，人工验证结果
- **文档**：AI 辅助生成 README 和设计文档

**所有 AI 输出均经过人工校验**，确保代码正确性和设计合理性。

---

## 已知限制

1. **表格跨页**：PyMuPDF `find_tables()` 不处理跨页表格
2. **多轮对话**：当前仅支持单轮问答
3. **Rerank**：未接入在线 Rerank API

---

## License

MIT