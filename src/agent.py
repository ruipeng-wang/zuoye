"""
Agent 问答引擎：
- 接收用户问题
- 检索相关证据
- 生成答案 + 来源引用
- 自检：是否有依据、是否幻觉、是否拒答
"""
import logging
import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_config
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class RAGAgent:
    """智能文档问答 Agent"""

    def __init__(self, kb: KnowledgeBase):
        cfg = get_config()
        self.kb = kb
        self.self_check_enabled = cfg.self_check_enabled

        # LLM
        kwargs = {
            "model": cfg.llm_model,
            "temperature": cfg.llm_temperature,
        }
        if cfg.openai_base_url:
            kwargs["base_url"] = cfg.openai_base_url
        self.llm = ChatOpenAI(
            api_key=cfg.openai_api_key,  # type: ignore
            **kwargs
        )

    def answer(self, question: str) -> Dict[str, Any]:
        """
        主问答流程：
        1. 检索相关证据
        2. 生成答案
        3. 自检
        """
        # Step 1: 检索
        evidence = self.kb.search(question)
        logger.info(f"检索到 {len(evidence)} 条证据，内容如下：{evidence}")

        # Step 2: 生成
        context = self._format_context(evidence)
        answer = self._generate(question, context, evidence)
        logger.info(f"生成答案: {answer['text']}...")

        # Step 3: 自检
        if self.self_check_enabled:
            check_result = self._self_check(question, answer["text"], evidence)
        else:
            check_result = {"pass": True, "issues": [], "confidence": 1.0}

        return {
            "question": question,
            "answer": answer["text"],
            "sources": answer["sources"],
            "evidence_count": len(evidence),
            "self_check": check_result,
        }

    def _format_context(self, evidence: List[Dict[str, Any]]) -> str:
        """格式化证据为 LLM 上下文"""
        parts = []
        for i, ev in enumerate(evidence):
            meta = ev.get("meta", {})
            source = f"[来源{i+1}] 第{meta.get('page','?')}页, 类型:{meta.get('chunk_type','text')}"
            parts.append(f"{source}\n{ev['doc']}")
        return "\n\n---\n\n".join(parts)

    def _generate(
        self, question: str, context: str, evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """使用 LLM 生成答案"""
        system_prompt = """你是一个严谨的文档问答助手。请根据提供的文档上下文回答问题。

规则：
1. 只能根据提供的文档内容回答，不要使用外部知识。
2. 如果文档中没有足够信息，请明确说"文档中未找到相关信息"，不要编造。
3. 回答时引用来源编号，例如 [来源1]。
4. 如果涉及表格数据，请保持原始数值不变。
5. 回答要简洁、准确。"""

        user_prompt = f"""文档上下文：
{context}

问题：{question}

请回答上述问题，并在末尾列出引用的来源（页码和片段）。"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        text = response.content if hasattr(response, "content") else str(response)

        # 提取引用来源
        sources = self._extract_sources(evidence)

        return {"text": text, "sources": sources}

    def _extract_sources(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取去重的来源列表"""
        seen = set()
        sources = []
        for ev in evidence:
            meta = ev.get("meta", {})
            key = f"{meta.get('page')}_{meta.get('chunk_type')}_{ev['doc'][:60]}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "page": meta.get("page", "?"),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "snippet": ev["doc"][:200],
                })
        return sources

    def _self_check(
        self, question: str, answer: str, evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        自检模块：
        - 检查答案是否有依据
        - 检查是否存在幻觉
        - 是否需要拒答
        """
        context_snippets = "\n".join([
            f"[{i+1}] {ev['doc'][:200]}" for i, ev in enumerate(evidence[:3])
        ])

        check_prompt = f"""你是答案质量审核员。请对以下问答进行审核。

问题：{question}

答案：{answer}

检索到的文档证据（前3条）：
{context_snippets}

请按以下 JSON 格式输出审核结果：
{{
    "has_evidence": true/false,      // 答案是否能在证据中找到支撑
    "confidence": 0.0-1.0,           // 答案可信度
    "is_hallucination": true/false,   // 是否存在幻觉（捏造不存在的信息）
    "should_refuse": true/false,     // 是否应该拒答（知识库无相关信息）
    "issues": ["问题描述"],
    "explanation": "审核说明"
}}

只输出 JSON，不要输出其他内容。"""

        response = self.llm.invoke([
            HumanMessage(content=check_prompt),
        ])
        text = response.content if hasattr(response, "content") else str(response)

        try:
            # 提取 JSON
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"自检 JSON 解析失败，原始输出: {text[:200]}")
            result = {
                "has_evidence": True,
                "confidence": 0.5,
                "is_hallucination": False,
                "should_refuse": False,
                "issues": ["自检 JSON 解析失败"],
                "explanation": "自动审核异常，请人工确认",
            }

        result["pass"] = (
            result.get("has_evidence", True)
            and not result.get("is_hallucination", False)
            and result.get("confidence", 0) >= 0.5
        )

        if result.get("should_refuse"):
            result["pass"] = False

        logger.info(f"自检结果: pass={result['pass']}, confidence={result.get('confidence')}")
        return result