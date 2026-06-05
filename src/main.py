"""
CLI 入口：文档解析 → 知识库构建 → 交互问答
"""
import argparse
import logging
import sys
import json
from pathlib import Path

from .config import get_config
from .pdf_parser import PDFParser
from .knowledge_base import KnowledgeBase
from .agent import RAGAgent


def setup_logging():
    cfg = get_config()
    log_file = cfg.log_dir / "rag_agent.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_build(args):
    """构建知识库"""
    cfg = get_config()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"错误: PDF 文件不存在: {pdf_path}")
        sys.exit(1)

    print(f"正在解析 PDF: {pdf_path.name}")
    parser = PDFParser()
    records = parser.parse(pdf_path)

    if not records:
        print("错误: 未解析到任何内容")
        sys.exit(1)

    print(f"解析完成: {len(records)} 条记录")
    text_count = sum(1 for r in records if r.get("chunk_type") == "text")
    table_count = sum(1 for r in records if r.get("chunk_type") == "table")
    print(f"  - 正文块: {text_count}")
    print(f"  - 表格块: {table_count}")

    # 保存解析结果
    output_file = cfg.output_dir / f"{pdf_path.stem}_parsed.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)
    print(f"解析结果已保存: {output_file}")

    print("正在构建知识库...")
    kb = KnowledgeBase()
    kb.build(records, force_rebuild=args.force)
    print("知识库构建完成!")


def cmd_ask(args):
    """问答模式"""
    print("正在加载知识库...")
    kb = KnowledgeBase()
    kb._load_bm25_if_needed()

    agent = RAGAgent(kb)

    if args.question:
        # 单次问答
        result = agent.answer(args.question)
        print_result(result, verbose=args.verbose)
    else:
        # 交互模式
        print("\n智能文档问答 Agent (输入 'quit' 退出, 'verbose' 切换详细模式)")
        verbose = args.verbose
        while True:
            try:
                q = input("\n请输入问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见!")
                break

            if not q:
                continue
            if q.lower() == "quit":
                print("再见!")
                break
            if q.lower() == "verbose":
                verbose = not verbose
                print(f"详细模式: {'开启' if verbose else '关闭'}")
                continue

            result = agent.answer(q)
            print_result(result, verbose=verbose)


def print_result(result: dict, verbose: bool = False):
    """格式化输出结果"""
    print(f"\n{'='*60}")
    print(f"问题: {result['question']}")
    print(f"{'='*60}")
    print(f"答案:\n{result['answer']}")
    print(f"\n{'-'*40}")

    # 自检结果
    sc = result.get("self_check", {})
    status = "✓ 通过" if sc.get("pass") else "✗ 未通过"
    print(f"自检: {status}")
    if sc.get("issues"):
        print(f"问题: {', '.join(sc['issues'])}")
    if "confidence" in sc:
        print(f"可信度: {sc['confidence']:.2f}")

    if verbose:
        print(f"\n{'-'*40}")
        print(f"检索证据数: {result.get('evidence_count', 0)}")
        print("来源引用:")
        for i, src in enumerate(result.get("sources", [])):
            print(f"  [{i+1}] 第{src['page']}页 [{src['chunk_type']}]")
            print(f"      {src['snippet'][:100]}...")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="智能文档问答 Agent")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # build 子命令
    build_parser = subparsers.add_parser("build", help="构建知识库")
    build_parser.add_argument("pdf", help="PDF 文件路径")
    build_parser.add_argument("--force", action="store_true", help="强制重建知识库")

    # ask 子命令
    ask_parser = subparsers.add_parser("ask", help="问答")
    ask_parser.add_argument("-q", "--question", help="直接提问")
    ask_parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "ask":
        cmd_ask(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()