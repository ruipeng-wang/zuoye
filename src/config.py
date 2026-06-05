"""
全局配置管理
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Config:
    """应用配置，支持环境变量覆盖"""

    # --- 路径 ---
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    log_dir: Path = field(default=None)

    # --- OCR ---
    ocr_lang: str = "chi_sim+eng"
    ocr_dpi: int = 300

    # --- 向量库 ---
    chroma_persist_dir: str = "outputs/chroma_db"
    embedding_model: str = "text-embedding-v3"
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- LLM ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.1

    # --- 检索 ---
    retrieval_top_k: int = 5
    bm25_weight: float = 0.3

    # --- 自检 ---
    self_check_enabled: bool = True
    hallucination_threshold: float = 0.5

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.output_dir is None:
            self.output_dir = self.project_root / "outputs"
        if self.log_dir is None:
            self.log_dir = self.project_root / "logs"
        for d in [self.data_dir, self.output_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config