"""
PDF 解析模块：
- 判断 PDF 类型（扫描件/文本型）
- 扫描件 → OCR 提取文字
- 表格区域检测与提取
- 输出结构化 Document 列表
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF 解析器：自动判断类型并选择解析策略"""

    def __init__(self):
        cfg = get_config()
        self.ocr_lang = cfg.ocr_lang
        self.ocr_dpi = cfg.ocr_dpi
        self._ocr_engine = None

    def _get_ocr_engine(self):
        """延迟加载 OCR 引擎"""
        if self._ocr_engine is None:
            try:
                import pytesseract
                from pdf2image import convert_from_path
                self._ocr_engine = "tesseract"
                self._pytesseract = pytesseract
                self._pdf2image = convert_from_path
                logger.info("OCR 引擎: Tesseract")
            except ImportError:
                try:
                    from paddleocr import PaddleOCR
                    self._ocr_engine = "paddleocr"
                    self._paddleocr = PaddleOCR(lang="ch")
                    logger.info("OCR 引擎: PaddleOCR")
                except ImportError:
                    logger.warning("未安装任何 OCR 引擎，仅支持文本型 PDF")
                    self._ocr_engine = None
        return self._ocr_engine

    def _is_scanned_pdf(self, doc: fitz.Document) -> bool:
        """
        判断 PDF 是否为扫描件：
        - 检查前 5 页的文本长度
        - 如果平均每页文字 < 100 字符，判定为扫描件
        """
        text_pages = 0
        total_chars = 0
        for i in range(min(5, len(doc))):
            text = doc[i].get_text("text").strip()
            total_chars += len(text)
            text_pages += 1
        avg_chars = total_chars / max(text_pages, 1)
        is_scanned = avg_chars < 100
        logger.info(f"PDF 类型判断: 前{text_pages}页平均 {avg_chars:.0f} 字符 → {'扫描件' if is_scanned else '文本型PDF'}")
        return is_scanned

    def parse(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        解析 PDF，返回结构化文档列表
        每条记录包含: text, page, source, chunk_type
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        is_scanned = self._is_scanned_pdf(doc)
        records: List[Dict[str, Any]] = []

        if is_scanned:
            records = self._parse_scanned(doc, pdf_path)
        else:
            records = self._parse_text(doc, pdf_path)

        # 提取表格
        table_records = self._extract_tables(doc, pdf_path, is_scanned)
        records.extend(table_records)

        doc.close()
        logger.info(f"PDF 解析完成: {len(records)} 条记录 (含 {len(table_records)} 条表格)")
        return records

    def _parse_text(self, doc: fitz.Document, source: Path) -> List[Dict[str, Any]]:
        """解析文本型 PDF"""
        records = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                continue
            # 按段落分块
            blocks = page.get_text("blocks")
            for block in blocks:
                block_text = block[4].strip() if len(block) > 4 else ""
                if not block_text:
                    continue
                records.append({
                    "text": block_text,
                    "page": page_num + 1,
                    "source": str(source.name),
                    "chunk_type": "text",
                })
        return records

    def _parse_scanned(self, doc: fitz.Document, source: Path) -> List[Dict[str, Any]]:
        """解析扫描件 PDF → OCR"""
        engine = self._get_ocr_engine()
        if engine is None:
            logger.warning("未安装 OCR 引擎，无法处理扫描件，回退到文本提取")
            return self._parse_text(doc, source)

        records = []
        if engine == "tesseract":
            from pdf2image import convert_from_path
            images = convert_from_path(str(source), dpi=self.ocr_dpi)
            for page_num, image in enumerate(images):
                text = self._pytesseract.image_to_string(image, lang=self.ocr_lang)
                if text.strip():
                    records.append({
                        "text": text.strip(),
                        "page": page_num + 1,
                        "source": str(source.name),
                        "chunk_type": "text",
                    })
        elif engine == "paddleocr":
            from pdf2image import convert_from_path
            images = convert_from_path(str(source), dpi=self.ocr_dpi)
            for page_num, image in enumerate(images):
                img_array = np.array(image)
                result = self._paddleocr.ocr(img_array, cls=True)
                if result and result[0]:
                    lines = [line[1][0] for line in result[0] if line and len(line) > 1]
                    text = "\n".join(lines)
                    if text.strip():
                        records.append({
                            "text": text.strip(),
                            "page": page_num + 1,
                            "source": str(source.name),
                            "chunk_type": "text",
                        })
        return records

    def _extract_tables(
        self, doc: fitz.Document, source: Path, is_scanned: bool
    ) -> List[Dict[str, Any]]:
        """提取 PDF 中的表格"""
        table_records = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    for table in tables.tables:
                        table_data = table.extract()
                        if not table_data:
                            continue
                        # 转换为 Markdown 表格格式
                        md_table = self._table_to_markdown(table_data)
                        if md_table:
                            table_records.append({
                                "text": md_table,
                                "page": page_num + 1,
                                "source": str(source.name),
                                "chunk_type": "table",
                            })
            except Exception as e:
                logger.debug(f"第 {page_num + 1} 页表格提取跳过: {e}")
        return table_records

    @staticmethod
    def _table_to_markdown(table_data: List[List[Any]]) -> str:
        """将表格数据转为 Markdown 表格字符串"""
        if not table_data or not table_data[0]:
            return ""
        # 过滤全空行
        table_data = [row for row in table_data if any(cell is not None and str(cell).strip() != "" for cell in row)]
        if not table_data:
            return ""
        header = table_data[0]
        col_count = len(header)
        header_str = "| " + " | ".join(str(c) if c is not None else "" for c in header) + " |"
        sep = "| " + " | ".join(["---"] * col_count) + " |"
        rows = [header_str, sep]
        for row in table_data[1:]:
            padded = list(row) + [""] * (col_count - len(row))
            row_str = "| " + " | ".join(str(c) if c is not None else "" for c in padded[:col_count]) + " |"
            rows.append(row_str)
        return "\n".join(rows)