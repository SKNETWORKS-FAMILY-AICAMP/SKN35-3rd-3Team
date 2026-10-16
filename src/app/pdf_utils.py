import fitz  # PyMuPDF

MAX_PDF_PAGES = 5  # 비용·토큰 보호용 상한


def pdf_to_images(pdf_bytes: bytes, max_pages: int = MAX_PDF_PAGES, dpi: int = 120) -> list[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        count = min(len(doc), max_pages)
        return [doc[i].get_pixmap(dpi=dpi).tobytes("png") for i in range(count)]
    finally:
        doc.close()