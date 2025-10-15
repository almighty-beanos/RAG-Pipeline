# extractor.py
from readability import Document
from bs4 import BeautifulSoup
import re
from typing import Dict

def extract_main_text(html: str, url: str = "") -> Dict:
    """
    Return {'title': str, 'text': str, 'cleaned_html': str}
    Uses readability to extract main content, then cleans with BeautifulSoup.
    """
    if not html:
        return {"title": "", "text": "", "cleaned_html": ""}

    doc = Document(html)
    title = doc.short_title()
    content_html = doc.summary()

    # lightweight cleaning
    soup = BeautifulSoup(content_html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "iframe", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # normalize whitespace
    text = re.sub(r'\n{2,}', '\n\n', text).strip()
    # remove super long repeated lines
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    return {"title": title, "text": text, "cleaned_html": str(soup)}
