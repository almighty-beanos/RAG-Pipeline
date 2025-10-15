# crawler.py
import time
import requests
from urllib.parse import urljoin, urldefrag, urlparse
import tldextract
from bs4 import BeautifulSoup
import urllib.robotparser
from typing import List, Dict, Tuple, Set
from utils import logger

DEFAULT_USER_AGENT = "polite-rag-bot/1.0 (+https://example.com/bot)"

class PoliteCrawler:
    def __init__(self, start_url: str, user_agent: str = DEFAULT_USER_AGENT):
        self.start_url = start_url
        self.start_netloc = urlparse(start_url).netloc
        self.reg_domain = tldextract.extract(start_url).registered_domain
        self.user_agent = user_agent
        self.robots = self._load_robots(start_url)

    def _load_robots(self, url: str) -> urllib.robotparser.RobotFileParser:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
            logger.info(f"Loaded robots from {robots_url}")
        except Exception as e:
            logger.warning(f"Could not load robots.txt: {e}")
        return rp

    def _allowed(self, url: str) -> bool:
        try:
            return self.robots.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _same_registrable_domain(self, url: str) -> bool:
        try:
            return tldextract.extract(url).registered_domain == self.reg_domain
        except Exception:
            return False

    def _normalize(self, base: str, link: str) -> str:
        # join, remove fragments
        joined = urljoin(base, link)
        nofrag, _ = urldefrag(joined)
        return nofrag

    def crawl(self, max_pages: int = 50, max_depth: int = 3, crawl_delay_ms: int = 500) -> Tuple[List[Dict], int]:
        """BFS crawl starting from start_url, returning list of {url, status_code, html}"""
        to_visit = [(self.start_url, 0)]
        visited: Set[str] = set()
        results = []
        skipped = 0

        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            if not self._same_registrable_domain(url):
                logger.debug(f"Skipping out-of-domain {url}")
                skipped += 1
                continue
            if not self._allowed(url):
                logger.info(f"Disallowed by robots: {url}")
                skipped += 1
                continue

            try:
                headers = {"User-Agent": self.user_agent}
                resp = requests.get(url, headers=headers, timeout=10)
                status = resp.status_code
                html = resp.text if resp.ok else ""
                results.append({"url": url, "status": status, "html": html})
                logger.info(f"Fetched {url} ({status})")
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                skipped += 1
                continue

            # parse links if depth < max_depth
            if depth < max_depth and html:
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    try:
                        nxt = self._normalize(url, a["href"])
                        if nxt not in visited and self._same_registrable_domain(nxt):
                            to_visit.append((nxt, depth + 1))
                    except Exception:
                        continue

            time.sleep(crawl_delay_ms / 1000.0)

        return results, skipped
