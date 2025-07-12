# app/core/web_crawler.py
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Optional, Tuple

# --- Configuration ---
# Headers to mimic a browser and avoid being blocked
CRAWLER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# Timeout for requests
REQUEST_TIMEOUT = 10

class WebCrawler:
    """
    An asynchronous web crawler designed to fetch and parse documentation websites,
    extracting textual content and code blocks.
    """
    def __init__(self, max_depth: int = 2, max_pages: int = 50):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.crawled_content: List[Dict[str, Any]] = []

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """Checks if a URL is valid, within the same domain, and not an anchor/mailto link."""
        parsed_url = urlparse(url)
        # Ensure it's a web URL, has a domain, and belongs to the same domain as the start URL
        return (
            parsed_url.scheme in ['http', 'https'] and
            parsed_url.netloc == base_domain and
            '#' not in url and
            not url.startswith('mailto:')
        )

    def _extract_content(self, soup: BeautifulSoup) -> Tuple[str, List[str]]:
        """Extracts the main text and all code blocks from a BeautifulSoup object."""
        # Simple heuristic for main content: look for <main>, <article>, or fall back to <body>
        main_content_area = soup.find('main') or soup.find('article') or soup.body
        if not main_content_area:
            return "", []

        # Extract code blocks first and remove them to not duplicate content
        code_blocks = []
        for code_tag in main_content_area.find_all(['pre', 'code']):
            # Often code is in <pre><code>...</code></pre>, sometimes just <code>
            # We prefer <pre> as it preserves formatting.
            parent = code_tag.find_parent('pre')
            target_tag = parent if parent else code_tag

            code_text = target_tag.get_text(strip=True)
            if code_text:
                code_blocks.append(code_text)
            # Decompose the tag to prevent its text from being included in the main text extraction
            target_tag.decompose()

        # Extract remaining text, which should now be mostly prose
        text = main_content_area.get_text(separator='\n', strip=True)
        return text, code_blocks

    async def _crawl_page(self, session: aiohttp.ClientSession, url: str, base_domain: str, depth: int):
        """Recursively crawls a single page."""
        if depth > self.max_depth or len(self.crawled_content) >= self.max_pages or url in self.visited_urls:
            return

        print(f"Crawling (depth {depth}): {url}")
        self.visited_urls.add(url)

        try:
            async with session.get(url, headers=CRAWLER_HEADERS, timeout=REQUEST_TIMEOUT) as response:
                if response.status != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                    print(f"Skipping non-HTML or failed request: {url} (Status: {response.status})")
                    return

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                page_title = soup.title.string if soup.title else "No Title"
                main_text, code_snippets = self._extract_content(soup)

                if main_text or code_snippets:
                    self.crawled_content.append({
                        "url": url,
                        "title": page_title,
                        "text_content": main_text,
                        "code_snippets": code_snippets
                    })

                if depth < self.max_depth:
                    # Find all links and queue them for crawling
                    tasks = []
                    for link in soup.find_all('a', href=True):
                        absolute_link = urljoin(url, link['href'])
                        if self._is_valid_url(absolute_link, base_domain):
                            tasks.append(self._crawl_page(session, absolute_link, base_domain, depth + 1))

                    if tasks:
                        await asyncio.gather(*tasks)

        except aiohttp.ClientError as e:
            print(f"Network error crawling {url}: {e}")
        except asyncio.TimeoutError:
            print(f"Timeout error crawling {url}")
        except Exception as e:
            print(f"An unexpected error occurred while crawling {url}: {e}")

    async def run(self, start_url: str) -> List[Dict[str, Any]]:
        """Starts the crawling process from a given URL."""
        base_domain = urlparse(start_url).netloc
        if not base_domain:
            raise ValueError("Invalid start URL provided.")

        self.visited_urls.clear()
        self.crawled_content.clear()

        async with aiohttp.ClientSession() as session:
            await self._crawl_page(session, start_url, base_domain, depth=0)

        print(f"\nCrawling finished. Visited {len(self.visited_urls)} URLs and extracted content from {len(self.crawled_content)} pages.")
        return self.crawled_content

# Example Usage
if __name__ == '__main__':
    async def main():
        # A documentation site that is likely to be crawler-friendly
        # Using LangChain's Python docs as an example.
        start_url = "https://python.langchain.com/v0.2/docs/introduction/"

        # Limit for testing purposes to not crawl the entire site
        crawler = WebCrawler(max_depth=1, max_pages=5)

        crawled_data = await crawler.run(start_url)

        print("\n--- Sample of Crawled Data ---")
        if crawled_data:
            # Print details of the first crawled page
            first_page = crawled_data[0]
            print(f"URL: {first_page['url']}")
            print(f"Title: {first_page['title']}")
            print(f"Text Preview: {first_page['text_content'][:400]}...")
            print(f"Found {len(first_page['code_snippets'])} code snippets.")
            if first_page['code_snippets']:
                print(f"First Code Snippet: {first_page['code_snippets'][0]}")
        else:
            print("No data was crawled. Check the start URL and network connection.")

    asyncio.run(main())
```
