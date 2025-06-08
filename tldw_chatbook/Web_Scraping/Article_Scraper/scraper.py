# article_scraper/scraper.py
#
# Imports
import asyncio
import logging
from typing import Any, Dict, List, Optional
#
# Third-Party Libraries
from playwright.async_api import async_playwright, Browser, BrowserContext
import trafilatura
#
# Local Imports
from .config import ScraperConfig
from .utils import convert_html_to_markdown, ContentMetadataHandler
#
#######################################################################################################################
#
# Functions:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Scraper:
    """Manages a Playwright browser instance for efficient, repeated scraping."""

    def __init__(self, config: Optional[ScraperConfig] = None, custom_cookies: Optional[List[Dict]] = None):
        self.config = config or ScraperConfig()
        self.custom_cookies = custom_cookies
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self):
        """Initializes the browser context when entering an `async with` block."""
        logging.info("Starting Playwright browser...")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the browser and stops Playwright when exiting the block."""
        logging.info("Closing Playwright browser...")
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _fetch_html(self, context: BrowserContext, url: str) -> str:
        """Fetches HTML from a URL with retries."""
        for attempt in range(self.config.retries):
            try:
                page = await context.new_page()
                if self.config.stealth:
                    try:
                        from playwright_stealth import stealth_async
                        await stealth_async(page)
                    except ImportError:
                        logging.warning("playwright-stealth not installed. Running without stealth.")

                await page.goto(url, wait_until="domcontentloaded", timeout=self.config.request_timeout_ms)

                if self.config.stealth:
                    await page.wait_for_timeout(self.config.stealth_wait_ms)
                else:
                    await page.wait_for_load_state("networkidle", timeout=self.config.request_timeout_ms)

                content = await page.content()
                await page.close()
                return content
            except Exception as e:
                logging.error(f"Error fetching {url} on attempt {attempt + 1}: {e}")
                if attempt >= self.config.retries - 1:
                    return ""  # Return empty on final failure
                await asyncio.sleep(2)
        return ""

    def _extract_data(self, html: str, url: str) -> Dict[str, Any]:
        """Extracts article data from HTML using Trafilatura."""
        if not html:
            return {'extraction_successful': False, 'error': 'HTML content was empty.'}

        main_content_html = trafilatura.extract(
            html,
            include_comments=self.config.include_comments,
            include_tables=self.config.include_tables,
            include_images=self.config.include_images
        )
        metadata = trafilatura.extract_metadata(html)

        if not main_content_html or not metadata:
            return {'extraction_successful': False, 'error': 'Trafilatura failed to extract content or metadata.'}

        # Convert the extracted HTML content to clean Markdown
        main_content_md = convert_html_to_markdown(main_content_html)

        article_data = {
            'url': url,
            'title': metadata.title or 'N/A',
            'author': metadata.author or 'N/A',
            'date': metadata.date or 'N/A',
            'content': main_content_md,
            'extraction_successful': True
        }

        # Add our own metadata wrapper
        article_data['content_with_meta'] = ContentMetadataHandler.format_content_with_metadata(
            url=url,
            content=main_content_md,
            pipeline="trafilatura-playwright",
            additional_metadata={'author': article_data['author'], 'extracted_date': article_data['date']}
        )
        return article_data

    async def scrape(self, url: str) -> Dict[str, Any]:
        """Scrapes a single article from a URL."""
        if not self._browser:
            raise RuntimeError("Scraper must be used within an `async with` block.")

        logging.info(f"Scraping article from: {url}")

        context = await self._browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1280, "height": 720},
        )
        if self.custom_cookies:
            await context.add_cookies(self.custom_cookies)

        html = await self._fetch_html(context, url)
        await context.close()

        result = self._extract_data(html, url)
        if result['extraction_successful']:
            logging.info(f"Successfully extracted article: '{result.get('title', 'N/A')}'")
        else:
            logging.warning(f"Failed to extract article from {url}. Reason: {result.get('error')}")

        return result

    async def scrape_many(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrapes multiple URLs concurrently for maximum performance."""
        tasks = [self.scrape(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

#
# End of article_scraper/scraper.py
#######################################################################################################################
