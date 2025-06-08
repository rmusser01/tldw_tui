# article_scraper/processors.py
import logging
from typing import List, Dict, Any, Callable, Optional, Coroutine
from tqdm.asyncio import tqdm_asyncio
#
# Third-Party Libraries
import asyncio
#
# Local Imports
from .scraper import Scraper
from .config import ProcessorConfig
#
#######################################################################################################################
#
# Functions:

# Define types for our injectable functions
Summarizer = Callable[[str, ProcessorConfig], Coroutine[Any, Any, str]]
DBLogger = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


async def default_summarizer(content: str, config: ProcessorConfig) -> str:
    # This is where you would put your actual LLM call.
    # We are simulating it for decoupling.
    # e.g., from tldw_chatbook.LLM_Calls import analyze
    logging.info(f"Simulating summarization for content of length {len(content)}...")
    # summary = await analyze(...)
    await asyncio.sleep(0.1)  # Simulate network latency
    return f"This is a simulated summary based on prompt: '{config.custom_prompt}'"


async def default_db_logger(article_data: Dict[str, Any]):
    # This is where you would put your DB ingestion logic.
    # e.g., from tldw_chatbook.DB import ingest_article_to_db
    logging.info(f"Simulating DB ingestion for article: '{article_data.get('title')}'")
    # await ingest_article_to_db(...)
    await asyncio.sleep(0.05)  # Simulate DB latency


async def scrape_and_process_urls(
        urls: List[str],
        proc_config: ProcessorConfig,
        scraper: Scraper,
        summarizer: Summarizer = default_summarizer,
        db_logger: Optional[DBLogger] = None
) -> List[Dict[str, Any]]:
    """
    A high-level pipeline to scrape, optionally summarize, and log articles.

    Args:
        urls: A list of URLs to process.
        proc_config: Configuration for the processing step (e.g., summarization).
        scraper: An initialized Scraper instance.
        summarizer: An async function to call for summarization.
        db_logger: An optional async function to log results to a database.
    """
    results = []

    # Scrape all URLs concurrently
    scraped_articles = await scraper.scrape_many(urls)

    successful_articles = [art for art in scraped_articles if art.get('extraction_successful')]

    async def process_one(article: Dict[str, Any]):
        if proc_config.summarize and article.get('content'):
            article['summary'] = await summarizer(article['content'], proc_config)
        else:
            article['summary'] = None

        if db_logger:
            await db_logger(article)

        return article

    # Process all successful articles concurrently
    tasks = [process_one(art) for art in successful_articles]
    processed_results = await tqdm_asyncio.gather(*tasks, desc="Summarizing and Processing")

    # Add failed articles back in for a complete report
    failed_articles = [art for art in scraped_articles if not art.get('extraction_successful')]

    return processed_results + failed_articles

#
# End of article_scraper/processors.py
#######################################################################################################################
