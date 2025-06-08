# article_scraper/config.py
#
# Imports
from dataclasses import dataclass, field
from typing import List, Dict, Any
#
# Third-Party Imports
#
# Imports
#
#######################################################################################################################
#
# Functions:

@dataclass
class ScraperConfig:
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    request_timeout_ms: int = 60000  # 60 seconds
    retries: int = 3
    stealth: bool = True
    # Time to wait after page load if stealth is enabled
    stealth_wait_ms: int = 5000

    # Trafilatura settings
    include_comments: bool = False
    include_tables: bool = False
    include_images: bool = False


@dataclass
class ProcessorConfig:
    api_name: str
    api_key: str
    summarize: bool = False
    custom_prompt: str = "Please provide a concise summary of the following article."
    system_message: str = "You are an expert summarization assistant."
    temperature: float = 0.7
    keywords: List[str] = field(default_factory=list)

#
# End of article_scraper/config.py
#######################################################################################################################
