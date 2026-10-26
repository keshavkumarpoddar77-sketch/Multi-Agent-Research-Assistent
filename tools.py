import os
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

import config

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs, and snippets."""
    query = query.strip()[:config.MAX_QUERY_CHARS]

    last_error = None
    for attempt in range(1, config.SEARCH_RETRIES + 1):
        try:
            results = tavily.search(
                query=query,
                max_results=config.SEARCH_RESULTS,
                search_depth=config.SEARCH_DEPTH,
            )
            out = [
                f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:config.SNIPPET_CHARS]}\n"
                for r in results.get("results", [])
            ]
            if not out:
                return "No results found. Try a shorter or simpler query."
            return "\n----\n".join(out)
        except Exception as e:
            last_error = e
            time.sleep(1.5 * attempt)

    return f"Search failed after {config.SEARCH_RETRIES} attempts: {last_error}. Try a shorter query."


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=config.SCRAPE_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if not text:
            return "The page had no readable text content."
        return f"SOURCE: {url}\n{text[:config.SCRAPE_CHARS]}"
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"