"""
All the settings you may want to change live here.
Edit a value, save, and the server reloads (if started with --reload).
"""

# ---- Model (Groq) ----------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-120b"   # smaller/faster: "openai/gpt-oss-20b"
TEMPERATURE = 0.3                    # lower = more factual, higher = more creative
TIMEOUT = 90                         # seconds before a stuck call fails
MAX_RETRIES = 3

# ---- Web search (Tavily) ---------------------------------------------------
SEARCH_RESULTS = 4                   # how many sources to fetch
SEARCH_DEPTH = "advanced"            # "basic" (fast) or "advanced" (better)
SNIPPET_CHARS = 300                  # characters kept per search result
MAX_QUERY_CHARS = 380                # long queries break search APIs
SEARCH_RETRIES = 3

# ---- Reader / scraper ------------------------------------------------------
URLS_TO_READ = 1                     # how many pages the reader scrapes
SCRAPE_CHARS = 3000                  # characters kept per scraped page
SCRAPE_TIMEOUT = 10

# ---- Writer ----------------------------------------------------------------
REPORT_WORDS = "700-900"           # target length of the report

# ---- Pipeline --------------------------------------------------------------
SEARCH_TEXT_TO_READER = 1500         # default for the UI slider (characters)

# ---- Token budget (Groq free tier allows ~8000 tokens per minute) ----------
RESEARCH_CHARS = 6000                # max research text sent to writer and reviser