"""
Paleontology Research Assistant -- Agent Loop
V0.9: Replace arXiv retrieval with Semantic Scholar. arXiv returns HTTP 429
      on Streamlit Cloud due to shared-IP rate limiting. Semantic Scholar is
      free, requires no authentication, is cloud-friendly, and covers the
      same paleontology literature via a plain JSON REST API.
"""

import json
import logging
import re
import urllib.parse
import urllib.request
import anthropic
from dotenv import load_dotenv

load_dotenv()

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -- Client --------------------------------------------------------------------
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env automatically

# -- Config --------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
MAX_LOOP_ITERATIONS = 5    # guard against runaway loops
S2_MAX_RESULTS = 5
S2_TIMEOUT = 15            # seconds

# -- Tools ---------------------------------------------------------------------
TOOLS = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
    }
]

# -- System prompt -------------------------------------------------------------
SYSTEM_PROMPT = """You are a paleontology research assistant with access to live web search
and pre-fetched Semantic Scholar abstracts supplied in the user message.
Use web search to retrieve current findings, then synthesise both the web
results and the provided academic abstracts into a structured summary using
this exact format:

## Research Summary

**Query:** [restate the query]

### Key Findings
- [finding 1]
- [finding 2]
- [finding 3]

### arXiv Papers
- [paper title, authors, year, paper ID]
- [paper title, authors, year, paper ID]

### Sources
- [web source or reference 1]
- [web source or reference 2]

### Open Questions
- [open question 1]
- [open question 2]

Always use web search before answering. Incorporate Semantic Scholar abstracts where relevant.
Cite academic papers in the arXiv Papers section using their IDs.
Cite web results in Sources. Be specific and accurate. If you are uncertain
about something, say so rather than speculating.
Keep findings concise -- one to two sentences each."""


# -- Semantic Scholar query construction ---------------------------------------
# Strip conversational prefixes ("What is", "Tell me about", etc.) so the
# core subject terms reach the S2 search engine. S2 uses natural-language
# relevance ranking, not arXiv-style boolean syntax, so we append a single
# paleontology anchor term only when the query doesn't already contain one.
_QUESTION_PREFIX = re.compile(
    r"(?i)^\s*"
    r"(?:what(?:\s+(?:is|are|was|were|do|does|did))?|how|why|when|where|who|which|"
    r"tell\s+me\s+(?:about|of)|explain|describe|give\s+me|find|search\s+for)\s+"
)

_PALEO_TERMS = {
    "paleontology", "palaeontology", "paleobiology", "palaeobiology",
    "fossil", "dinosaur", "extinct", "prehistoric",
}

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,abstract,externalIds"


def _build_s2_query(user_query: str) -> str:
    """
    Build a Semantic Scholar search query from a natural-language user question.

    1. Strip leading question words / conversational prefixes.
    2. Remove trailing punctuation.
    3. Append "paleontology" if no domain term is already present, so generic
       subjects (e.g. "locomotion", "bone structure") return on-topic papers.

    S2 uses its own relevance ranking over plain keyword queries; boolean
    operators (AND/OR) are not needed and can hurt precision here.
    """
    q = _QUESTION_PREFIX.sub("", user_query).rstrip("?.! ").strip()
    if not q:
        q = user_query.strip()
    if not any(t in q.lower() for t in _PALEO_TERMS):
        q = q + " paleontology"
    logger.info("S2 query constructed: %r", q)
    return q


# -- Semantic Scholar pre-fetch ------------------------------------------------
def _fetch_semantic_scholar(query: str) -> str:
    """
    Fetch academic abstracts from the Semantic Scholar Graph API.

    Endpoint: GET https://api.semanticscholar.org/graph/v1/paper/search
    Fields:   title, authors, year, abstract, externalIds
    Auth:     none required for unauthenticated access (rate-limited to
              ~100 req/s across all anonymous callers — well within app usage)

    Returns a formatted string ready to inject into the user message, or an
    empty string if no results are found or the request fails (fails loudly
    with exception type logged so cloud-side errors are diagnosable).

    Paper identifier priority: ArXiv ID > DOI > S2 paperId.
    """
    logger.info("Fetching Semantic Scholar abstracts for: %r", query)
    try:
        params = urllib.parse.urlencode({
            "query": _build_s2_query(query),
            "fields": _S2_FIELDS,
            "limit": S2_MAX_RESULTS,
        })
        url = f"{_S2_API}?{params}"
        logger.info("S2 URL: %s", url)

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "paleo-research-assistant/0.9 (research tool)"},
        )
        with urllib.request.urlopen(req, timeout=S2_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        papers = payload.get("data", [])
        if not papers:
            logger.info("S2: no results for this query.")
            return ""

        logger.info("S2: %d result(s) retrieved.", len(papers))
        sections = []
        for paper in papers:
            title = (paper.get("title") or "Untitled").strip()
            abstract = (paper.get("abstract") or "").strip()
            year = str(paper.get("year") or "n.d.")

            raw_authors = paper.get("authors") or []
            author_names = [(a.get("name") or "").strip() for a in raw_authors if a.get("name")]
            author_str = ", ".join(author_names[:3])
            if len(author_names) > 3:
                author_str += " et al."

            ext = paper.get("externalIds") or {}
            if ext.get("ArXiv"):
                paper_id = "arXiv:" + ext["ArXiv"]
            elif ext.get("DOI"):
                paper_id = "DOI:" + ext["DOI"]
            else:
                paper_id = "S2:" + (paper.get("paperId") or "unknown")

            block = (
                "Title: " + title + "\n"
                + "Authors: " + author_str + " (" + year + ")\n"
                + "ID: " + paper_id + "\n"
                + "Abstract: " + abstract
            )
            sections.append(block)

        return "\n\n---\n\n".join(sections)

    except Exception as exc:
        logger.warning("S2 fetch failed (%s): %s", type(exc).__name__, exc)
        return ""


# -- Helpers -------------------------------------------------------------------
def _log_content_blocks(content: list) -> None:
    """Emit a diagnostic line showing each block type returned by the API."""
    summary = [getattr(b, "type", type(b).__name__) for b in content]
    logger.info("Content blocks received: %s", summary)


def _extract_final_text(content: list) -> str | None:
    """
    Join ALL text blocks into one string.

    When web search fires, Claude writes text fragments between each search:
      text -> server_tool_use -> web_search_tool_result -> text -> ...
    Each fragment is a piece of the structured response (headings, bullets,
    sources). Concatenating them reconstructs the complete summary.
    """
    text_blocks = [b for b in content if hasattr(b, "text")]
    logger.info("Text blocks found: %d", len(text_blocks))
    return "".join(b.text for b in text_blocks) if text_blocks else None


# -- Agent entry point ---------------------------------------------------------
def run_agent(query: str, history: list | None = None) -> str:
    """
    Accept a natural language research query and optional conversation history.

    history: list of {"role": "user"/"assistant", "content": str} pairs from
             prior turns. Entries use plain text only (no tool-use blocks).
             Pass None or [] for the first turn.

    Steps:
      1. Pre-fetch Semantic Scholar abstracts for the current query and inject
         into the current user message (prior turns are not re-augmented).
      2. Build messages = history + [current user message].
      3. Run the agentic loop with native web search.
      4. Concatenate all text blocks from the final response.

    stop_reason flow:
      end_turn   -> done; extract and return text.
      pause_turn -> server hit its iteration cap; append and continue.
    """
    logger.info("Agent received query: %r (history turns: %d)", query, len(history or []))

    s2_context = _fetch_semantic_scholar(query)
    if s2_context:
        user_content = (
            query
            + "\n\nThe following Semantic Scholar abstracts are provided as additional context. "
            + "Incorporate relevant findings and cite them in the arXiv Papers "
            + "section:\n\n"
            + s2_context
        )
    else:
        user_content = query

    messages = list(history or []) + [{"role": "user", "content": user_content}]

    try:
        for iteration in range(MAX_LOOP_ITERATIONS):
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                tool_choice={"type": "any"},   # force web_search invocation
                messages=messages,
            )

            logger.info(
                "Iteration %d: stop_reason=%r", iteration + 1, response.stop_reason
            )
            _log_content_blocks(response.content)

            if response.stop_reason == "end_turn":
                text = _extract_final_text(response.content)
                if text:
                    logger.info("Agent completed successfully.")
                    return text
                return "## Research Summary\n\n**Status:** No text response received."

            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            logger.warning("Unexpected stop_reason: %r", response.stop_reason)
            messages.append({"role": "assistant", "content": response.content})

        logger.warning("Agent loop exhausted without reaching end_turn.")
        return (
            "## Research Summary\n\n"
            "**Status:** The research assistant reached its iteration limit. "
            "Please try a more specific query."
        )

    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return (
            "## Research Summary\n\n**Status:** The research assistant encountered "
            "an error and could not complete your request. Please try again.\n\n"
            "**Error:** " + str(exc)
        )
