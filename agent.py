"""
Paleontology Research Assistant -- Agent Loop
V0.7: Multi-turn support. run_agent accepts a history list of prior
      {"role", "content"} pairs and prepends them to the messages array
      so follow-up questions have full prior context.
      History entries use plain text only -- no server_tool_use or
      web_search_tool_result blocks -- keeping context compact.
"""

import logging
import re
import arxiv
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
ARXIV_MAX_RESULTS = 5

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
and pre-fetched arXiv abstracts supplied in the user message.
Use web search to retrieve current findings, then synthesise both the web
results and the provided arXiv abstracts into a structured summary using
this exact format:

## Research Summary

**Query:** [restate the query]

### Key Findings
- [finding 1]
- [finding 2]
- [finding 3]

### arXiv Papers
- [paper title, authors, year, arXiv ID]
- [paper title, authors, year, arXiv ID]

### Sources
- [web source or reference 1]
- [web source or reference 2]

### Open Questions
- [open question 1]
- [open question 2]

Always use web search before answering. Incorporate arXiv abstracts where relevant.
Cite arXiv papers in the arXiv Papers section using their IDs.
Cite web results in Sources. Be specific and accurate. If you are uncertain
about something, say so rather than speculating.
Keep findings concise -- one to two sentences each."""


# -- arXiv query construction --------------------------------------------------
# Strips conversational prefixes so "What dinosaurs were found in 2024?"
# becomes "dinosaurs were found in 2024", then ANDs with a paleontology anchor
# so generic terms (evolution, species) return on-topic papers.
# q-bio.PE alone is too narrow; many paleo papers carry no arXiv category.
_QUESTION_PREFIX = re.compile(
    r"(?i)^\s*"
    r"(?:what(?:\s+(?:is|are|was|were|do|does|did))?|how|why|when|where|who|which|"
    r"tell\s+me\s+(?:about|of)|explain|describe|give\s+me|find|search\s+for)\s+"
)

_PALEO_ANCHOR = (
    "(paleontology OR paleobiology OR fossil OR dinosaur OR "
    "palaeontology OR palaeobiology OR extinct OR prehistoric)"
)


def _build_arxiv_query(user_query: str) -> str:
    """
    Convert a natural-language user query into a focused arXiv search string.

    1. Strip leading question words / conversational prefixes.
    2. Remove trailing punctuation.
    3. AND with a paleontology domain anchor so generic keywords stay on-topic.

    Example:
      "What new theropod species were named in 2024?"
      -> "(new theropod species were named in 2024) AND (paleontology OR ...)"
    """
    q = _QUESTION_PREFIX.sub("", user_query).rstrip("?.! ").strip()
    if not q:
        q = user_query.strip()
    arxiv_query = f"({q}) AND {_PALEO_ANCHOR}"
    logger.info("arXiv query constructed: %r", arxiv_query)
    return arxiv_query


# -- arXiv pre-fetch -----------------------------------------------------------
def _fetch_arxiv_abstracts(query: str) -> str:
    """
    Search arXiv for abstracts relevant to the query.
    Returns a formatted string ready to inject into the user message, or an
    empty string if no results are found or the request fails.
    """
    logger.info("Fetching arXiv abstracts for: %r", query)
    try:
        search = arxiv.Search(
            query=_build_arxiv_query(query),
            max_results=ARXIV_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = list(arxiv.Client().results(search))
        if not papers:
            logger.info("arXiv: no results.")
            return ""
        logger.info("arXiv: %d result(s) retrieved.", len(papers))
        sections = []
        for paper in papers:
            authors = ", ".join(str(a) for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            year = paper.published.year if paper.published else "n.d."
            block = (
                "Title: " + paper.title + "\n"
                + "Authors: " + authors + " (" + str(year) + ")\n"
                + "arXiv ID: " + paper.entry_id + "\n"
                + "Abstract: " + paper.summary
            )
            sections.append(block)
        return "\n\n---\n\n".join(sections)
    except Exception as exc:
        logger.warning("arXiv fetch failed: %s", exc)
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
      1. Pre-fetch arXiv abstracts for the current query and inject into the
         current user message (prior turns are not re-augmented).
      2. Build messages = history + [current user message].
      3. Run the agentic loop with native web search.
      4. Concatenate all text blocks from the final response.

    stop_reason flow:
      end_turn   -> done; extract and return text.
      pause_turn -> server hit its iteration cap; append and continue.
    """
    logger.info("Agent received query: %r (history turns: %d)", query, len(history or []))

    arxiv_context = _fetch_arxiv_abstracts(query)
    if arxiv_context:
        user_content = (
            query
            + "\n\nThe following arXiv abstracts are provided as additional context. "
            + "Incorporate relevant findings and cite them in the arXiv Papers "
            + "section:\n\n"
            + arxiv_context
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
