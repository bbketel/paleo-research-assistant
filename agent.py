"""
Paleontology Research Assistant -- Agent Loop
V1.1: Autonomous multi-search loop. After the initial research pass, a
      deterministic gap analysis scans the output for uncertainty signals and
      Open Questions bullets, generates up to 2 targeted follow-up queries,
      runs them as additional search passes (3 passes total max), and merges
      results into a 'Further Research' section before final tier annotation.
      No LLM involvement in gap detection or query generation; all pattern
      matching is pure Python.
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
MAX_LOOP_ITERATIONS = 5    # guard against runaway inner loops
MAX_PASSES = 3             # 1 initial + up to 2 follow-up searches
MAX_FOLLOWUP_QUERIES = 2   # follow-up searches per run_agent call
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

# -- Source tier keyword lists -------------------------------------------------
# Matched in priority order (T1 first) against the lowercased source line,
# which typically contains both the URL and the title as written by Claude.
# T4 is the fallback when nothing matches.

_TIER1_KEYWORDS = [
    # Publisher domains
    "nature.com", "sciencemag.org", "science.org/doi",
    "plos.org", "plosone.org",
    "cell.com", "thecellpress",
    "elifesciences.org",
    "royalsocietypublishing.org",
    "academic.oup.com",
    "cambridge.org/core",
    "biorxiv.org",
    "peerj.com",
    "jstor.org",
    "pubmed", "ncbi.nlm.nih.gov",
    "doi.org/10.", "doi:10.",
    "vertpaleo.org",
    "paleosoc.org",
    "pensoft.net",
    # Specific journal names (as Claude would write them in a citation)
    "journal of vertebrate paleontology",
    "journal of systematic palaeontology",
    "journal of systematic paleontology",
    "journal of paleontology",
    "cretaceous research",
    "palaeogeography",
    "acta palaeontologica polonica",
    "proceedings of the royal society",
    "proc. r. soc",
    "plos one", "plos biology",
    "science advances",
    "current biology",
    "nature ecology", "nature communications",
    "scientific reports",
    "zookeys",
    "peerj",
    "elife",
    "palaeo-electronica.org",
    "palaeoelectronica",
    "palaeontologia electronica",
]

_TIER2_KEYWORDS = [
    # TLD patterns — reliable institutional indicators
    ".edu", ".ac.uk", ".ac.jp", ".ac.au", ".ac.nz",
    ".gov",
    # Specific museum / institution domains
    "nhm.ac.uk", "nhm.org",
    "amnh.org",
    "fieldmuseum.org",
    "si.edu",
    "ucmp.berkeley",
    "peabody.yale",
    "paleobiodb.org",
    "fossilworks.org",
    "usgs.gov", "nps.gov",
    "ufhealth.org",
    # Text patterns that appear in Claude's citations
    "natural history museum",
    "university press",
    "smithsonian institution",
    "carnegie museum",
    "royal tyrrell",
    "american museum of natural history",
    "vertebrate paleontology lab",
]

_TIER3_KEYWORDS = [
    # Science journalism outlets (URL fragments and publication names)
    "bbc.com", "bbc.co.uk",
    "npr.org",
    "nationalgeographic.com", "natgeo.com",
    "sciencedaily.com", "sciencedaily",
    "scientificamerican.com", "scientific american",
    "sciencenews.org", "science news",
    "smithsonianmag.com", "smithsonian magazine",
    "livescience.com", "livescience",
    "phys.org",
    "newscientist.com", "new scientist",
    "wired.com",
    "discovermagazine.com", "discover magazine",
    "eos.org",
    "theatlantic.com",
    "popsci.com", "popular science",
    "popularmechanics.com",
    # Names as written in citations (no URL present)
    "national geographic",
    "science daily",
]

# Regex to strip any existing [T1]-[T4] label before re-annotating,
# preventing double-labelling when Claude follows the system prompt instruction.
_EXISTING_TIER_LABEL = re.compile(r"\s*\[T[1-4]\]")

# -- Gap analysis patterns -----------------------------------------------------
# Matched against the full response text to detect uncertainty signals.
# Used for logging the gap signal count; Open Questions bullets are the primary
# source of follow-up queries.
_GAP_SIGNAL_RE = re.compile(
    r"\b(?:remains?\s+unclear|is\s+debated|further\s+research\s+(?:is\s+)?needed|"
    r"unknown|uncertain(?:ty)?|not\s+yet\s+(?:known|determined|understood)|"
    r"poorly\s+understood|little\s+is\s+known|limited\s+evidence|"
    r"more\s+research|needs?\s+(?:further|more)\s+(?:study|research|investigation))\b",
    re.IGNORECASE,
)

# Captures all bullet lines under the '### Open Questions' heading.
_OPEN_Q_RE = re.compile(
    r"###\s+Open\s+Questions\s*\n((?:[ \t]*-[ \t]+.+\n?)*)",
    re.IGNORECASE,
)

# Guards _annotate_tier_labels: a bullet line is only scored if it looks like
# an actual citation — contains a URL or a 4-digit publication year.
_CITATION_LINE_RE = re.compile(r"https?://|\b(19|20)\d{2}\b")


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
- [Title](URL)
- Title (Publication, Year)

Format each source as a markdown link - [Title](URL) - when a URL is available.
If no URL is available, write - Title (Publication, Year) with no placeholder characters.

### Open Questions
- [open question 1]
- [open question 2]

Source quality tiers — apply [T1]-[T4] labels in the Sources section:
  [T1] Peer-reviewed journal (Nature, Science, PLOS, Cell, Royal Society,
       Oxford/Cambridge journals, Journal of Vertebrate Paleontology,
       Journal of Systematic Palaeontology, Cretaceous Research, etc.)
  [T2] Institutional or museum source (.edu, .gov, natural history museum,
       university press release, government agency, paleontology database)
  [T3] Science journalism (NPR, BBC, National Geographic, Scientific American,
       ScienceDaily, Smithsonian Magazine, New Scientist, LiveScience)
  [T4] General news or blog

When T1 or T2 sources conflict with T3 or T4 on the same finding, weight
the higher-tier source. Add a [T1]-[T4] label to every Sources bullet.

Always use web search before answering. Incorporate Semantic Scholar abstracts where relevant.
Cite academic papers in the arXiv Papers section using their IDs.
Cite web results in Sources. Be specific and accurate. If you are uncertain
about something, say so rather than speculating.
Keep findings concise -- one to two sentences each."""


# -- Follow-up search system prompt --------------------------------------------
_FOLLOWUP_SYSTEM_PROMPT = """You are a paleontology research assistant filling a specific knowledge gap
identified in a prior research pass. Use web search to find current information
directly addressing the provided gap or open question. Return your findings in
this exact format:

**Gap addressed:** [restate the gap or question]

**New Findings:**
- [finding 1]
- [finding 2]

### Sources
- [source 1]
- [source 2]

Keep findings concise (one to two sentences each) and directly relevant to the gap.
Add a [T1]-[T4] tier label to every Sources bullet:
  [T1] Peer-reviewed journal
  [T2] Institutional / museum source
  [T3] Science journalism
  [T4] General news or blog"""


# -- Source quality scoring ----------------------------------------------------
def _score_source(line: str) -> tuple[int, str]:
    """
    Deterministically assign a quality tier to a source line.

    Checks keyword lists in priority order (T1 -> T2 -> T3 -> T4 default).
    Matching is case-insensitive substring search on the full line, which
    typically contains both URL and title as written by Claude.

    Returns (tier: int, matched_keyword: str) for logging.
    """
    t = line.lower()
    for kw in _TIER1_KEYWORDS:
        if kw in t:
            return 1, kw
    for kw in _TIER2_KEYWORDS:
        if kw in t:
            return 2, kw
    for kw in _TIER3_KEYWORDS:
        if kw in t:
            return 3, kw
    return 4, "(no keyword matched)"


def _annotate_tier_labels(text: str) -> str:
    """
    Post-process the response text to add deterministic [T1]-[T4] labels.

    Locates the '### Sources' section, scores every bullet line via
    _score_source(), strips any label Claude may have already added to
    prevent duplicates, appends the authoritative label, and logs each
    assignment.  Lines outside the Sources section are returned unchanged.
    Handles multiple '### Sources' sections (e.g. in Further Research).
    """
    lines = text.split("\n")
    result = []
    in_sources = False

    for line in lines:
        stripped = line.strip()

        # Detect section boundaries
        if stripped.startswith("### Sources"):
            in_sources = True
            result.append(line)
            continue
        if in_sources and stripped.startswith("##"):
            in_sources = False

        if in_sources and stripped.startswith("- "):
            if not _CITATION_LINE_RE.search(stripped):
                result.append(line)
                continue
            base = _EXISTING_TIER_LABEL.sub("", line.rstrip())
            tier, matched = _score_source(base)
            logger.info("Source tier T%d (matched %r): %s", tier, matched, base.strip()[:100])
            result.append(base + " [T" + str(tier) + "]")
        else:
            result.append(line)

    return "\n".join(result)


# -- Semantic Scholar query construction ---------------------------------------
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
    3. Append "paleontology" if no domain term is already present.
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

    Returns a formatted string ready to inject into the user message, or an
    empty string if no results are found or the request fails.
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
            headers={"User-Agent": "paleo-research-assistant/1.0 (research tool)"},
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


# -- Query cleaning ------------------------------------------------------------
_MD_EMPHASIS_RE = re.compile(r"\*+(.+?)\*+")


def _clean_query(text: str) -> str:
    """Strip markdown bold (**text**) and italic (*text*) markers from a query string."""
    return _MD_EMPHASIS_RE.sub(r"\1", text).strip()


# -- Gap analysis --------------------------------------------------------------
def _extract_followup_queries(text: str) -> list[str]:
    """
    Extract up to MAX_FOLLOWUP_QUERIES follow-up search queries from the text.

    Priority 1: Open Questions bullets — used directly as queries since they
                are already in question form and represent the most explicit gaps.
    Priority 2: Lines containing uncertainty signal phrases — used if Priority 1
                yields fewer than MAX_FOLLOWUP_QUERIES queries.

    Returns a deduplicated list capped at MAX_FOLLOWUP_QUERIES.
    """
    queries: list[str] = []

    # Priority 1 — Open Questions bullets
    oq_match = _OPEN_Q_RE.search(text)
    if oq_match:
        for raw in oq_match.group(1).splitlines():
            q = raw.strip().lstrip("- ").strip()
            if q and q not in queries:
                queries.append(q)
                if len(queries) >= MAX_FOLLOWUP_QUERIES:
                    return queries

    # Priority 2 — lines with uncertainty signal phrases
    for line in text.splitlines():
        if len(queries) >= MAX_FOLLOWUP_QUERIES:
            break
        stripped = line.strip().lstrip("- ").strip()
        if _GAP_SIGNAL_RE.search(stripped) and 20 < len(stripped) < 250:
            if stripped not in queries:
                queries.append(stripped)

    return queries[:MAX_FOLLOWUP_QUERIES]


# -- Inner agentic loop --------------------------------------------------------
def _run_pass(messages: list, system_prompt: str, max_tokens: int = MAX_TOKENS) -> str | None:
    """
    Execute one agentic pass: call the API, handle pause_turn continuations,
    and return concatenated text on end_turn. Returns None if end_turn is
    never reached within MAX_LOOP_ITERATIONS.

    Mutates `messages` in place when appending assistant turns during
    pause_turn handling (same pattern as the original inline loop).
    """
    for iteration in range(MAX_LOOP_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=TOOLS,
            tool_choice={"type": "any"},
            messages=messages,
        )

        logger.info("  iteration %d: stop_reason=%r", iteration + 1, response.stop_reason)
        _log_content_blocks(response.content)

        if response.stop_reason == "end_turn":
            return _extract_final_text(response.content)

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        logger.warning("Unexpected stop_reason: %r", response.stop_reason)
        messages.append({"role": "assistant", "content": response.content})

    logger.warning("Pass exhausted %d iterations without end_turn.", MAX_LOOP_ITERATIONS)
    return None


# -- Agent entry point ---------------------------------------------------------
def run_agent(query: str, history: list | None = None) -> str:
    """
    Accept a natural language research query and optional conversation history.

    history: list of {"role": "user"/"assistant", "content": str} pairs from
             prior turns. Entries use plain text only (no tool-use blocks).
             Pass None or [] for the first turn.

    Steps:
      1. Pre-fetch Semantic Scholar abstracts; inject into current user message.
      2. Pass 1 — run the initial agentic loop with native web search.
      3. Gap analysis — count uncertainty signals; extract Open Questions as
         follow-up queries (up to MAX_FOLLOWUP_QUERIES).
      4. Passes 2–MAX_PASSES — run one targeted follow-up search per query.
      5. Merge initial text + supplemental results into a 'Further Research'
         section (if any follow-ups returned content).
      6. Apply deterministic tier labels to all Sources sections and return.

    stop_reason flow inside each pass:
      end_turn   -> done; extract text and return to caller.
      pause_turn -> server hit its iteration cap; append and continue.
    """
    logger.info("Agent received query: %r (history turns: %d)", query, len(history or []))

    # -- Pass 1: initial research ----------------------------------------------
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
        logger.info("Pass 1 — initial research")
        initial_text = _run_pass(messages, SYSTEM_PROMPT)
        if not initial_text:
            logger.warning("Pass 1 returned no text.")
            return (
                "## Research Summary\n\n"
                "**Status:** The research assistant reached its iteration limit. "
                "Please try a more specific query."
            )
        logger.info("Pass 1 complete.")

        # -- Gap analysis ------------------------------------------------------
        signal_count = len(_GAP_SIGNAL_RE.findall(initial_text))
        followup_queries = _extract_followup_queries(initial_text)
        logger.info(
            "Gap analysis: %d uncertainty signal(s), %d follow-up quer%s generated",
            signal_count,
            len(followup_queries),
            "y" if len(followup_queries) == 1 else "ies",
        )

        # -- Passes 2–MAX_PASSES: targeted follow-up searches ------------------
        supplementals: list[str] = []
        for pass_num, fq in enumerate(followup_queries[:MAX_PASSES - 1], start=2):
            fq = _clean_query(fq)
            logger.info("Pass %d — follow-up search: %r", pass_num, fq)
            fq_messages = [
                {
                    "role": "user",
                    "content": (
                        f"In research about '{query}', this gap or open question was identified:\n\n"
                        f"{fq}\n\n"
                        f"Please search for current information directly addressing this gap."
                    ),
                }
            ]
            supp = _run_pass(fq_messages, _FOLLOWUP_SYSTEM_PROMPT, max_tokens=2048)
            if supp:
                supplementals.append(supp)
                logger.info("Pass %d complete.", pass_num)
            else:
                logger.warning("Pass %d returned no text.", pass_num)

        # -- Merge and annotate ------------------------------------------------
        if supplementals:
            further = "\n\n### Further Research\n\n" + "\n\n---\n\n".join(supplementals)
            final_text = initial_text + further
        else:
            final_text = initial_text

        final_text = _annotate_tier_labels(final_text)
        logger.info(
            "Agent completed successfully (%d pass(es) total).", 1 + len(supplementals)
        )
        return final_text

    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return (
            "## Research Summary\n\n**Status:** The research assistant encountered "
            "an error and could not complete your request. Please try again.\n\n"
            "**Error:** " + str(exc)
        )
