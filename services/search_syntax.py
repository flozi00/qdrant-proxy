"""Google dork style query parsing and result filtering helpers."""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Sequence
from urllib.parse import urlsplit


_QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "’": "'",
        "‘": "'",
    }
)

_FIELD_PATTERN = re.compile(
    r"(?P<prefix>[+-]?)"
    r"(?P<operator>"
    r"allintext|intext|allinurl|inurl|allintitle|intitle|site|filetype|ext|"
    r"link|allinanchor|inanchor|allinpostauthor|inpostauthor|related|cache|"
    r"before|after|numrange"
    r"):(?P<value>\"[^\"]*\"|\([^)]*\)|[^\s()]+)",
    flags=re.IGNORECASE,
)

SEARCH_CANDIDATE_MULTIPLIER = 5
SEARCH_CANDIDATE_BUFFER = 20
SEARCH_CANDIDATE_MAX_LIMIT = 200
SCROLL_BATCH_MULTIPLIER = 3
SCROLL_BATCH_MIN_LIMIT = 25


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        candidate = (value or "").strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(candidate)
    return ordered


def _normalize_clause_value(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1]
    if value.startswith("(") and value.endswith(")") and len(value) >= 2:
        value = value[1:-1]
    return value.strip()


def _split_clause_terms(raw_value: str, *, split_words: bool = False) -> list[str]:
    value = _normalize_clause_value(raw_value)
    if not value:
        return []

    if "|" in value:
        parts = [part.strip().strip('"').strip("'") for part in value.split("|")]
        return _dedupe(parts)

    if split_words:
        return _dedupe(value.split())

    return [value]


def _parse_date(raw_value: str) -> Optional[datetime]:
    value = _normalize_clause_value(raw_value)
    if not value:
        return None

    try:
        if len(value) == 10:
            parsed = date.fromisoformat(value)
            return datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)

        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _parse_free_terms(query: str) -> tuple[list[str], list[str]]:
    if not query.strip():
        return [], []

    try:
        tokens = shlex.split(query)
    except ValueError:
        tokens = query.split()

    positive_terms: list[str] = []
    negative_terms: list[str] = []

    for token in tokens:
        cleaned = token.strip()
        if not cleaned or cleaned in {"&", "|", "&&", "||"}:
            continue

        sign = ""
        while cleaned and cleaned[0] in "+-~":
            if not sign and cleaned[0] in "+-":
                sign = cleaned[0]
            cleaned = cleaned[1:]

        cleaned = cleaned.strip("() ")
        if not cleaned or cleaned in {"&", "|"}:
            continue

        parts = _split_clause_terms(cleaned) if "|" in cleaned else [cleaned]
        for part in parts:
            target = negative_terms if sign == "-" else positive_terms
            target.append(part)

    return _dedupe(positive_terms), _dedupe(negative_terms)


def expanded_candidate_limit(limit: int) -> int:
    return min(
        max(limit * SEARCH_CANDIDATE_MULTIPLIER, limit + SEARCH_CANDIDATE_BUFFER),
        SEARCH_CANDIDATE_MAX_LIMIT,
    )


def scroll_batch_limit(limit: int) -> int:
    return min(
        max(limit * SCROLL_BATCH_MULTIPLIER, SCROLL_BATCH_MIN_LIMIT),
        SEARCH_CANDIDATE_MAX_LIMIT,
    )


@dataclass
class ParsedSearchQuery:
    original_query: str
    free_terms: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    content_any: list[str] = field(default_factory=list)
    content_all: list[str] = field(default_factory=list)
    excluded_content: list[str] = field(default_factory=list)
    title_any: list[str] = field(default_factory=list)
    title_all: list[str] = field(default_factory=list)
    excluded_title: list[str] = field(default_factory=list)
    url_any: list[str] = field(default_factory=list)
    url_all: list[str] = field(default_factory=list)
    excluded_url: list[str] = field(default_factory=list)
    site_patterns: list[str] = field(default_factory=list)
    excluded_site_patterns: list[str] = field(default_factory=list)
    filetypes: list[str] = field(default_factory=list)
    excluded_filetypes: list[str] = field(default_factory=list)
    link_any: list[str] = field(default_factory=list)
    link_all: list[str] = field(default_factory=list)
    excluded_links: list[str] = field(default_factory=list)
    before: Optional[datetime] = None
    after: Optional[datetime] = None

    @property
    def retrieval_terms(self) -> list[str]:
        return _dedupe(
            [
                *self.free_terms,
                *self.content_any,
                *self.content_all,
                *self.title_any,
                *self.title_all,
            ]
        )

    @property
    def has_text_query(self) -> bool:
        return bool(self.retrieval_terms)

    @property
    def has_structured_filters(self) -> bool:
        return any(
            [
                self.excluded_terms,
                self.content_any,
                self.content_all,
                self.excluded_content,
                self.title_any,
                self.title_all,
                self.excluded_title,
                self.url_any,
                self.url_all,
                self.excluded_url,
                self.site_patterns,
                self.excluded_site_patterns,
                self.filetypes,
                self.excluded_filetypes,
                self.link_any,
                self.link_all,
                self.excluded_links,
                self.before,
                self.after,
            ]
        )

    @property
    def semantic_query(self) -> str:
        return " ".join(self.retrieval_terms).strip()


def parse_google_dork_query(query: str) -> ParsedSearchQuery:
    normalized_query = (query or "").translate(_QUOTE_TRANSLATION)
    parsed = ParsedSearchQuery(original_query=query)
    spans: list[tuple[int, int]] = []

    for match in _FIELD_PATTERN.finditer(normalized_query):
        spans.append(match.span())
        prefix = match.group("prefix")
        operator = match.group("operator").lower()
        raw_value = match.group("value")
        negative = prefix == "-"

        def assign_filter_values(
            values: Sequence[str], positive_attr: str, negative_attr: str
        ) -> None:
            target_attr = negative_attr if negative else positive_attr
            current = getattr(parsed, target_attr)
            current.extend(values)
            setattr(parsed, target_attr, _dedupe(current))

        if operator in {"intext", "allintext"}:
            assign_filter_values(
                _split_clause_terms(raw_value, split_words=operator == "allintext"),
                "content_any" if operator == "intext" else "content_all",
                "excluded_content",
            )
        elif operator in {"intitle", "allintitle"}:
            assign_filter_values(
                _split_clause_terms(raw_value, split_words=operator == "allintitle"),
                "title_any" if operator == "intitle" else "title_all",
                "excluded_title",
            )
        elif operator in {"inurl", "allinurl", "related", "cache"}:
            assign_filter_values(
                _split_clause_terms(raw_value, split_words=operator == "allinurl"),
                "url_any" if operator != "allinurl" else "url_all",
                "excluded_url",
            )
        elif operator == "site":
            assign_filter_values(
                _split_clause_terms(raw_value),
                "site_patterns",
                "excluded_site_patterns",
            )
        elif operator in {"filetype", "ext"}:
            values = [value.lstrip(".").lower() for value in _split_clause_terms(raw_value)]
            assign_filter_values(values, "filetypes", "excluded_filetypes")
        elif operator in {"link", "inanchor", "allinanchor"}:
            assign_filter_values(
                _split_clause_terms(raw_value, split_words=operator == "allinanchor"),
                "link_any" if operator != "allinanchor" else "link_all",
                "excluded_links",
            )
        elif operator in {"inpostauthor", "allinpostauthor"}:
            assign_filter_values(
                _split_clause_terms(raw_value, split_words=operator == "allinpostauthor"),
                "content_any" if operator != "allinpostauthor" else "content_all",
                "excluded_content",
            )
        elif operator == "before":
            if parsed_before := _parse_date(raw_value):
                parsed.before = parsed_before
        elif operator == "after":
            if parsed_after := _parse_date(raw_value):
                parsed.after = parsed_after
        elif operator == "numrange":
            parsed.free_terms.extend(_split_clause_terms(raw_value))

    remaining_chars = list(normalized_query)
    for start, end in spans:
        for idx in range(start, end):
            remaining_chars[idx] = " "

    free_terms, excluded_terms = _parse_free_terms("".join(remaining_chars))
    parsed.free_terms = _dedupe([*parsed.free_terms, *free_terms])
    parsed.excluded_terms = _dedupe([*parsed.excluded_terms, *excluded_terms])
    return parsed


def _normalize_text(value: Any) -> str:
    return str(value or "").lower()


def _matches_any(text: str, terms: Sequence[str]) -> bool:
    return not terms or any(term.lower() in text for term in terms)


def _matches_all(text: str, terms: Sequence[str]) -> bool:
    return not terms or all(term.lower() in text for term in terms)


def _matches_none(text: str, terms: Sequence[str]) -> bool:
    return not any(term.lower() in text for term in terms)


def _extract_filetype(url: str) -> str:
    path = urlsplit(url).path or ""
    if "." not in path.rsplit("/", 1)[-1]:
        return ""
    return path.rsplit(".", 1)[-1].lower()


def _site_matches(hostname: str, pattern: str) -> bool:
    host = (hostname or "").lower().lstrip(".")
    normalized = (pattern or "").strip().lower().lstrip(".")
    if not host or not normalized:
        return False
    if normalized.startswith("www."):
        normalized = normalized[4:]
    if host.startswith("www."):
        host = host[4:]
    return fnmatch.fnmatch(host, normalized) or host == normalized


def _parse_payload_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def document_matches_query(parsed: ParsedSearchQuery, payload: dict[str, Any]) -> bool:
    payload = payload or {}
    metadata = payload.get("metadata") or {}
    content = _normalize_text(payload.get("content"))
    title = _normalize_text(payload.get("title") or metadata.get("title"))
    url = _normalize_text(payload.get("url"))
    hostname = (urlsplit(payload.get("url", "")).hostname or metadata.get("domain") or "").lower()
    hyperlinks = " ".join(payload.get("hyperlinks") or [])
    links = _normalize_text(hyperlinks)
    filetype = _extract_filetype(payload.get("url", ""))
    indexed_at = _parse_payload_datetime(metadata.get("indexed_at"))

    generic_text = "\n".join(part for part in [content, title, url] if part)

    if not _matches_all(content, parsed.content_all):
        return False
    if not _matches_any(content, parsed.content_any):
        return False
    if not _matches_none(content, parsed.excluded_content):
        return False

    if not _matches_all(title, parsed.title_all):
        return False
    if not _matches_any(title, parsed.title_any):
        return False
    if not _matches_none(title, parsed.excluded_title):
        return False

    if not _matches_all(url, parsed.url_all):
        return False
    if not _matches_any(url, parsed.url_any):
        return False
    if not _matches_none(url, parsed.excluded_url):
        return False

    if parsed.site_patterns and not any(
        _site_matches(hostname, pattern) for pattern in parsed.site_patterns
    ):
        return False
    if any(_site_matches(hostname, pattern) for pattern in parsed.excluded_site_patterns):
        return False

    if parsed.filetypes and filetype not in parsed.filetypes:
        return False
    if filetype and filetype in parsed.excluded_filetypes:
        return False

    if not _matches_all(links, parsed.link_all):
        return False
    if not _matches_any(links, parsed.link_any):
        return False
    if not _matches_none(links, parsed.excluded_links):
        return False

    if not _matches_all(generic_text, parsed.free_terms):
        return False
    if not _matches_none(generic_text, parsed.excluded_terms):
        return False

    if parsed.before and indexed_at and not indexed_at < parsed.before:
        return False
    if parsed.after and indexed_at and not indexed_at > parsed.after:
        return False
    if (parsed.before or parsed.after) and indexed_at is None:
        return False

    return True


def faq_matches_query(parsed: ParsedSearchQuery, faq: dict[str, Any]) -> bool:
    text = _normalize_text(f"{faq.get('question', '')}\n{faq.get('answer', '')}")
    urls = [
        str(source.get("url", "")).strip()
        for source in faq.get("source_documents") or []
        if isinstance(source, dict)
    ]
    url_blob = _normalize_text(" ".join(urls))
    link_blob = url_blob
    hostnames = [(urlsplit(url).hostname or "").lower() for url in urls]
    filetypes = {_extract_filetype(url) for url in urls if url}

    if not _matches_all(text, [*parsed.content_all, *parsed.title_all]):
        return False
    if not _matches_any(text, [*parsed.content_any, *parsed.title_any]):
        return False
    if not _matches_none(text, [*parsed.excluded_content, *parsed.excluded_title]):
        return False

    if not _matches_all(url_blob, parsed.url_all):
        return False
    if not _matches_any(url_blob, parsed.url_any):
        return False
    if not _matches_none(url_blob, parsed.excluded_url):
        return False

    if parsed.site_patterns and not any(
        _site_matches(hostname, pattern)
        for hostname in hostnames
        for pattern in parsed.site_patterns
    ):
        return False
    if any(
        _site_matches(hostname, pattern)
        for hostname in hostnames
        for pattern in parsed.excluded_site_patterns
    ):
        return False

    if parsed.filetypes and not any(filetype in parsed.filetypes for filetype in filetypes):
        return False
    if any(filetype in parsed.excluded_filetypes for filetype in filetypes):
        return False

    if not _matches_all(link_blob, parsed.link_all):
        return False
    if not _matches_any(link_blob, parsed.link_any):
        return False
    if not _matches_none(link_blob, parsed.excluded_links):
        return False

    if not _matches_all("\n".join([text, url_blob]), parsed.free_terms):
        return False
    if not _matches_none("\n".join([text, url_blob]), parsed.excluded_terms):
        return False

    faq_timestamp = _parse_payload_datetime(faq.get("last_updated") or faq.get("first_seen"))
    if parsed.before and faq_timestamp and not faq_timestamp < parsed.before:
        return False
    if parsed.after and faq_timestamp and not faq_timestamp > parsed.after:
        return False
    if (parsed.before or parsed.after) and faq_timestamp is None:
        return False

    return True


def filter_document_points(parsed: ParsedSearchQuery, points: Sequence[Any]) -> list[Any]:
    if not parsed.has_structured_filters:
        return list(points)
    return [point for point in points if document_matches_query(parsed, point.payload or {})]


def filter_faq_dicts(parsed: ParsedSearchQuery, faqs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not parsed.has_structured_filters:
        return list(faqs)
    return [faq for faq in faqs if faq_matches_query(parsed, faq)]


def scroll_matching_documents(
    qdrant_client,
    collection_name: str,
    parsed: ParsedSearchQuery,
    *,
    limit: int,
    scroll_filter: Any = None,
) -> list[Any]:
    if not qdrant_client.collection_exists(collection_name):
        return []

    matches: list[Any] = []
    offset = None
    batch_limit = scroll_batch_limit(limit)

    while len(matches) < limit:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=batch_limit,
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            if document_matches_query(parsed, point.payload or {}):
                matches.append(
                    SimpleNamespace(id=point.id, payload=point.payload or {}, score=0.0)
                )
                if len(matches) >= limit:
                    break

        if offset is None:
            break

    return matches


def scroll_matching_faqs(
    qdrant_client,
    collection_name: str,
    parsed: ParsedSearchQuery,
    *,
    limit: int,
    scroll_filter: Any = None,
) -> list[dict[str, Any]]:
    if not qdrant_client.collection_exists(collection_name):
        return []

    matches: list[dict[str, Any]] = []
    offset = None
    batch_limit = scroll_batch_limit(limit)

    while len(matches) < limit:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=batch_limit,
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            faq = {
                "id": str(point.id),
                "question": point.payload.get("question", ""),
                "answer": point.payload.get("answer", ""),
                "score": 0.0,
                "source_documents": point.payload.get("source_documents", []),
                "first_seen": point.payload.get("first_seen"),
                "last_updated": point.payload.get("last_updated"),
            }
            if faq_matches_query(parsed, faq):
                matches.append(faq)
                if len(matches) >= limit:
                    break

        if offset is None:
            break

    return matches
