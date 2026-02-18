"""Native Docling integration for URL scraping and file conversion.

Uses the docling Python library directly instead of the remote docling-serve container.
This gives fine-grained control over conversion pipelines and avoids network overhead.

Provides functions for:
- Scraping URLs to markdown via native Docling DocumentConverter
- Converting uploaded files to markdown
- Resolving URL redirects
- Extracting structural layouts and hyperlinks from Docling documents

CUDA memory management: torch.cuda.empty_cache() is called after every conversion
to prevent GPU memory leaks from accumulating model tensors.
"""

import asyncio
import gc
import logging
import re
import threading
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

import httpx
import torch
from config import settings
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.io import DocumentStream
from fastapi import HTTPException, status

from utils.timings import linetimer

logger = logging.getLogger(__name__)

_PICTURE_DESCRIPTION_PROMPT = (
    "Describe that image as detailed as possible, including any text it contains. "
    "Be concise but thorough."
)

# Lock to serialize converter access (models are not thread-safe)
_converter_lock = threading.Lock()
_converter: Optional[DocumentConverter] = None
_converter_no_pictures: Optional[DocumentConverter] = None


def _clear_cuda_cache() -> None:
    """Clear CUDA cache and run garbage collection to prevent memory leaks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache after Docling conversion")


def _build_picture_description_options() -> PictureDescriptionApiOptions:
    """Build API-based picture description options using LiteLLM."""
    base_url = settings.litellm_base_url.rstrip("/")
    api_url = f"{base_url}/chat/completions"
    headers: Dict[str, str] = {}
    if settings.litellm_api_key:
        headers["Authorization"] = f"Bearer {settings.litellm_api_key}"
    return PictureDescriptionApiOptions(
        url=api_url,
        params={"model": "default"},
        headers=headers,
        concurrency=4,
        timeout=60,
        prompt=_PICTURE_DESCRIPTION_PROMPT,
        scale=2.0,
    )


def _build_pipeline_options(enable_picture_enrichment: bool = True) -> PdfPipelineOptions:
    """Build PDF pipeline options for native Docling conversion."""
    opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
        ),
        enable_remote_services=True,
        images_scale=2.0,
        generate_picture_images=enable_picture_enrichment,
        do_picture_description=enable_picture_enrichment,
        do_picture_classification=enable_picture_enrichment,
    )
    if enable_picture_enrichment:
        opts.picture_description_options = _build_picture_description_options()
    return opts


def _get_converter(enable_picture_enrichment: bool = True) -> DocumentConverter:
    """Get or create a singleton DocumentConverter instance.

    Two converter instances are maintained: one with picture enrichment, one without.
    """
    global _converter, _converter_no_pictures

    if enable_picture_enrichment:
        if _converter is None:
            pipeline_options = _build_pipeline_options(enable_picture_enrichment=True)
            _converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.HTML,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.IMAGE,
                    InputFormat.MD,
                    InputFormat.CSV,
                    InputFormat.ASCIIDOC,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                },
            )
            logger.info("Initialized Docling DocumentConverter (with picture enrichment)")
        return _converter
    else:
        if _converter_no_pictures is None:
            pipeline_options = _build_pipeline_options(enable_picture_enrichment=False)
            _converter_no_pictures = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.HTML,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.IMAGE,
                    InputFormat.MD,
                    InputFormat.CSV,
                    InputFormat.ASCIIDOC,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                },
            )
            logger.info("Initialized Docling DocumentConverter (without picture enrichment)")
        return _converter_no_pictures


@dataclass
class DoclingResult:
    """Result from Docling scraping with enriched metadata."""

    content: str
    title: Optional[str] = None
    hyperlinks: List[str] = field(default_factory=list)
    docling_layout: List[Dict[str, Any]] = field(default_factory=list)


def _build_docling_index(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for key in (
        "texts",
        "groups",
        "pictures",
        "tables",
        "key_value_items",
        "form_items",
    ):
        for item in doc.get(key, []) or []:
            if isinstance(item, dict) and item.get("self_ref"):
                index[item["self_ref"]] = item
    return index


def _resolve_ref(node: Any, index: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if isinstance(node, dict) and "$ref" in node:
        return index.get(node["$ref"])
    if isinstance(node, dict):
        return node
    return None


def _format_link(text: str, hyperlink: Optional[str], base_url: Optional[str]) -> str:
    if hyperlink:
        href = urljoin(base_url, hyperlink) if base_url else hyperlink
        return f"[{text}]({href})"
    return text


def _node_text(node: Dict[str, Any], base_url: Optional[str]) -> str:
    text = (node.get("text") or "").strip()
    if not text:
        return ""
    return _format_link(text, node.get("hyperlink"), base_url)


def _is_low_signal_text(text: str, label: Optional[str]) -> bool:
    if not text:
        return True
    if label in {"caption"}:
        return True

    compact = " ".join(text.replace("\u00a0", " ").split())
    lower = compact.lower()

    if compact in {"+", "-"}:
        return True
    if lower in {"follow", "upvote", "share"}:
        return True
    if "avatar" in lower:
        return True
    if compact.startswith("[Follow]("):
        return True
    if compact.startswith("[Upvote"):
        return True

    if re.fullmatch(r"-\s*\+\d+", compact):
        return True
    if re.fullmatch(r"\+\d+", compact):
        return True

    stripped = compact.lstrip("+-")
    if stripped.isdigit() and len(stripped) <= 4:
        return True

    if len(text) <= 2:
        return True
    return False


def _extract_text_value(value: Any) -> Optional[str]:
    """Extract the first non-empty text from nested values."""
    if isinstance(value, str):
        text = value.strip()
        return text or None

    if isinstance(value, dict):
        for key in ("text", "description", "content", "value", "caption"):
            text = _extract_text_value(value.get(key))
            if text:
                return text
        for nested in value.values():
            text = _extract_text_value(nested)
            if text:
                return text
        return None

    if isinstance(value, list):
        for item in value:
            text = _extract_text_value(item)
            if text:
                return text

    return None


def _extract_text_values(value: Any) -> List[str]:
    """Extract all non-empty text fragments from nested values."""
    values: List[str] = []

    if isinstance(value, str):
        text = value.strip()
        if text:
            values.append(text)
        return values

    if isinstance(value, dict):
        for key in ("text", "description", "content", "value", "caption"):
            values.extend(_extract_text_values(value.get(key)))
        for nested in value.values():
            values.extend(_extract_text_values(nested))
        return values

    if isinstance(value, list):
        for item in value:
            values.extend(_extract_text_values(item))

    return values


def _is_low_signal_picture_description(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return True

    generic_tokens = {
        "image",
        "figure",
        "chart",
        "diagram",
        "photo",
        "picture",
        "graphic",
        "illustration",
        "logo",
        "icon",
        "benchmark",
    }

    words = normalized.split()
    if len(normalized) < 24 and len(words) <= 4:
        return True
    if len(words) <= 5 and all(word in generic_tokens for word in words):
        return True
    if normalized in {
        "benchmark image",
        "chart image",
        "community evals benchmark image",
        "figure image",
        "image",
    }:
        return True

    return False


def _select_best_description(candidates: List[str]) -> Optional[str]:
    unique: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)

    if not unique:
        return None

    rich = [text for text in unique if not _is_low_signal_picture_description(text)]
    if rich:
        return max(rich, key=len)

    return None


def _get_picture_description(
    node: Dict[str, Any],
    index: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Extract VLM-generated description from a picture node.

    Checks (in order):
    1. meta.description / meta.annotations (PictureMeta, current format)
    2. top-level annotations[] (deprecated format)
    """
    candidates: List[str] = []
    meta = node.get("meta")
    if isinstance(meta, dict):
        candidates.extend(_extract_text_values(meta.get("description")))

        for ann in meta.get("annotations", []) or []:
            if isinstance(ann, dict) and ann.get("kind") == "description":
                candidates.extend(_extract_text_values(ann))

        for key in ("caption", "captions"):
            candidates.extend(_extract_text_values(meta.get(key)))

    candidates.extend(_extract_text_values(node.get("description")))

    for ann in node.get("annotations", []) or []:
        if isinstance(ann, dict) and ann.get("kind") == "description":
            candidates.extend(_extract_text_values(ann))

    return _select_best_description(candidates)


def _get_picture_caption(
    node: Dict[str, Any],
    index: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Resolve caption text from a picture node's captions refs."""
    parts: List[str] = []
    for cap_ref in node.get("captions", []) or []:
        resolved = _resolve_ref(cap_ref, index)
        if resolved:
            text = (resolved.get("text") or "").strip()
            if text:
                parts.append(text)
    caption = " ".join(parts).strip() or None
    if caption and _is_low_signal_picture_description(caption):
        return None
    return caption


def _is_navigation_list(items: List[Tuple[str, Optional[str]]]) -> bool:
    if len(items) < 3:
        return False

    link_count = sum(1 for _, link in items if link)
    avg_len = sum(len(text) for text, _ in items) / max(len(items), 1)

    if link_count == len(items):
        fragment_links = 0
        for _, link in items:
            href = (link or "").strip()
            if href.startswith("#"):
                fragment_links += 1
            elif "#" in href:
                left, _sep, right = href.partition("#")
                if right and (left.startswith("http://") or left.startswith("https://")):
                    fragment_links += 1

        if fragment_links == len(items):
            return True

    return link_count == len(items) and avg_len <= 40


def _render_inline(
    node: Dict[str, Any],
    index: Dict[str, Dict[str, Any]],
    base_url: Optional[str],
) -> str:
    if node.get("content_layer") != "body":
        return ""

    if "text" in node:
        return _node_text(node, base_url)

    if node.get("label") == "inline":
        parts = []
        for child in node.get("children", []) or []:
            resolved = _resolve_ref(child, index)
            if not resolved:
                continue
            segment = _render_inline(resolved, index, base_url)
            if segment:
                parts.append(segment)
        return " ".join(parts).strip()

    return ""


def _collect_list_items(
    group_node: Dict[str, Any],
    index: Dict[str, Dict[str, Any]],
    base_url: Optional[str],
) -> List[Tuple[str, Optional[str]]]:
    items: List[Tuple[str, Optional[str]]] = []
    for child in group_node.get("children", []) or []:
        resolved = _resolve_ref(child, index)
        if not resolved or resolved.get("content_layer") != "body":
            continue
        if resolved.get("label") == "list_item":
            text = (resolved.get("text") or "").strip()
            if text:
                items.append((text, resolved.get("hyperlink")))
        elif resolved.get("label") == "inline":
            inline_text = _render_inline(resolved, index, base_url)
            if inline_text:
                items.append((inline_text, None))
    return items


def _render_blocks(
    node: Dict[str, Any],
    index: Dict[str, Dict[str, Any]],
    base_url: Optional[str],
) -> List[str]:
    if node.get("content_layer") != "body":
        return []

    label = node.get("label")
    blocks: List[str] = []

    # Handle picture/chart nodes FIRST: they may have an empty "text" key
    # which would cause early return in the generic text branch below,
    # skipping the VLM-generated description entirely.
    if label in {"picture", "chart"}:
        description = _get_picture_description(node, index)
        caption = _get_picture_caption(node, index)
        if description:
            blocks.append(f"[Image: {description}]")
        elif caption:
            blocks.append(f"[Image: {caption}]")
        return blocks

    if "text" in node:
        text = _node_text(node, base_url)
        if _is_low_signal_text(text, label):
            return []

        if label in {"title", "section_header"}:
            level = node.get("level") or (1 if label == "title" else 2)
            level = max(1, min(6, int(level)))
            blocks.append(f"{'#' * level} {text}")
        elif label == "list_item":
            marker = "-"
            if node.get("enumerated"):
                marker = "1."
            blocks.append(f"{marker} {text}")
        else:
            blocks.append(text)

        for child in node.get("children", []) or []:
            resolved = _resolve_ref(child, index)
            if resolved:
                blocks.extend(_render_blocks(resolved, index, base_url))
        return blocks

    group_label = node.get("label")
    group_name = node.get("name")

    if group_label == "list" or group_name == "list":
        items = _collect_list_items(node, index, base_url)
        if items and not _is_navigation_list(items):
            for text, link in items:
                line = _format_link(text, link, base_url)
                blocks.append(f"- {line}")
        return blocks

    if group_label == "inline":
        inline_text = _render_inline(node, index, base_url)
        if inline_text and not _is_low_signal_text(inline_text, None):
            blocks.append(inline_text)
        return blocks

    for child in node.get("children", []) or []:
        resolved = _resolve_ref(child, index)
        if resolved:
            blocks.extend(_render_blocks(resolved, index, base_url))

    return blocks


def _extract_markdown_link(block: str) -> Optional[Tuple[str, str]]:
    match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", block.strip())
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _is_profile_link_block(block: str, base_url: Optional[str]) -> bool:
    link = _extract_markdown_link(block)
    if not link:
        return False

    _text, href = link
    parsed = urlsplit(href)
    if parsed.scheme not in {"http", "https"}:
        return False

    if base_url:
        base_host = urlsplit(base_url).netloc.lower()
        if parsed.netloc.lower() != base_host:
            return False

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) != 1:
        return False

    reserved = {
        "blog",
        "datasets",
        "models",
        "spaces",
        "docs",
        "join",
        "login",
        "settings",
        "pricing",
    }
    return path_parts[0].lower() not in reserved


def _is_comment_or_ui_block(block: str) -> bool:
    compact = " ".join(block.lower().split())
    if not compact:
        return True

    noisy_exact = {
        "reply",
        "see translation",
        "deleted",
        "comment",
        "edit preview",
        "this comment has been hidden",
        "tap or paste here to upload images",
    }
    if compact in noisy_exact:
        return True

    noisy_contains = (
        "sign up",
        "log in",
        "to comment",
        "upload images",
        "article author",
    )
    return any(token in compact for token in noisy_contains)


# Regex to strip markdown link syntax: [text](url) → text
_STRIP_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _block_plain_text(block: str) -> str:
    """Extract plain text content from a markdown block.

    Strips list markers, link syntax, and heading markers so we can
    inspect the actual visible text.
    """
    text = block.strip()
    # Strip list markers
    text = re.sub(r"^(?:[-*]|\d+\.)\s+", "", text)
    # Strip heading markers
    text = re.sub(r"^#{1,6}\s+", "", text)
    # Strip markdown links, keep display text
    text = _STRIP_MD_LINK_RE.sub(r"\1", text)
    return text.strip()


def _heading_text(block: str) -> Optional[str]:
    stripped = block.strip()
    if not stripped.startswith("#"):
        return None
    return stripped.lstrip("#").strip().lower()


def _clean_rendered_blocks(blocks: List[str], base_url: Optional[str]) -> List[str]:
    """Drop UI chrome/noise while keeping the main article content."""
    if not blocks:
        return []

    cutoff_headings = {
        "more articles from our blog",
        "community",
        "references",
    }

    cutoff_plain_blocks = {
        "more articles from our blog",
    }

    cleaned: List[str] = []
    for idx, block in enumerate(blocks):
        heading = _heading_text(block)
        if heading in cutoff_headings:
            break

        block_lower = " ".join(block.lower().split())
        if block_lower in cutoff_plain_blocks:
            break

        if "avatar" in block.lower():
            continue

        # Author/profile chips are usually near the top of scraped blog pages.
        if idx < 120 and _is_profile_link_block(block, base_url):
            continue

        if _is_comment_or_ui_block(block):
            continue

        cleaned.append(block)

    # Remove alphabet-index navigation: runs of 5+ consecutive blocks
    # where each block's visible text is a single character (e.g. A-Z filters)
    if cleaned:
        remove_indices: set = set()
        run_start = 0
        while run_start < len(cleaned):
            run_end = run_start
            while run_end < len(cleaned):
                plain = _block_plain_text(cleaned[run_end])
                if len(plain) == 1:
                    run_end += 1
                else:
                    break
            run_length = run_end - run_start
            if run_length >= 5:
                remove_indices.update(range(run_start, run_end))
            run_start = run_end if run_end > run_start else run_start + 1
        if remove_indices:
            cleaned = [b for i, b in enumerate(cleaned) if i not in remove_indices]

    return cleaned


def convert_docling_document_to_markdown(
    doc: Dict[str, Any], base_url: Optional[str] = None
) -> str:
    body = doc.get("body")
    if not isinstance(body, dict):
        return ""

    index = _build_docling_index(doc)
    blocks: List[str] = []
    for child in body.get("children", []) or []:
        resolved = _resolve_ref(child, index)
        if resolved:
            blocks.extend(_render_blocks(resolved, index, base_url))

    cleaned_blocks = [block.strip() for block in blocks if block and block.strip()]
    cleaned_blocks = _clean_rendered_blocks(cleaned_blocks, base_url)
    return "\n\n".join(cleaned_blocks).strip()


def _get_docling_json(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(doc.get("json_content"), dict):
        return doc["json_content"]
    if doc.get("schema_name") == "DoclingDocument":
        return doc
    return None


def extract_all_hyperlinks(
    doc: Dict[str, Any], base_url: Optional[str] = None
) -> List[str]:
    """Extract ALL hyperlinks from a Docling document across all content layers.

    Walks every node in the document index (body, header, footer, navigation)
    and collects hyperlinks as absolute URLs. This captures links that the
    markdown renderer filters out (e.g. navigation menus).
    """
    index = _build_docling_index(doc)
    seen: set[str] = set()
    links: List[str] = []

    for node in index.values():
        hyperlink = node.get("hyperlink")
        if not hyperlink:
            continue
        # Skip non-http links
        if hyperlink.startswith(("javascript:", "mailto:", "tel:", "data:")):
            continue
        href = urljoin(base_url, hyperlink) if base_url else hyperlink
        # Normalize: remove fragment
        parts = urlsplit(href)
        if parts.scheme not in ("http", "https"):
            continue
        normalized = parts._replace(fragment="").geturl()
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)

    return links


def extract_docling_layout(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract a compact structural layout from a Docling document.

    Returns a list of block descriptors representing the page skeleton.
    This can be used to detect repeating templates across pages from the
    same domain and separate boilerplate from main content.
    """
    body = doc.get("body")
    if not isinstance(body, dict):
        return []

    index = _build_docling_index(doc)
    layout: List[Dict[str, Any]] = []

    def _walk(node: Dict[str, Any]) -> None:
        label = node.get("label")
        content_layer = node.get("content_layer")

        if label and content_layer:
            entry: Dict[str, Any] = {
                "label": label,
                "content_layer": content_layer,
            }
            level = node.get("level")
            if level is not None:
                entry["level"] = level
            children = node.get("children") or []
            if label in ("list", "ordered_list") and children:
                entry["items"] = len(children)
            if node.get("hyperlink"):
                entry["has_link"] = True
            layout.append(entry)

        for child in node.get("children", []) or []:
            resolved = _resolve_ref(child, index)
            if resolved:
                _walk(resolved)

    _walk(body)
    return layout


def extract_docling_title(doc: Dict[str, Any]) -> Optional[str]:
    """Extract the page title from a Docling document.

    Looks for a text node with label="title" in any content layer
    (body or furniture). Falls back to the first section_header if
    no explicit title node is found.
    """
    for item in doc.get("texts", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("label") == "title":
            text = (item.get("text") or item.get("orig") or "").strip()
            if text:
                return text

    # Fallback: first section_header in body
    for item in doc.get("texts", []) or []:
        if not isinstance(item, dict):
            continue
        if (
            item.get("label") == "section_header"
            and item.get("content_layer") == "body"
        ):
            text = (item.get("text") or "").strip()
            if text:
                return text

    return None


def _convert_native(
    source: Any,
    headers: Optional[Dict[str, str]] = None,
    enable_picture_enrichment: bool = True,
) -> Dict[str, Any]:
    """Run native Docling conversion and return the document as a dict.

    Handles fallback: if picture enrichment fails, retries without it.
    Always clears CUDA cache after conversion to prevent memory leaks.
    """
    with _converter_lock:
        try:
            converter = _get_converter(enable_picture_enrichment=enable_picture_enrichment)
            result = converter.convert(source, headers=headers)
            doc_dict = result.document.export_to_dict()
            return doc_dict
        except Exception as e:
            if enable_picture_enrichment:
                logger.warning(
                    "Docling conversion failed with picture enrichment, retrying without: %s", e
                )
                try:
                    converter = _get_converter(enable_picture_enrichment=False)
                    result = converter.convert(source, headers=headers)
                    doc_dict = result.document.export_to_dict()
                    return doc_dict
                except Exception as e2:
                    raise e2
            raise
        finally:
            _clear_cuda_cache()


@linetimer()
async def scrape_url_with_docling(url: str) -> DoclingResult:
    """Scrape a URL using native Docling library.

    Args:
        url: The URL to scrape

    Returns:
        DoclingResult with markdown content, all hyperlinks, and structural layout

    Raises:
        HTTPException: If scraping fails
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # Run synchronous Docling conversion in a thread to avoid blocking the event loop
        doc_dict = await asyncio.to_thread(
            _convert_native,
            source=url,
            headers=headers,
            enable_picture_enrichment=True,
        )

        doc_json = _get_docling_json(doc_dict)

        # Extract hyperlinks, layout, and title from the raw docling document
        hyperlinks: List[str] = []
        docling_layout: List[Dict[str, Any]] = []
        title: Optional[str] = None
        if doc_json:
            hyperlinks = extract_all_hyperlinks(doc_json, base_url=url)
            docling_layout = extract_docling_layout(doc_json)
            title = extract_docling_title(doc_json)

        converted = (
            convert_docling_document_to_markdown(doc_json, base_url=url)
            if doc_json
            else ""
        )
        if converted:
            return DoclingResult(
                content=converted,
                title=title,
                hyperlinks=hyperlinks,
                docling_layout=docling_layout,
            )

        raise ValueError("Docling returned empty content")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to scrape URL {url} with Docling: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to scrape URL: {str(e)}",
        )


_PLAINTEXT_EXTENSIONS = {
    ".txt", ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
    ".log", ".md", ".markdown", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".xml", ".svg", ".env",
    ".sh", ".bash", ".py", ".js", ".ts", ".css", ".html", ".htm",
    ".sql", ".r", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".pl", ".lua", ".php", ".swift", ".kt",
}


def _is_plaintext_file(filename: str) -> bool:
    """Check if file can be read as plain text without Docling."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return f".{ext}" in _PLAINTEXT_EXTENSIONS


def _wrap_plaintext(text: str, filename: str) -> str:
    """Wrap plain text content with a code fence for structured formats."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("md", "markdown"):
        return text
    if ext in ("html", "htm", "xml", "svg"):
        return f"```{ext}\n{text}\n```"
    if ext in ("json", "jsonl", "ndjson"):
        return f"```json\n{text}\n```"
    if ext in ("yaml", "yml"):
        return f"```yaml\n{text}\n```"
    if ext in ("csv", "tsv"):
        return f"```\n{text}\n```"
    if ext in ("py", "js", "ts", "go", "rs", "java", "c", "cpp", "h", "hpp",
               "rb", "pl", "lua", "php", "swift", "kt", "r", "sql",
               "sh", "bash", "css"):
        return f"```{ext}\n{text}\n```"
    return text


@linetimer()
async def convert_file_with_docling(file_content: bytes, filename: str) -> str:
    """Convert a file to markdown using native Docling library.

    Plain-text formats (.txt, .csv, .json, .md, etc.) are read directly
    without Docling to avoid unnecessary processing overhead.

    Args:
        file_content: Raw file bytes
        filename: Original filename for format detection

    Returns:
        Markdown content extracted from the file

    Raises:
        HTTPException: If conversion fails
    """
    # Fast path: read plain-text files directly
    if _is_plaintext_file(filename):
        try:
            text = file_content.decode("utf-8").strip()
            if text:
                return _wrap_plaintext(text, filename)
        except UnicodeDecodeError:
            pass  # Fall through to Docling for binary files with misleading extensions

    try:
        # Create a DocumentStream from the file bytes
        stream = DocumentStream(name=filename, stream=BytesIO(file_content))

        # Run synchronous Docling conversion in a thread
        doc_dict = await asyncio.to_thread(
            _convert_native,
            source=stream,
            enable_picture_enrichment=True,
        )

        doc_json = _get_docling_json(doc_dict)
        converted = (
            convert_docling_document_to_markdown(doc_json)
            if doc_json
            else ""
        )
        if converted:
            return converted

        raise ValueError(
            f"Docling returned empty content for file: {filename}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to convert file {filename}: {e}")
        # Fallback: if docling fails, try to decode as text
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Docling conversion failed: {e}",
            )
