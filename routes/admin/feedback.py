"""Admin feedback routes.

Provides:
- Search feedback endpoints (thumbs up/down)
- Feedback export for training
"""

import logging
from json import JSONDecodeError, dumps, loads
from typing import List, Optional

from auth import verify_admin_auth
from config import settings
from fastapi import APIRouter, Depends, HTTPException, status
from knowledge_graph import (
    FeedbackExportResponse,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from models import LLMSearchRankingRequest, LLMSearchRankingResponse
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
from qdrant_client import models
from state import get_app_state

from services import get_feedback_collection_name

router = APIRouter()

COLLECTION_NAME = settings.collection_name


def _clamp_stars(value: int) -> int:
    return max(1, min(5, value))


# ============================================================================
# SEARCH FEEDBACK ENDPOINTS
# ============================================================================
# Note: POST /feedback is defined in app.py for end-user access (no admin auth)
# This module only handles admin-protected feedback management endpoints


@router.post("/search/llm-rank", response_model=LLMSearchRankingResponse)
async def llm_rank_search_results(
    body: LLMSearchRankingRequest,
    _: bool = Depends(verify_admin_auth),
):
    """Generate LLM-based star rating hints for search result documents.

    The model receives all result options in one prompt and returns relative
    ranking + a short reason per option so users can calibrate manual rating.
    """
    options = body.documents
    if not options:
        return LLMSearchRankingResponse(query=body.query, model=settings.llm_ranking_model, hints=[])

    client = AsyncOpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )

    compact_options = []
    for item in options:
        compact_options.append(
            {
                "option_id": item.option_id,
                "doc_id": item.doc_id,
                "url": item.url,
                "search_score": item.search_score,
                "content": item.content[:1600],
            }
        )

    response_schema = {
        "name": "search_ranking_hints",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "hints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option_id": {"type": "string"},
                            "stars": {"type": "integer", "minimum": 1, "maximum": 5},
                            "relative_rank": {"type": "integer", "minimum": 1},
                            "reason": {"type": "string"},
                        },
                        "required": ["option_id", "stars", "relative_rank", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["hints"],
            "additionalProperties": False,
        },
    }

    try:
        completion = await client.chat.completions.create(
            model=settings.llm_ranking_model,
            temperature=0.1,
            max_tokens=2200,
            response_format={
                "type": "json_schema",
                "json_schema": response_schema,
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate search result relevance. "
                        "Given one query and multiple candidate documents, score each candidate with 1-5 stars. "
                        "Rank candidates relative to each other where rank 1 is best. "
                        "Return concise reasons that mention relevance, specificity, and evidence from content."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query:\n{body.query}\n\n"
                        "Candidate options (JSON array):\n"
                        f"{dumps(compact_options, ensure_ascii=True)}\n\n"
                        "Return one hint per option_id. "
                        "Use stars=5 only for clearly excellent matches and stars=1 for poor/off-topic results."
                    ),
                },
            ],
        )
    except Exception as exc:
        logger.error(f"LLM ranking failed for query '{body.query}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM ranking failed: {exc}",
        )

    raw_content = (completion.choices[0].message.content or "").strip()
    try:
        parsed = loads(raw_content) if raw_content else {"hints": []}
    except JSONDecodeError as exc:
        logger.error(f"Failed parsing LLM ranking output: {exc}; content={raw_content[:400]}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM ranking returned invalid JSON",
        )

    parsed_hints = parsed.get("hints", []) if isinstance(parsed, dict) else []
    hints_by_id = {}
    for hint in parsed_hints:
        if not isinstance(hint, dict):
            continue
        option_id = str(hint.get("option_id", "")).strip()
        if not option_id:
            continue
        try:
            stars = _clamp_stars(int(hint.get("stars", 3)))
        except (TypeError, ValueError):
            stars = 3
        try:
            rank = max(1, int(hint.get("relative_rank", 1)))
        except (TypeError, ValueError):
            rank = 1
        reason = str(hint.get("reason", "")).strip() or "No reason provided"
        hints_by_id[option_id] = {
            "option_id": option_id,
            "stars": stars,
            "relative_rank": rank,
            "reason": reason[:400],
        }

    # Guarantee one hint per option, preserving input order.
    normalized_hints = []
    for idx, option in enumerate(options, start=1):
        hint = hints_by_id.get(option.option_id)
        if hint is None:
            hint = {
                "option_id": option.option_id,
                "stars": 3,
                "relative_rank": idx,
                "reason": "Fallback hint: this result appears moderately relevant.",
            }
        normalized_hints.append(hint)

    return LLMSearchRankingResponse(
        query=body.query,
        model=settings.llm_ranking_model,
        hints=normalized_hints,
    )


@router.get("/feedback", response_model=List[FeedbackResponse])
async def list_feedback(
    collection_name: Optional[str] = None,
    limit: int = 100,
    offset: Optional[str] = None,
    user_rating: Optional[int] = None,
    _: bool = Depends(verify_admin_auth),
):
    """List feedback records with optional filters."""
    target_collection = collection_name or COLLECTION_NAME
    feedback_collection = get_feedback_collection_name(target_collection)
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    if not qdrant_client.collection_exists(feedback_collection):
        return []

    try:
        filter_conditions = []

        if user_rating is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="user_rating",
                    match=models.MatchValue(value=user_rating),
                )
            )

        scroll_filter = None
        if filter_conditions:
            scroll_filter = models.Filter(must=filter_conditions)

        results, _ = qdrant_client.scroll(
            collection_name=feedback_collection,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        feedback_list = []
        for point in results:
            payload = point.payload
            feedback_list.append(
                FeedbackResponse(
                    id=str(point.id),
                    query=payload.get("query", ""),
                    faq_id=payload.get("faq_id") or payload.get("fact_id"),
                    faq_text=payload.get("faq_text") or payload.get("fact_text"),
                    doc_id=payload.get("doc_id"),
                    doc_url=payload.get("doc_url"),
                    doc_content=payload.get("doc_content"),
                    search_score=payload.get("search_score", 0.0),
                    user_rating=payload.get("user_rating", 0),
                    ranking_score=payload.get("ranking_score"),
                    content_type=payload.get("content_type", "faq"),
                    collection_name=payload.get("collection_name", ""),
                    created_at=payload.get("created_at", ""),
                )
            )

        return feedback_list

    except Exception as e:
        logger.error(f"Failed to list feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list feedback: {str(e)}",
        )


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    collection_name: Optional[str] = None,
    _: bool = Depends(verify_admin_auth),
):
    """Get aggregated feedback statistics for quality assessment."""
    target_collection = collection_name or COLLECTION_NAME
    feedback_collection = get_feedback_collection_name(target_collection)
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    if not qdrant_client.collection_exists(feedback_collection):
        return FeedbackStatsResponse(
            collection_name=target_collection,
            total_feedback=0,
        )

    try:
        all_feedback = []
        offset = None

        while True:
            results, next_offset = qdrant_client.scroll(
                collection_name=feedback_collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not results:
                break

            all_feedback.extend(results)
            offset = next_offset

            if not offset:
                break

        total = len(all_feedback)
        positive = sum(1 for f in all_feedback if f.payload.get("user_rating", 0) == 1)
        neutral = sum(1 for f in all_feedback if f.payload.get("user_rating", 0) == 0)
        negative = sum(1 for f in all_feedback if f.payload.get("user_rating", 0) == -1)

        recommendations = []
        if total >= 10:
            high_score_irrelevant = [
                f.payload.get("search_score", 0)
                for f in all_feedback
                if f.payload.get("user_rating", 0) == -1
            ]

            if high_score_irrelevant:
                min_false_pos = min(high_score_irrelevant)
                recommendations.append(
                    {
                        "type": "threshold_suggestion",
                        "message": f"Consider raising fact score threshold above {min_false_pos:.1f}",
                        "rationale": f"{len(high_score_irrelevant)} false positives found with scores >= {min_false_pos:.1f}",
                        "requires_human_approval": True,
                    }
                )

        patterns = []

        return FeedbackStatsResponse(
            collection_name=target_collection,
            total_feedback=total,
            positive_feedback=positive,
            neutral_feedback=neutral,
            negative_feedback=negative,
            score_threshold_recommendations=recommendations,
            common_failure_patterns=patterns,
        )

    except Exception as e:
        logger.error(f"Failed to compute feedback stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute feedback stats: {str(e)}",
        )


@router.get("/feedback/export")
async def export_feedback(
    collection_name: Optional[str] = None,
    format: str = "contrastive",
    include_neutral: bool = False,
    _: bool = Depends(verify_admin_auth),
):
    """Export feedback data for embedding model fine-tuning."""
    target_collection = collection_name or COLLECTION_NAME
    feedback_collection = get_feedback_collection_name(target_collection)
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    if not qdrant_client.collection_exists(feedback_collection):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feedback collection '{feedback_collection}' not found",
        )

    try:
        all_feedback = []
        offset = None

        while True:
            results, next_offset = qdrant_client.scroll(
                collection_name=feedback_collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not results:
                break

            all_feedback.extend(results)
            offset = next_offset

            if not offset:
                break

        positives = []
        negatives = []

        for f in all_feedback:
            payload = f.payload
            user_rating = payload.get("user_rating", 0)

            if not include_neutral and user_rating == 0:
                continue

            content_type = payload.get("content_type", "faq")
            if content_type == "document":
                text_content = payload.get("doc_content", "")
            else:
                text_content = payload.get("faq_text") or payload.get("fact_text", "")

            record = {
                "query": payload.get("query", ""),
                "text": text_content,
                "content_type": content_type,
                "search_score": payload.get("search_score", 0.0),
                "user_rating": user_rating,
                "ranking_score": payload.get("ranking_score"),
            }

            if user_rating == 1:
                positives.append(record)
            elif user_rating == -1:
                negatives.append(record)

        if format == "contrastive":
            contrastive_pairs = []

            positive_by_query = {}
            negative_by_query = {}
            ranked_by_query = {}

            for p in positives:
                q = p["query"]
                if q not in positive_by_query:
                    positive_by_query[q] = []
                positive_by_query[q].append(p)

            for n in negatives:
                q = n["query"]
                if q not in negative_by_query:
                    negative_by_query[q] = []
                negative_by_query[q].append(n)

            # Collect all records that have a ranking_score for ranked pairs
            for record in positives + negatives:
                if record.get("ranking_score") is not None:
                    q = record["query"]
                    if q not in ranked_by_query:
                        ranked_by_query[q] = []
                    ranked_by_query[q].append(record)

            # 1) Binary pairs: positive vs negative (existing logic)
            for query in positive_by_query:
                if query in negative_by_query:
                    for pos in positive_by_query[query]:
                        for neg in negative_by_query[query]:
                            contrastive_pairs.append(
                                {
                                    "query": query,
                                    "positive": pos["text"],
                                    "negative": neg["text"],
                                    "positive_type": pos["content_type"],
                                    "negative_type": neg["content_type"],
                                    "positive_score": pos["search_score"],
                                    "negative_score": neg["search_score"],
                                    "pair_source": "binary",
                                    "score_gap": None,
                                }
                            )

            # 2) Ranked pairs: higher ranking_score vs lower ranking_score
            #    This lets "good" results serve as hard negatives to "very good" results
            for query, records in ranked_by_query.items():
                # Sort by ranking_score descending
                sorted_records = sorted(
                    records, key=lambda r: r["ranking_score"], reverse=True
                )
                for i, higher in enumerate(sorted_records):
                    for lower in sorted_records[i + 1 :]:
                        if higher["ranking_score"] > lower["ranking_score"]:
                            contrastive_pairs.append(
                                {
                                    "query": query,
                                    "positive": higher["text"],
                                    "negative": lower["text"],
                                    "positive_type": higher["content_type"],
                                    "negative_type": lower["content_type"],
                                    "positive_score": higher["search_score"],
                                    "negative_score": lower["search_score"],
                                    "pair_source": "ranked",
                                    "score_gap": higher["ranking_score"]
                                    - lower["ranking_score"],
                                }
                            )

            binary_count = sum(1 for p in contrastive_pairs if p["pair_source"] == "binary")
            ranked_count = sum(1 for p in contrastive_pairs if p["pair_source"] == "ranked")

            return FeedbackExportResponse(
                format=format,
                total_records=len(all_feedback),
                positive_pairs=len(positives),
                negative_pairs=len(negatives),
                contrastive_pairs=len(contrastive_pairs),
                binary_pairs=binary_count,
                ranked_pairs=ranked_count,
                data=contrastive_pairs,
            )

        elif format == "jsonl":
            all_records = []
            for f in all_feedback:
                payload = f.payload
                content_type = payload.get("content_type", "faq")
                text_content = (
                    payload.get("doc_content", "")
                    if content_type == "document"
                    else payload.get("faq_text") or payload.get("fact_text", "")
                )
                all_records.append(
                    {
                        "query": payload.get("query", ""),
                        "text": text_content,
                        "content_type": content_type,
                        "search_score": payload.get("search_score", 0.0),
                        "user_rating": payload.get("user_rating", 0),
                        "ranking_score": payload.get("ranking_score"),
                    }
                )

            return FeedbackExportResponse(
                format=format,
                total_records=len(all_feedback),
                positive_pairs=len(positives),
                negative_pairs=len(negatives),
                contrastive_pairs=0,
                data=all_records,
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown format: {format}. Use 'contrastive' or 'jsonl'",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export feedback: {str(e)}",
        )


@router.delete("/feedback/{feedback_id}")
async def delete_feedback(
    feedback_id: str,
    collection_name: Optional[str] = None,
    _: bool = Depends(verify_admin_auth),
):
    """Delete a feedback record."""
    target_collection = collection_name or COLLECTION_NAME
    feedback_collection = get_feedback_collection_name(target_collection)
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    if not qdrant_client.collection_exists(feedback_collection):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feedback collection '{feedback_collection}' not found",
        )

    try:
        qdrant_client.delete(
            collection_name=feedback_collection,
            points_selector=models.PointIdsList(points=[feedback_id]),
        )

        return {"message": f"Feedback {feedback_id} deleted"}

    except Exception as e:
        logger.error(f"Failed to delete feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete feedback: {str(e)}",
        )

