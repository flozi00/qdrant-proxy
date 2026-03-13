from pathlib import Path
import sys


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from services import build_contrastive_pairs


def test_build_contrastive_pairs_only_uses_adjacent_star_buckets():
    records = [
        {
            "query": "vector database",
            "text": "five star",
            "search_score": 0.95,
            "user_rating": 1,
            "ranking_score": 5,
            "rating_session_id": "session-1",
            "content_type": "document",
        },
        {
            "query": "vector database",
            "text": "four star",
            "search_score": 0.85,
            "user_rating": 1,
            "ranking_score": 4,
            "rating_session_id": "session-1",
            "content_type": "document",
        },
        {
            "query": "vector database",
            "text": "three star",
            "search_score": 0.7,
            "user_rating": 0,
            "ranking_score": 3,
            "rating_session_id": "session-1",
            "content_type": "document",
        },
        {
            "query": "vector database",
            "text": "one star",
            "search_score": 0.1,
            "user_rating": -1,
            "ranking_score": 1,
            "rating_session_id": "session-1",
            "content_type": "document",
        },
    ]

    pairs = build_contrastive_pairs(records)

    assert {(pair["positive"], pair["negative"]) for pair in pairs} == {
        ("five star", "four star"),
        ("four star", "three star"),
    }
    assert all(pair["pair_source"] == "ranked" for pair in pairs)
    assert all(pair["score_gap"] == 1 for pair in pairs)


def test_build_contrastive_pairs_falls_back_to_next_lower_star_bucket():
    records = [
        {
            "query": "vector database",
            "text": "five star",
            "search_score": 0.95,
            "user_rating": 1,
            "ranking_score": 5,
            "rating_session_id": "session-1",
            "content_type": "document",
        },
        {
            "query": "vector database",
            "text": "three star",
            "search_score": 0.7,
            "user_rating": 0,
            "ranking_score": 3,
            "rating_session_id": "session-1",
            "content_type": "document",
        },
    ]

    pairs = build_contrastive_pairs(records)

    assert pairs == [
        {
            "query": "vector database",
            "positive": "five star",
            "negative": "three star",
            "positive_score": 0.95,
            "negative_score": 0.7,
            "rating_session_id": "session-1",
            "pair_source": "ranked",
            "score_gap": 2,
            "positive_type": "document",
            "negative_type": "document",
        }
    ]


def test_build_contrastive_pairs_keeps_binary_pairs_for_thumbs_only_feedback():
    records = [
        {
            "query": "vector database",
            "text": "thumbs up",
            "search_score": 0.82,
            "user_rating": 1,
            "ranking_score": None,
            "rating_session_id": "session-2",
        },
        {
            "query": "vector database",
            "text": "thumbs down",
            "search_score": 0.21,
            "user_rating": -1,
            "ranking_score": None,
            "rating_session_id": "session-2",
        },
    ]

    pairs = build_contrastive_pairs(records)

    assert pairs == [
        {
            "query": "vector database",
            "positive": "thumbs up",
            "negative": "thumbs down",
            "positive_score": 0.82,
            "negative_score": 0.21,
            "rating_session_id": "session-2",
            "pair_source": "binary",
            "score_gap": None,
        }
    ]
