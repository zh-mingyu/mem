from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_memgallery import load_memgallery


def test_memgallery_official_schema(tmp_path):
    root = tmp_path / "data"
    dialog_dir = root / "dialog"
    dialog_dir.mkdir(parents=True)
    payload = {
        "character_profile": {"name": "Demo"},
        "multi_session_dialogues": [{
            "session_id": "session-0",
            "date": "2024-01-01",
            "dialogues": [{
                "round": 1,
                "user": "I saw a lake.",
                "assistant": "Tell me more.",
                "image_id": ["D1:IMG_001"],
                "image_caption": ["sunrise lake"],
            }],
        }],
        "human-annotated QAs": [{
            "question": "What did I see?",
            "answer": "a lake",
            "point": "AR",
            "session_id": "session-0",
            "clue": ["1"],
        }],
    }
    (dialog_dir / "Demo.json").write_text(json.dumps(payload), encoding="utf-8")

    samples = load_memgallery(str(root))

    assert len(samples) == 1
    assert samples[0]["sample_id"] == "memgallery_00_Demo"
    assert [turn["speaker"] for turn in samples[0]["turns"]] == [
        "user (Demo)",
        "assistant",
    ]
    assert "sunrise lake" in samples[0]["turns"][0]["text"]
    assert "D1:IMG_001" in samples[0]["turns"][0]["text"]
    assert samples[0]["qas"][0]["category"] == "AR"


def test_memgallery_query_image_caption_is_kept_for_prompting(tmp_path):
    root = tmp_path / "data"
    dialog_dir = root / "dialog"
    dialog_dir.mkdir(parents=True)
    payload = {
        "character_profile": {"name": "Demo"},
        "multi_session_dialogues": [{
            "session_id": "session-0",
            "date": "2024-01-01",
            "dialogues": [{"round": "D1:1", "user": "I visited a city.", "assistant": "Nice."}],
        }],
        "human-annotated QAs": [{
            "point": "VR",
            "question": "Which city is shown in this image?",
            "image_caption": "A dense city beside a blue strait.",
            "answer": "Istanbul",
        }],
    }
    (dialog_dir / "Demo.json").write_text(json.dumps(payload), encoding="utf-8")

    samples = load_memgallery(str(root))

    assert samples[0]["qas"][0]["query_image_caption"] == "A dense city beside a blue strait."
