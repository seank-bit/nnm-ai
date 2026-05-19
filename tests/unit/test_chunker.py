from __future__ import annotations
from unittest.mock import MagicMock

from nnm.services.chunker import Chunker, _split_recursive


def _fake_tokenizer():
    tk = MagicMock()
    tk.encode = lambda s, add_special_tokens=False: list(range(max(1, len(s) // 4)))
    return tk


def test_split_recursive_respects_token_limit():
    parts = _split_recursive(
        "abcd" * 200, target_tokens=20, overlap_tokens=4, tokenizer=_fake_tokenizer()
    )
    for p in parts:
        assert len(_fake_tokenizer().encode(p)) <= 20 + 4


def test_chunker_drops_references_section():
    doc = {
        "elements": [
            {"type": "heading", "level": 1, "text": "Introduction", "page": 1},
            {"type": "paragraph", "text": "Intro body here.", "page": 1},
            {"type": "heading", "level": 1, "text": "References", "page": 5},
            {"type": "paragraph", "text": "[1] Some et al.", "page": 5},
        ],
    }
    ch = Chunker(tokenizer=_fake_tokenizer(), target_tokens=100, overlap_tokens=10)
    chunks = list(ch.chunk(doc, paper_title="딥러닝 OO"))
    assert {c.section for c in chunks} == {"Introduction"}
    assert all("[딥러닝 OO] [Introduction]" in c.text_for_embed for c in chunks)


def test_chunker_omits_title_prefix_when_missing():
    doc = {
        "elements": [
            {"type": "heading", "level": 1, "text": "Methods", "page": 2},
            {"type": "paragraph", "text": "Method body.", "page": 2},
        ],
    }
    ch = Chunker(tokenizer=_fake_tokenizer(), target_tokens=100, overlap_tokens=10)
    chunks = list(ch.chunk(doc, paper_title=None))
    assert chunks[0].text_for_embed.startswith("[Methods]")
    assert "[None]" not in chunks[0].text_for_embed
