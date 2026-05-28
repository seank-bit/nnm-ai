from __future__ import annotations
import csv
import uuid

import pytest

from nnm.services.publication_mapper import PublicationMapping, PublicationMapper


def _write_csv(path, rows, encoding="cp949"):
    with path.open("w", encoding=encoding, newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "s3_key", "title", "legacy_aid"])
        for r in rows:
            w.writerow(r)


@pytest.mark.asyncio
async def test_mapper_returns_mapping_for_full_key(tmp_path):
    pub_id = uuid.uuid4()
    csv_path = tmp_path / "publications.csv"
    _write_csv(csv_path, [(str(pub_id), "abc123def456", "딥러닝 OO", 7)])

    m = PublicationMapper(csv_path=csv_path)
    result = await m.lookup("abc123def456")

    assert result == PublicationMapping(
        publication_id=pub_id, title="딥러닝 OO", legacy_aid=7,
    )


@pytest.mark.asyncio
async def test_mapper_strips_prefix_and_extension(tmp_path):
    pub_id = uuid.uuid4()
    csv_path = tmp_path / "publications.csv"
    _write_csv(csv_path, [(str(pub_id), "abc123def456", "Title X", 1)])

    m = PublicationMapper(csv_path=csv_path)
    result = await m.lookup("papers/abc123def456.pdf")

    assert result is not None
    assert result.publication_id == pub_id


@pytest.mark.asyncio
async def test_mapper_returns_none_on_miss(tmp_path):
    csv_path = tmp_path / "publications.csv"
    _write_csv(csv_path, [(str(uuid.uuid4()), "other_key", "x", 1)])

    m = PublicationMapper(csv_path=csv_path)
    assert await m.lookup("missing_key") is None


@pytest.mark.asyncio
async def test_mapper_returns_none_when_csv_unconfigured():
    m = PublicationMapper(csv_path=None)
    assert await m.lookup("anything") is None


@pytest.mark.asyncio
async def test_mapper_parses_legacy_aid(tmp_path):
    pub_id = uuid.uuid4()
    csv_path = tmp_path / "publications.csv"
    _write_csv(csv_path, [(str(pub_id), "k1", "t1", 42)])

    m = PublicationMapper(csv_path=csv_path)
    result = await m.lookup("k1")

    assert result is not None
    assert result.legacy_aid == 42
    assert m.legacy_aid_for("k1") == 42


def test_mapper_legacy_aid_for_unknown_key(tmp_path):
    csv_path = tmp_path / "publications.csv"
    _write_csv(csv_path, [(str(uuid.uuid4()), "known", "t", 9)])

    m = PublicationMapper(csv_path=csv_path)
    assert m.legacy_aid_for("missing") is None


def test_mapper_legacy_aid_falls_back_to_none_on_garbage(tmp_path):
    pub_id = uuid.uuid4()
    csv_path = tmp_path / "publications.csv"
    _write_csv(csv_path, [(str(pub_id), "k1", "t", "not-an-int")])

    m = PublicationMapper(csv_path=csv_path)
    assert m.legacy_aid_for("k1") is None


@pytest.mark.asyncio
async def test_mapper_skips_rows_with_invalid_uuid(tmp_path):
    pub_id = uuid.uuid4()
    csv_path = tmp_path / "publications.csv"
    _write_csv(csv_path, [
        ("not-a-uuid", "bad_row", "x", 1),
        (str(pub_id), "good_row", "ok", 2),
    ])

    m = PublicationMapper(csv_path=csv_path)
    assert await m.lookup("bad_row") is None
    good = await m.lookup("good_row")
    assert good is not None and good.publication_id == pub_id
