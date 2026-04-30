"""Unit tests for SelfModifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from eva.autonomy.self_modifier import Proposal, SelfModifier


def _make(tmp_path: Path) -> SelfModifier:
    (tmp_path / "configs").mkdir()
    (tmp_path / "eva" / "guidance").mkdir(parents=True)
    (tmp_path / "eva" / "transparency").mkdir(parents=True)
    (tmp_path / "eva" / "guidance" / "covenant.py").write_text("# covenant", encoding="utf-8")
    return SelfModifier(repo_root=tmp_path)


class TestSelfModifier:
    def test_allowed_target_applied(self, tmp_path):
        sm = _make(tmp_path)
        prop = Proposal(
            target="configs/custom.yaml",
            new_content="model:\n  d_model: 64\n",
            reason="tune",
        )
        rec = sm.evaluate(prop)
        assert rec.applied is True
        assert (tmp_path / "configs" / "custom.yaml").exists()

    def test_forbidden_target_rejected(self, tmp_path):
        sm = _make(tmp_path)
        prop = Proposal(
            target="eva/guidance/covenant.py",
            new_content="# try to rewrite covenant",
        )
        rec = sm.evaluate(prop)
        assert rec.applied is False
        assert "forbidden" in rec.rejection_reason

    def test_non_whitelisted_rejected(self, tmp_path):
        sm = _make(tmp_path)
        prop = Proposal(
            target="eva/training/loop.py",
            new_content="# random code",
        )
        rec = sm.evaluate(prop)
        assert rec.applied is False
        assert "not in the allow-list" in rec.rejection_reason

    def test_syntax_check_rejects_bad_yaml(self, tmp_path):
        sm = _make(tmp_path)
        prop = Proposal(
            target="configs/broken.yaml",
            new_content="key: value:\n  -bad indent",
        )
        rec = sm.evaluate(prop)
        assert rec.applied is False
        assert "yaml" in rec.rejection_reason.lower()

    def test_approval_fn_can_veto(self, tmp_path):
        approved = []
        sm = _make(tmp_path)
        sm._approval = lambda p: False
        prop = Proposal(
            target="configs/custom.yaml",
            new_content="a: 1\n",
        )
        rec = sm.evaluate(prop)
        assert rec.applied is False
        assert "approval" in rec.rejection_reason.lower()
