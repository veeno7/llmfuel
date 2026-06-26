# tests/test_receipts.py
"""
Basic tests for fuel.receipts.ReceiptChain.
Run with: pytest tests/test_receipts.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest
from fuel.receipts import ReceiptChain, _sha256, _hash_content


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def chain():
    return ReceiptChain(agent="test-agent", principal="pytest")


# ── Core chain integrity ───────────────────────────────────────────────────────

def test_verify_chain_two_steps(chain):
    chain.record("step_1", input_data="Solve 2x+3=11", output_data="x=4", input_tokens=8, output_tokens=3)
    chain.record("step_2", input_data="Check: 2(4)+3=11", output_data="✓ correct", input_tokens=9, output_tokens=2)
    assert chain.verify_chain() is True


def test_verify_chain_single_step(chain):
    chain.record("step_1", input_data="hello", output_data="world")
    assert chain.verify_chain() is True


def test_verify_chain_empty(chain):
    assert chain.verify_chain() is True


def test_prev_hash_chains_correctly(chain):
    r1 = chain.record("step_1", input_data="a", output_data="b")
    r2 = chain.record("step_2", input_data="c", output_data="d")
    expected = _sha256(json.dumps(r1, sort_keys=True))
    assert r2["prev_hash"] == expected


def test_tamper_detection(chain):
    chain.record("step_1", input_data="a", output_data="b")
    chain.record("step_2", input_data="c", output_data="d")
    chain._chain[0]["action"] = "TAMPERED"
    assert chain.verify_chain() is False


# ── Receipt schema ─────────────────────────────────────────────────────────────

def test_receipt_has_required_fields(chain):
    r = chain.record("cot_step", input_data="test", output_data="result")
    required = {"id", "ts", "agent", "principal", "action",
                "input_hash", "output_hash", "prev_hash", "ext"}
    assert required.issubset(r.keys())


def test_ext_has_required_fields(chain):
    r = chain.record("cot_step", input_data="test", output_data="result",
                     input_tokens=5, output_tokens=3)
    ext = r["ext"]
    assert ext["input_tokens"] == 5
    assert ext["output_tokens"] == 3
    assert ext["step_id"] == 1
    assert "run_id" in ext
    assert "version" in ext


def test_ts_is_integer_epoch_ms(chain):
    r = chain.record("step", input_data="x", output_data="y")
    assert isinstance(r["ts"], int)
    assert r["ts"] > 1_700_000_000_000


def test_genesis_prev_hash(chain):
    r = chain.record("step_1", input_data="x", output_data="y")
    assert r["prev_hash"] == "GENESIS"


# ── Privacy defaults ───────────────────────────────────────────────────────────

def test_no_plaintext_by_default(chain):
    r = chain.record("step", input_data="secret", output_data="result")
    assert "input" not in r["ext"]
    assert "output" not in r["ext"]


def test_plaintext_opt_in():
    chain = ReceiptChain(agent="test", store_plaintext=True)
    r = chain.record("step", input_data="secret", output_data="result")
    assert r["ext"]["input"] == "secret"
    assert r["ext"]["output"] == "result"


# ── JSONL flush ────────────────────────────────────────────────────────────────

def test_jsonl_flush_writes_valid_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "receipts.jsonl"
        chain = ReceiptChain(agent="test", output_path=path)
        chain.record("step_1", input_data="a", output_data="b")
        chain.record("step_2", input_data="c", output_data="d")

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "id" in obj


# ── agentreceipts.ai v1 adapter ───────────────────────────────────────────────

def test_to_agentreceipts_v1_iso_timestamp(chain):
    r = chain.record("step", input_data="x", output_data="y")
    v1 = chain.to_agentreceipts_v1(r)
    assert isinstance(v1["ts"], str)
    assert "T" in v1["ts"] and "Z" in v1["ts"]
    assert v1["id"] == r["id"]
    assert v1["agent"] == r["agent"]
