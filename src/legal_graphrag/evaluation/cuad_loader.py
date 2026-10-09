"""
CUAD (Contract Understanding Atticus Dataset) loader.

CUAD ships as a single SQuAD-2.0-format JSON file (commonly `CUAD_v1.json`).
This loader will use a local copy if you point it at one, or auto-download
it from Hugging Face (theatticusproject/cuad) otherwise — see download_cuad().

    {
      "data": [
        {
          "title": "<doc_id>__<Contract Type>",
          "paragraphs": [
            {
              "context": "<full contract text>",
              "qas": [
                {
                  "id": "...",
                  "question": "Highlight the parts (if any) of this contract
                                related to \"<Clause Category>\" that ...",
                  "answers": [{"text": "...", "answer_start": int}, ...],
                  "is_impossible": bool
                },
                ...
              ]
            }
          ]
        },
        ...
      ]
    }

Each `title` in CUAD encodes the contract type as the text after the last
"__" (e.g. "...__Sponsorship_Agreement" -> "Sponsorship Agreement"), and
CUAD's 41 clause categories line up with retrieval/contract_metadata.py's
`contract_type` taxonomy, which is why this project can filter CUAD
contracts by the same `contract_type` field HybridSearchAgent already
supports — no separate mapping table needed.

We keep the CUAD ground truth (the `answers` list, `is_impossible`, and the
clause-category question text) alongside each example specifically so
evaluation/ragas_eval.py can score "was the extraction correct" — not just
"did the pipeline produce an answer" — against a real human-annotated
reference, which is the whole point of using CUAD instead of a synthetic
eval set.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Hugging Face configuration
# ---------------------------------------------------------------------------

CUAD_REPO_ID = "theatticusproject/cuad"
CUAD_FILENAME = "CUAD_v1/CUAD_v1.json"
CUAD_REVISION = "main"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CUADExample:
    """One (contract, clause-category question) pair from CUAD, with ground truth."""

    qas_id: str
    contract_title: str            # CUAD's raw `title`, e.g. "...__Sponsorship_Agreement"
    contract_type: str             # parsed out of contract_title; matches contract_metadata.py's taxonomy
    contract_text: str             # full contract text (CUAD's `context`)
    question: str                  # CUAD's clause-category question, e.g. Highlight the parts about "Governing Law"...
    clause_category: str           # the quoted category pulled out of `question`, e.g. "Governing Law"
    ground_truth_spans: list[str] = field(default_factory=list)  # verbatim answer spans from CUAD annotators
    is_impossible: bool = False    # True = CUAD annotators found NO clause of this category in this contract

    @property
    def collection_name(self) -> str:
        """Chroma collection name this contract is ingested into — see cuad_ingest.py."""
        return re.sub(r"\W+", "_", self.contract_title).strip("_").lower()

    @property
    def reference_answer(self) -> str:
        """
        A single reference string for RAGAS's answer_correctness (which wants
        one `reference` per example): the concatenated ground-truth spans,
        or an explicit "no such clause" statement for is_impossible cases.
        """
        if self.is_impossible or not self.ground_truth_spans:
            return f'This contract contains no clause related to "{self.clause_category}".'
        return " / ".join(self.ground_truth_spans)


_CATEGORY_RE = re.compile(r'related to\s+"([^"]+)"', flags=re.IGNORECASE)


def _parse_contract_type(title: str) -> str:
    # CUAD titles look like "<doc-id>__<Contract_Type>"; the type segment
    # uses underscores for spaces.
    tail = title.rsplit("__", 1)[-1]
    return tail.replace("_", " ").strip()


def _parse_clause_category(question: str) -> str:
    match = _CATEGORY_RE.search(question)
    return match.group(1) if match else question.strip()


# ---------------------------------------------------------------------------
# Hugging Face download
# ---------------------------------------------------------------------------


def download_cuad(
    cache_dir: str | Path | None = None,
    revision: str = CUAD_REVISION,
    force_download: bool = False,
) -> Path:
    """
    Download or retrieve CUAD_v1.json from the Hugging Face cache.

    Parameters:
        cache_dir:
            Optional directory for the Hugging Face cache.

        revision:
            Hugging Face branch, tag, or commit hash.
            "main" is convenient, but a commit hash is better for
            reproducible experiments.

        force_download:
            If True, download the file again instead of using the cache.

    Returns:
        Path to the locally cached CUAD_v1.json file.
    """
    print("[CUAD] Loading dataset from Hugging Face...")
    print(f"[CUAD] Repository: {CUAD_REPO_ID}")
    print(f"[CUAD] File: {CUAD_FILENAME}")
    print(f"[CUAD] Revision: {revision}")

    downloaded_path = hf_hub_download(
        repo_id=CUAD_REPO_ID,
        filename=CUAD_FILENAME,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        force_download=force_download,
    )

    cuad_path = Path(downloaded_path)

    if not cuad_path.exists():
        raise FileNotFoundError(
            f"CUAD was not found after download: {cuad_path}"
        )

    print(f"[CUAD] Dataset available at: {cuad_path}")

    return cuad_path


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def _load_raw_json(
    json_path: str | Path | None = None,
) -> dict:
    """
    Load CUAD JSON from a local path or Hugging Face.

    If json_path is None, CUAD is automatically retrieved from Hugging Face.
    """
    if json_path is None:
        json_path = download_cuad()

    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"CUAD JSON file does not exist: {json_path}"
        )

    with json_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        raise ValueError("CUAD JSON root must be an object.")

    if "data" not in raw:
        raise ValueError(
            'Invalid CUAD JSON: expected a top-level "data" field.'
        )

    if not isinstance(raw["data"], list):
        raise ValueError(
            'Invalid CUAD JSON: "data" must be a list.'
        )

    return raw


def load_cuad(json_path: str | Path | None = None) -> list[CUADExample]:
    """
    Parse a full CUAD_v1.json into a flat list of CUADExample, one per
    (contract, question). Pass a local path to use your own copy, or
    omit it entirely to auto-download from Hugging Face (see download_cuad()).
    """
    raw = _load_raw_json(json_path)

    examples: list[CUADExample] = []
    for doc in raw["data"]:
        contract_title = doc["title"]
        contract_type = _parse_contract_type(contract_title)
        for paragraph in doc["paragraphs"]:
            contract_text = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append(
                    CUADExample(
                        qas_id=qa["id"],
                        contract_title=contract_title,
                        contract_type=contract_type,
                        contract_text=contract_text,
                        question=qa["question"],
                        clause_category=_parse_clause_category(qa["question"]),
                        ground_truth_spans=[a["text"] for a in qa.get("answers", [])],
                        is_impossible=bool(qa.get("is_impossible", False)),
                    )
                )
    return examples


def unique_contracts(examples: list[CUADExample]) -> dict[str, CUADExample]:
    """One representative example per unique contract (by collection_name), for ingestion."""
    by_contract: dict[str, CUADExample] = {}
    for ex in examples:
        by_contract.setdefault(ex.collection_name, ex)
    return by_contract


def split_answerable(examples: list[CUADExample]) -> tuple[list[CUADExample], list[CUADExample]]:
    """
    Returns (answerable, unanswerable).

    This split matters for evaluation: CUAD's `is_impossible` questions mark
    clause categories that genuinely do NOT appear in that contract. RAGAS's
    context_recall/answer_correctness assume a real reference answer exists
    to recall against — scoring an "absence" example the same way as a real
    one would tank those metrics for a case where a low/no-evidence answer
    is actually the CORRECT behavior. See ragas_eval.py, which scores these
    two groups with different metrics: RAGAS metrics for `answerable`, and
    a separate "correctly identified absence" accuracy for `unanswerable`.
    """
    answerable = [e for e in examples if not e.is_impossible]
    unanswerable = [e for e in examples if e.is_impossible]
    return answerable, unanswerable


def sample_examples(
    examples: list[CUADExample],
    n: int,
    seed: int = 42,
    keep_ratio_impossible: bool = True,
) -> list[CUADExample]:
    """
    Deterministic random subsample of size n, for keeping evaluation runs
    (and LLM spend) bounded on the full ~13,000-question CUAD set.

    keep_ratio_impossible: if True (default), samples proportionally from
    the answerable/unanswerable groups so the subsample's is_impossible
    ratio roughly matches the full dataset's, instead of accidentally
    sampling mostly-answerable (or mostly-impossible) questions.
    """
    rng = random.Random(seed)
    if not keep_ratio_impossible:
        return rng.sample(examples, min(n, len(examples)))

    answerable, unanswerable = split_answerable(examples)
    total = len(examples) or 1
    n_answerable = round(n * len(answerable) / total)
    n_unanswerable = n - n_answerable

    sampled = rng.sample(answerable, min(n_answerable, len(answerable))) + rng.sample(
        unanswerable, min(n_unanswerable, len(unanswerable))
    )
    rng.shuffle(sampled)
    return sampled


def sample_contracts(
    examples: list[CUADExample],
    n_contracts: int,
    seed: int = 42,
) -> list[CUADExample]:
    """
    Restrict to all questions belonging to a random sample of n_contracts
    unique contracts — the right way to bound ingestion cost (fewer
    documents to embed) while still evaluating every question CUAD asks
    about each one, rather than truncating questions arbitrarily.
    """
    rng = random.Random(seed)
    contracts = list(unique_contracts(examples).keys())
    chosen = set(rng.sample(contracts, min(n_contracts, len(contracts))))
    return [e for e in examples if e.collection_name in chosen]
