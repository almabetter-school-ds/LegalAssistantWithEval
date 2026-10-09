"""
Contract-level metadata extraction.

Populates the fields that HybridSearchAgent's metadata_filter can filter on
(contract_type, parties, dates, monetary value, governing law) — the same
dimensions the uploaded ContractSearchTool filtered on, but computed here
via a single LLM pass over the document instead of assuming they already
exist as separate database columns. The result is merged into every chunk's
Chroma metadata at ingestion time (see pdf_pipeline.embed_and_store's
extra_metadata parameter).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from ..llm_client import call_json

_METADATA_SYSTEM_PROMPT = """You are a contract analyst. Given the text of a \
contract (or the first portion of it), extract high-level metadata about it.

Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "contract_type": string|null,        // one of: Affiliate Agreement, Development, Distributor,
                                        // Endorsement, Franchise, Hosting, IP, Joint Venture,
                                        // License Agreement, Maintenance, Manufacturing, Marketing,
                                        // Non-Compete/Solicit, Outsourcing, Promotion, Reseller,
                                        // Service, Sponsorship, Strategic Alliance, Supply,
                                        // Transportation, or null if none fit
  "parties": [string],                 // party/company names on the contract
  "effective_date": string|null,       // ISO 8601 YYYY-MM-DD, or null if not stated
  "end_date": string|null,             // ISO 8601 YYYY-MM-DD, or null if not stated / perpetual
  "monetary_value": number|null,       // total contract value if stated, in the currency's numeric amount
  "governing_law_country": string|null // two-letter ISO country code, or null if not stated
}
If a field cannot be determined from the text, use null (or [] for parties)."""


def extract_contract_metadata(document_text: str, max_chars: int = 6000) -> dict:
    """
    document_text: ideally the first few chunks concatenated (governing law,
    parties, and contract type are almost always stated early; truncating
    keeps the prompt small without losing the fields that matter).
    """
    result = call_json(_METADATA_SYSTEM_PROMPT, document_text[:max_chars])
    return result if isinstance(result, dict) else {}


def to_epoch(date_str: Optional[str]) -> Optional[float]:
    """
    Convert an ISO 8601 date string to a Unix epoch (seconds), so it can be
    stored as a numeric Chroma metadata value and filtered with $gte/$lte —
    Chroma's metadata filters support numeric comparisons but not native
    date types or string range queries.
    """
    if not date_str:
        return None
    try:
        return datetime.combine(date.fromisoformat(date_str), datetime.min.time()).timestamp()
    except ValueError:
        return None


def build_chunk_metadata(contract_metadata: dict) -> dict:
    """
    Flatten extract_contract_metadata()'s output into the scalar key/value
    pairs Chroma metadata requires (no nested lists/dicts) — `parties` is
    joined into a single string for substring filtering, since Chroma
    metadata values must be str/int/float/bool, never a list.
    """
    parties = contract_metadata.get("parties") or []
    return {
        "contract_type": contract_metadata.get("contract_type") or "",
        "parties": ", ".join(parties),
        "effective_date_epoch": to_epoch(contract_metadata.get("effective_date")) or 0,
        "end_date_epoch": to_epoch(contract_metadata.get("end_date")) or 0,
        "monetary_value": float(contract_metadata.get("monetary_value") or 0),
        "governing_law_country": (contract_metadata.get("governing_law_country") or "").upper(),
    }
