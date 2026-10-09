"""
RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

  1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
     metrics — faithfulness, context_precision, context_recall, and
     answer_correctness — against CUAD's human-annotated answer spans as
     the `reference`.

  2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
     clause of that category in the contract, so there's no real reference
     to recall against. These are excluded from the RAGAS dataset entirely
     and instead scored with a separate "correctly identified absence"
     metric: did the pipeline's own evidence auditor / answer correctly
     signal that nothing was found, rather than hallucinating a clause?
     Mixing the two would silently tank context_recall/answer_correctness
     on questions that were never answerable in the first place.

Requires (not in the base requirements.txt — see requirements.txt's
"Evaluation" section): `ragas`, `langchain-huggingface`, and ONE of
`langchain-anthropic` / `langchain-groq` depending on RAGAS_JUDGE_PROVIDER
(see _build_ragas_judge_llm below). RAGAS's LLM-judged metrics (faithfulness,
context precision/recall, answer_correctness) need a judge LLM + embeddings
model; the judge defaults to Claude but is swappable via env vars
(RAGAS_JUDGE_PROVIDER=anthropic|groq, RAGAS_JUDGE_MODEL=<model id>) —
embeddings always use the SAME model resources.py's HybridSearchAgent uses
(all-mpnet-base-v2) so "similar enough" judgments are consistent with what
retrieval itself considers similar, regardless of which provider judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..llm_client import MODEL_NAME
from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
from ..tracing import traceable
from .cuad_loader import CUADExample
from .scripted_reviewer import ScriptedRunResult

# ---------------------------------------------------------------------------
# Absence-detection scoring (unanswerable / is_impossible questions)
# ---------------------------------------------------------------------------

_ABSENCE_PHRASES = (
    "no clause", "does not contain", "no such clause", "not found", "no provision",
    "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
)


def _looks_like_absence_answer(answer: Optional[str]) -> bool:
    if not answer:
        return False
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


@dataclass
class AbsenceDetectionResult:
    qas_id: str
    contract_title: str
    clause_category: str
    evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
    evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
    answer_stated_absence: bool            # final answer text reads as "no such clause"
    correctly_identified_absence: bool     # any of the above three


def score_absence_detection(
    results: list[ScriptedRunResult], examples: list[CUADExample]
) -> list[AbsenceDetectionResult]:
    """
    Scores the `is_impossible` subset. `correctly_identified_absence` is
    True if EITHER the evidence auditor flagged insufficient evidence, the
    evidence checkpoint rejected outright, or the synthesized answer itself
    reads as an absence statement — any of these means the pipeline did NOT
    fabricate a clause that CUAD's annotators confirmed doesn't exist.
    """
    scored = []
    for example, result in zip(examples, results):
        assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
        evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
        evidence_rejected = result.status == "evidence_rejected"
        answer_absence = _looks_like_absence_answer(result.final_answer)
        scored.append(
            AbsenceDetectionResult(
                qas_id=example.qas_id,
                contract_title=example.contract_title,
                clause_category=example.clause_category,
                evidence_marked_insufficient=evidence_insufficient,
                evidence_rejected=evidence_rejected,
                answer_stated_absence=answer_absence,
                correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
            )
        )
    return scored


def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
    if not scored:
        return float("nan")
    return sum(s.correctly_identified_absence for s in scored) / len(scored)


# ---------------------------------------------------------------------------
# RAGAS dataset construction (answerable questions only)
# ---------------------------------------------------------------------------

@dataclass
class RagasRow:
    qas_id: str
    contract_title: str
    clause_category: str
    user_input: str
    retrieved_contexts: list[str]
    response: str
    reference: str
    evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


def build_ragas_rows(
    results: list[ScriptedRunResult], examples: list[CUADExample]
) -> list[RagasRow]:
    """
    Builds one row per answerable example. `retrieved_contexts` comes from
    `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
    are forced onto the `hybrid` route — see scripted_reviewer.py's
    force_route default and cuad_ingest.py's module docstring for why.
    Rows where the pipeline never produced a final_answer (evidence or
    answer rejected) still get a row with response="" so RAGAS scores that
    as a real failure rather than silently dropping it from the average.
    """
    rows = []
    for example, result in zip(examples, results):
        assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
        contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
            hit.get("text", "") for hit in result.graph_hits
        ]
        rows.append(
            RagasRow(
                qas_id=example.qas_id,
                contract_title=example.contract_title,
                clause_category=example.clause_category,
                user_input=example.question,
                retrieved_contexts=contexts,
                response=result.final_answer or "",
                reference=example.reference_answer,
                evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
            )
        )
    return rows


def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
    """
    Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
    ragas lazily so the rest of this module (and the non-RAGAS parts of the
    eval harness) works even if `ragas` isn't installed.
    """
    from ragas import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r.user_input,
            retrieved_contexts=r.retrieved_contexts,
            response=r.response,
            reference=r.reference,
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)
def _build_ragas_judge_llm():
    """
    Build the RAGAS judge LLM.

    Provider is auto-selected from environment variables so you can point the
    judge at whatever you have working, WITHOUT hardcoding keys in source:

        RAGAS_JUDGE_PROVIDER = gemini | openai | anthropic | hf   (optional;
                               otherwise auto-detected from the keys below)

        gemini    -> GOOGLE_API_KEY (or GEMINI_API_KEY)   model: RAGAS_JUDGE_MODEL or gemini-2.5-flash
        openai    -> OPENAI_API_KEY                        model: RAGAS_JUDGE_MODEL or gpt-4o-mini
        anthropic -> ANTHROPIC_API_KEY                     model: RAGAS_JUDGE_MODEL or claude-3-5-haiku-latest
        hf        -> HF_TOKEN (original path)              model: RAGAS_JUDGE_MODEL or openai/gpt-oss-120b:groq

    NOTE: the earlier all-NaN metrics were caused by the HF Inference router
    returning HTTP 402 (credits depleted) + timeouts for the judge. Switching
    to any funded provider here makes the four metrics compute normally.
    """
    import os

    from ragas.llms import LangchainLLMWrapper

    provider = os.getenv("RAGAS_JUDGE_PROVIDER", "").strip().lower()
    model_override = os.getenv("RAGAS_JUDGE_MODEL")
    temperature = float(os.getenv("RAGAS_JUDGE_TEMPERATURE", "0"))

    def _has(*names: str) -> bool:
        return any(os.getenv(n) for n in names)

    # auto-detect provider if not explicitly set
    if not provider:
        if _has("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            provider = "gemini"
        elif _has("OPENAI_API_KEY"):
            provider = "openai"
        elif _has("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif _has("HF_TOKEN"):
            provider = "hf"

    # Pick the model for THIS provider. A global RAGAS_JUDGE_MODEL is only
    # applied to the HF path (that's where those ids like
    # "openai/gpt-oss-120b:groq" belong) — otherwise a stale HF id would be
    # sent to Gemini/OpenAI/Anthropic and 404. Each provider also has its own
    # override env var, and any override is ignored if it clearly belongs to a
    # different provider.
    def _model_for(prov: str, default: str, own_env: str) -> str:
        own = os.getenv(own_env)
        if own:
            return own
        if model_override and _looks_like(prov, model_override):
            return model_override
        return default

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = _model_for("gemini", "gemini-2.5-flash", "RAGAS_GEMINI_MODEL")
        chat = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature)
        print(f"[ragas] judge provider=gemini model={model}")
        return LangchainLLMWrapper(chat)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = _model_for("openai", "gpt-4o-mini", "RAGAS_OPENAI_MODEL")
        chat = ChatOpenAI(model=model, temperature=temperature)
        print(f"[ragas] judge provider=openai model={model}")
        return LangchainLLMWrapper(chat)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = _model_for("anthropic", "claude-3-5-haiku-latest", "RAGAS_ANTHROPIC_MODEL")
        chat = ChatAnthropic(model=model, temperature=temperature)
        print(f"[ragas] judge provider=anthropic model={model}")
        return LangchainLLMWrapper(chat)

    # ---- default / fallback: Hugging Face Inference API (original path) ----
    return _build_ragas_judge_llm_hf()


def _looks_like(provider: str, model: str) -> bool:
    """True if `model` id plausibly belongs to `provider` (guards against a
    stale RAGAS_JUDGE_MODEL from one provider leaking into another)."""
    m = model.lower()
    if provider == "gemini":
        return m.startswith("gemini") or m.startswith("models/")
    if provider == "openai":
        return m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("chatgpt")
    if provider == "anthropic":
        return m.startswith("claude")
    return True  # hf: anything goes


def _build_ragas_judge_llm_hf():
    """
    Original Hugging Face Inference judge. Uses HF_TOKEN from the environment.

    Example:
        HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
        RAGAS_JUDGE_MODEL=openai/gpt-oss-120b:groq
    """
    import os

    from huggingface_hub import InferenceClient
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    from ragas.llms import LangchainLLMWrapper

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Add HF_TOKEN=hf_... to your .env file."
        )

    model = os.getenv(
        "RAGAS_JUDGE_MODEL",
        "openai/gpt-oss-120b:groq",
    )

    class HuggingFaceChatModel(BaseChatModel):
        client: object = Field(exclude=True)
        model: str

        @property
        def _llm_type(self) -> str:
            return "huggingface_inference"

        @property
        def _identifying_params(self) -> dict:
            return {
                "model": self.model,
            }

        def _generate(
            self,
            messages,
            stop=None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:

            hf_messages = []

            for message in messages:

                if isinstance(message, SystemMessage):
                    role = "system"

                elif isinstance(message, HumanMessage):
                    role = "user"

                elif isinstance(message, AIMessage):
                    role = "assistant"

                else:
                    role = "user"

                hf_messages.append(
                    {
                        "role": role,
                        "content": message.content,
                    }
                )

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=hf_messages,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.0),
            )

            text = completion.choices[0].message.content or ""

            generation = ChatGeneration(
                message=AIMessage(content=text)
            )

            return ChatResult(
                generations=[generation]
            )

    client = InferenceClient(
        api_key=hf_token,
        provider="auto",
    )

    hf_llm = HuggingFaceChatModel(
        client=client,
        model=model,
    )

    return LangchainLLMWrapper(hf_llm)

# def _build_ragas_judge_llm():
#     """
#     RAGAS's LLM-judged metrics need a langchain-wrapped chat model to act as
#     judge. Configurable via env vars so you can point this at whatever the
#     rest of your eval stack already uses, independent of llm_client.py's
#     Claude model:

#         RAGAS_JUDGE_PROVIDER = "anthropic" (default) | "groq"
#         RAGAS_JUDGE_MODEL    = provider-specific model id
#                                 (default: llm_client.MODEL_NAME for anthropic,
#                                  "openai/gpt-oss-120b" for groq)

#     For provider="groq": if RAGAS_JUDGE_MODEL carries a trailing
#     ":<routing-hint>" suffix (e.g. "openai/gpt-oss-120b:groq" — an
#     OpenRouter-style provider suffix), it's stripped before being passed to
#     ChatGroq, since Groq's own API takes the bare model id
#     ("openai/gpt-oss-120b") and rejects the suffixed form. Requires
#     GROQ_API_KEY to be set.
#     """
#     import os

#     from ragas.llms import LangchainLLMWrapper

#     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "anthropic").lower()

#     if provider == "groq":
#         from langchain_groq import ChatGroq

#         model = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
#         model = model.split(":", 1)[0]  # strip any ":<routing-hint>" suffix — see docstring
#         return LangchainLLMWrapper(ChatGroq(model=model))

#     if provider == "anthropic":
#         from langchain_anthropic import ChatAnthropic

#         model = os.getenv("RAGAS_JUDGE_MODEL", MODEL_NAME)
#         return LangchainLLMWrapper(ChatAnthropic(model=model))

#     raise ValueError(f"Unknown RAGAS_JUDGE_PROVIDER={provider!r} — expected 'anthropic' or 'groq'.")


# def _build_ragas_llm_and_embeddings():
#     """
#     Judge LLM (see _build_ragas_judge_llm) + embeddings model. Embeddings
#     use the SAME model HybridSearchAgent uses (EMBEDDING_MODEL_NAME), so
#     RAGAS's notion of "similar" for answer_correctness matches what
#     retrieval itself uses, regardless of which provider judges the LLM
#     metrics.
#     """
#     from langchain_huggingface import HuggingFaceEmbeddings
#     from ragas.embeddings import LangchainEmbeddingsWrapper

#     llm = _build_ragas_judge_llm()
#     embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
#     return llm, embeddings
def _build_ragas_llm_and_embeddings():
    """
    Build the Hugging Face judge LLM + Hugging Face embeddings.
    """

    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm = _build_ragas_judge_llm()

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
    )

    return llm, embeddings

@traceable(name="evaluation.run_ragas", run_type="chain")
def run_ragas_evaluation(rows: list[RagasRow]):
    """
    Runs the RAGAS metric set over `rows` (answerable questions only — see this
    module's docstring). Returns ragas's EvaluationResult (has both an
    aggregate `.to_pandas()` table and per-metric scores).

    Metric set is now the lean pair the UI shows by default:
        - Faithfulness        — is every claim in the answer backed by context?
        - AnswerCorrectness   — semantic alignment vs the CUAD ground-truth answer

    Set RAGAS_FULL_METRICS=1 to also compute LLMContextPrecisionWithReference
    and LLMContextRecall (the two extra retrieval metrics).
    """
    import os

    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        AnswerCorrectness,
    )

    # [FIX] Make evaluate() robust to RAGAS's empty-trace IndexError (scores are
    # computed fine; only the debug-only .traces parsing crashes). See ragas_compat.
    from .ragas_compat import install_ragas_trace_guard
    install_ragas_trace_guard()

    dataset = _build_ragas_evaluation_dataset(rows)
    llm, embeddings = _build_ragas_llm_and_embeddings()

    # Lean default: the two metrics the frontend renders.
    metrics = [
        Faithfulness(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]
    if os.getenv("RAGAS_FULL_METRICS", "").strip() in ("1", "true", "yes"):
        metrics += [
            LLMContextPrecisionWithReference(llm=llm),
            LLMContextRecall(llm=llm),
        ]

    evaluation = evaluate(dataset=dataset, metrics=metrics)

    # ---- optional debug dump (set RAGAS_DEBUG=1) ----
    if os.getenv("RAGAS_DEBUG", "").strip() in ("1", "true", "yes"):
        print("\n================ RAGAS DEBUG ================")
        print("[DEBUG RAGAS TYPE]:", type(evaluation))
        if hasattr(evaluation, "_repr_dict"):
            print("[DEBUG _repr_dict]:", dict(evaluation._repr_dict))
        try:
            df = evaluation.to_pandas()
            print("[DEBUG DataFrame Columns]:", list(df.columns))
            print("\n[DEBUG DataFrame]:\n", df.to_string())
            if len(df):
                print("\n[DEBUG First Row]:", df.iloc[0].to_dict())
            df.to_csv("ragas_evaluation_results.csv", index=False)
            print("[DEBUG] wrote ragas_evaluation_results.csv")
        except Exception as exc:  # noqa: BLE001
            print("[DEBUG] to_pandas() failed:", exc)
        print("==============================================\n")

    return evaluation


# """
# RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

#   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
#      metrics — faithfulness, context_precision, context_recall, and
#      answer_correctness — against CUAD's human-annotated answer spans as
#      the `reference`.

#   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
#      clause of that category in the contract, so there's no real reference
#      to recall against. These are excluded from the RAGAS dataset entirely
#      and instead scored with a separate "correctly identified absence"
#      metric: did the pipeline's own evidence auditor / answer correctly
#      signal that nothing was found, rather than hallucinating a clause?
#      Mixing the two would silently tank context_recall/answer_correctness
#      on questions that were never answerable in the first place.

# Requires (not in the base requirements.txt — see requirements.txt's
# "Evaluation" section): `ragas`, `langchain-HF`, `langchain-huggingface`.
# RAGAS's LLM-judged metrics (faithfulness, context precision/recall,
# answer_correctness) need an LLM + embeddings model; this module reuses
# Claude (via langchain_HF, matching llm_client.py's MODEL_NAME) as
# the judge, and the SAME embedding model resources.py's HybridSearchAgent
# uses (all-mpnet-base-v2) so "similar enough" judgments are consistent with
# what retrieval itself considers similar.
# """

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Optional

# from ..llm_client import MODEL_NAME
# from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# from ..tracing import traceable
# from .cuad_loader import CUADExample
# from .scripted_reviewer import ScriptedRunResult

# import os

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_huggingface import HuggingFaceEmbeddings

# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# ---------------------------------------------------------------------------
# # Absence-detection scoring (unanswerable / is_impossible questions)
# # ---------------------------------------------------------------------------
# MODEL_NAME = "openai/gpt-oss-120b:groq"
# _ABSENCE_PHRASES = (
#     "no clause", "does not contain", "no such clause", "not found", "no provision",
#     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# )


# def _looks_like_absence_answer(answer: Optional[str]) -> bool:
#     if not answer:
#         return False
#     lowered = answer.lower()
#     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# @dataclass
# class AbsenceDetectionResult:
#     qas_id: str
#     contract_title: str
#     clause_category: str
#     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
#     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
#     answer_stated_absence: bool            # final answer text reads as "no such clause"
#     correctly_identified_absence: bool     # any of the above three


# def score_absence_detection(
#     results: list[ScriptedRunResult], examples: list[CUADExample]
# ) -> list[AbsenceDetectionResult]:
#     """
#     Scores the `is_impossible` subset. `correctly_identified_absence` is
#     True if EITHER the evidence auditor flagged insufficient evidence, the
#     evidence checkpoint rejected outright, or the synthesized answer itself
#     reads as an absence statement — any of these means the pipeline did NOT
#     fabricate a clause that CUAD's annotators confirmed doesn't exist.
#     """
#     scored = []
#     for example, result in zip(examples, results):
#         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
#         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
#         evidence_rejected = result.status == "evidence_rejected"
#         answer_absence = _looks_like_absence_answer(result.final_answer)
#         scored.append(
#             AbsenceDetectionResult(
#                 qas_id=example.qas_id,
#                 contract_title=example.contract_title,
#                 clause_category=example.clause_category,
#                 evidence_marked_insufficient=evidence_insufficient,
#                 evidence_rejected=evidence_rejected,
#                 answer_stated_absence=answer_absence,
#                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
#             )
#         )
#     return scored


# def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
#     if not scored:
#         return float("nan")
#     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # ---------------------------------------------------------------------------
# # RAGAS dataset construction (answerable questions only)
# # ---------------------------------------------------------------------------

# @dataclass
# class RagasRow:
#     qas_id: str
#     contract_title: str
#     clause_category: str
#     user_input: str
#     retrieved_contexts: list[str]
#     response: str
#     reference: str
#     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# def build_ragas_rows(
#     results: list[ScriptedRunResult], examples: list[CUADExample]
# ) -> list[RagasRow]:
#     """
#     Builds one row per answerable example. `retrieved_contexts` comes from
#     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
#     are forced onto the `hybrid` route — see scripted_reviewer.py's
#     force_route default and cuad_ingest.py's module docstring for why.
#     Rows where the pipeline never produced a final_answer (evidence or
#     answer rejected) still get a row with response="" so RAGAS scores that
#     as a real failure rather than silently dropping it from the average.
#     """
#     rows = []
#     for example, result in zip(examples, results):
#         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
#         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
#             hit.get("text", "") for hit in result.graph_hits
#         ]
#         rows.append(
#             RagasRow(
#                 qas_id=example.qas_id,
#                 contract_title=example.contract_title,
#                 clause_category=example.clause_category,
#                 user_input=example.question,
#                 retrieved_contexts=contexts,
#                 response=result.final_answer or "",
#                 reference=example.reference_answer,
#                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
#             )
#         )
#     return rows


# def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
#     """
#     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
#     ragas lazily so the rest of this module (and the non-RAGAS parts of the
#     eval harness) works even if `ragas` isn't installed.
#     """
#     from ragas import EvaluationDataset, SingleTurnSample

#     samples = [
#         SingleTurnSample(
#             user_input=r.user_input,
#             retrieved_contexts=r.retrieved_contexts,
#             response=r.response,
#             reference=r.reference,
#         )
#         for r in rows
#     ]
#     return EvaluationDataset(samples=samples)


# def _build_ragas_llm_and_embeddings():
#     """
#     RAGAS's LLM-judged metrics need a langchain-wrapped LLM + embeddings
#     model. Uses the same Claude model as the rest of the pipeline
#     (llm_client.MODEL_NAME) and the same embedding model HybridSearchAgent
#     uses (EMBEDDING_MODEL_NAME), so judgments are made with the same notion
#     of "similar" that retrieval itself uses.
#     """
#     llm_endpoint = HuggingFaceEndpoint(
#         repo_id="openai/gpt-oss-120b:groq",
#         huggingfacehub_api_token=os.getenv("HF_TOKEN"),
#         max_new_tokens=1000,
#         temperature=0.1,
#     )

#     hf_llm = ChatHuggingFace(
#         llm=llm_endpoint
#     )

#     llm = LangchainLLMWrapper(hf_llm)

#     # -----------------------------
#     # Hugging Face Embeddings
#     # -----------------------------
#     embeddings_model = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME
#     )

#     embeddings = LangchainEmbeddingsWrapper(
#         embeddings_model
#     )
#     return llm, embeddings
#     # from langchain_anthropic import ChatAnthropic
#     # from langchain_huggingface import HuggingFaceEmbeddings
#     # from ragas.llms import LangchainLLMWrapper
#     # from ragas.embeddings import LangchainEmbeddingsWrapper

#     # llm = LangchainLLMWrapper(ChatAnthropic(model=MODEL_NAME))
#     # embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    


# @traceable(name="evaluation.run_ragas", run_type="chain")
# def run_ragas_evaluation(rows: list[RagasRow]):
#     """
#     Runs faithfulness, context_precision, context_recall, and
#     answer_correctness over `rows` (answerable questions only — see this
#     module's docstring). Returns ragas's EvaluationResult (has both an
#     aggregate `.to_pandas()` table and per-metric scores).
#     """
#     from ragas import evaluate
#     from ragas.metrics import (
#         Faithfulness,
#         LLMContextPrecisionWithReference,
#         LLMContextRecall,
#         AnswerCorrectness,
#     )

#     dataset = _build_ragas_evaluation_dataset(rows)
#     llm, embeddings = _build_ragas_llm_and_embeddings()

#     metrics = [
#         Faithfulness(llm=llm),
#         LLMContextPrecisionWithReference(llm=llm),
#         LLMContextRecall(llm=llm),
#         AnswerCorrectness(llm=llm, embeddings=embeddings),
#     ]
#     return evaluate(dataset=dataset, metrics=metrics)
# """
# RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

#   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
#      metrics — faithfulness, context_precision, context_recall, and
#      answer_correctness — against CUAD's human-annotated answer spans as
#      the `reference`.

#   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
#      clause of that category in the contract, so there's no real reference
#      to recall against. These are excluded from the RAGAS dataset entirely
#      and instead scored with a separate "correctly identified absence"
#      metric: did the pipeline's own evidence auditor / answer correctly
#      signal that nothing was found, rather than hallucinating a clause?
#      Mixing the two would silently tank context_recall/answer_correctness
#      on questions that were never answerable in the first place.

# Requires (not in the base requirements.txt — see requirements.txt's
# "Evaluation" section): `ragas`, `langchain-huggingface`, and ONE of
# `langchain-anthropic` / `langchain-groq` depending on RAGAS_JUDGE_PROVIDER
# (see _build_ragas_judge_llm below). RAGAS's LLM-judged metrics (faithfulness,
# context precision/recall, answer_correctness) need a judge LLM + embeddings
# model; the judge defaults to Claude but is swappable via env vars
# (RAGAS_JUDGE_PROVIDER=anthropic|groq, RAGAS_JUDGE_MODEL=<model id>) —
# embeddings always use the SAME model resources.py's HybridSearchAgent uses
# (all-mpnet-base-v2) so "similar enough" judgments are consistent with what
# retrieval itself considers similar, regardless of which provider judges.
# """

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Optional

# from ..llm_client import MODEL_NAME
# from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# from ..tracing import traceable
# from .cuad_loader import CUADExample
# from .scripted_reviewer import ScriptedRunResult

# # ---------------------------------------------------------------------------
# # Absence-detection scoring (unanswerable / is_impossible questions)
# # ---------------------------------------------------------------------------

# _ABSENCE_PHRASES = (
#     "no clause", "does not contain", "no such clause", "not found", "no provision",
#     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# )


# def _looks_like_absence_answer(answer: Optional[str]) -> bool:
#     if not answer:
#         return False
#     lowered = answer.lower()
#     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# @dataclass
# class AbsenceDetectionResult:
#     qas_id: str
#     contract_title: str
#     clause_category: str
#     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
#     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
#     answer_stated_absence: bool            # final answer text reads as "no such clause"
#     correctly_identified_absence: bool     # any of the above three


# def score_absence_detection(
#     results: list[ScriptedRunResult], examples: list[CUADExample]
# ) -> list[AbsenceDetectionResult]:
#     """
#     Scores the `is_impossible` subset. `correctly_identified_absence` is
#     True if EITHER the evidence auditor flagged insufficient evidence, the
#     evidence checkpoint rejected outright, or the synthesized answer itself
#     reads as an absence statement — any of these means the pipeline did NOT
#     fabricate a clause that CUAD's annotators confirmed doesn't exist.
#     """
#     scored = []
#     for example, result in zip(examples, results):
#         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
#         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
#         evidence_rejected = result.status == "evidence_rejected"
#         answer_absence = _looks_like_absence_answer(result.final_answer)
#         scored.append(
#             AbsenceDetectionResult(
#                 qas_id=example.qas_id,
#                 contract_title=example.contract_title,
#                 clause_category=example.clause_category,
#                 evidence_marked_insufficient=evidence_insufficient,
#                 evidence_rejected=evidence_rejected,
#                 answer_stated_absence=answer_absence,
#                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
#             )
#         )
#     return scored


# def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
#     if not scored:
#         return float("nan")
#     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # ---------------------------------------------------------------------------
# # RAGAS dataset construction (answerable questions only)
# # ---------------------------------------------------------------------------

# @dataclass
# class RagasRow:
#     qas_id: str
#     contract_title: str
#     clause_category: str
#     user_input: str
#     retrieved_contexts: list[str]
#     response: str
#     reference: str
#     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# def build_ragas_rows(
#     results: list[ScriptedRunResult], examples: list[CUADExample]
# ) -> list[RagasRow]:
#     """
#     Builds one row per answerable example. `retrieved_contexts` comes from
#     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
#     are forced onto the `hybrid` route — see scripted_reviewer.py's
#     force_route default and cuad_ingest.py's module docstring for why.
#     Rows where the pipeline never produced a final_answer (evidence or
#     answer rejected) still get a row with response="" so RAGAS scores that
#     as a real failure rather than silently dropping it from the average.
#     """
#     rows = []
#     for example, result in zip(examples, results):
#         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
#         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
#             hit.get("text", "") for hit in result.graph_hits
#         ]
#         rows.append(
#             RagasRow(
#                 qas_id=example.qas_id,
#                 contract_title=example.contract_title,
#                 clause_category=example.clause_category,
#                 user_input=example.question,
#                 retrieved_contexts=contexts,
#                 response=result.final_answer or "",
#                 reference=example.reference_answer,
#                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
#             )
#         )
#     return rows


# def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
#     """
#     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
#     ragas lazily so the rest of this module (and the non-RAGAS parts of the
#     eval harness) works even if `ragas` isn't installed.
#     """
#     from ragas import EvaluationDataset, SingleTurnSample

#     samples = [
#         SingleTurnSample(
#             user_input=r.user_input,
#             retrieved_contexts=r.retrieved_contexts,
#             response=r.response,
#             reference=r.reference,
#         )
#         for r in rows
#     ]
#     return EvaluationDataset(samples=samples)
# def _build_ragas_judge_llm():
#     """
#     Build the RAGAS judge LLM.

#     Provider is auto-selected from environment variables so you can point the
#     judge at whatever you have working, WITHOUT hardcoding keys in source:

#         RAGAS_JUDGE_PROVIDER = gemini | openai | anthropic | hf   (optional;
#                                otherwise auto-detected from the keys below)

#         gemini    -> GOOGLE_API_KEY (or GEMINI_API_KEY)   model: RAGAS_JUDGE_MODEL or gemini-2.5-flash
#         openai    -> OPENAI_API_KEY                        model: RAGAS_JUDGE_MODEL or gpt-4o-mini
#         anthropic -> ANTHROPIC_API_KEY                     model: RAGAS_JUDGE_MODEL or claude-3-5-haiku-latest
#         hf        -> HF_TOKEN (original path)              model: RAGAS_JUDGE_MODEL or openai/gpt-oss-120b:groq

#     NOTE: the earlier all-NaN metrics were caused by the HF Inference router
#     returning HTTP 402 (credits depleted) + timeouts for the judge. Switching
#     to any funded provider here makes the four metrics compute normally.
#     """
#     import os

#     from ragas.llms import LangchainLLMWrapper

#     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "").strip().lower()
#     model_override = os.getenv("RAGAS_JUDGE_MODEL")
#     temperature = float(os.getenv("RAGAS_JUDGE_TEMPERATURE", "0"))

#     def _has(*names: str) -> bool:
#         return any(os.getenv(n) for n in names)

#     # auto-detect provider if not explicitly set
#     if not provider:
#         if _has("GOOGLE_API_KEY", "GEMINI_API_KEY"):
#             provider = "gemini"
#         elif _has("OPENAI_API_KEY"):
#             provider = "openai"
#         elif _has("ANTHROPIC_API_KEY"):
#             provider = "anthropic"
#         elif _has("HF_TOKEN"):
#             provider = "hf"

#     # Pick the model for THIS provider. A global RAGAS_JUDGE_MODEL is only
#     # applied to the HF path (that's where those ids like
#     # "openai/gpt-oss-120b:groq" belong) — otherwise a stale HF id would be
#     # sent to Gemini/OpenAI/Anthropic and 404. Each provider also has its own
#     # override env var, and any override is ignored if it clearly belongs to a
#     # different provider.
#     def _model_for(prov: str, default: str, own_env: str) -> str:
#         own = os.getenv(own_env)
#         if own:
#             return own
#         if model_override and _looks_like(prov, model_override):
#             return model_override
#         return default

#     if provider == "gemini":
#         from langchain_google_genai import ChatGoogleGenerativeAI
#         key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
#         model = _model_for("gemini", "gemini-2.5-flash", "RAGAS_GEMINI_MODEL")
#         chat = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature)
#         print(f"[ragas] judge provider=gemini model={model}")
#         return LangchainLLMWrapper(chat)

#     if provider == "openai":
#         from langchain_openai import ChatOpenAI
#         model = _model_for("openai", "gpt-4o-mini", "RAGAS_OPENAI_MODEL")
#         chat = ChatOpenAI(model=model, temperature=temperature)
#         print(f"[ragas] judge provider=openai model={model}")
#         return LangchainLLMWrapper(chat)

#     if provider == "anthropic":
#         from langchain_anthropic import ChatAnthropic
#         model = _model_for("anthropic", "claude-3-5-haiku-latest", "RAGAS_ANTHROPIC_MODEL")
#         chat = ChatAnthropic(model=model, temperature=temperature)
#         print(f"[ragas] judge provider=anthropic model={model}")
#         return LangchainLLMWrapper(chat)

#     # ---- default / fallback: Hugging Face Inference API (original path) ----
#     return _build_ragas_judge_llm_hf()


# def _looks_like(provider: str, model: str) -> bool:
#     """True if `model` id plausibly belongs to `provider` (guards against a
#     stale RAGAS_JUDGE_MODEL from one provider leaking into another)."""
#     m = model.lower()
#     if provider == "gemini":
#         return m.startswith("gemini") or m.startswith("models/")
#     if provider == "openai":
#         return m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("chatgpt")
#     if provider == "anthropic":
#         return m.startswith("claude")
#     return True  # hf: anything goes


# def _build_ragas_judge_llm_hf():
#     """
#     Original Hugging Face Inference judge. Uses HF_TOKEN from the environment.

#     Example:
#         HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
#         RAGAS_JUDGE_MODEL=openai/gpt-oss-120b:groq
#     """
#     import os

#     from huggingface_hub import InferenceClient
#     from langchain_core.language_models.chat_models import BaseChatModel
#     from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
#     from langchain_core.outputs import ChatGeneration, ChatResult
#     from pydantic import Field

#     from ragas.llms import LangchainLLMWrapper

#     hf_token = os.getenv("HF_TOKEN")

#     if not hf_token:
#         raise RuntimeError(
#             "HF_TOKEN environment variable is not set. "
#             "Add HF_TOKEN=hf_... to your .env file."
#         )

#     model = os.getenv(
#         "RAGAS_JUDGE_MODEL",
#         "openai/gpt-oss-120b:groq",
#     )

#     class HuggingFaceChatModel(BaseChatModel):
#         client: object = Field(exclude=True)
#         model: str

#         @property
#         def _llm_type(self) -> str:
#             return "huggingface_inference"

#         @property
#         def _identifying_params(self) -> dict:
#             return {
#                 "model": self.model,
#             }

#         def _generate(
#             self,
#             messages,
#             stop=None,
#             run_manager=None,
#             **kwargs,
#         ) -> ChatResult:

#             hf_messages = []

#             for message in messages:

#                 if isinstance(message, SystemMessage):
#                     role = "system"

#                 elif isinstance(message, HumanMessage):
#                     role = "user"

#                 elif isinstance(message, AIMessage):
#                     role = "assistant"

#                 else:
#                     role = "user"

#                 hf_messages.append(
#                     {
#                         "role": role,
#                         "content": message.content,
#                     }
#                 )

#             completion = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=hf_messages,
#                 max_tokens=kwargs.get("max_tokens", 2000),
#                 temperature=kwargs.get("temperature", 0.0),
#             )

#             text = completion.choices[0].message.content or ""

#             generation = ChatGeneration(
#                 message=AIMessage(content=text)
#             )

#             return ChatResult(
#                 generations=[generation]
#             )

#     client = InferenceClient(
#         api_key=hf_token,
#         provider="auto",
#     )

#     hf_llm = HuggingFaceChatModel(
#         client=client,
#         model=model,
#     )

#     return LangchainLLMWrapper(hf_llm)

# # def _build_ragas_judge_llm():
# #     """
# #     RAGAS's LLM-judged metrics need a langchain-wrapped chat model to act as
# #     judge. Configurable via env vars so you can point this at whatever the
# #     rest of your eval stack already uses, independent of llm_client.py's
# #     Claude model:

# #         RAGAS_JUDGE_PROVIDER = "anthropic" (default) | "groq"
# #         RAGAS_JUDGE_MODEL    = provider-specific model id
# #                                 (default: llm_client.MODEL_NAME for anthropic,
# #                                  "openai/gpt-oss-120b" for groq)

# #     For provider="groq": if RAGAS_JUDGE_MODEL carries a trailing
# #     ":<routing-hint>" suffix (e.g. "openai/gpt-oss-120b:groq" — an
# #     OpenRouter-style provider suffix), it's stripped before being passed to
# #     ChatGroq, since Groq's own API takes the bare model id
# #     ("openai/gpt-oss-120b") and rejects the suffixed form. Requires
# #     GROQ_API_KEY to be set.
# #     """
# #     import os

# #     from ragas.llms import LangchainLLMWrapper

# #     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "anthropic").lower()

# #     if provider == "groq":
# #         from langchain_groq import ChatGroq

# #         model = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
# #         model = model.split(":", 1)[0]  # strip any ":<routing-hint>" suffix — see docstring
# #         return LangchainLLMWrapper(ChatGroq(model=model))

# #     if provider == "anthropic":
# #         from langchain_anthropic import ChatAnthropic

# #         model = os.getenv("RAGAS_JUDGE_MODEL", MODEL_NAME)
# #         return LangchainLLMWrapper(ChatAnthropic(model=model))

# #     raise ValueError(f"Unknown RAGAS_JUDGE_PROVIDER={provider!r} — expected 'anthropic' or 'groq'.")


# # def _build_ragas_llm_and_embeddings():
# #     """
# #     Judge LLM (see _build_ragas_judge_llm) + embeddings model. Embeddings
# #     use the SAME model HybridSearchAgent uses (EMBEDDING_MODEL_NAME), so
# #     RAGAS's notion of "similar" for answer_correctness matches what
# #     retrieval itself uses, regardless of which provider judges the LLM
# #     metrics.
# #     """
# #     from langchain_huggingface import HuggingFaceEmbeddings
# #     from ragas.embeddings import LangchainEmbeddingsWrapper

# #     llm = _build_ragas_judge_llm()
# #     embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
# #     return llm, embeddings
# def _build_ragas_llm_and_embeddings():
#     """
#     Build the Hugging Face judge LLM + Hugging Face embeddings.
#     """

#     from langchain_huggingface import HuggingFaceEmbeddings
#     from ragas.embeddings import LangchainEmbeddingsWrapper

#     llm = _build_ragas_judge_llm()

#     embeddings = LangchainEmbeddingsWrapper(
#         HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL_NAME
#         )
#     )

#     return llm, embeddings

# @traceable(name="evaluation.run_ragas", run_type="chain")
# def run_ragas_evaluation(rows: list[RagasRow]):
#     """
#     Runs the RAGAS metric set over `rows` (answerable questions only — see this
#     module's docstring). Returns ragas's EvaluationResult (has both an
#     aggregate `.to_pandas()` table and per-metric scores).

#     Metric set is now the lean pair the UI shows by default:
#         - Faithfulness        — is every claim in the answer backed by context?
#         - AnswerCorrectness   — semantic alignment vs the CUAD ground-truth answer

#     Set RAGAS_FULL_METRICS=1 to also compute LLMContextPrecisionWithReference
#     and LLMContextRecall (the two extra retrieval metrics).
#     """
#     import os

#     from ragas import evaluate
#     from ragas.metrics import (
#         Faithfulness,
#         LLMContextPrecisionWithReference,
#         LLMContextRecall,
#         AnswerCorrectness,
#     )

#     # [FIX] Make evaluate() robust to RAGAS's empty-trace IndexError (scores are
#     # computed fine; only the debug-only .traces parsing crashes). See ragas_compat.
#     from .ragas_compat import install_ragas_trace_guard
#     install_ragas_trace_guard()

#     dataset = _build_ragas_evaluation_dataset(rows)
#     llm, embeddings = _build_ragas_llm_and_embeddings()

#     # Lean default: the two metrics the frontend renders.
#     metrics = [
#         Faithfulness(llm=llm),
#         AnswerCorrectness(llm=llm, embeddings=embeddings),
#     ]
#     if os.getenv("RAGAS_FULL_METRICS", "").strip() in ("1", "true", "yes"):
#         metrics += [
#             LLMContextPrecisionWithReference(llm=llm),
#             LLMContextRecall(llm=llm),
#         ]

#     evaluation = evaluate(dataset=dataset, metrics=metrics)

#     # ---- optional debug dump (set RAGAS_DEBUG=1) ----
#     if os.getenv("RAGAS_DEBUG", "").strip() in ("1", "true", "yes"):
#         print("\n================ RAGAS DEBUG ================")
#         print("[DEBUG RAGAS TYPE]:", type(evaluation))
#         if hasattr(evaluation, "_repr_dict"):
#             print("[DEBUG _repr_dict]:", dict(evaluation._repr_dict))
#         try:
#             df = evaluation.to_pandas()
#             print("[DEBUG DataFrame Columns]:", list(df.columns))
#             print("\n[DEBUG DataFrame]:\n", df.to_string())
#             if len(df):
#                 print("\n[DEBUG First Row]:", df.iloc[0].to_dict())
#             df.to_csv("ragas_evaluation_results.csv", index=False)
#             print("[DEBUG] wrote ragas_evaluation_results.csv")
#         except Exception as exc:  # noqa: BLE001
#             print("[DEBUG] to_pandas() failed:", exc)
#         print("==============================================\n")

#     return evaluation


# # """
# # RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# # Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

# #   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
# #      metrics — faithfulness, context_precision, context_recall, and
# #      answer_correctness — against CUAD's human-annotated answer spans as
# #      the `reference`.

# #   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
# #      clause of that category in the contract, so there's no real reference
# #      to recall against. These are excluded from the RAGAS dataset entirely
# #      and instead scored with a separate "correctly identified absence"
# #      metric: did the pipeline's own evidence auditor / answer correctly
# #      signal that nothing was found, rather than hallucinating a clause?
# #      Mixing the two would silently tank context_recall/answer_correctness
# #      on questions that were never answerable in the first place.

# # Requires (not in the base requirements.txt — see requirements.txt's
# # "Evaluation" section): `ragas`, `langchain-HF`, `langchain-huggingface`.
# # RAGAS's LLM-judged metrics (faithfulness, context precision/recall,
# # answer_correctness) need an LLM + embeddings model; this module reuses
# # Claude (via langchain_HF, matching llm_client.py's MODEL_NAME) as
# # the judge, and the SAME embedding model resources.py's HybridSearchAgent
# # uses (all-mpnet-base-v2) so "similar enough" judgments are consistent with
# # what retrieval itself considers similar.
# # """

# # from __future__ import annotations

# # from dataclasses import dataclass, field
# # from typing import Optional

# # from ..llm_client import MODEL_NAME
# # from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# # from ..tracing import traceable
# # from .cuad_loader import CUADExample
# # from .scripted_reviewer import ScriptedRunResult

# # import os

# # from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# # from langchain_huggingface import HuggingFaceEmbeddings

# # from ragas.llms import LangchainLLMWrapper
# # from ragas.embeddings import LangchainEmbeddingsWrapper
# # ---------------------------------------------------------------------------
# # # Absence-detection scoring (unanswerable / is_impossible questions)
# # # ---------------------------------------------------------------------------
# # MODEL_NAME = "openai/gpt-oss-120b:groq"
# # _ABSENCE_PHRASES = (
# #     "no clause", "does not contain", "no such clause", "not found", "no provision",
# #     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# # )


# # def _looks_like_absence_answer(answer: Optional[str]) -> bool:
# #     if not answer:
# #         return False
# #     lowered = answer.lower()
# #     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# # @dataclass
# # class AbsenceDetectionResult:
# #     qas_id: str
# #     contract_title: str
# #     clause_category: str
# #     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
# #     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
# #     answer_stated_absence: bool            # final answer text reads as "no such clause"
# #     correctly_identified_absence: bool     # any of the above three


# # def score_absence_detection(
# #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # ) -> list[AbsenceDetectionResult]:
# #     """
# #     Scores the `is_impossible` subset. `correctly_identified_absence` is
# #     True if EITHER the evidence auditor flagged insufficient evidence, the
# #     evidence checkpoint rejected outright, or the synthesized answer itself
# #     reads as an absence statement — any of these means the pipeline did NOT
# #     fabricate a clause that CUAD's annotators confirmed doesn't exist.
# #     """
# #     scored = []
# #     for example, result in zip(examples, results):
# #         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
# #         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
# #         evidence_rejected = result.status == "evidence_rejected"
# #         answer_absence = _looks_like_absence_answer(result.final_answer)
# #         scored.append(
# #             AbsenceDetectionResult(
# #                 qas_id=example.qas_id,
# #                 contract_title=example.contract_title,
# #                 clause_category=example.clause_category,
# #                 evidence_marked_insufficient=evidence_insufficient,
# #                 evidence_rejected=evidence_rejected,
# #                 answer_stated_absence=answer_absence,
# #                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
# #             )
# #         )
# #     return scored


# # def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
# #     if not scored:
# #         return float("nan")
# #     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # # ---------------------------------------------------------------------------
# # # RAGAS dataset construction (answerable questions only)
# # # ---------------------------------------------------------------------------

# # @dataclass
# # class RagasRow:
# #     qas_id: str
# #     contract_title: str
# #     clause_category: str
# #     user_input: str
# #     retrieved_contexts: list[str]
# #     response: str
# #     reference: str
# #     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# # def build_ragas_rows(
# #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # ) -> list[RagasRow]:
# #     """
# #     Builds one row per answerable example. `retrieved_contexts` comes from
# #     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
# #     are forced onto the `hybrid` route — see scripted_reviewer.py's
# #     force_route default and cuad_ingest.py's module docstring for why.
# #     Rows where the pipeline never produced a final_answer (evidence or
# #     answer rejected) still get a row with response="" so RAGAS scores that
# #     as a real failure rather than silently dropping it from the average.
# #     """
# #     rows = []
# #     for example, result in zip(examples, results):
# #         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
# #         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
# #             hit.get("text", "") for hit in result.graph_hits
# #         ]
# #         rows.append(
# #             RagasRow(
# #                 qas_id=example.qas_id,
# #                 contract_title=example.contract_title,
# #                 clause_category=example.clause_category,
# #                 user_input=example.question,
# #                 retrieved_contexts=contexts,
# #                 response=result.final_answer or "",
# #                 reference=example.reference_answer,
# #                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
# #             )
# #         )
# #     return rows


# # def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
# #     """
# #     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
# #     ragas lazily so the rest of this module (and the non-RAGAS parts of the
# #     eval harness) works even if `ragas` isn't installed.
# #     """
# #     from ragas import EvaluationDataset, SingleTurnSample

# #     samples = [
# #         SingleTurnSample(
# #             user_input=r.user_input,
# #             retrieved_contexts=r.retrieved_contexts,
# #             response=r.response,
# #             reference=r.reference,
# #         )
# #         for r in rows
# #     ]
# #     return EvaluationDataset(samples=samples)


# # def _build_ragas_llm_and_embeddings():
# #     """
# #     RAGAS's LLM-judged metrics need a langchain-wrapped LLM + embeddings
# #     model. Uses the same Claude model as the rest of the pipeline
# #     (llm_client.MODEL_NAME) and the same embedding model HybridSearchAgent
# #     uses (EMBEDDING_MODEL_NAME), so judgments are made with the same notion
# #     of "similar" that retrieval itself uses.
# #     """
# #     llm_endpoint = HuggingFaceEndpoint(
# #         repo_id="openai/gpt-oss-120b:groq",
# #         huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# #         max_new_tokens=1000,
# #         temperature=0.1,
# #     )

# #     hf_llm = ChatHuggingFace(
# #         llm=llm_endpoint
# #     )

# #     llm = LangchainLLMWrapper(hf_llm)

# #     # -----------------------------
# #     # Hugging Face Embeddings
# #     # -----------------------------
# #     embeddings_model = HuggingFaceEmbeddings(
# #         model_name=EMBEDDING_MODEL_NAME
# #     )

# #     embeddings = LangchainEmbeddingsWrapper(
# #         embeddings_model
# #     )
# #     return llm, embeddings
# #     # from langchain_anthropic import ChatAnthropic
# #     # from langchain_huggingface import HuggingFaceEmbeddings
# #     # from ragas.llms import LangchainLLMWrapper
# #     # from ragas.embeddings import LangchainEmbeddingsWrapper

# #     # llm = LangchainLLMWrapper(ChatAnthropic(model=MODEL_NAME))
# #     # embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    


# # @traceable(name="evaluation.run_ragas", run_type="chain")
# # def run_ragas_evaluation(rows: list[RagasRow]):
# #     """
# #     Runs faithfulness, context_precision, context_recall, and
# #     answer_correctness over `rows` (answerable questions only — see this
# #     module's docstring). Returns ragas's EvaluationResult (has both an
# #     aggregate `.to_pandas()` table and per-metric scores).
# #     """
# #     from ragas import evaluate
# #     from ragas.metrics import (
# #         Faithfulness,
# #         LLMContextPrecisionWithReference,
# #         LLMContextRecall,
# #         AnswerCorrectness,
# #     )

# #     dataset = _build_ragas_evaluation_dataset(rows)
# #     llm, embeddings = _build_ragas_llm_and_embeddings()

# #     metrics = [
# #         Faithfulness(llm=llm),
# #         LLMContextPrecisionWithReference(llm=llm),
# #         LLMContextRecall(llm=llm),
# #         AnswerCorrectness(llm=llm, embeddings=embeddings),
# #     ]
# #     return evaluate(dataset=dataset, metrics=metrics)

# # """
# # RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# # Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

# #   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
# #      metrics — faithfulness, context_precision, context_recall, and
# #      answer_correctness — against CUAD's human-annotated answer spans as
# #      the `reference`.

# #   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
# #      clause of that category in the contract, so there's no real reference
# #      to recall against. These are excluded from the RAGAS dataset entirely
# #      and instead scored with a separate "correctly identified absence"
# #      metric: did the pipeline's own evidence auditor / answer correctly
# #      signal that nothing was found, rather than hallucinating a clause?
# #      Mixing the two would silently tank context_recall/answer_correctness
# #      on questions that were never answerable in the first place.

# # Requires (not in the base requirements.txt — see requirements.txt's
# # "Evaluation" section): `ragas`, `langchain-huggingface`, and ONE of
# # `langchain-anthropic` / `langchain-groq` depending on RAGAS_JUDGE_PROVIDER
# # (see _build_ragas_judge_llm below). RAGAS's LLM-judged metrics (faithfulness,
# # context precision/recall, answer_correctness) need a judge LLM + embeddings
# # model; the judge defaults to Claude but is swappable via env vars
# # (RAGAS_JUDGE_PROVIDER=anthropic|groq, RAGAS_JUDGE_MODEL=<model id>) —
# # embeddings always use the SAME model resources.py's HybridSearchAgent uses
# # (all-mpnet-base-v2) so "similar enough" judgments are consistent with what
# # retrieval itself considers similar, regardless of which provider judges.
# # """

# # from __future__ import annotations

# # from dataclasses import dataclass, field
# # from typing import Optional

# # from ..llm_client import MODEL_NAME
# # from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# # from ..tracing import traceable
# # from .cuad_loader import CUADExample
# # from .scripted_reviewer import ScriptedRunResult

# # # ---------------------------------------------------------------------------
# # # Absence-detection scoring (unanswerable / is_impossible questions)
# # # ---------------------------------------------------------------------------

# # _ABSENCE_PHRASES = (
# #     "no clause", "does not contain", "no such clause", "not found", "no provision",
# #     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# # )


# # def _looks_like_absence_answer(answer: Optional[str]) -> bool:
# #     if not answer:
# #         return False
# #     lowered = answer.lower()
# #     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# # @dataclass
# # class AbsenceDetectionResult:
# #     qas_id: str
# #     contract_title: str
# #     clause_category: str
# #     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
# #     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
# #     answer_stated_absence: bool            # final answer text reads as "no such clause"
# #     correctly_identified_absence: bool     # any of the above three


# # def score_absence_detection(
# #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # ) -> list[AbsenceDetectionResult]:
# #     """
# #     Scores the `is_impossible` subset. `correctly_identified_absence` is
# #     True if EITHER the evidence auditor flagged insufficient evidence, the
# #     evidence checkpoint rejected outright, or the synthesized answer itself
# #     reads as an absence statement — any of these means the pipeline did NOT
# #     fabricate a clause that CUAD's annotators confirmed doesn't exist.
# #     """
# #     scored = []
# #     for example, result in zip(examples, results):
# #         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
# #         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
# #         evidence_rejected = result.status == "evidence_rejected"
# #         answer_absence = _looks_like_absence_answer(result.final_answer)
# #         scored.append(
# #             AbsenceDetectionResult(
# #                 qas_id=example.qas_id,
# #                 contract_title=example.contract_title,
# #                 clause_category=example.clause_category,
# #                 evidence_marked_insufficient=evidence_insufficient,
# #                 evidence_rejected=evidence_rejected,
# #                 answer_stated_absence=answer_absence,
# #                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
# #             )
# #         )
# #     return scored


# # def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
# #     if not scored:
# #         return float("nan")
# #     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # # ---------------------------------------------------------------------------
# # # RAGAS dataset construction (answerable questions only)
# # # ---------------------------------------------------------------------------

# # @dataclass
# # class RagasRow:
# #     qas_id: str
# #     contract_title: str
# #     clause_category: str
# #     user_input: str
# #     retrieved_contexts: list[str]
# #     response: str
# #     reference: str
# #     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# # def build_ragas_rows(
# #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # ) -> list[RagasRow]:
# #     """
# #     Builds one row per answerable example. `retrieved_contexts` comes from
# #     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
# #     are forced onto the `hybrid` route — see scripted_reviewer.py's
# #     force_route default and cuad_ingest.py's module docstring for why.
# #     Rows where the pipeline never produced a final_answer (evidence or
# #     answer rejected) still get a row with response="" so RAGAS scores that
# #     as a real failure rather than silently dropping it from the average.
# #     """
# #     rows = []
# #     for example, result in zip(examples, results):
# #         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
# #         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
# #             hit.get("text", "") for hit in result.graph_hits
# #         ]
# #         rows.append(
# #             RagasRow(
# #                 qas_id=example.qas_id,
# #                 contract_title=example.contract_title,
# #                 clause_category=example.clause_category,
# #                 user_input=example.question,
# #                 retrieved_contexts=contexts,
# #                 response=result.final_answer or "",
# #                 reference=example.reference_answer,
# #                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
# #             )
# #         )
# #     return rows


# # def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
# #     """
# #     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
# #     ragas lazily so the rest of this module (and the non-RAGAS parts of the
# #     eval harness) works even if `ragas` isn't installed.
# #     """
# #     from ragas import EvaluationDataset, SingleTurnSample

# #     samples = [
# #         SingleTurnSample(
# #             user_input=r.user_input,
# #             retrieved_contexts=r.retrieved_contexts,
# #             response=r.response,
# #             reference=r.reference,
# #         )
# #         for r in rows
# #     ]
# #     return EvaluationDataset(samples=samples)
# # def _build_ragas_judge_llm():
# #     """
# #     Build the RAGAS judge LLM.

# #     Provider is auto-selected from environment variables so you can point the
# #     judge at whatever you have working, WITHOUT hardcoding keys in source:

# #         RAGAS_JUDGE_PROVIDER = gemini | openai | anthropic | hf   (optional;
# #                                otherwise auto-detected from the keys below)

# #         gemini    -> GOOGLE_API_KEY (or GEMINI_API_KEY)   model: RAGAS_JUDGE_MODEL or gemini-2.5-flash
# #         openai    -> OPENAI_API_KEY                        model: RAGAS_JUDGE_MODEL or gpt-4o-mini
# #         anthropic -> ANTHROPIC_API_KEY                     model: RAGAS_JUDGE_MODEL or claude-3-5-haiku-latest
# #         hf        -> HF_TOKEN (original path)              model: RAGAS_JUDGE_MODEL or openai/gpt-oss-120b:groq

# #     NOTE: the earlier all-NaN metrics were caused by the HF Inference router
# #     returning HTTP 402 (credits depleted) + timeouts for the judge. Switching
# #     to any funded provider here makes the four metrics compute normally.
# #     """
# #     import os

# #     from ragas.llms import LangchainLLMWrapper

# #     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "").strip().lower()
# #     model_override = os.getenv("RAGAS_JUDGE_MODEL")
# #     temperature = float(os.getenv("RAGAS_JUDGE_TEMPERATURE", "0"))

# #     def _has(*names: str) -> bool:
# #         return any(os.getenv(n) for n in names)

# #     # auto-detect provider if not explicitly set
# #     if not provider:
# #         if _has("GOOGLE_API_KEY", "GEMINI_API_KEY"):
# #             provider = "gemini"
# #         elif _has("OPENAI_API_KEY"):
# #             provider = "openai"
# #         elif _has("ANTHROPIC_API_KEY"):
# #             provider = "anthropic"
# #         elif _has("HF_TOKEN"):
# #             provider = "hf"

# #     # Pick the model for THIS provider. A global RAGAS_JUDGE_MODEL is only
# #     # applied to the HF path (that's where those ids like
# #     # "openai/gpt-oss-120b:groq" belong) — otherwise a stale HF id would be
# #     # sent to Gemini/OpenAI/Anthropic and 404. Each provider also has its own
# #     # override env var, and any override is ignored if it clearly belongs to a
# #     # different provider.
# #     def _model_for(prov: str, default: str, own_env: str) -> str:
# #         own = os.getenv(own_env)
# #         if own:
# #             return own
# #         if model_override and _looks_like(prov, model_override):
# #             return model_override
# #         return default

# #     if provider == "gemini":
# #         from langchain_google_genai import ChatGoogleGenerativeAI
# #         key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# #         model = _model_for("gemini", "gemini-3.6-flash", "RAGAS_GEMINI_MODEL")
# #         chat = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature)
# #         print(f"[ragas] judge provider=gemini model={model}")
# #         return LangchainLLMWrapper(chat)

# #     if provider == "openai":
# #         from langchain_openai import ChatOpenAI
# #         model = _model_for("openai", "gpt-4o-mini", "RAGAS_OPENAI_MODEL")
# #         chat = ChatOpenAI(model=model, temperature=temperature)
# #         print(f"[ragas] judge provider=openai model={model}")
# #         return LangchainLLMWrapper(chat)

# #     if provider == "anthropic":
# #         from langchain_anthropic import ChatAnthropic
# #         model = _model_for("anthropic", "claude-3-5-haiku-latest", "RAGAS_ANTHROPIC_MODEL")
# #         chat = ChatAnthropic(model=model, temperature=temperature)
# #         print(f"[ragas] judge provider=anthropic model={model}")
# #         return LangchainLLMWrapper(chat)

# #     # ---- default / fallback: Hugging Face Inference API (original path) ----
# #     return _build_ragas_judge_llm_hf()


# # def _looks_like(provider: str, model: str) -> bool:
# #     """True if `model` id plausibly belongs to `provider` (guards against a
# #     stale RAGAS_GEMINI_MODEL from one provider leaking into another)."""
# #     m = model.lower()
# #     if provider == "gemini":
# #         return m.startswith("gemini") or m.startswith("models/")
# #     if provider == "openai":
# #         return m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("chatgpt")
# #     if provider == "anthropic":
# #         return m.startswith("claude")
# #     return True  # hf: anything goes


# # def _build_ragas_judge_llm_hf():
# #     """
# #     Original Hugging Face Inference judge. Uses HF_TOKEN from the environment.

# #     Example:
# #         HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
# #         RAGAS_JUDGE_MODEL=openai/gpt-oss-120b:groq
# #     """
# #     import os

# #     from huggingface_hub import InferenceClient
# #     from langchain_core.language_models.chat_models import BaseChatModel
# #     from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# #     from langchain_core.outputs import ChatGeneration, ChatResult
# #     from pydantic import Field

# #     from ragas.llms import LangchainLLMWrapper

# #     hf_token = os.getenv("HF_TOKEN")

# #     if not hf_token:
# #         raise RuntimeError(
# #             "HF_TOKEN environment variable is not set. "
# #             "Add HF_TOKEN=hf_... to your .env file."
# #         )

# #     model = os.getenv(
# #         "RAGAS_JUDGE_MODEL",
# #         "openai/gpt-oss-120b:groq",
# #     )

# #     class HuggingFaceChatModel(BaseChatModel):
# #         client: object = Field(exclude=True)
# #         model: str

# #         @property
# #         def _llm_type(self) -> str:
# #             return "huggingface_inference"

# #         @property
# #         def _identifying_params(self) -> dict:
# #             return {
# #                 "model": self.model,
# #             }

# #         def _generate(
# #             self,
# #             messages,
# #             stop=None,
# #             run_manager=None,
# #             **kwargs,
# #         ) -> ChatResult:

# #             hf_messages = []

# #             for message in messages:

# #                 if isinstance(message, SystemMessage):
# #                     role = "system"

# #                 elif isinstance(message, HumanMessage):
# #                     role = "user"

# #                 elif isinstance(message, AIMessage):
# #                     role = "assistant"

# #                 else:
# #                     role = "user"

# #                 hf_messages.append(
# #                     {
# #                         "role": role,
# #                         "content": message.content,
# #                     }
# #                 )

# #             completion = self.client.chat.completions.create(
# #                 model=self.model,
# #                 messages=hf_messages,
# #                 max_tokens=kwargs.get("max_tokens", 2000),
# #                 temperature=kwargs.get("temperature", 0.0),
# #             )

# #             text = completion.choices[0].message.content or ""

# #             generation = ChatGeneration(
# #                 message=AIMessage(content=text)
# #             )

# #             return ChatResult(
# #                 generations=[generation]
# #             )

# #     client = InferenceClient(
# #         api_key=hf_token,
# #         provider="auto",
# #     )

# #     hf_llm = HuggingFaceChatModel(
# #         client=client,
# #         model=model,
# #     )

# #     return LangchainLLMWrapper(hf_llm)

# # # def _build_ragas_judge_llm():
# # #     """
# # #     RAGAS's LLM-judged metrics need a langchain-wrapped chat model to act as
# # #     judge. Configurable via env vars so you can point this at whatever the
# # #     rest of your eval stack already uses, independent of llm_client.py's
# # #     Claude model:

# # #         RAGAS_JUDGE_PROVIDER = "anthropic" (default) | "groq"
# # #         RAGAS_JUDGE_MODEL    = provider-specific model id
# # #                                 (default: llm_client.MODEL_NAME for anthropic,
# # #                                  "openai/gpt-oss-120b" for groq)

# # #     For provider="groq": if RAGAS_JUDGE_MODEL carries a trailing
# # #     ":<routing-hint>" suffix (e.g. "openai/gpt-oss-120b:groq" — an
# # #     OpenRouter-style provider suffix), it's stripped before being passed to
# # #     ChatGroq, since Groq's own API takes the bare model id
# # #     ("openai/gpt-oss-120b") and rejects the suffixed form. Requires
# # #     GROQ_API_KEY to be set.
# # #     """
# # #     import os

# # #     from ragas.llms import LangchainLLMWrapper

# # #     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "anthropic").lower()

# # #     if provider == "groq":
# # #         from langchain_groq import ChatGroq

# # #         model = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
# # #         model = model.split(":", 1)[0]  # strip any ":<routing-hint>" suffix — see docstring
# # #         return LangchainLLMWrapper(ChatGroq(model=model))

# # #     if provider == "anthropic":
# # #         from langchain_anthropic import ChatAnthropic

# # #         model = os.getenv("RAGAS_JUDGE_MODEL", MODEL_NAME)
# # #         return LangchainLLMWrapper(ChatAnthropic(model=model))

# # #     raise ValueError(f"Unknown RAGAS_JUDGE_PROVIDER={provider!r} — expected 'anthropic' or 'groq'.")


# # # def _build_ragas_llm_and_embeddings():
# # #     """
# # #     Judge LLM (see _build_ragas_judge_llm) + embeddings model. Embeddings
# # #     use the SAME model HybridSearchAgent uses (EMBEDDING_MODEL_NAME), so
# # #     RAGAS's notion of "similar" for answer_correctness matches what
# # #     retrieval itself uses, regardless of which provider judges the LLM
# # #     metrics.
# # #     """
# # #     from langchain_huggingface import HuggingFaceEmbeddings
# # #     from ragas.embeddings import LangchainEmbeddingsWrapper

# # #     llm = _build_ragas_judge_llm()
# # #     embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
# # #     return llm, embeddings
# # def _build_ragas_llm_and_embeddings():
# #     """
# #     Build the Hugging Face judge LLM + Hugging Face embeddings.
# #     """

# #     from langchain_huggingface import HuggingFaceEmbeddings
# #     from ragas.embeddings import LangchainEmbeddingsWrapper

# #     llm = _build_ragas_judge_llm()

# #     embeddings = LangchainEmbeddingsWrapper(
# #         HuggingFaceEmbeddings(
# #             model_name=EMBEDDING_MODEL_NAME
# #         )
# #     )

# #     return llm, embeddings

# # @traceable(name="evaluation.run_ragas", run_type="chain")
# # def run_ragas_evaluation(rows: list[RagasRow]):
# #     """
# #     Runs the RAGAS metric set over `rows` (answerable questions only — see this
# #     module's docstring). Returns ragas's EvaluationResult (has both an
# #     aggregate `.to_pandas()` table and per-metric scores).

# #     Metric set is now the lean pair the UI shows by default:
# #         - Faithfulness        — is every claim in the answer backed by context?
# #         - AnswerCorrectness   — semantic alignment vs the CUAD ground-truth answer

# #     Set RAGAS_FULL_METRICS=1 to also compute LLMContextPrecisionWithReference
# #     and LLMContextRecall (the two extra retrieval metrics).
# #     """
# #     import os

# #     from ragas import evaluate
# #     from ragas.metrics import (
# #         Faithfulness,
# #         LLMContextPrecisionWithReference,
# #         LLMContextRecall,
# #         AnswerCorrectness,
# #     )

# #     # [FIX] Make evaluate() robust to RAGAS's empty-trace IndexError (scores are
# #     # computed fine; only the debug-only .traces parsing crashes). See ragas_compat.
# #     from .ragas_compat import install_ragas_trace_guard
# #     install_ragas_trace_guard()

# #     dataset = _build_ragas_evaluation_dataset(rows)
# #     llm, embeddings = _build_ragas_llm_and_embeddings()

# #     # Lean default: the two metrics the frontend renders.
# #     metrics = [
# #         Faithfulness(llm=llm),
# #         AnswerCorrectness(llm=llm, embeddings=embeddings),
# #     ]
# #     if os.getenv("RAGAS_FULL_METRICS", "").strip() in ("1", "true", "yes"):
# #         metrics += [
# #             LLMContextPrecisionWithReference(llm=llm),
# #             LLMContextRecall(llm=llm),
# #         ]

# #     evaluation = evaluate(dataset=dataset, metrics=metrics)

# #     # ---- optional debug dump (set RAGAS_DEBUG=1) ----
# #     if os.getenv("RAGAS_DEBUG", "").strip() in ("1", "true", "yes"):
# #         print("\n================ RAGAS DEBUG ================")
# #         print("[DEBUG RAGAS TYPE]:", type(evaluation))
# #         if hasattr(evaluation, "_repr_dict"):
# #             print("[DEBUG _repr_dict]:", dict(evaluation._repr_dict))
# #         try:
# #             df = evaluation.to_pandas()
# #             print("[DEBUG DataFrame Columns]:", list(df.columns))
# #             print("\n[DEBUG DataFrame]:\n", df.to_string())
# #             if len(df):
# #                 print("\n[DEBUG First Row]:", df.iloc[0].to_dict())
# #             df.to_csv("ragas_evaluation_results.csv", index=False)
# #             print("[DEBUG] wrote ragas_evaluation_results.csv")
# #         except Exception as exc:  # noqa: BLE001
# #             print("[DEBUG] to_pandas() failed:", exc)
# #         print("==============================================\n")

# #     return evaluation


# # # """
# # # RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# # # Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

# # #   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
# # #      metrics — faithfulness, context_precision, context_recall, and
# # #      answer_correctness — against CUAD's human-annotated answer spans as
# # #      the `reference`.

# # #   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
# # #      clause of that category in the contract, so there's no real reference
# # #      to recall against. These are excluded from the RAGAS dataset entirely
# # #      and instead scored with a separate "correctly identified absence"
# # #      metric: did the pipeline's own evidence auditor / answer correctly
# # #      signal that nothing was found, rather than hallucinating a clause?
# # #      Mixing the two would silently tank context_recall/answer_correctness
# # #      on questions that were never answerable in the first place.

# # # Requires (not in the base requirements.txt — see requirements.txt's
# # # "Evaluation" section): `ragas`, `langchain-HF`, `langchain-huggingface`.
# # # RAGAS's LLM-judged metrics (faithfulness, context precision/recall,
# # # answer_correctness) need an LLM + embeddings model; this module reuses
# # # Claude (via langchain_HF, matching llm_client.py's MODEL_NAME) as
# # # the judge, and the SAME embedding model resources.py's HybridSearchAgent
# # # uses (all-mpnet-base-v2) so "similar enough" judgments are consistent with
# # # what retrieval itself considers similar.
# # # """

# # # from __future__ import annotations

# # # from dataclasses import dataclass, field
# # # from typing import Optional

# # # from ..llm_client import MODEL_NAME
# # # from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# # # from ..tracing import traceable
# # # from .cuad_loader import CUADExample
# # # from .scripted_reviewer import ScriptedRunResult

# # # import os

# # # from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# # # from langchain_huggingface import HuggingFaceEmbeddings

# # # from ragas.llms import LangchainLLMWrapper
# # # from ragas.embeddings import LangchainEmbeddingsWrapper
# # # ---------------------------------------------------------------------------
# # # # Absence-detection scoring (unanswerable / is_impossible questions)
# # # # ---------------------------------------------------------------------------
# # # MODEL_NAME = "openai/gpt-oss-120b:groq"
# # # _ABSENCE_PHRASES = (
# # #     "no clause", "does not contain", "no such clause", "not found", "no provision",
# # #     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# # # )


# # # def _looks_like_absence_answer(answer: Optional[str]) -> bool:
# # #     if not answer:
# # #         return False
# # #     lowered = answer.lower()
# # #     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# # # @dataclass
# # # class AbsenceDetectionResult:
# # #     qas_id: str
# # #     contract_title: str
# # #     clause_category: str
# # #     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
# # #     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
# # #     answer_stated_absence: bool            # final answer text reads as "no such clause"
# # #     correctly_identified_absence: bool     # any of the above three


# # # def score_absence_detection(
# # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # ) -> list[AbsenceDetectionResult]:
# # #     """
# # #     Scores the `is_impossible` subset. `correctly_identified_absence` is
# # #     True if EITHER the evidence auditor flagged insufficient evidence, the
# # #     evidence checkpoint rejected outright, or the synthesized answer itself
# # #     reads as an absence statement — any of these means the pipeline did NOT
# # #     fabricate a clause that CUAD's annotators confirmed doesn't exist.
# # #     """
# # #     scored = []
# # #     for example, result in zip(examples, results):
# # #         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
# # #         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
# # #         evidence_rejected = result.status == "evidence_rejected"
# # #         answer_absence = _looks_like_absence_answer(result.final_answer)
# # #         scored.append(
# # #             AbsenceDetectionResult(
# # #                 qas_id=example.qas_id,
# # #                 contract_title=example.contract_title,
# # #                 clause_category=example.clause_category,
# # #                 evidence_marked_insufficient=evidence_insufficient,
# # #                 evidence_rejected=evidence_rejected,
# # #                 answer_stated_absence=answer_absence,
# # #                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
# # #             )
# # #         )
# # #     return scored


# # # def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
# # #     if not scored:
# # #         return float("nan")
# # #     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # # # ---------------------------------------------------------------------------
# # # # RAGAS dataset construction (answerable questions only)
# # # # ---------------------------------------------------------------------------

# # # @dataclass
# # # class RagasRow:
# # #     qas_id: str
# # #     contract_title: str
# # #     clause_category: str
# # #     user_input: str
# # #     retrieved_contexts: list[str]
# # #     response: str
# # #     reference: str
# # #     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# # # def build_ragas_rows(
# # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # ) -> list[RagasRow]:
# # #     """
# # #     Builds one row per answerable example. `retrieved_contexts` comes from
# # #     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
# # #     are forced onto the `hybrid` route — see scripted_reviewer.py's
# # #     force_route default and cuad_ingest.py's module docstring for why.
# # #     Rows where the pipeline never produced a final_answer (evidence or
# # #     answer rejected) still get a row with response="" so RAGAS scores that
# # #     as a real failure rather than silently dropping it from the average.
# # #     """
# # #     rows = []
# # #     for example, result in zip(examples, results):
# # #         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
# # #         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
# # #             hit.get("text", "") for hit in result.graph_hits
# # #         ]
# # #         rows.append(
# # #             RagasRow(
# # #                 qas_id=example.qas_id,
# # #                 contract_title=example.contract_title,
# # #                 clause_category=example.clause_category,
# # #                 user_input=example.question,
# # #                 retrieved_contexts=contexts,
# # #                 response=result.final_answer or "",
# # #                 reference=example.reference_answer,
# # #                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
# # #             )
# # #         )
# # #     return rows


# # # def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
# # #     """
# # #     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
# # #     ragas lazily so the rest of this module (and the non-RAGAS parts of the
# # #     eval harness) works even if `ragas` isn't installed.
# # #     """
# # #     from ragas import EvaluationDataset, SingleTurnSample

# # #     samples = [
# # #         SingleTurnSample(
# # #             user_input=r.user_input,
# # #             retrieved_contexts=r.retrieved_contexts,
# # #             response=r.response,
# # #             reference=r.reference,
# # #         )
# # #         for r in rows
# # #     ]
# # #     return EvaluationDataset(samples=samples)


# # # def _build_ragas_llm_and_embeddings():
# # #     """
# # #     RAGAS's LLM-judged metrics need a langchain-wrapped LLM + embeddings
# # #     model. Uses the same Claude model as the rest of the pipeline
# # #     (llm_client.MODEL_NAME) and the same embedding model HybridSearchAgent
# # #     uses (EMBEDDING_MODEL_NAME), so judgments are made with the same notion
# # #     of "similar" that retrieval itself uses.
# # #     """
# # #     llm_endpoint = HuggingFaceEndpoint(
# # #         repo_id="openai/gpt-oss-120b:groq",
# # #         huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# # #         max_new_tokens=1000,
# # #         temperature=0.1,
# # #     )

# # #     hf_llm = ChatHuggingFace(
# # #         llm=llm_endpoint
# # #     )

# # #     llm = LangchainLLMWrapper(hf_llm)

# # #     # -----------------------------
# # #     # Hugging Face Embeddings
# # #     # -----------------------------
# # #     embeddings_model = HuggingFaceEmbeddings(
# # #         model_name=EMBEDDING_MODEL_NAME
# # #     )

# # #     embeddings = LangchainEmbeddingsWrapper(
# # #         embeddings_model
# # #     )
# # #     return llm, embeddings
# # #     # from langchain_anthropic import ChatAnthropic
# # #     # from langchain_huggingface import HuggingFaceEmbeddings
# # #     # from ragas.llms import LangchainLLMWrapper
# # #     # from ragas.embeddings import LangchainEmbeddingsWrapper

# # #     # llm = LangchainLLMWrapper(ChatAnthropic(model=MODEL_NAME))
# # #     # embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    


# # # @traceable(name="evaluation.run_ragas", run_type="chain")
# # # def run_ragas_evaluation(rows: list[RagasRow]):
# # #     """
# # #     Runs faithfulness, context_precision, context_recall, and
# # #     answer_correctness over `rows` (answerable questions only — see this
# # #     module's docstring). Returns ragas's EvaluationResult (has both an
# # #     aggregate `.to_pandas()` table and per-metric scores).
# # #     """
# # #     from ragas import evaluate
# # #     from ragas.metrics import (
# # #         Faithfulness,
# # #         LLMContextPrecisionWithReference,
# # #         LLMContextRecall,
# # #         AnswerCorrectness,
# # #     )

# # #     dataset = _build_ragas_evaluation_dataset(rows)
# # #     llm, embeddings = _build_ragas_llm_and_embeddings()

# # #     metrics = [
# # #         Faithfulness(llm=llm),
# # #         LLMContextPrecisionWithReference(llm=llm),
# # #         LLMContextRecall(llm=llm),
# # #         AnswerCorrectness(llm=llm, embeddings=embeddings),
# # #     ]
# # #     return evaluate(dataset=dataset, metrics=metrics)

# # # """
# # # RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# # # Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

# # #   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
# # #      metrics — faithfulness, context_precision, context_recall, and
# # #      answer_correctness — against CUAD's human-annotated answer spans as
# # #      the `reference`.

# # #   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
# # #      clause of that category in the contract, so there's no real reference
# # #      to recall against. These are excluded from the RAGAS dataset entirely
# # #      and instead scored with a separate "correctly identified absence"
# # #      metric: did the pipeline's own evidence auditor / answer correctly
# # #      signal that nothing was found, rather than hallucinating a clause?
# # #      Mixing the two would silently tank context_recall/answer_correctness
# # #      on questions that were never answerable in the first place.

# # # Requires (not in the base requirements.txt — see requirements.txt's
# # # "Evaluation" section): `ragas`, `langchain-huggingface`, and ONE of
# # # `langchain-anthropic` / `langchain-groq` depending on RAGAS_JUDGE_PROVIDER
# # # (see _build_ragas_judge_llm below). RAGAS's LLM-judged metrics (faithfulness,
# # # context precision/recall, answer_correctness) need a judge LLM + embeddings
# # # model; the judge defaults to Claude but is swappable via env vars
# # # (RAGAS_JUDGE_PROVIDER=anthropic|groq, RAGAS_JUDGE_MODEL=<model id>) —
# # # embeddings always use the SAME model resources.py's HybridSearchAgent uses
# # # (all-mpnet-base-v2) so "similar enough" judgments are consistent with what
# # # retrieval itself considers similar, regardless of which provider judges.
# # # """

# # # from __future__ import annotations

# # # from dataclasses import dataclass, field
# # # from typing import Optional

# # # from ..llm_client import MODEL_NAME
# # # from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# # # from ..tracing import traceable
# # # from .cuad_loader import CUADExample
# # # from .scripted_reviewer import ScriptedRunResult

# # # # ---------------------------------------------------------------------------
# # # # Absence-detection scoring (unanswerable / is_impossible questions)
# # # # ---------------------------------------------------------------------------

# # # _ABSENCE_PHRASES = (
# # #     "no clause", "does not contain", "no such clause", "not found", "no provision",
# # #     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# # # )


# # # def _looks_like_absence_answer(answer: Optional[str]) -> bool:
# # #     if not answer:
# # #         return False
# # #     lowered = answer.lower()
# # #     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# # # @dataclass
# # # class AbsenceDetectionResult:
# # #     qas_id: str
# # #     contract_title: str
# # #     clause_category: str
# # #     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
# # #     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
# # #     answer_stated_absence: bool            # final answer text reads as "no such clause"
# # #     correctly_identified_absence: bool     # any of the above three


# # # def score_absence_detection(
# # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # ) -> list[AbsenceDetectionResult]:
# # #     """
# # #     Scores the `is_impossible` subset. `correctly_identified_absence` is
# # #     True if EITHER the evidence auditor flagged insufficient evidence, the
# # #     evidence checkpoint rejected outright, or the synthesized answer itself
# # #     reads as an absence statement — any of these means the pipeline did NOT
# # #     fabricate a clause that CUAD's annotators confirmed doesn't exist.
# # #     """
# # #     scored = []
# # #     for example, result in zip(examples, results):
# # #         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
# # #         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
# # #         evidence_rejected = result.status == "evidence_rejected"
# # #         answer_absence = _looks_like_absence_answer(result.final_answer)
# # #         scored.append(
# # #             AbsenceDetectionResult(
# # #                 qas_id=example.qas_id,
# # #                 contract_title=example.contract_title,
# # #                 clause_category=example.clause_category,
# # #                 evidence_marked_insufficient=evidence_insufficient,
# # #                 evidence_rejected=evidence_rejected,
# # #                 answer_stated_absence=answer_absence,
# # #                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
# # #             )
# # #         )
# # #     return scored


# # # def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
# # #     if not scored:
# # #         return float("nan")
# # #     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # # # ---------------------------------------------------------------------------
# # # # RAGAS dataset construction (answerable questions only)
# # # # ---------------------------------------------------------------------------

# # # @dataclass
# # # class RagasRow:
# # #     qas_id: str
# # #     contract_title: str
# # #     clause_category: str
# # #     user_input: str
# # #     retrieved_contexts: list[str]
# # #     response: str
# # #     reference: str
# # #     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# # # def build_ragas_rows(
# # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # ) -> list[RagasRow]:
# # #     """
# # #     Builds one row per answerable example. `retrieved_contexts` comes from
# # #     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
# # #     are forced onto the `hybrid` route — see scripted_reviewer.py's
# # #     force_route default and cuad_ingest.py's module docstring for why.
# # #     Rows where the pipeline never produced a final_answer (evidence or
# # #     answer rejected) still get a row with response="" so RAGAS scores that
# # #     as a real failure rather than silently dropping it from the average.
# # #     """
# # #     rows = []
# # #     for example, result in zip(examples, results):
# # #         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
# # #         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
# # #             hit.get("text", "") for hit in result.graph_hits
# # #         ]
# # #         rows.append(
# # #             RagasRow(
# # #                 qas_id=example.qas_id,
# # #                 contract_title=example.contract_title,
# # #                 clause_category=example.clause_category,
# # #                 user_input=example.question,
# # #                 retrieved_contexts=contexts,
# # #                 response=result.final_answer or "",
# # #                 reference=example.reference_answer,
# # #                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
# # #             )
# # #         )
# # #     return rows


# # # def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
# # #     """
# # #     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
# # #     ragas lazily so the rest of this module (and the non-RAGAS parts of the
# # #     eval harness) works even if `ragas` isn't installed.
# # #     """
# # #     from ragas import EvaluationDataset, SingleTurnSample

# # #     samples = [
# # #         SingleTurnSample(
# # #             user_input=r.user_input,
# # #             retrieved_contexts=r.retrieved_contexts,
# # #             response=r.response,
# # #             reference=r.reference,
# # #         )
# # #         for r in rows
# # #     ]
# # #     return EvaluationDataset(samples=samples)
# # # def _build_ragas_judge_llm():
# # #     """
# # #     Build the RAGAS judge LLM.

# # #     Provider is auto-selected from environment variables so you can point the
# # #     judge at whatever you have working, WITHOUT hardcoding keys in source:

# # #         RAGAS_JUDGE_PROVIDER = gemini | openai | anthropic | hf   (optional;
# # #                                otherwise auto-detected from the keys below)

# # #         gemini    -> GOOGLE_API_KEY (or GEMINI_API_KEY)   model: RAGAS_JUDGE_MODEL or gemini-3.6-flash
# # #         openai    -> OPENAI_API_KEY                        model: RAGAS_JUDGE_MODEL or gpt-4o-mini
# # #         anthropic -> ANTHROPIC_API_KEY                     model: RAGAS_JUDGE_MODEL or claude-3-5-haiku-latest
# # #         hf        -> HF_TOKEN (original path)              model: RAGAS_JUDGE_MODEL or openai/gpt-oss-120b:groq

# # #     NOTE: the earlier all-NaN metrics were caused by the HF Inference router
# # #     returning HTTP 402 (credits depleted) + timeouts for the judge. Switching
# # #     to any funded provider here makes the four metrics compute normally.
# # #     """
# # #     import os

# # #     from ragas.llms import LangchainLLMWrapper

# # #     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "").strip().lower()
# # #     model_override = os.getenv("RAGAS_JUDGE_MODEL")
# # #     temperature = float(os.getenv("RAGAS_JUDGE_TEMPERATURE", "0"))

# # #     def _has(*names: str) -> bool:
# # #         return any(os.getenv(n) for n in names)

# # #     # auto-detect provider if not explicitly set
# # #     if not provider:
# # #         if _has("GOOGLE_API_KEY", "GEMINI_API_KEY"):
# # #             provider = "gemini"
# # #         elif _has("OPENAI_API_KEY"):
# # #             provider = "openai"
# # #         elif _has("ANTHROPIC_API_KEY"):
# # #             provider = "anthropic"
# # #         elif _has("new_HF_TOKEN"):
# # #             provider = "hf"

# # #     if provider == "gemini":
# # #         from langchain_google_genai import ChatGoogleGenerativeAI
# # #         key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# # #         chat = ChatGoogleGenerativeAI(
# # #             model=model_override or "gemini-3.6-flash",
# # #             google_api_key=key,
# # #             temperature=temperature,
# # #         )
# # #         print(f"[ragas] judge provider=gemini model={chat.model}")
# # #         return LangchainLLMWrapper(chat)

# # #     if provider == "openai":
# # #         from langchain_openai import ChatOpenAI
# # #         chat = ChatOpenAI(
# # #             model=model_override or "gpt-4o-mini",
# # #             temperature=temperature,
# # #         )
# # #         print(f"[ragas] judge provider=openai model={chat.model_name}")
# # #         return LangchainLLMWrapper(chat)

# # #     if provider == "anthropic":
# # #         from langchain_anthropic import ChatAnthropic
# # #         chat = ChatAnthropic(
# # #             model=model_override or "claude-3-5-haiku-latest",
# # #             temperature=temperature,
# # #         )
# # #         print(f"[ragas] judge provider=anthropic model={chat.model}")
# # #         return LangchainLLMWrapper(chat)

# # #     # ---- default / fallback: Hugging Face Inference API (original path) ----
# # #     return _build_ragas_judge_llm_hf()


# # # def _build_ragas_judge_llm_hf():
# # #     """
# # #     Original Hugging Face Inference judge. Uses new_HF_TOKEN from the environment.

# # #     Example:
        
# # #         RAGAS_JUDGE_MODEL=openai/gpt-oss-120b:groq
# # #     """
# # #     import os

# # #     from huggingface_hub import InferenceClient
# # #     from langchain_core.language_models.chat_models import BaseChatModel
# # #     from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# # #     from langchain_core.outputs import ChatGeneration, ChatResult
# # #     from pydantic import Field

# # #     from ragas.llms import LangchainLLMWrapper

# # #     hf_token = os.getenv("new_HF_TOKEN")

# # #     if not hf_token:
# # #         raise RuntimeError(
# # #             "new_HF_TOKEN environment variable is not set. "
# # #             "Add HF_TOKEN=hf_... to your .env file."
# # #         )

# # #     model = os.getenv(
# # #         "RAGAS_JUDGE_MODEL",
# # #         "openai/gpt-oss-120b:groq",
# # #     )

# # #     class HuggingFaceChatModel(BaseChatModel):
# # #         client: object = Field(exclude=True)
# # #         model: str

# # #         @property
# # #         def _llm_type(self) -> str:
# # #             return "huggingface_inference"

# # #         @property
# # #         def _identifying_params(self) -> dict:
# # #             return {
# # #                 "model": self.model,
# # #             }

# # #         def _generate(
# # #             self,
# # #             messages,
# # #             stop=None,
# # #             run_manager=None,
# # #             **kwargs,
# # #         ) -> ChatResult:

# # #             hf_messages = []

# # #             for message in messages:

# # #                 if isinstance(message, SystemMessage):
# # #                     role = "system"

# # #                 elif isinstance(message, HumanMessage):
# # #                     role = "user"

# # #                 elif isinstance(message, AIMessage):
# # #                     role = "assistant"

# # #                 else:
# # #                     role = "user"

# # #                 hf_messages.append(
# # #                     {
# # #                         "role": role,
# # #                         "content": message.content,
# # #                     }
# # #                 )

# # #             completion = self.client.chat.completions.create(
# # #                 model=self.model,
# # #                 messages=hf_messages,
# # #                 max_tokens=kwargs.get("max_tokens", 2000),
# # #                 temperature=kwargs.get("temperature", 0.0),
# # #             )

# # #             text = completion.choices[0].message.content or ""

# # #             generation = ChatGeneration(
# # #                 message=AIMessage(content=text)
# # #             )

# # #             return ChatResult(
# # #                 generations=[generation]
# # #             )

# # #     client = InferenceClient(
# # #         api_key=hf_token,
# # #         provider="auto",
# # #     )

# # #     hf_llm = HuggingFaceChatModel(
# # #         client=client,
# # #         model=model,
# # #     )

# # #     return LangchainLLMWrapper(hf_llm)

# # # # def _build_ragas_judge_llm():
# # # #     """
# # # #     RAGAS's LLM-judged metrics need a langchain-wrapped chat model to act as
# # # #     judge. Configurable via env vars so you can point this at whatever the
# # # #     rest of your eval stack already uses, independent of llm_client.py's
# # # #     Claude model:

# # # #         RAGAS_JUDGE_PROVIDER = "anthropic" (default) | "groq"
# # # #         RAGAS_JUDGE_MODEL    = provider-specific model id
# # # #                                 (default: llm_client.MODEL_NAME for anthropic,
# # # #                                  "openai/gpt-oss-120b" for groq)

# # # #     For provider="groq": if RAGAS_JUDGE_MODEL carries a trailing
# # # #     ":<routing-hint>" suffix (e.g. "openai/gpt-oss-120b:groq" — an
# # # #     OpenRouter-style provider suffix), it's stripped before being passed to
# # # #     ChatGroq, since Groq's own API takes the bare model id
# # # #     ("openai/gpt-oss-120b") and rejects the suffixed form. Requires
# # # #     GROQ_API_KEY to be set.
# # # #     """
# # # #     import os

# # # #     from ragas.llms import LangchainLLMWrapper

# # # #     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "anthropic").lower()

# # # #     if provider == "groq":
# # # #         from langchain_groq import ChatGroq

# # # #         model = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
# # # #         model = model.split(":", 1)[0]  # strip any ":<routing-hint>" suffix — see docstring
# # # #         return LangchainLLMWrapper(ChatGroq(model=model))

# # # #     if provider == "anthropic":
# # # #         from langchain_anthropic import ChatAnthropic

# # # #         model = os.getenv("RAGAS_JUDGE_MODEL", MODEL_NAME)
# # # #         return LangchainLLMWrapper(ChatAnthropic(model=model))

# # # #     raise ValueError(f"Unknown RAGAS_JUDGE_PROVIDER={provider!r} — expected 'anthropic' or 'groq'.")


# # # # def _build_ragas_llm_and_embeddings():
# # # #     """
# # # #     Judge LLM (see _build_ragas_judge_llm) + embeddings model. Embeddings
# # # #     use the SAME model HybridSearchAgent uses (EMBEDDING_MODEL_NAME), so
# # # #     RAGAS's notion of "similar" for answer_correctness matches what
# # # #     retrieval itself uses, regardless of which provider judges the LLM
# # # #     metrics.
# # # #     """
# # # #     from langchain_huggingface import HuggingFaceEmbeddings
# # # #     from ragas.embeddings import LangchainEmbeddingsWrapper

# # # #     llm = _build_ragas_judge_llm()
# # # #     embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
# # # #     return llm, embeddings
# # # def _build_ragas_llm_and_embeddings():
# # #     """
# # #     Build the Hugging Face judge LLM + Hugging Face embeddings.
# # #     """

# # #     from langchain_huggingface import HuggingFaceEmbeddings
# # #     from ragas.embeddings import LangchainEmbeddingsWrapper

# # #     llm = _build_ragas_judge_llm()

# # #     embeddings = LangchainEmbeddingsWrapper(
# # #         HuggingFaceEmbeddings(
# # #             model_name=EMBEDDING_MODEL_NAME
# # #         )
# # #     )

# # #     return llm, embeddings

# # # @traceable(name="evaluation.run_ragas", run_type="chain")
# # # def run_ragas_evaluation(rows: list[RagasRow]):
# # #     """
# # #     Runs faithfulness, context_precision, context_recall, and
# # #     answer_correctness over `rows` (answerable questions only — see this
# # #     module's docstring). Returns ragas's EvaluationResult (has both an
# # #     aggregate `.to_pandas()` table and per-metric scores).
# # #     """
# # #     from ragas import evaluate
# # #     from ragas.metrics import (
# # #         Faithfulness,
# # #         LLMContextPrecisionWithReference,
# # #         LLMContextRecall,
# # #         AnswerCorrectness,
# # #     )

# # #     # [FIX] Make evaluate() robust to RAGAS's empty-trace IndexError (scores are
# # #     # computed fine; only the debug-only .traces parsing crashes). See ragas_compat.
# # #     from .ragas_compat import install_ragas_trace_guard
# # #     install_ragas_trace_guard()

# # #     dataset = _build_ragas_evaluation_dataset(rows)
# # #     llm, embeddings = _build_ragas_llm_and_embeddings()

# # #     metrics = [
# # #         Faithfulness(llm=llm),
# # #         LLMContextPrecisionWithReference(llm=llm),
# # #         LLMContextRecall(llm=llm),
# # #         AnswerCorrectness(llm=llm, embeddings=embeddings),
# # #     ]
# # #     return evaluate(dataset=dataset, metrics=metrics)


# # # # """
# # # # RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# # # # Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

# # # #   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
# # # #      metrics — faithfulness, context_precision, context_recall, and
# # # #      answer_correctness — against CUAD's human-annotated answer spans as
# # # #      the `reference`.

# # # #   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
# # # #      clause of that category in the contract, so there's no real reference
# # # #      to recall against. These are excluded from the RAGAS dataset entirely
# # # #      and instead scored with a separate "correctly identified absence"
# # # #      metric: did the pipeline's own evidence auditor / answer correctly
# # # #      signal that nothing was found, rather than hallucinating a clause?
# # # #      Mixing the two would silently tank context_recall/answer_correctness
# # # #      on questions that were never answerable in the first place.

# # # # Requires (not in the base requirements.txt — see requirements.txt's
# # # # "Evaluation" section): `ragas`, `langchain-HF`, `langchain-huggingface`.
# # # # RAGAS's LLM-judged metrics (faithfulness, context precision/recall,
# # # # answer_correctness) need an LLM + embeddings model; this module reuses
# # # # Claude (via langchain_HF, matching llm_client.py's MODEL_NAME) as
# # # # the judge, and the SAME embedding model resources.py's HybridSearchAgent
# # # # uses (all-mpnet-base-v2) so "similar enough" judgments are consistent with
# # # # what retrieval itself considers similar.
# # # # """

# # # # from __future__ import annotations

# # # # from dataclasses import dataclass, field
# # # # from typing import Optional

# # # # from ..llm_client import MODEL_NAME
# # # # from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# # # # from ..tracing import traceable
# # # # from .cuad_loader import CUADExample
# # # # from .scripted_reviewer import ScriptedRunResult

# # # # import os

# # # # from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# # # # from langchain_huggingface import HuggingFaceEmbeddings

# # # # from ragas.llms import LangchainLLMWrapper
# # # # from ragas.embeddings import LangchainEmbeddingsWrapper
# # # # ---------------------------------------------------------------------------
# # # # # Absence-detection scoring (unanswerable / is_impossible questions)
# # # # # ---------------------------------------------------------------------------
# # # # MODEL_NAME = "openai/gpt-oss-120b:groq"
# # # # _ABSENCE_PHRASES = (
# # # #     "no clause", "does not contain", "no such clause", "not found", "no provision",
# # # #     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# # # # )


# # # # def _looks_like_absence_answer(answer: Optional[str]) -> bool:
# # # #     if not answer:
# # # #         return False
# # # #     lowered = answer.lower()
# # # #     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# # # # @dataclass
# # # # class AbsenceDetectionResult:
# # # #     qas_id: str
# # # #     contract_title: str
# # # #     clause_category: str
# # # #     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
# # # #     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
# # # #     answer_stated_absence: bool            # final answer text reads as "no such clause"
# # # #     correctly_identified_absence: bool     # any of the above three


# # # # def score_absence_detection(
# # # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # # ) -> list[AbsenceDetectionResult]:
# # # #     """
# # # #     Scores the `is_impossible` subset. `correctly_identified_absence` is
# # # #     True if EITHER the evidence auditor flagged insufficient evidence, the
# # # #     evidence checkpoint rejected outright, or the synthesized answer itself
# # # #     reads as an absence statement — any of these means the pipeline did NOT
# # # #     fabricate a clause that CUAD's annotators confirmed doesn't exist.
# # # #     """
# # # #     scored = []
# # # #     for example, result in zip(examples, results):
# # # #         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
# # # #         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
# # # #         evidence_rejected = result.status == "evidence_rejected"
# # # #         answer_absence = _looks_like_absence_answer(result.final_answer)
# # # #         scored.append(
# # # #             AbsenceDetectionResult(
# # # #                 qas_id=example.qas_id,
# # # #                 contract_title=example.contract_title,
# # # #                 clause_category=example.clause_category,
# # # #                 evidence_marked_insufficient=evidence_insufficient,
# # # #                 evidence_rejected=evidence_rejected,
# # # #                 answer_stated_absence=answer_absence,
# # # #                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
# # # #             )
# # # #         )
# # # #     return scored


# # # # def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
# # # #     if not scored:
# # # #         return float("nan")
# # # #     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # # # # ---------------------------------------------------------------------------
# # # # # RAGAS dataset construction (answerable questions only)
# # # # # ---------------------------------------------------------------------------

# # # # @dataclass
# # # # class RagasRow:
# # # #     qas_id: str
# # # #     contract_title: str
# # # #     clause_category: str
# # # #     user_input: str
# # # #     retrieved_contexts: list[str]
# # # #     response: str
# # # #     reference: str
# # # #     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# # # # def build_ragas_rows(
# # # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # # ) -> list[RagasRow]:
# # # #     """
# # # #     Builds one row per answerable example. `retrieved_contexts` comes from
# # # #     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
# # # #     are forced onto the `hybrid` route — see scripted_reviewer.py's
# # # #     force_route default and cuad_ingest.py's module docstring for why.
# # # #     Rows where the pipeline never produced a final_answer (evidence or
# # # #     answer rejected) still get a row with response="" so RAGAS scores that
# # # #     as a real failure rather than silently dropping it from the average.
# # # #     """
# # # #     rows = []
# # # #     for example, result in zip(examples, results):
# # # #         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
# # # #         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
# # # #             hit.get("text", "") for hit in result.graph_hits
# # # #         ]
# # # #         rows.append(
# # # #             RagasRow(
# # # #                 qas_id=example.qas_id,
# # # #                 contract_title=example.contract_title,
# # # #                 clause_category=example.clause_category,
# # # #                 user_input=example.question,
# # # #                 retrieved_contexts=contexts,
# # # #                 response=result.final_answer or "",
# # # #                 reference=example.reference_answer,
# # # #                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
# # # #             )
# # # #         )
# # # #     return rows


# # # # def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
# # # #     """
# # # #     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
# # # #     ragas lazily so the rest of this module (and the non-RAGAS parts of the
# # # #     eval harness) works even if `ragas` isn't installed.
# # # #     """
# # # #     from ragas import EvaluationDataset, SingleTurnSample

# # # #     samples = [
# # # #         SingleTurnSample(
# # # #             user_input=r.user_input,
# # # #             retrieved_contexts=r.retrieved_contexts,
# # # #             response=r.response,
# # # #             reference=r.reference,
# # # #         )
# # # #         for r in rows
# # # #     ]
# # # #     return EvaluationDataset(samples=samples)


# # # # def _build_ragas_llm_and_embeddings():
# # # #     """
# # # #     RAGAS's LLM-judged metrics need a langchain-wrapped LLM + embeddings
# # # #     model. Uses the same Claude model as the rest of the pipeline
# # # #     (llm_client.MODEL_NAME) and the same embedding model HybridSearchAgent
# # # #     uses (EMBEDDING_MODEL_NAME), so judgments are made with the same notion
# # # #     of "similar" that retrieval itself uses.
# # # #     """
# # # #     llm_endpoint = HuggingFaceEndpoint(
# # # #         repo_id="openai/gpt-oss-120b:groq",
# # # #         huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# # # #         max_new_tokens=1000,
# # # #         temperature=0.1,
# # # #     )

# # # #     hf_llm = ChatHuggingFace(
# # # #         llm=llm_endpoint
# # # #     )

# # # #     llm = LangchainLLMWrapper(hf_llm)

# # # #     # -----------------------------
# # # #     # Hugging Face Embeddings
# # # #     # -----------------------------
# # # #     embeddings_model = HuggingFaceEmbeddings(
# # # #         model_name=EMBEDDING_MODEL_NAME
# # # #     )

# # # #     embeddings = LangchainEmbeddingsWrapper(
# # # #         embeddings_model
# # # #     )
# # # #     return llm, embeddings
# # # #     # from langchain_anthropic import ChatAnthropic
# # # #     # from langchain_huggingface import HuggingFaceEmbeddings
# # # #     # from ragas.llms import LangchainLLMWrapper
# # # #     # from ragas.embeddings import LangchainEmbeddingsWrapper

# # # #     # llm = LangchainLLMWrapper(ChatAnthropic(model=MODEL_NAME))
# # # #     # embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    


# # # # @traceable(name="evaluation.run_ragas", run_type="chain")
# # # # def run_ragas_evaluation(rows: list[RagasRow]):
# # # #     """
# # # #     Runs faithfulness, context_precision, context_recall, and
# # # #     answer_correctness over `rows` (answerable questions only — see this
# # # #     module's docstring). Returns ragas's EvaluationResult (has both an
# # # #     aggregate `.to_pandas()` table and per-metric scores).
# # # #     """
# # # #     from ragas import evaluate
# # # #     from ragas.metrics import (
# # # #         Faithfulness,
# # # #         LLMContextPrecisionWithReference,
# # # #         LLMContextRecall,
# # # #         AnswerCorrectness,
# # # #     )

# # # #     dataset = _build_ragas_evaluation_dataset(rows)
# # # #     llm, embeddings = _build_ragas_llm_and_embeddings()

# # # #     metrics = [
# # # #         Faithfulness(llm=llm),
# # # #         LLMContextPrecisionWithReference(llm=llm),
# # # #         LLMContextRecall(llm=llm),
# # # #         AnswerCorrectness(llm=llm, embeddings=embeddings),
# # # #     ]
# # # #     return evaluate(dataset=dataset, metrics=metrics)
# # # # """
# # # # RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# # # # Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

# # # #   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
# # # #      metrics — faithfulness, context_precision, context_recall, and
# # # #      answer_correctness — against CUAD's human-annotated answer spans as
# # # #      the `reference`.

# # # #   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
# # # #      clause of that category in the contract, so there's no real reference
# # # #      to recall against. These are excluded from the RAGAS dataset entirely
# # # #      and instead scored with a separate "correctly identified absence"
# # # #      metric: did the pipeline's own evidence auditor / answer correctly
# # # #      signal that nothing was found, rather than hallucinating a clause?
# # # #      Mixing the two would silently tank context_recall/answer_correctness
# # # #      on questions that were never answerable in the first place.

# # # # Requires (not in the base requirements.txt — see requirements.txt's
# # # # "Evaluation" section): `ragas`, `langchain-huggingface`, and ONE of
# # # # `langchain-anthropic` / `langchain-groq` depending on RAGAS_JUDGE_PROVIDER
# # # # (see _build_ragas_judge_llm below). RAGAS's LLM-judged metrics (faithfulness,
# # # # context precision/recall, answer_correctness) need a judge LLM + embeddings
# # # # model; the judge defaults to Claude but is swappable via env vars
# # # # (RAGAS_JUDGE_PROVIDER=anthropic|groq, RAGAS_JUDGE_MODEL=<model id>) —
# # # # embeddings always use the SAME model resources.py's HybridSearchAgent uses
# # # # (all-mpnet-base-v2) so "similar enough" judgments are consistent with what
# # # # retrieval itself considers similar, regardless of which provider judges.
# # # # """

# # # # from __future__ import annotations

# # # # from dataclasses import dataclass, field
# # # # from typing import Optional

# # # # from ..llm_client import MODEL_NAME
# # # # from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# # # # from ..tracing import traceable
# # # # from .cuad_loader import CUADExample
# # # # from .scripted_reviewer import ScriptedRunResult

# # # # from legal_graphrag.evaluation.ragas_compat import (
# # # #     install_ragas_trace_guard,
# # # # )
# # # # # ---------------------------------------------------------------------------
# # # # # Absence-detection scoring (unanswerable / is_impossible questions)
# # # # # ---------------------------------------------------------------------------

# # # # _ABSENCE_PHRASES = (
# # # #     "no clause", "does not contain", "no such clause", "not found", "no provision",
# # # #     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# # # # )


# # # # def _looks_like_absence_answer(answer: Optional[str]) -> bool:
# # # #     if not answer:
# # # #         return False
# # # #     lowered = answer.lower()
# # # #     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# # # # @dataclass
# # # # class AbsenceDetectionResult:
# # # #     qas_id: str
# # # #     contract_title: str
# # # #     clause_category: str
# # # #     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
# # # #     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
# # # #     answer_stated_absence: bool            # final answer text reads as "no such clause"
# # # #     correctly_identified_absence: bool     # any of the above three


# # # # def score_absence_detection(
# # # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # # ) -> list[AbsenceDetectionResult]:
# # # #     """
# # # #     Scores the `is_impossible` subset. `correctly_identified_absence` is
# # # #     True if EITHER the evidence auditor flagged insufficient evidence, the
# # # #     evidence checkpoint rejected outright, or the synthesized answer itself
# # # #     reads as an absence statement — any of these means the pipeline did NOT
# # # #     fabricate a clause that CUAD's annotators confirmed doesn't exist.
# # # #     """
# # # #     scored = []
# # # #     for example, result in zip(examples, results):
# # # #         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
# # # #         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
# # # #         evidence_rejected = result.status == "evidence_rejected"
# # # #         answer_absence = _looks_like_absence_answer(result.final_answer)
# # # #         scored.append(
# # # #             AbsenceDetectionResult(
# # # #                 qas_id=example.qas_id,
# # # #                 contract_title=example.contract_title,
# # # #                 clause_category=example.clause_category,
# # # #                 evidence_marked_insufficient=evidence_insufficient,
# # # #                 evidence_rejected=evidence_rejected,
# # # #                 answer_stated_absence=answer_absence,
# # # #                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
# # # #             )
# # # #         )
# # # #     return scored


# # # # def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
# # # #     if not scored:
# # # #         return float("nan")
# # # #     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # # # # ---------------------------------------------------------------------------
# # # # # RAGAS dataset construction (answerable questions only)
# # # # # ---------------------------------------------------------------------------

# # # # @dataclass
# # # # class RagasRow:
# # # #     qas_id: str
# # # #     contract_title: str
# # # #     clause_category: str
# # # #     user_input: str
# # # #     retrieved_contexts: list[str]
# # # #     response: str
# # # #     reference: str
# # # #     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# # # # def build_ragas_rows(
# # # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # # ) -> list[RagasRow]:
# # # #     """
# # # #     Builds one row per answerable example. `retrieved_contexts` comes from
# # # #     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
# # # #     are forced onto the `hybrid` route — see scripted_reviewer.py's
# # # #     force_route default and cuad_ingest.py's module docstring for why.
# # # #     Rows where the pipeline never produced a final_answer (evidence or
# # # #     answer rejected) still get a row with response="" so RAGAS scores that
# # # #     as a real failure rather than silently dropping it from the average.
# # # #     """
# # # #     rows = []
# # # #     for example, result in zip(examples, results):
# # # #         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
# # # #         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
# # # #             hit.get("text", "") for hit in result.graph_hits
# # # #         ]
# # # #         rows.append(
# # # #             RagasRow(
# # # #                 qas_id=example.qas_id,
# # # #                 contract_title=example.contract_title,
# # # #                 clause_category=example.clause_category,
# # # #                 user_input=example.question,
# # # #                 retrieved_contexts=contexts,
# # # #                 response=result.final_answer or "",
# # # #                 reference=example.reference_answer,
# # # #                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
# # # #             )
# # # #         )
# # # #     return rows


# # # # def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
# # # #     """
# # # #     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
# # # #     ragas lazily so the rest of this module (and the non-RAGAS parts of the
# # # #     eval harness) works even if `ragas` isn't installed.
# # # #     """
# # # #     from ragas import EvaluationDataset, SingleTurnSample

# # # #     samples = [
# # # #         SingleTurnSample(
# # # #             user_input=r.user_input,
# # # #             retrieved_contexts=r.retrieved_contexts,
# # # #             response=r.response,
# # # #             reference=r.reference,
# # # #         )
# # # #         for r in rows
# # # #     ]
# # # #     return EvaluationDataset(samples=samples)
# # # # def _build_ragas_judge_llm():
# # # #     """
# # # #     Build the RAGAS judge using Hugging Face Inference API.

# # # #     Uses HF_TOKEN from the environment / .env file.

# # # #     Example:
# # # #         
# # # #         RAGAS_JUDGE_MODEL=meta-llama/Llama-3.3-70B-Instruct
# # # #     """
# # # #     import os
# # # #     import json
# # # #     import re
# # # #     from huggingface_hub import InferenceClient
# # # #     from langchain_core.language_models.chat_models import BaseChatModel
# # # #     from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# # # #     from langchain_core.outputs import ChatGeneration, ChatResult
# # # #     from pydantic import Field

# # # #     from ragas.llms import LangchainLLMWrapper

# # # #     hf_token = os.getenv("HF_TOKEN")

# # # #     if not hf_token:
# # # #         raise RuntimeError(
# # # #             "HF_TOKEN environment variable is not set. "
# # # #             "Add HF_TOKEN=hf_... to your .env file."
# # # #         )

# # # #     model = os.getenv(
# # # #         "RAGAS_JUDGE_MODEL",
# # # #         "openai/gpt-oss-120b:groq",
# # # #     )

# # # #     class HuggingFaceChatModel(BaseChatModel):
# # # #         client: object = Field(exclude=True)
# # # #         model: str

# # # #         @property
# # # #         def _llm_type(self) -> str:
# # # #             return "huggingface_inference"

# # # #         @property
# # # #         def _identifying_params(self) -> dict:
# # # #             return {
# # # #                 "model": self.model,
# # # #             }

# # # #         def _generate(
# # # #             self,
# # # #             messages,
# # # #             stop=None,
# # # #             run_manager=None,
# # # #             **kwargs,
# # # #         ) -> ChatResult:

# # # #             hf_messages = []

# # # #             for message in messages:

# # # #                 if isinstance(message, SystemMessage):
# # # #                     role = "system"

# # # #                 elif isinstance(message, HumanMessage):
# # # #                     role = "user"

# # # #                 elif isinstance(message, AIMessage):
# # # #                     role = "assistant"

# # # #                 else:
# # # #                     role = "user"

# # # #                 hf_messages.append(
# # # #                     {
# # # #                         "role": role,
# # # #                         "content": message.content,
# # # #                     }
# # # #                 )

# # # #             completion = self.client.chat.completions.create(
# # # #                 model=self.model,
# # # #                 messages=hf_messages,
# # # #                 max_tokens=kwargs.get("max_tokens", 2000),
# # # #                 temperature=kwargs.get("temperature", 0.0),
# # # #             )

# # # #             text = completion.choices[0].message.content or ""
# # # #             # Clean markdown code fences if present (e.g. ```json ... ```)
# # # #             cleaned_text = text.strip()
# # # #             if cleaned_text.startswith("```"):
# # # #             # Remove opening and closing code blocks
# # # #                 cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.IGNORECASE)
# # # #                 cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
# # # #             generation = ChatGeneration(
# # # #                 message=AIMessage(content=text)
# # # #             )

# # # #             return ChatResult(
# # # #                 generations=[generation]
# # # #             )

# # # #     client = InferenceClient(
# # # #         api_key=hf_token,
# # # #         provider="auto",
# # # #     )

# # # #     hf_llm = HuggingFaceChatModel(
# # # #         client=client,
# # # #         model=model,
# # # #     )

# # # #     return LangchainLLMWrapper(hf_llm)

# # # # # def _build_ragas_judge_llm():
# # # # #     """
# # # # #     RAGAS's LLM-judged metrics need a langchain-wrapped chat model to act as
# # # # #     judge. Configurable via env vars so you can point this at whatever the
# # # # #     rest of your eval stack already uses, independent of llm_client.py's
# # # # #     Claude model:

# # # # #         RAGAS_JUDGE_PROVIDER = "anthropic" (default) | "groq"
# # # # #         RAGAS_JUDGE_MODEL    = provider-specific model id
# # # # #                                 (default: llm_client.MODEL_NAME for anthropic,
# # # # #                                  "openai/gpt-oss-120b" for groq)

# # # # #     For provider="groq": if RAGAS_JUDGE_MODEL carries a trailing
# # # # #     ":<routing-hint>" suffix (e.g. "openai/gpt-oss-120b:groq" — an
# # # # #     OpenRouter-style provider suffix), it's stripped before being passed to
# # # # #     ChatGroq, since Groq's own API takes the bare model id
# # # # #     ("openai/gpt-oss-120b") and rejects the suffixed form. Requires
# # # # #     GROQ_API_KEY to be set.
# # # # #     """
# # # # #     import os

# # # # #     from ragas.llms import LangchainLLMWrapper

# # # # #     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "anthropic").lower()

# # # # #     if provider == "groq":
# # # # #         from langchain_groq import ChatGroq

# # # # #         model = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
# # # # #         model = model.split(":", 1)[0]  # strip any ":<routing-hint>" suffix — see docstring
# # # # #         return LangchainLLMWrapper(ChatGroq(model=model))

# # # # #     if provider == "anthropic":
# # # # #         from langchain_anthropic import ChatAnthropic

# # # # #         model = os.getenv("RAGAS_JUDGE_MODEL", MODEL_NAME)
# # # # #         return LangchainLLMWrapper(ChatAnthropic(model=model))

# # # # #     raise ValueError(f"Unknown RAGAS_JUDGE_PROVIDER={provider!r} — expected 'anthropic' or 'groq'.")


# # # # # def _build_ragas_llm_and_embeddings():
# # # # #     """
# # # # #     Judge LLM (see _build_ragas_judge_llm) + embeddings model. Embeddings
# # # # #     use the SAME model HybridSearchAgent uses (EMBEDDING_MODEL_NAME), so
# # # # #     RAGAS's notion of "similar" for answer_correctness matches what
# # # # #     retrieval itself uses, regardless of which provider judges the LLM
# # # # #     metrics.
# # # # #     """
# # # # #     from langchain_huggingface import HuggingFaceEmbeddings
# # # # #     from ragas.embeddings import LangchainEmbeddingsWrapper

# # # # #     llm = _build_ragas_judge_llm()
# # # # #     embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
# # # # #     return llm, embeddings
# # # # def _build_ragas_llm_and_embeddings():
# # # #     """
# # # #     Build the Hugging Face judge LLM + Hugging Face embeddings.
# # # #     """

# # # #     from langchain_huggingface import HuggingFaceEmbeddings
# # # #     from ragas.embeddings import LangchainEmbeddingsWrapper

# # # #     llm = _build_ragas_judge_llm()

# # # #     embeddings = LangchainEmbeddingsWrapper(
# # # #         HuggingFaceEmbeddings(
# # # #             model_name=EMBEDDING_MODEL_NAME
# # # #         )
# # # #     )

# # # #     return llm, embeddings

# # # # @traceable(name="evaluation.run_ragas", run_type="chain")
# # # # def run_ragas_evaluation(rows: list[RagasRow]):
# # # #     """
# # # #     Runs faithfulness, context_precision, context_recall, and
# # # #     answer_correctness over `rows` (answerable questions only — see this
# # # #     module's docstring). Returns ragas's EvaluationResult (has both an
# # # #     aggregate `.to_pandas()` table and per-metric scores).
# # # #     """
# # # #     from legal_graphrag.evaluation.ragas_compat import (
# # # #     install_ragas_trace_guard,
# # # #     )
# # # #     from ragas import evaluate
# # # #     from ragas.metrics import (
# # # #         Faithfulness,
# # # #         LLMContextPrecisionWithReference,
# # # #         LLMContextRecall,
# # # #         AnswerCorrectness,
# # # #     )

# # # #     dataset = _build_ragas_evaluation_dataset(rows)
# # # #     llm, embeddings = _build_ragas_llm_and_embeddings()

# # # #     metrics = [
# # # #         Faithfulness(llm=llm),
# # # #         LLMContextPrecisionWithReference(llm=llm),
# # # #         LLMContextRecall(llm=llm),
# # # #         AnswerCorrectness(llm=llm, embeddings=embeddings),
# # # #     ]

# # # #     installed = install_ragas_trace_guard()
# # # #     print(f"[ragas] trace guard installed={installed}")

# # # #     print(f"[ragas] trace guard installed={installed}")
# # # # ##=====
# # # #     # 1. Execute RAGAS evaluation
# # # #     evaluation = evaluate(dataset=dataset, metrics=metrics)

# # # #     # 2. Add diagnostic logs
# # # #     print("[DEBUG RAGAS TYPE]:", type(evaluation))
# # # #     if hasattr(evaluation, "_repr_dict"):
# # # #         print("[DEBUG _repr_dict]:", dict(evaluation._repr_dict))
        
# # # #     frame = evaluation.to_pandas()
# # # #     print("[DEBUG DataFrame Columns]:", list(frame.columns))
# # # #     print("[DEBUG DataFrame Row 0]:", frame.iloc[0].to_dict() if len(frame) else {})
# # # # #=====

# # # #     return evaluate(dataset=dataset, metrics=metrics)


# # # # # """
# # # # # RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# # # # # Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

# # # # #   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
# # # # #      metrics — faithfulness, context_precision, context_recall, and
# # # # #      answer_correctness — against CUAD's human-annotated answer spans as
# # # # #      the `reference`.

# # # # #   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
# # # # #      clause of that category in the contract, so there's no real reference
# # # # #      to recall against. These are excluded from the RAGAS dataset entirely
# # # # #      and instead scored with a separate "correctly identified absence"
# # # # #      metric: did the pipeline's own evidence auditor / answer correctly
# # # # #      signal that nothing was found, rather than hallucinating a clause?
# # # # #      Mixing the two would silently tank context_recall/answer_correctness
# # # # #      on questions that were never answerable in the first place.

# # # # # Requires (not in the base requirements.txt — see requirements.txt's
# # # # # "Evaluation" section): `ragas`, `langchain-HF`, `langchain-huggingface`.
# # # # # RAGAS's LLM-judged metrics (faithfulness, context precision/recall,
# # # # # answer_correctness) need an LLM + embeddings model; this module reuses
# # # # # Claude (via langchain_HF, matching llm_client.py's MODEL_NAME) as
# # # # # the judge, and the SAME embedding model resources.py's HybridSearchAgent
# # # # # uses (all-mpnet-base-v2) so "similar enough" judgments are consistent with
# # # # # what retrieval itself considers similar.
# # # # # """

# # # # # from __future__ import annotations

# # # # # from dataclasses import dataclass, field
# # # # # from typing import Optional

# # # # # from ..llm_client import MODEL_NAME
# # # # # from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# # # # # from ..tracing import traceable
# # # # # from .cuad_loader import CUADExample
# # # # # from .scripted_reviewer import ScriptedRunResult

# # # # # import os

# # # # # from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# # # # # from langchain_huggingface import HuggingFaceEmbeddings

# # # # # from ragas.llms import LangchainLLMWrapper
# # # # # from ragas.embeddings import LangchainEmbeddingsWrapper
# # # # # ---------------------------------------------------------------------------
# # # # # # Absence-detection scoring (unanswerable / is_impossible questions)
# # # # # # ---------------------------------------------------------------------------
# # # # # MODEL_NAME = "openai/gpt-oss-120b:groq"
# # # # # _ABSENCE_PHRASES = (
# # # # #     "no clause", "does not contain", "no such clause", "not found", "no provision",
# # # # #     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# # # # # )


# # # # # def _looks_like_absence_answer(answer: Optional[str]) -> bool:
# # # # #     if not answer:
# # # # #         return False
# # # # #     lowered = answer.lower()
# # # # #     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# # # # # @dataclass
# # # # # class AbsenceDetectionResult:
# # # # #     qas_id: str
# # # # #     contract_title: str
# # # # #     clause_category: str
# # # # #     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
# # # # #     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
# # # # #     answer_stated_absence: bool            # final answer text reads as "no such clause"
# # # # #     correctly_identified_absence: bool     # any of the above three


# # # # # def score_absence_detection(
# # # # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # # # ) -> list[AbsenceDetectionResult]:
# # # # #     """
# # # # #     Scores the `is_impossible` subset. `correctly_identified_absence` is
# # # # #     True if EITHER the evidence auditor flagged insufficient evidence, the
# # # # #     evidence checkpoint rejected outright, or the synthesized answer itself
# # # # #     reads as an absence statement — any of these means the pipeline did NOT
# # # # #     fabricate a clause that CUAD's annotators confirmed doesn't exist.
# # # # #     """
# # # # #     scored = []
# # # # #     for example, result in zip(examples, results):
# # # # #         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
# # # # #         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
# # # # #         evidence_rejected = result.status == "evidence_rejected"
# # # # #         answer_absence = _looks_like_absence_answer(result.final_answer)
# # # # #         scored.append(
# # # # #             AbsenceDetectionResult(
# # # # #                 qas_id=example.qas_id,
# # # # #                 contract_title=example.contract_title,
# # # # #                 clause_category=example.clause_category,
# # # # #                 evidence_marked_insufficient=evidence_insufficient,
# # # # #                 evidence_rejected=evidence_rejected,
# # # # #                 answer_stated_absence=answer_absence,
# # # # #                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
# # # # #             )
# # # # #         )
# # # # #     return scored


# # # # # def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
# # # # #     if not scored:
# # # # #         return float("nan")
# # # # #     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # # # # # ---------------------------------------------------------------------------
# # # # # # RAGAS dataset construction (answerable questions only)
# # # # # # ---------------------------------------------------------------------------

# # # # # @dataclass
# # # # # class RagasRow:
# # # # #     qas_id: str
# # # # #     contract_title: str
# # # # #     clause_category: str
# # # # #     user_input: str
# # # # #     retrieved_contexts: list[str]
# # # # #     response: str
# # # # #     reference: str
# # # # #     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# # # # # def build_ragas_rows(
# # # # #     results: list[ScriptedRunResult], examples: list[CUADExample]
# # # # # ) -> list[RagasRow]:
# # # # #     """
# # # # #     Builds one row per answerable example. `retrieved_contexts` comes from
# # # # #     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
# # # # #     are forced onto the `hybrid` route — see scripted_reviewer.py's
# # # # #     force_route default and cuad_ingest.py's module docstring for why.
# # # # #     Rows where the pipeline never produced a final_answer (evidence or
# # # # #     answer rejected) still get a row with response="" so RAGAS scores that
# # # # #     as a real failure rather than silently dropping it from the average.
# # # # #     """
# # # # #     rows = []
# # # # #     for example, result in zip(examples, results):
# # # # #         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
# # # # #         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
# # # # #             hit.get("text", "") for hit in result.graph_hits
# # # # #         ]
# # # # #         rows.append(
# # # # #             RagasRow(
# # # # #                 qas_id=example.qas_id,
# # # # #                 contract_title=example.contract_title,
# # # # #                 clause_category=example.clause_category,
# # # # #                 user_input=example.question,
# # # # #                 retrieved_contexts=contexts,
# # # # #                 response=result.final_answer or "",
# # # # #                 reference=example.reference_answer,
# # # # #                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
# # # # #             )
# # # # #         )
# # # # #     return rows


# # # # # def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
# # # # #     """
# # # # #     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
# # # # #     ragas lazily so the rest of this module (and the non-RAGAS parts of the
# # # # #     eval harness) works even if `ragas` isn't installed.
# # # # #     """
# # # # #     from ragas import EvaluationDataset, SingleTurnSample

# # # # #     samples = [
# # # # #         SingleTurnSample(
# # # # #             user_input=r.user_input,
# # # # #             retrieved_contexts=r.retrieved_contexts,
# # # # #             response=r.response,
# # # # #             reference=r.reference,
# # # # #         )
# # # # #         for r in rows
# # # # #     ]
# # # # #     return EvaluationDataset(samples=samples)


# # # # # def _build_ragas_llm_and_embeddings():
# # # # #     """
# # # # #     RAGAS's LLM-judged metrics need a langchain-wrapped LLM + embeddings
# # # # #     model. Uses the same Claude model as the rest of the pipeline
# # # # #     (llm_client.MODEL_NAME) and the same embedding model HybridSearchAgent
# # # # #     uses (EMBEDDING_MODEL_NAME), so judgments are made with the same notion
# # # # #     of "similar" that retrieval itself uses.
# # # # #     """
# # # # #     llm_endpoint = HuggingFaceEndpoint(
# # # # #         repo_id="openai/gpt-oss-120b:groq",
# # # # #         huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# # # # #         max_new_tokens=1000,
# # # # #         temperature=0.1,
# # # # #     )

# # # # #     hf_llm = ChatHuggingFace(
# # # # #         llm=llm_endpoint
# # # # #     )

# # # # #     llm = LangchainLLMWrapper(hf_llm)

# # # # #     # -----------------------------
# # # # #     # Hugging Face Embeddings
# # # # #     # -----------------------------
# # # # #     embeddings_model = HuggingFaceEmbeddings(
# # # # #         model_name=EMBEDDING_MODEL_NAME
# # # # #     )

# # # # #     embeddings = LangchainEmbeddingsWrapper(
# # # # #         embeddings_model
# # # # #     )
# # # # #     return llm, embeddings
# # # # #     # from langchain_anthropic import ChatAnthropic
# # # # #     # from langchain_huggingface import HuggingFaceEmbeddings
# # # # #     # from ragas.llms import LangchainLLMWrapper
# # # # #     # from ragas.embeddings import LangchainEmbeddingsWrapper

# # # # #     # llm = LangchainLLMWrapper(ChatAnthropic(model=MODEL_NAME))
# # # # #     # embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    


# # # # # @traceable(name="evaluation.run_ragas", run_type="chain")
# # # # # def run_ragas_evaluation(rows: list[RagasRow]):
# # # # #     """
# # # # #     Runs faithfulness, context_precision, context_recall, and
# # # # #     answer_correctness over `rows` (answerable questions only — see this
# # # # #     module's docstring). Returns ragas's EvaluationResult (has both an
# # # # #     aggregate `.to_pandas()` table and per-metric scores).
# # # # #     """
# # # # #     from ragas import evaluate
# # # # #     from ragas.metrics import (
# # # # #         Faithfulness,
# # # # #         LLMContextPrecisionWithReference,
# # # # #         LLMContextRecall,
# # # # #         AnswerCorrectness,
# # # # #     )

# # # # #     dataset = _build_ragas_evaluation_dataset(rows)
# # # # #     llm, embeddings = _build_ragas_llm_and_embeddings()

# # # # #     metrics = [
# # # # #         Faithfulness(llm=llm),
# # # # #         LLMContextPrecisionWithReference(llm=llm),
# # # # #         LLMContextRecall(llm=llm),
# # # # #         AnswerCorrectness(llm=llm, embeddings=embeddings),
# # # # #     ]
# # # # #     return evaluate(dataset=dataset, metrics=metrics)
