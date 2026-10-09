"""
Shared, low-level LLM call helpers.

Uses Groq Llama 3.1 8B Instant.
"""

from __future__ import annotations

import json
import re
import os

from groq import Groq

from .tracing import traceable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# "

MODEL_NAME = "llama-3.1-8b-instant"


if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file or environment."
    )


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

_client = Groq(
    api_key=GROQ_API_KEY
)


# ---------------------------------------------------------------------------
# JSON LLM call
# ---------------------------------------------------------------------------

@traceable(name="llm.call_json", run_type="llm")
def call_json(
    system: str,
    user: str,
    max_tokens: int = 2000,
) -> dict | list:

    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""

    cleaned = raw.strip()

    # Remove <think>...</think> if returned
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Remove markdown code fences
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        match = re.search(
            r"(\{.*\}|\[.*\])",
            cleaned,
            flags=re.DOTALL,
        )

        if match:
            return json.loads(match.group(1))

        raise ValueError(
            "Model returned invalid JSON.\n\n"
            f"Raw response:\n{raw}"
        )


# ---------------------------------------------------------------------------
# Plain-text LLM call
# ---------------------------------------------------------------------------

@traceable(name="llm.call_text", run_type="llm")
def call_text(
    system: str,
    user: str,
    max_tokens: int = 1500,
) -> str:

    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )

    raw = response.choices[0].message.content or ""

    # Remove reasoning blocks if present
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    return cleaned
# """
# Shared, low-level LLM call helpers.

# All LLM calls in the Legal GraphRAG system go through this module:
# - clause extraction
# - text-to-Cypher
# - router
# - auditor
# - synthesizer

# Uses Hugging Face InferenceClient.
# """

# from __future__ import annotations

# import json
# import os
# import re

# from huggingface_hub import InferenceClient

# from .tracing import traceable

# import os
# import json
# import re

# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient
# from openai import OpenAI

# from .tracing import traceable
# from .tracing import trace_huggingface_call

# load_dotenv(override=True)
# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------



# # load_dotenv(override=True)

# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Jg")
# MODEL_NAME = "gemini-3.6-flash"

# if not GEMINI_API_KEY:
#     raise RuntimeError(
#         "GEMINI_API_KEY environment variable is not set. "
#         "Please provide a valid Google Gemini API key."
#     )

# # ---------------------------------------------------------------------------
# # Gemini OpenAI-Compatible Client Initialization
# # ---------------------------------------------------------------------------

# _client = OpenAI(
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     api_key=GEMINI_API_KEY,
# )


# # ---------------------------------------------------------------------------
# # JSON LLM call
# # ---------------------------------------------------------------------------

# @traceable(name="llm.call_json", run_type="llm")
# def call_json(
#     system: str,
#     user: str,
#     max_tokens: int = 2000,
# ) -> dict | list:
#     """
#     Call Google Gemini expecting JSON output.

#     Returns:
#         Parsed dict or list.
#     """

#     response = _client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system,
#             },
#             {
#                 "role": "user",
#                 "content": user,
#             },
#         ],
#         max_tokens=max_tokens,
#         temperature=0.0,
#         response_format={"type": "json_object"},
#     )

#     raw = response.choices[0].message.content or ""

#     cleaned = raw.strip()

#     # Remove markdown code fences if the model still returns them.
#     cleaned = re.sub(
#         r"^```(?:json)?\s*",
#         "",
#         cleaned,
#         flags=re.IGNORECASE,
#     )

#     cleaned = re.sub(
#         r"\s*```$",
#         "",
#         cleaned,
#     )

#     cleaned = cleaned.strip()

#     try:
#         return json.loads(cleaned)

#     except json.JSONDecodeError:

#         # Last-resort recovery:
#         # find the first JSON object or array.
#         match = re.search(
#             r"(\{.*\}|\[.*\])",
#             cleaned,
#             flags=re.DOTALL,
#         )

#         if match:
#             return json.loads(match.group(1))

#         raise ValueError(
#             "Model returned invalid JSON.\n\n"
#             f"Raw response:\n{raw}"
#         )


# # ---------------------------------------------------------------------------
# # Plain-text LLM call
# # ---------------------------------------------------------------------------

# @traceable(name="llm.call_text", run_type="llm")
# def call_text(
#     system: str,
#     user: str,
#     max_tokens: int = 1500,
# ) -> str:
#     """
#     Call Google Gemini expecting plain text.
#     """

#     response = _client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system,
#             },
#             {
#                 "role": "user",
#                 "content": user,
#             },
#         ],
#         max_tokens=max_tokens,
#         temperature=0.2,
#     )

#     return (
#         response.choices[0].message.content or ""
#     ).strip()
# # # HF_TOKEN = os.getenv("new_HF_TOKEN")
# # # Choose the Hugging Face model you want to use.
# # # Example: meta-llama/Llama-3.3-70B-Instruct
# # MODEL_NAME = "openai/gpt-oss-120b:groq"

# # if not HF_TOKEN:
# #     raise RuntimeError(
# #         "HF_TOKEN environment variable is not set. "
# #         "Create a Hugging Face token with inference permissions."
# #     )


# # # ---------------------------------------------------------------------------
# # # Hugging Face client
# # # ---------------------------------------------------------------------------

# # _client = InferenceClient(
# #     api_key=HF_TOKEN
# # )


# # # ---------------------------------------------------------------------------
# # # JSON LLM call
# # # ---------------------------------------------------------------------------
# # @traceable(name="llm.call_json", run_type="llm")
# # def call_json(
# #     system: str,
# #     user: str,
# #     max_tokens: int = 2000,
# # ) -> dict | list:
# #     """
# #     Call the Hugging Face model expecting JSON output.

# #     Returns:
# #         Parsed dict or list.
# #     """

# #     response = _client.chat.completions.create(
# #         model=MODEL_NAME,
# #         messages=[
# #             {
# #                 "role": "system",
# #                 "content": system,
# #             },
# #             {
# #                 "role": "user",
# #                 "content": user,
# #             },
# #         ],
# #         max_tokens=max_tokens,
# #         temperature=0.0,
# #     )

# #     raw = response.choices[0].message.content or ""

# #     cleaned = raw.strip()

# #     # Remove markdown code fences if the model still returns them.
# #     cleaned = re.sub(
# #         r"^```(?:json)?\s*",
# #         "",
# #         cleaned,
# #         flags=re.IGNORECASE,
# #     )

# #     cleaned = re.sub(
# #         r"\s*```$",
# #         "",
# #         cleaned,
# #     )

# #     cleaned = cleaned.strip()

# #     try:
# #         return json.loads(cleaned)

# #     except json.JSONDecodeError:

# #         # Last-resort recovery:
# #         # find the first JSON object or array.
# #         match = re.search(
# #             r"(\{.*\}|\[.*\])",
# #             cleaned,
# #             flags=re.DOTALL,
# #         )

# #         if match:
# #             return json.loads(match.group(1))

# #         raise ValueError(
# #             "Model returned invalid JSON.\n\n"
# #             f"Raw response:\n{raw}"
# #         )


# # # ---------------------------------------------------------------------------
# # # Plain-text LLM call
# # # ---------------------------------------------------------------------------
# # @traceable(name="llm.call_text", run_type="llm")
# # def call_text(
# #     system: str,
# #     user: str,
# #     max_tokens: int = 1500,
# # ) -> str:
# #     """
# #     Call the Hugging Face model expecting plain text.
# #     """

# #     response = _client.chat.completions.create(
# #         model=MODEL_NAME,
# #         messages=[
# #             {
# #                 "role": "system",
# #                 "content": system,
# #             },
# #             {
# #                 "role": "user",
# #                 "content": user,
# #             },
# #         ],
# #         max_tokens=max_tokens,
# #         temperature=0.2,
# #     )

# #     return (
# #         response.choices[0].message.content or ""
# #     ).strip()

