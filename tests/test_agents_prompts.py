from legal_graphrag.agents.prompts import (
    classify_query_style,
    classify_route,
    ALPHA_BY_QUERY_STYLE,
)


def test_style_detects_quoted_phrase_as_exact_match():
    assert classify_query_style('What does the clause containing "time is of the essence" say?') == "exact_match"


def test_style_detects_section_reference_as_exact_match():
    assert classify_query_style("What does Section 4.2 say about indemnification?") == "exact_match"


def test_style_detects_case_citation_as_exact_match():
    assert classify_query_style("How was this clause treated in Smith v. Jones?") == "exact_match"


def test_style_detects_open_ended_question_as_semantic():
    assert classify_query_style("What happens if the vendor misses a delivery deadline?") == "semantic"


def test_style_defaults_to_balanced():
    assert classify_query_style("Tell me about the payment terms.") == "balanced"


def test_route_graph_keyword_fast_path_still_works_with_alpha_returned():
    route, reasoning, alpha = classify_route(
        "Show all contracts where ABC Ltd. is the vendor, the same clause appears in "
        "another contract, and that clause has been interpreted by multiple judgments."
    )
    assert route == "graph"
    assert alpha == ALPHA_BY_QUERY_STYLE[classify_query_style(
        "Show all contracts where ABC Ltd. is the vendor, the same clause appears in "
        "another contract, and that clause has been interpreted by multiple judgments."
    )]


def test_route_hybrid_exact_match_fast_path_skips_llm():
    # A confidently-classified style should short-circuit to "hybrid" with the
    # matching alpha, without ever needing to call the LLM fallback.
    route, reasoning, alpha = classify_route('What does "for cause" mean in Section 9?')
    assert route == "hybrid"
    assert alpha == ALPHA_BY_QUERY_STYLE["exact_match"]


def test_route_hybrid_semantic_fast_path_skips_llm():
    route, reasoning, alpha = classify_route("Explain what happens if the contract is breached.")
    assert route == "hybrid"
    assert alpha == ALPHA_BY_QUERY_STYLE["semantic"]
