from legal_graphrag.graphrag.extraction import is_read_only_cypher


def test_read_only_cypher_allows_match_return():
    assert is_read_only_cypher("MATCH (c:Contract) RETURN c.name LIMIT 10")


def test_read_only_cypher_rejects_delete():
    assert not is_read_only_cypher("MATCH (c:Contract) DETACH DELETE c")


def test_read_only_cypher_rejects_create():
    assert not is_read_only_cypher("CREATE (c:Contract {name: 'x'})")


def test_read_only_cypher_rejects_set():
    assert not is_read_only_cypher("MATCH (c:Contract) SET c.approved = true RETURN c")


def test_read_only_cypher_rejects_case_insensitively():
    assert not is_read_only_cypher("match (c) detach delete c")
