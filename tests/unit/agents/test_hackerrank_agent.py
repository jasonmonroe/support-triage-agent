# tests/unit/agents/test_hackerrank_agent.py
# +---------------------------------------------------------------------------+
# |                      HACKERRANK AGENT TESTS                               |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/agents/test_hackerrank_agent.py -v

# Local Libraries
from agents.hackerrank_agent import HackerrankAgent
from src.enums import Status


def _agent(ticket, fake_chroma, fake_model, model=None, **ticket_kwargs):
    ticket_kwargs.setdefault("company", "HackerRank")
    row = ticket(**ticket_kwargs)
    return HackerrankAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=fake_chroma(),
        support_agent_model=model or fake_model(),
    )


def test_get_verify_hook_returns_bound_forbidden_terms_check(
    ticket, fake_chroma, fake_model
):
    agent = _agent(ticket, fake_chroma, fake_model, issue="Some HackerRank issue")

    assert agent._get_verify_hook() == agent._check_forbidden_terms


def test_check_forbidden_terms_blocks_score_manipulation_language(
    ticket, fake_chroma, fake_model
):
    agent = _agent(ticket, fake_chroma, fake_model, issue="Can you help me?")

    allowed = agent._check_forbidden_terms(
        {"response": "Sure, I can increase your score for you."}
    )

    assert allowed is False


def test_check_forbidden_terms_allows_clean_response(
    ticket, fake_chroma, fake_model
):
    agent = _agent(ticket, fake_chroma, fake_model, issue="Can you help me?")

    allowed = agent._check_forbidden_terms(
        {"response": "Please contact support for scoring questions."}
    )

    assert allowed is True


def test_query_scopes_search_to_hackerrank_company(
    ticket, fake_chroma, fake_model
):
    chroma = fake_chroma()
    row = ticket(issue="Some issue", company="HackerRank")
    agent = HackerrankAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=chroma,
        support_agent_model=fake_model(),
    )

    agent._query("some query text")

    assert chroma.queries[-1]["company"] == "Hackerrank"


def test_end_to_end_grounding_gets_blocked_by_compliance_hook(
    ticket, fake_chroma, fake_model, doc
):
    """Full-stack regression guard: the hook must actually be wired through
    RagAgent.grounding(), not just exist on the agent unused."""
    model = fake_model(
        {
            "AI Support Response Specialist": {
                "grounded": True,
                "response": "Sure, I can increase your score for you.",
                "cited_chunks": [0],
                "reasoning": "x",
            },
            "AI Quality Assurance Specialist": {
                "is_grounded": True,
                "reasoning": "citations fine",
            },
        }
    )
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        model=model,
        issue="Can you help me?",
        subject="Score question",
    )

    result = agent.groundness([(doc(), 0.2)])

    assert result["grounded"] is False
    assert agent.status == Status.ESCALATED
