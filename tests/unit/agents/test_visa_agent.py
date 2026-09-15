# tests/unit/agents/test_visa_agent.py
# +---------------------------------------------------------------------------+
# |                          VISA AGENT TESTS                                 |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/agents/test_visa_agent.py -v

# Local Libraries
from agents.visa_agent import VisaAgent
from src.enums import Status

COMPLIANCE_MARKER = "Visa Compliance & Safety Auditor"


def _agent(ticket, fake_chroma, fake_model, model=None, **ticket_kwargs):
    ticket_kwargs.setdefault("company", "Visa")
    row = ticket(**ticket_kwargs)
    return VisaAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=fake_chroma(),
        support_agent_model=model or fake_model(),
    )


def test_get_verify_hook_returns_bound_compliance_check(
    ticket, fake_chroma, fake_model
):
    agent = _agent(ticket, fake_chroma, fake_model, issue="Card issue")

    assert agent._get_verify_hook() == agent._compliance_check


def test_compliance_check_blocks_when_model_flags_violation(
    ticket, fake_chroma, fake_model
):
    model = fake_model(
        {COMPLIANCE_MARKER: {"is_compliant": False, "reasoning": "discloses PII"}}
    )
    agent = _agent(
        ticket, fake_chroma, fake_model, model=model, issue="What is my card number?"
    )

    allowed = agent._compliance_check(
        {"response": "Your card number is 4111 1111 1111 1111."}
    )

    assert allowed is False


def test_compliance_check_allows_compliant_response(
    ticket, fake_chroma, fake_model
):
    model = fake_model(
        {COMPLIANCE_MARKER: {"is_compliant": True, "reasoning": "clean"}}
    )
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        model=model,
        issue="How do I request a refund?",
    )

    allowed = agent._compliance_check(
        {"response": "Contact support for a refund review."}
    )

    assert allowed is True


def test_compliance_check_fails_closed_on_malformed_model_response(
    ticket, fake_chroma, fake_model
):
    """A None/non-dict compliance response must fail closed (escalate), not
    silently pass."""
    model = fake_model({})  # no marker matches -> get_response returns {}
    agent = _agent(
        ticket, fake_chroma, fake_model, model=model, issue="Card issue"
    )

    allowed = agent._compliance_check({"response": "some response"})

    assert allowed is False


def test_query_scopes_search_to_visa_company(ticket, fake_chroma, fake_model):
    chroma = fake_chroma()
    row = ticket(issue="Some issue", company="Visa")
    agent = VisaAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=chroma,
        support_agent_model=fake_model(),
    )

    agent._query("query text")

    assert chroma.queries[-1]["company"] == "Visa"


def test_end_to_end_grounding_gets_blocked_by_compliance_hook(
    ticket, fake_chroma, fake_model, doc
):
    model = fake_model(
        {
            "AI Support Response Specialist": {
                "grounded": True,
                "response": "Your card number is 4111 1111 1111 1111.",
                "cited_chunks": [0],
                "reasoning": "x",
            },
            "AI Quality Assurance Specialist": {
                "is_grounded": True,
                "reasoning": "citations fine",
            },
            COMPLIANCE_MARKER: {
                "is_compliant": False,
                "reasoning": "discloses PII",
            },
        }
    )
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        model=model,
        issue="What is my card number?",
    )

    result = agent.groundness([(doc(), 0.2)])

    assert result["grounded"] is False
    assert agent.status == Status.ESCALATED
