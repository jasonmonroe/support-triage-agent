# tests/unit/agents/test_support_agent.py
# +---------------------------------------------------------------------------+
# |                        SUPPORT AGENT TESTS                                |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/agents/test_support_agent.py -v

# Local Libraries
from agents.support_agent import SupportAgent
from src.enums import Risk, Status


def _agent(ticket, fake_chroma, fake_model, **ticket_kwargs):
    row = ticket(**ticket_kwargs)
    return SupportAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=fake_chroma(),
        support_agent_model=fake_model(),
    )


def test_classify_issue_flags_critical_risk(ticket, fake_chroma, fake_model):
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        issue="There has been a data leak, please help",
    )

    agent.classify_issue()

    assert agent._risk_level == Risk.CRITICAL


def test_classify_issue_flags_urgent_terms(ticket, fake_chroma, fake_model):
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        issue="The site is down and I need help immediately",
    )

    agent.classify_issue()

    assert agent._urgency == Risk.HIGH


def test_make_decision_hard_escalates_on_critical_risk(
    ticket, fake_chroma, fake_model
):
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        issue="This is a security vulnerability in your system",
    )
    agent.classify_issue()

    agent.make_decision()

    assert agent.status == Status.ESCALATED


def test_make_decision_replies_on_high_risk_terms_not_critical(
    ticket, fake_chroma, fake_model
):
    """HIGH_RISK_TERMS (e.g. "refund") deliberately do NOT hard-escalate —
    regression guard for the fix documented in make_decision()'s docstring:
    auto-escalating these skips retrieval and can block an answerable FAQ."""
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        issue="I would like to request a refund for my subscription",
    )
    agent.classify_issue()

    agent.make_decision()

    assert agent.status == Status.REPLIED


def test_rag_agent_receives_populated_issue_and_subject(
    ticket, fake_chroma, fake_model
):
    """Regression guard: RagAgent must be constructed AFTER _set_attrs()
    runs, otherwise it's built with issue/subject still None."""
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        issue="My account was compromised",
        subject="Urgent security issue",
    )

    assert agent.rag_agent._issue == "My account was compromised"
    assert agent.rag_agent._subject == "Urgent security issue"


def test_default_verify_hook_is_none(ticket, fake_chroma, fake_model):
    agent = _agent(ticket, fake_chroma, fake_model, issue="Some issue")

    assert agent.rag_agent._verify_hook is None


def test_company_detected_from_keywords_when_blank(
    ticket, fake_chroma, fake_model
):
    agent = _agent(
        ticket,
        fake_chroma,
        fake_model,
        issue="My HackerRank test invite expired",
        company="None",
    )

    assert agent.company == "hackerrank"


def test_groundness_copies_status_response_and_justification_back(
    ticket, fake_chroma, fake_model, doc
):
    """Regression guard: groundness() must mirror status/response/
    justification from RagAgent onto the agent, not just status — otherwise
    export() reads stale placeholder text set by make_decision()."""
    row = ticket(
        issue="I forgot my password and need to reset it",
        subject="Password reset",
    )
    model = fake_model(
        {
            "AI Support Response Specialist": {
                "grounded": True,
                "response": "Reset it in Settings.",
                "cited_chunks": [0],
                "reasoning": "covers it",
            },
            "AI Quality Assurance Specialist": {
                "is_grounded": True,
                "reasoning": "fine",
            },
            "AI Support Supervisor": {
                "precision_score": 0.95,
                "reasoning": "on topic",
            },
        }
    )
    agent = SupportAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=fake_chroma(),
        support_agent_model=model,
    )

    agent.groundness([(doc(), 0.2)])

    assert agent.status == Status.REPLIED
    assert agent.response == "Reset it in Settings."
    assert "grounded documentation" in agent.justification


def test_export_includes_ticket_columns_and_justification(
    ticket, fake_chroma, fake_model
):
    agent = _agent(
        ticket, fake_chroma, fake_model, issue="issue text", subject="subject text"
    )
    agent.justification = "some reason"

    exported = agent.export(["issue", "subject", "company"])

    assert exported["issue"] == "issue text"
    assert exported["subject"] == "subject text"
    assert exported["justification"] == "some reason"
    assert "response" not in exported
