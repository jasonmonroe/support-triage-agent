# tests/unit/src/test_ticket_analyzer.py
# +---------------------------------------------------------------------------+
# |                        TICKET ANALYZER TESTS                              |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/src/test_ticket_analyzer.py -v
#
# These drive the real SupportAgent/RagAgent grounding pipeline (via the
# same ticket/fake_chroma/fake_model fixtures used under tests/unit/agents/)
# rather than mocking TicketAnalyzer's collaborators, since orchestrating
# that pipeline correctly is exactly what TicketAnalyzer is for.

# Vendor Libraries
import pytest

# Local Libraries
from agents.claude_agent import ClaudeAgent
from agents.hackerrank_agent import HackerrankAgent
from agents.support_agent import SupportAgent
from agents.visa_agent import VisaAgent
from src.ticket_analyzer import TicketAnalyzer

DRAFT_MARKER = "AI Support Response Specialist"
VERIFY_MARKER = "AI Quality Assurance Specialist"
PRECISION_MARKER = "AI Support Supervisor"
FINAL_MARKER = "SUPPORT TICKET DATA FOR ANALYSIS"


def test_hard_escalation_skips_retrieval_and_grounding(
    ticket, fake_chroma, fake_model
):
    chroma = fake_chroma()
    model = fake_model(
        {
            FINAL_MARKER: {
                "status": "Escalated",
                "response": "escalated",
                "justification": "critical risk",
            }
        }
    )
    row = ticket(
        issue="There has been a data leak in our system", company="None"
    )
    analyzer = TicketAnalyzer({"chroma_model": chroma, "model": model})

    result = analyzer.process(row)

    assert chroma.queries == []  # retrieval never ran
    assert result["status"] == "Escalated"
    assert result["response"] == "escalated"


def test_normal_path_runs_full_grounding_pipeline(
    ticket, fake_chroma, fake_model, doc
):
    chroma = fake_chroma(documents=[(doc(), 0.2)])
    model = fake_model(
        {
            DRAFT_MARKER: {
                "grounded": True,
                "response": "Reset it in Settings.",
                "cited_chunks": [0],
                "reasoning": "covers it",
            },
            VERIFY_MARKER: {"is_grounded": True, "reasoning": "fine"},
            PRECISION_MARKER: {
                "precision_score": 0.95,
                "reasoning": "on topic",
            },
            FINAL_MARKER: {
                "status": "Replied",
                "product_area": "account_access",
                "request_type": "product_issue",
                "response": "final answer",
                "justification": "final justification",
            },
        }
    )
    row = ticket(
        issue="I forgot my password and need to reset it",
        subject="Password reset",
    )
    analyzer = TicketAnalyzer({"chroma_model": chroma, "model": model})

    result = analyzer.process(row)

    assert len(chroma.queries) == 1
    assert result["status"] == "Replied"
    assert result["response"] == "final answer"
    assert result["justification"] == "final justification"


def test_get_agent_routes_by_company(ticket, fake_chroma, fake_model):
    analyzer = TicketAnalyzer(
        {"chroma_model": fake_chroma(), "model": fake_model()}
    )

    assert isinstance(
        analyzer._get_agent(ticket(company="HackerRank")), HackerrankAgent
    )
    assert isinstance(
        analyzer._get_agent(ticket(company="Claude")), ClaudeAgent
    )
    assert isinstance(analyzer._get_agent(ticket(company="Visa")), VisaAgent)
    assert isinstance(
        analyzer._get_agent(ticket(company="None")), SupportAgent
    )


def test_get_agent_raises_for_unsupported_company(
    ticket, fake_chroma, fake_model
):
    analyzer = TicketAnalyzer(
        {"chroma_model": fake_chroma(), "model": fake_model()}
    )

    with pytest.raises(ValueError):
        analyzer._get_agent(ticket(company="SomeUnknownCompany"))


def test_format_output_overrides_ticket_values_with_response(
    ticket, fake_chroma, fake_model
):
    analyzer = TicketAnalyzer(
        {"chroma_model": fake_chroma(), "model": fake_model()}
    )
    analyzer._ticket = {
        "issue": "original issue",
        "status": "",
        "response": "",
    }

    output = analyzer._format_output(
        {"status": "Replied", "response": "answer"}
    )

    assert output == {
        "issue": "original issue",
        "status": "Replied",
        "response": "answer",
    }


def test_format_output_keeps_ticket_value_when_key_absent_from_response(
    ticket, fake_chroma, fake_model
):
    analyzer = TicketAnalyzer(
        {"chroma_model": fake_chroma(), "model": fake_model()}
    )
    analyzer._ticket = {"issue": "original issue", "status": "Replied"}

    output = analyzer._format_output({"status": "Escalated"})

    assert output["issue"] == "original issue"
    assert output["status"] == "Escalated"
