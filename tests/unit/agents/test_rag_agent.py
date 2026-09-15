# tests/unit/agents/test_rag_agent.py
# +---------------------------------------------------------------------------+
# |                          RAG AGENT TESTS                                  |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/agents/test_rag_agent.py -v

# Local Libraries
from agents.rag_agent import RagAgent
from src.enums import Status

DRAFT_MARKER = "AI Support Response Specialist"
VERIFY_MARKER = "AI Quality Assurance Specialist"
PRECISION_MARKER = "AI Support Supervisor"


def _agent(
    model,
    verify_hook=None,
    issue="I forgot my password",
    subject="Password reset help",
):
    return RagAgent(
        {
            "model": model,
            "row_index": 0,
            "issue": issue,
            "subject": subject,
            "status": None,
            "verify_hook": verify_hook,
        }
    )


def test_full_success_marks_grounded_and_precise(fake_model, doc):
    model = fake_model(
        {
            DRAFT_MARKER: {
                "grounded": True,
                "response": "Here is your answer.",
                "cited_chunks": [0],
                "reasoning": "Context covers it.",
            },
            VERIFY_MARKER: {
                "is_grounded": True,
                "reasoning": "citations check out",
            },
            PRECISION_MARKER: {
                "precision_score": 0.95,
                "reasoning": "on topic",
            },
        }
    )
    agent = _agent(model)

    result = agent.grounding([(doc(), 0.2)])

    assert result["grounded"] is True
    assert result["precise"] is True
    assert result["response"] == "Here is your answer."
    assert result["status"] == Status.REPLIED
    # Regression guard: grounding() must return the results dict, not None
    assert result is not None


def test_no_relevant_documents_returns_ungrounded(fake_model):
    model = fake_model({})
    agent = _agent(model)

    # Distance above MAX_RELEVANCE_DISTANCE -> filtered out before any LLM call.
    result = agent.grounding([(object(), 0.95)])

    assert result["grounded"] is False
    assert result["status"] == Status.ESCALATED
    assert model.calls == []


def test_citation_verification_failure_escalates_with_plain_string_reasoning(
    fake_model, doc
):
    model = fake_model(
        {
            DRAFT_MARKER: {
                "grounded": True,
                "response": "answer",
                "cited_chunks": [0],
                "reasoning": "x",
            },
            VERIFY_MARKER: {
                "is_grounded": False,
                "reasoning": "fabricated citation",
            },
        }
    )
    agent = _agent(model)

    result = agent.grounding([(doc(), 0.2)])

    assert result["grounded"] is False
    assert result["status"] == Status.ESCALATED
    # Regression guard: reasoning must be a plain string, not a 1-tuple.
    assert isinstance(result["reasoning"], str)


def test_precision_failure_keeps_grounded_true_but_not_precise(
    fake_model, doc
):
    model = fake_model(
        {
            DRAFT_MARKER: {
                "grounded": True,
                "response": "answer",
                "cited_chunks": [0],
                "reasoning": "x",
            },
            VERIFY_MARKER: {"is_grounded": True, "reasoning": "fine"},
            PRECISION_MARKER: {
                "precision_score": 0.0,
                "reasoning": "off topic",
            },
        }
    )
    agent = _agent(model)

    result = agent.grounding([(doc(), 0.2)])

    assert result["grounded"] is True
    assert result["precise"] is False
    assert result["status"] == Status.ESCALATED


def test_verify_hook_blocks_otherwise_valid_response(fake_model, doc):
    model = fake_model(
        {
            DRAFT_MARKER: {
                "grounded": True,
                "response": "I can increase your score.",
                "cited_chunks": [0],
                "reasoning": "x",
            },
            VERIFY_MARKER: {"is_grounded": True, "reasoning": "citations fine"},
        }
    )
    hook_calls = []

    def blocking_hook(draft):
        hook_calls.append(draft)
        return False

    agent = _agent(model, verify_hook=blocking_hook)

    result = agent.grounding([(doc(), 0.2)])

    assert result["grounded"] is False
    assert result["status"] == Status.ESCALATED
    assert hook_calls, "verify_hook was never invoked"
    assert "compliance" in result["reasoning"].lower()


def test_verify_hook_not_called_when_citations_already_failed(
    fake_model, doc
):
    model = fake_model(
        {
            DRAFT_MARKER: {
                "grounded": True,
                "response": "answer",
                "cited_chunks": [0],
                "reasoning": "x",
            },
            VERIFY_MARKER: {"is_grounded": False, "reasoning": "bad citation"},
        }
    )
    hook_calls = []
    agent = _agent(
        model, verify_hook=lambda draft: hook_calls.append(draft) or True
    )

    agent.grounding([(doc(), 0.2)])

    assert hook_calls == [], "hook must not run once citations already failed"


def test_format_documents_xml_wraps_content(doc):
    agent = _agent(model=None)
    documents = [doc(content="Some info", chunk_idx=3)]

    xml = agent._format_documents_xml(documents)

    assert isinstance(xml, str)  # regression guard: must not return None
    assert xml.startswith("<retrieved_context_documents>")
    assert xml.endswith("</retrieved_context_documents>")
    assert "id='3'" in xml
    assert "Some info" in xml


def test_format_documents_xml_empty_list_returns_empty_string():
    agent = _agent(model=None)
    assert agent._format_documents_xml([]) == ""
