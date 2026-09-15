# tests/unit/pipelines/test_process_tickets.py
# +---------------------------------------------------------------------------+
# |                 TICKET PROCESSING PIPELINE TESTS                          |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/pipelines/test_process_tickets.py -v
#
# SupportAgentModel and TicketAnalyzer are mocked — this pipeline's own job
# is iterating tickets and wiring collaborators together, not re-deriving
# grounding/classification behavior, which is already covered under
# tests/unit/agents/.

# Vendor Libraries
import pandas as pd

# Local Libraries
from pipelines.process_tickets import run_process_tickets_pipeline


def _tickets_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"Issue": [f"issue {i}" for i in range(n)]})


def test_processes_every_ticket_in_order(mocker):
    mocker.patch("pipelines.process_tickets.SupportAgentModel")
    mock_analyzer_cls = mocker.patch("pipelines.process_tickets.TicketAnalyzer")
    mock_analyzer = mock_analyzer_cls.return_value
    mock_analyzer.process.side_effect = lambda ticket: {"row": ticket.Index}

    result = run_process_tickets_pipeline(
        {}, {"support_tickets": _tickets_df(3)}, chroma_model=mocker.MagicMock()
    )

    assert result == [{"row": 0}, {"row": 1}, {"row": 2}]
    assert mock_analyzer.process.call_count == 3


def test_constructs_support_agent_model_with_row_count(mocker):
    mock_model_cls = mocker.patch("pipelines.process_tickets.SupportAgentModel")
    mocker.patch("pipelines.process_tickets.TicketAnalyzer")

    run_process_tickets_pipeline(
        {}, {"support_tickets": _tickets_df(2)}, chroma_model=mocker.MagicMock()
    )

    mock_model_cls.assert_called_once_with(2)


def test_analyzer_wired_with_chroma_model_and_support_agent_model(mocker):
    mock_model_cls = mocker.patch("pipelines.process_tickets.SupportAgentModel")
    mock_analyzer_cls = mocker.patch("pipelines.process_tickets.TicketAnalyzer")
    chroma = mocker.MagicMock()

    run_process_tickets_pipeline(
        {}, {"support_tickets": _tickets_df(1)}, chroma_model=chroma
    )

    mock_analyzer_cls.assert_called_once_with(
        {"chroma_model": chroma, "model": mock_model_cls.return_value}
    )


def test_empty_dataset_returns_empty_list_without_calling_analyzer(mocker):
    mocker.patch("pipelines.process_tickets.SupportAgentModel")
    mock_analyzer_cls = mocker.patch("pipelines.process_tickets.TicketAnalyzer")

    result = run_process_tickets_pipeline(
        {}, {"support_tickets": _tickets_df(0)}, chroma_model=mocker.MagicMock()
    )

    assert result == []
    mock_analyzer_cls.return_value.process.assert_not_called()
