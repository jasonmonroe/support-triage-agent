# tests/unit/agents/test_claude_agent.py
# +---------------------------------------------------------------------------+
# |                        CLAUDE AGENT TESTS                                 |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/agents/test_claude_agent.py -v

# Local Libraries
from agents.claude_agent import ClaudeAgent


def test_no_extra_verify_hook_by_default(ticket, fake_chroma, fake_model):
    row = ticket(issue="Some Claude issue", company="Claude")
    agent = ClaudeAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=fake_chroma(),
        support_agent_model=fake_model(),
    )

    # ClaudeAgent doesn't override _get_verify_hook(), so it inherits the
    # base "no extra compliance check" behavior — unlike HackerRank/Visa.
    assert agent._get_verify_hook() is None


def test_query_scopes_search_to_claude_company(ticket, fake_chroma, fake_model):
    chroma = fake_chroma()
    row = ticket(issue="Some issue", company="Claude")
    agent = ClaudeAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=chroma,
        support_agent_model=fake_model(),
    )

    agent._query("query text")

    assert chroma.queries[-1]["company"] == "Claude"


def test_title_and_company_set(ticket, fake_chroma, fake_model):
    row = ticket(issue="issue", company="Claude")
    agent = ClaudeAgent(
        row_index=row.Index,
        ticket_df=row,
        chroma_model=fake_chroma(),
        support_agent_model=fake_model(),
    )

    assert agent.title == "🤖 Claude Agent"
    assert agent.company == "Claude"
