"""Standalone MCP server for TriageForge. Fully isolated: does not import
from, modify, or depend on anything in /app. Runs as its own process,
independent of the FastAPI backend.

Run with: python -m mcp_server.server
"""

from mcp.server.fastmcp import FastMCP

from mcp_server.mock_data import lookup_account

mcp = FastMCP("triageforge-account-lookup")


@mcp.tool()
def check_account_status(account_id: str) -> dict:
    """Look up an account's plan tier, standing, and open disputes (mock CRM/billing lookup)."""
    return lookup_account(account_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
