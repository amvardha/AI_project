# Adithya Vardhan 32956089
"""
Script to run MCP servers as standalone processes.
This allows agents to connect as clients to running MCP servers.
"""
import os
import sys
import signal
import asyncio
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add MCPs directory to path
mcps_dir = Path(__file__).parent / "MCPs"
sys.path.insert(0, str(mcps_dir))

def run_orchestrator_server():
    """Run the orchestrator MCP server."""
    print("Starting Orchestrator MCP Server...", file=sys.stderr)
    from orchestrator_mcp import mcp as orchestrator_mcp
    orchestrator_mcp.run(transport="stdio")

def run_file_server():
    """Run the file MCP server."""
    print("Starting File MCP Server...", file=sys.stderr)
    from files_mcp import mcp as file_mcp
    file_mcp.run(transport="stdio")

def main():
    """Main entry point to run MCP servers."""
    # Determine which server to run based on command line argument
    server_type = None
    if len(sys.argv) == 2:
        server_type = sys.argv[1].lower()
    elif len(sys.argv) > 2:
        print("Usage: python run_mcp_servers.py [orchestrator|file]", file=sys.stderr)
        sys.exit(1)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down...", file=sys.stderr)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if server_type == "orchestrator":
            run_orchestrator_server()
        elif server_type == "file":
            run_file_server()
        else:
            print("Usage: python run_mcp_servers.py [orchestrator|file]", file=sys.stderr)
            print("Note: Run each server in a separate terminal/process", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error running MCP server: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

