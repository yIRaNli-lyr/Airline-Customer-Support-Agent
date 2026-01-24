#!/usr/bin/env python3
"""
CLI Interface - Interactive command-line interface for the agent.

Usage:
    agent-cli [MCP_SERVER_URL]...
    agent-cli --rag [MCP_SERVER_URL]...

Examples:
    agent-cli http://localhost:3000/mcp
    agent-cli --rag http://localhost:3000/mcp
"""

import argparse
import sys

from .agent import ToolCallingAgent
from .rag import register_query_policy_tool
from .tool_manager import ToolManager


def main():
    """Main CLI entrypoint"""
    ap = argparse.ArgumentParser(description="Agent CLI")
    ap.add_argument("--rag", action="store_true", help="Use RAG for policy (query_policy); require ingest")
    ap.add_argument("mcp_servers", nargs="*", help="MCP server URL(s), e.g. http://localhost:3000/mcp")
    args = ap.parse_args()
    mcp_servers = args.mcp_servers

    tool_manager = ToolManager.from_servers(mcp_servers)

    if args.rag:
        register_query_policy_tool(tool_manager)
        print("   📚 RAG enabled: query_policy registered\n")

    if not mcp_servers and not args.rag:
        print("Usage: agent-cli [--rag] [MCP_SERVER_URL]...")
        print("Example: agent-cli --rag http://localhost:3000/mcp\n")

    agent = ToolCallingAgent(tool_manager, use_rag=args.rag)

    print("\n" + "=" * 60)
    print("🤖 Agent ready! Type 'quit' or 'exit' to stop." + (" [RAG]" if args.rag else ""))
    print("=" * 60)

    try:
        while True:
            try:
                user_input = input('\n💬 You: ').strip()

                # Check for exit commands
                if not user_input or user_input.lower() in ['quit', 'exit']:
                    print('\n👋 Goodbye!\n')
                    break

                # Execute the user's task
                try:
                    response = agent.execute(user_input)

                    if response:
                        print(f"\n🤖 Agent: {response}\n")
                    else:
                        print('\n🤖 Agent: (no response)\n')
                except Exception as agent_error:
                    print(f"\n❌ Agent error: {agent_error}\n")
                    raise

            except KeyboardInterrupt:
                print('\n\n👋 Goodbye!\n')
                break
            except Exception as e:
                print(f'\n❌ Error: {e}\n')

    finally:
        tool_manager.disconnect()


if __name__ == '__main__':
    main()
