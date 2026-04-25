"""
A simple terminal chat with Claude, using your Claude.ai subscription
(via the Claude Agent SDK + an OAuth token from `claude setup-token`).

Tools enabled: Read (local files), WebSearch, WebFetch.

One-time setup:
    npm install -g @anthropic-ai/claude-code
    claude setup-token            # paste the printed sk-ant-oat01-... below
    pip install claude-agent-sdk python-dotenv
"""

import os
import sys
import anyio
from dotenv import load_dotenv

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
)

# Loads CLAUDE_CODE_OAUTH_TOKEN from a .env file in the project root.
load_dotenv()
CLAUDE_CODE_OAUTH_TOKEN = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")

# Claude model: Sonnet 4.6 (https://docs.anthropic.com/en/docs/about-claude/models)
CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a friendly, helpful assistant in a simple terminal chat. "
    "You have three tools available:\n"
    "- WebSearch: use this when the user asks about current events, specific "
    "places, recent things, or anything you don't know off the top of your head.\n"
    "- WebFetch: use this to read the full content of a web page (often a "
    "follow-up to WebSearch when a snippet isn't enough).\n"
    "- Read: use this when the user mentions a local file path and wants you "
    "to read it.\n"
    "When you use a tool, briefly tell the user what you're doing "
    "(e.g. 'Let me look that up...') so they know why you're pausing. "
    "Keep answers concise unless the user asks for more detail."
)


def _wants_end(user: str) -> bool:
    u = user.lower().strip()
    return u in ("quit", "exit", "bye", "q")


def _print_tool_use(block: ToolUseBlock) -> None:
    """Render a small note so the user can see why the bot is pausing."""
    name = block.name
    inp = block.input or {}
    if name == "WebSearch":
        icon, detail = "🔍 searching", inp.get("query", "")
    elif name == "WebFetch":
        icon, detail = "🌐 reading", inp.get("url", "")
    elif name == "Read":
        icon, detail = "📄 opening", inp.get("file_path", "")
    else:
        icon, detail = f"🔧 {name}", str(inp)[:60]
    print(f"  ({icon}: {detail})")

async def _collect_reply(client: ClaudeSDKClient) -> str:
    """Stream the reply token-by-token to stdout, return the full text."""
    parts: list[str] = []
    async for msg in client.receive_response():
        if isinstance(msg, StreamEvent):
            # Token-level text deltas — print as they arrive.
            evt = msg.event
            if evt.get("type") == "content_block_delta":
                delta = evt.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunk = delta.get("text", "")
                    parts.append(chunk)
                    print(chunk, end="", flush=True)
        elif isinstance(msg, AssistantMessage):
            # Text was already streamed above — here we only watch for
            # tool calls so we can show "(🔍 searching: ...)" notes.
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    print()  # break out of the current streaming line
                    _print_tool_use(block)
    print()  # final newline after the streamed text
    return "".join(parts).strip()


async def chat_with_claude(client: ClaudeSDKClient, user: str) -> str:
    """Send one user message and return Claude's reply.

    The open `client` keeps conversation context across calls,
    so we don't need to manage history manually.
    """
    try:
        await client.query(user)
        return await _collect_reply(client)
    except Exception as e:
        return f"[error: {e}]"


async def main() -> None:
    if not CLAUDE_CODE_OAUTH_TOKEN.strip():
        print(
            "Set CLAUDE_CODE_OAUTH_TOKEN in a .env file "
            "(run `claude setup-token` to generate one), then run again.",
            file=sys.stderr,
        )
        sys.exit(1)

    # The Agent SDK reads this env var to authenticate against your Claude
    # subscription instead of a Console API key.
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = CLAUDE_CODE_OAUTH_TOKEN.strip()

    options = ClaudeAgentOptions(
        model=CLAUDE_MODEL,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Read", "WebSearch", "WebFetch"],
        max_turns=5,            # cap the agentic loop so the bot can't spiral
        max_budget_usd=1.0,     # hard per-session spending cap (safety net)
        include_partial_messages=True, 
    )

    print(
        f"You're chatting with Claude ({CLAUDE_MODEL}). "
        "Type a message, or 'quit' to stop.\n"
    )

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                line = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if _wants_end(line):
                print("Bot: Alright, talk later!\n")
                break

            user_text = line.strip()
            if not user_text:
                print("Bot: I didn't catch that — what did you want to say?\n")
                continue

            print()
            print("-" * 30)
            print()
            print("Bot: ", end="", flush=True)        # 前缀先打
            await chat_with_claude(client, user_text)  # 流式打印发生在这里面
            print()  


if __name__ == "__main__":
    anyio.run(main)
