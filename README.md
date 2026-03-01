# ghostfolio-agent

AI-powered financial assistant for [Ghostfolio](https://ghostfol.io) portfolios. Ask questions about your holdings, performance, taxes, and compliance from the terminal.

## Quick Start

```bash
pip install ghostfolio-agent
```

You need two things:

1. **OpenRouter API key** — sign up at [openrouter.ai](https://openrouter.ai), go to Keys, create one
2. **Ghostfolio Security Token** — in your Ghostfolio instance, go to Settings and copy your Security Token

Set them up (pick one method):

### Option A: Environment variables

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export GHOSTFOLIO_ACCESS_TOKEN=your-security-token
```

### Option B: `.env` file

Create a `.env` file in your working directory:

```env
OPENROUTER_API_KEY=sk-or-v1-...
GHOSTFOLIO_ACCESS_TOKEN=your-security-token
```

### Option C: CLI flags

```bash
ghostfolio-agent --api-key sk-or-v1-... --token your-security-token
```

Then run:

```bash
ghostfolio-agent
```

## Example Session

```
You: Show my portfolio holdings
  Tools used: portfolio_analysis

| Symbol | Name           | Shares | Value     | Weight |
|--------|----------------|--------|-----------|--------|
| VTI    | Vanguard Total | 100    | $25,430   | 45.2%  |
| VXUS   | Vanguard Intl  | 80     | $15,200   | 27.0%  |
| BND    | Vanguard Bond  | 150    | $15,600   | 27.8%  |

  Confidence: 95% (high)

You: What's my sector exposure?
  Tools used: portfolio_analysis

Your portfolio is heavily weighted toward technology (38%) and healthcare (15%)...
```

## What It Can Do

- **Portfolio analysis** — holdings, allocations, unrealized gains/losses, sector/country exposure
- **Market data** — current prices, daily changes, symbol lookup
- **Transaction history** — trade log, dividend history, recent activity
- **Tax estimates** — realized gains/losses, capital gains, dividend income
- **Compliance checks** — portfolio health rules, concentration risk, diversification
- **Financial math** — percentages, projections, share calculations

## Prerequisites

1. A running [Ghostfolio](https://github.com/ghostfolio/ghostfolio) instance (self-hosted)
2. An [OpenRouter](https://openrouter.ai) API key for LLM access

## CLI Options

```
ghostfolio-agent [options]

Options:
  --api-key KEY     OpenRouter API key (or set OPENROUTER_API_KEY)
  --token TOKEN     Ghostfolio security token (or set GHOSTFOLIO_ACCESS_TOKEN)
  --url URL         Ghostfolio API URL (default: http://localhost:3333)
  --model MODEL     LLM model ID (default: anthropic/claude-haiku-4.5)
  --timeout SECS    Max seconds per request (default: 180)
  -v, --verbose     Show debug logging (LLM calls, tool execution)
```

## Configuration Reference

All settings can be provided via environment variables, a `.env` file, or CLI flags:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter API key |
| `GHOSTFOLIO_ACCESS_TOKEN` | Yes* | — | Security token from Ghostfolio Settings |
| `GHOSTFOLIO_JWT` | No | — | Direct JWT (advanced, skips token exchange) |
| `GHOSTFOLIO_API_URL` | No | `http://localhost:3333` | Ghostfolio API URL |
| `LLM_MODEL` | No | `anthropic/claude-haiku-4.5` | OpenRouter model ID |
| `MAX_TOOL_STEPS` | No | `5` | Max tool calls per turn |
| `MAX_RESPONSE_SECONDS` | No | `180` | Agent timeout in seconds |
| `LANGSMITH_API_KEY` | No | — | LangSmith tracing (optional) |
| `LANGSMITH_PROJECT` | No | — | LangSmith project name |

\* Either `GHOSTFOLIO_ACCESS_TOKEN` or `GHOSTFOLIO_JWT` is required. Most users should use `GHOSTFOLIO_ACCESS_TOKEN` — the package automatically exchanges it for a JWT.

## Server Mode

For programmatic access, run as a FastAPI server:

```bash
pip install "ghostfolio-agent[server]"
python -m ghostfolio_agent.server
```

Endpoints:
- `POST /chat` — send a message, get a response
- `POST /chat/stream` — Server-Sent Events streaming
- `GET /health` — health check

## How Authentication Works

You provide your **Ghostfolio Security Token** (found in Ghostfolio Settings). On startup, the agent automatically exchanges it for a short-lived JWT by calling your Ghostfolio instance's auth endpoint. You never need to manually handle JWTs.

## Development

```bash
git clone https://github.com/christensenca/ghostfolio-agent
cd ghostfolio-agent
pip install -e ".[dev,server]"
pytest
```

## License

AGPL-3.0 — see [LICENSE](LICENSE)
