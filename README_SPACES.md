---
title: EVA Telegram Bot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# EVA Telegram Bot

A fully agentic Telegram bot with dynamic personality adaptation and autonomous task execution.

## Features

- **Dynamic Personality Adaptation**: 7 personality types (Adviser, Therapist, Friend, Business Partner, Mentor, Course Mate, General)
- **Autonomous Task Execution**: Email, Calendar, Payments, Documents, Reminders, Web Search, Image Generation
- **Smart Memory System**: Multi-layer memory (short-term + long-term + semantic)
- **NVIDIA Model Orchestration**: Uses NVIDIA's advanced AI models for intelligent decision-making
- **Primary User vs Stranger Mode**: Different access levels for security

## Quick Start

### Environment Variables

Configure these secrets in your HuggingFace Space settings:

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from @BotFather
- `NVIDIA_API_KEY_MAIN`: Main NVIDIA API key
- `NVIDIA_API_KEY_TTS`: NVIDIA TTS API key
- `NVIDIA_API_KEY_STT`: NVIDIA STT API key
- `DATABASE_URL`: Database connection string (default: sqlite:///data/eva.db)
- `REDIS_URL`: Redis connection string (optional)
- `PINECONE_API_KEY`: Pinecone API key for vector storage (optional)
- `GOOGLE_CLIENT_ID`: Google OAuth client ID (optional)
- `GOOGLE_CLIENT_SECRET`: Google OAuth client secret (optional)
- `STRIPE_API_KEY`: Stripe API key (optional)
- `SERP_API_KEY`: SerpAPI key for web search (optional)
- `PRIMARY_USER_TELEGRAM_ID`: Your Telegram ID for primary user access

### Deployment

1. Create a new HuggingFace Space with Docker runtime
2. Upload your code or connect to your GitHub repository
3. Set the environment variables in the Space settings
4. Wait for the build to complete
5. Your bot will be accessible at the Space URL

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /webhook/telegram` - Telegram webhook endpoint
- `GET /` - Root endpoint with API info

## Setting up Telegram Webhook

Once your Space is running, set the Telegram webhook:

```bash
curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
  -d "url=https://<your-space-name>.hf.space/webhook/telegram"
```

## Architecture

```
┌─────────────────┐
│  Telegram Bot   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI App   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Orchestrator   │ ◄─── NVIDIA AI Models
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Manager  │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┬────────┐
    ▼         ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Email │ │Cal   │ │Pay   │ │Doc   │ │Search│
│Agent │ │Agent │ │Agent │ │Agent │ │Agent │
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘
```

## Development

For local development:

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your credentials

# Initialize database
python init_db.py

# Run the bot
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## License

MIT License

## Support

For issues or questions, please open an issue in the repository.
