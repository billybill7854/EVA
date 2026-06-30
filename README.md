# README for EVA Telegram Bot

## Overview
EVA is a fully agentic Telegram bot with:
- Dynamic personality adaptation (adviser, therapist, friend, mentor, business partner, course mate)
- Autonomous task execution (emails, calendar, payments, documents, reminders, web search, image generation)
- Smart multi-layer memory system (short-term + long-term + semantic)
- NVIDIA model orchestration for intelligent decision-making
- Primary user vs. stranger mode

## Quick Start

### 1. Installation

```bash
cd eva-telegram-bot

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from @BotFather
- `TELEGRAM_WEBHOOK_URL`: Your public webhook URL
- `NVIDIA_API_KEY_*`: Your NVIDIA API keys
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `PRIMARY_USER_TELEGRAM_ID`: Your Telegram ID (for primary user access)

### 3. Database Setup

```bash
# PostgreSQL should be running and accessible via DATABASE_URL
# Tables will be created automatically on first run
```

### 4. Run the Bot

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The bot will start on `http://localhost:8000`

## API Endpoints

### Public Endpoints
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /webhook/telegram` - Telegram webhook (receives messages)

### Admin Endpoints (add authentication in production)
- `POST /admin/set-personality?user_id=X&personality=Y` - Set user personality
- `GET /admin/user-memory/{user_id}` - View user's memories

## Project Structure

```
eva-telegram-bot/
├── config/
│   ├── settings.py - Environment configuration
│   ├── models_config.py - NVIDIA model mappings
│   └── prompts.yaml - Personality prompts
├── core/
│   ├── orchestrator.py - Main decision-making layer
│   ├── personality_engine.py - Personality detection & adaptation
│   ├── memory_manager.py - Multi-layer memory system
│   └── tool_registry.py - Tool discovery & execution
├── services/
│   ├── nvidia_service.py - NVIDIA API calls
│   ├── telegram_service.py - Telegram bot interactions
│   └── storage_service.py - Database, Redis, Pinecone
├── models/
│   └── database.py - SQLAlchemy models
├── agents/
│   └── (base agents - to be implemented)
├── utils/
│   └── (utility functions)
├── main.py - FastAPI app entry point
├── requirements.txt - Python dependencies
└── .env.example - Environment template
```

## How It Works

1. **Message Reception**: Telegram webhook receives user message
2. **User Identification**: Checks if primary user or stranger
3. **Personality Detection**: Analyzes message to detect personality type
4. **Orchestration**: Nemotron-3-Super-120B decides what tools/actions needed
5. **Memory Retrieval**: Fetches relevant context from long/short-term memory
6. **Response Generation**: Generates response in detected personality
7. **Autonomous Execution** (if primary user):
   - Executes tools (send email, schedule calendar, etc.)
   - Sends notifications to user
   - Logs tool calls
8. **Memory Storage**: Stores conversation and learns from interactions

## Personality Types

- **Adviser**: Strategic consultation, business recommendations
- **Therapist**: Emotional support, empathetic listening
- **Friend**: Casual conversation, genuine companionship
- **Business Partner**: Results-driven, professional partnership
- **Mentor**: Guiding, teaching, encouraging growth
- **Course Mate**: Collaborative learning, academic support
- **General**: Balanced, adaptive responses

## Autonomous Tools (Primary User Only)

1. **Email** - Send emails via Gmail
2. **Calendar** - Schedule meetings via Google Calendar
3. **Payment** - Transfer money via Stripe/Wise
4. **Document** - Create/edit documents via Google Drive
5. **Reminder** - Set reminders and notifications
6. **Search** - Search the web for information
7. **Image** - Generate images with FLUX.1-dev

## Deployment

### Using Railway or Render

1. Push code to GitHub
2. Connect repo to Railway/Render
3. Set environment variables
4. Set Telegram webhook to: `https://your-app-url.railway.app/webhook/telegram`
5. Deploy

## Security Notes

- In production, add authentication for admin endpoints
- Use environment variables for all secrets
- Validate all user inputs
- Use HTTPS for webhook
- Implement rate limiting
- Add proper logging and monitoring

## Next Steps

1. Implement individual tool agents (EmailAgent, CalendarAgent, etc.)
2. Add WhatsApp integration
3. Implement voice call handling via TTS/STT
4. Add more sophisticated memory analysis
5. Implement user preferences learning
6. Add analytics dashboard
7. Implement multi-language support

## Support

For issues or questions, check the logs and refer to the NVIDIA API documentation and Telegram Bot API docs.
