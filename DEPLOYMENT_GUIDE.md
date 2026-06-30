# EVA Telegram Bot - HuggingFace Spaces Deployment Guide

This guide will help you deploy the EVA Telegram Bot to HuggingFace Spaces.

## Prerequisites

1. **HuggingFace Account**: Create a free account at https://huggingface.co
2. **Telegram Bot Token**: Get your bot token from @BotFather on Telegram
3. **NVIDIA API Keys**: Have your NVIDIA API keys ready (from implementation_guide.txt)
4. **GitHub Account**: (Optional) For connecting your repository

## Step 1: Prepare Your Code

### Option A: Direct Upload to HuggingFace

1. Organize your project structure:
   ```
   eva-telegram-bot/
   ├── agents/
   ├── config/
   ├── core/
   ├── models/
   ├── services/
   ├── utils/
   ├── main.py
   ├── requirements.txt
   ├── Dockerfile
   ├── start.sh
   ├── README_SPACES.md
   └── .env.example
   ```

2. Ensure all files are in place:
   - `Dockerfile` - For container configuration
   - `start.sh` - Startup script
   - `requirements.txt` - Python dependencies
   - `README_SPACES.md` - HuggingFace Spaces README

### Option B: Connect GitHub Repository

1. Push your code to GitHub
2. Ensure all necessary files are committed
3. Make sure `.gitignore` excludes sensitive files

## Step 2: Create HuggingFace Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in the details:
   - **Space name**: e.g., `eva-telegram-bot`
   - **License**: MIT
   - **SDK**: Docker
   - **Visibility**: Public or Private
4. Click "Create Space"

## Step 3: Upload Your Code

### If using Direct Upload:

1. In your Space, click "Files"
2. Upload all files from your project
3. Make sure to upload:
   - All Python files (agents/, config/, core/, models/, services/, utils/)
   - `main.py`
   - `requirements.txt`
   - `Dockerfile`
   - `start.sh`
   - `README_SPACES.md`
   - `.env.example`

### If using GitHub:

1. In your Space settings, click "Connect to GitHub"
2. Select your repository
3. Choose the branch to deploy
4. Click "Connect"

## Step 4: Configure Environment Variables

In your Space settings, go to "Variables and secrets" and add:

### Required Variables:

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from @BotFather
- `NVIDIA_API_KEY_MAIN`: Your main NVIDIA API key
- `NVIDIA_API_KEY_TTS`: Your NVIDIA TTS API key
- `NVIDIA_API_KEY_STT`: Your NVIDIA STT API key
- `PRIMARY_USER_TELEGRAM_ID`: Your Telegram user ID (for primary user access)

### Optional Variables (for full functionality):

- `DATABASE_URL`: Database connection string (defaults to `sqlite:///data/eva.db`)
- `REDIS_URL`: Redis connection string
- `PINECONE_API_KEY`: Pinecone API key for vector storage
- `GOOGLE_CLIENT_ID`: Google OAuth client ID
- `GOOGLE_CLIENT_SECRET`: Google OAuth client secret
- `STRIPE_API_KEY`: Stripe API key for payments
- `SERP_API_KEY`: SerpAPI key for web search

### Server Configuration (already set in Dockerfile):

- `HOST`: `0.0.0.0`
- `PORT`: `7860`
- `DEBUG`: `False`

## Step 5: Build and Deploy

1. After uploading files and setting variables, HuggingFace will automatically build
2. Monitor the build logs in the "Logs" tab
3. Wait for the build to complete (may take 5-10 minutes)
4. Once built, your Space will be available at `https://your-space-name.hf.space`

## Step 6: Set Telegram Webhook

Once your Space is running, set the Telegram webhook:

```bash
curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
  -d "url=https://<your-space-name>.hf.space/webhook/telegram"
```

Replace:
- `<YOUR_BOT_TOKEN>` with your actual bot token
- `<your-space-name>` with your Space name

## Step 7: Test Your Bot

1. Open Telegram and search for your bot
2. Send a message to test it
3. Check the logs in HuggingFace Spaces to see the bot's response

## Troubleshooting

### Build Fails:

- Check the build logs for errors
- Ensure `requirements.txt` is complete
- Verify `Dockerfile` syntax is correct
- Make sure all Python files are properly formatted

### Bot Not Responding:

- Verify `TELEGRAM_BOT_TOKEN` is correct
- Check that the webhook is set correctly
- Review the application logs for errors
- Ensure NVIDIA API keys are valid

### Database Issues:

- The bot uses SQLite by default (`sqlite:///data/eva.db`)
- The database is automatically created in `/app/data/`
- If you need PostgreSQL, set `DATABASE_URL` accordingly

### NVIDIA API Errors:

- Verify your API keys are correct
- Check that the API keys have sufficient credits
- Review NVIDIA API status at https://status.nvidia.com

## Scaling and Performance

- HuggingFace Spaces runs on shared infrastructure
- For better performance, consider upgrading to a paid Space
- Monitor resource usage in the Space settings
- Consider using external services (Redis, PostgreSQL) for production

## Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use environment variables** for all sensitive data
3. **Set your Space to Private** if containing sensitive functionality
4. **Implement rate limiting** for API endpoints
5. **Use HTTPS** for all communications
6. **Regularly rotate API keys**

## Monitoring

- Monitor application logs in the "Logs" tab
- Check resource usage in the "Settings" tab
- Set up external monitoring (e.g., Uptime Robot) for webhook endpoint
- Track API usage and costs in NVIDIA console

## Updating Your Bot

To update your bot after deployment:

1. Make changes to your code
2. Upload new files or push to GitHub
3. HuggingFace will automatically rebuild
4. Wait for the build to complete
5. Test the new version

## Backup and Recovery

- HuggingFace Spaces uses persistent storage for `/app/data/`
- Regularly export your SQLite database if using it
- Consider using external database services for production
- Keep backups of your environment variables configuration

## Cost Considerations

- HuggingFace Spaces free tier has limited resources
- NVIDIA API usage is charged separately
- Consider API rate limits and costs
- Monitor usage to avoid unexpected charges

## Support

For issues specific to:
- **HuggingFace Spaces**: https://huggingface.co/docs/hub/spaces
- **Telegram Bot API**: https://core.telegram.org/bots/api
- **NVIDIA API**: https://ai.nvidia.com

## Next Steps

1. Set up monitoring and alerts
2. Implement proper error handling
3. Add analytics to track usage
4. Consider adding authentication for admin endpoints
5. Set up CI/CD pipeline for automated deployments
6. Document your specific configuration and customizations
