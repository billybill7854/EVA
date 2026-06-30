# EVA Autonomous System Guide

EVA is now a fully autonomous AI entity that can think, plan, and execute actions independently without user initiation. This guide explains how EVA's autonomous system works and how to control it.

## Overview

EVA's autonomous system allows her to:
- **Think independently**: Analyze situations and decide on actions without being asked
- **Execute proactively**: Take actions like sending emails, checking on users, gathering information
- **Schedule intelligently**: Plan actions for optimal times
- **Learn and adapt**: Use memory to improve autonomous decisions over time
- **Communicate autonomously**: Reach out to users when appropriate

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Autonomous Decision Engine                    │
│  - Analyzes user data and patterns                       │
│  - Identifies opportunities for proactive action          │
│  - Prioritizes and schedules autonomous decisions       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Background Scheduler                        │
│  - Runs periodic thinking cycles                        │
│  - Executes pending autonomous decisions                │
│  - Manages timing and priorities                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Agent Manager                            │
│  - Executes autonomous actions via agents               │
│  - Email, Calendar, Payment, Search, etc.               │
└─────────────────────────────────────────────────────────┘
```

## Autonomous Action Types

### 1. User Check
**When**: Morning (9 AM) and Evening (6 PM) if user hasn't been active
**What**: EVA sends a message checking on the user
**Example**: "Good morning! How are you doing today?"

### 2. Proactive Email
**When**: EVA identifies an opportunity to send an email
**What**: Sends email autonomously
**Example**: Following up on a meeting, sending information user might need

### 3. Scheduled Reminder
**When**: A reminder becomes due
**What**: Sends the reminder message
**Example**: "Reminder: Your meeting with John is in 30 minutes"

### 4. Information Gathering
**When**: User has shown interest in a topic and it's been a while since update
**What**: Searches for and gathers relevant information
**Example**: User interested in AI trends, EVA searches for latest news

### 5. Conversation Initiation
**When**: EVA notices patterns suggesting user might want to talk
**What**: Initiates conversation proactively
**Example**: "I noticed you've been working late. How are things going?"

### 6. Task Execution
**When**: Tasks are scheduled or become urgent
**What**: Executes tasks without user prompt
**Example**: Sending scheduled emails, updating databases

## How EVA Thinks

### The Thinking Process

Every 5 minutes, EVA runs a "thinking cycle":

1. **Scheduled Actions Check**
   - Is it time for morning/evening user checks?
   - Are any reminders due?
   - Are any scheduled tasks pending?

2. **Proactive Analysis**
   - Analyze user's recent activities and patterns
   - Look for opportunities to be helpful
   - Consider user's interests and preferences

3. **AI-Powered Decision Making**
   - Uses NVIDIA models to evaluate potential actions
   - Prioritizes based on importance and urgency
   - Considers user's current context and mood

4. **Decision Queue**
   - High-priority decisions are executed immediately
   - Lower-priority decisions are scheduled for optimal times
   - User is notified of autonomous actions taken

### Example Autonomous Decision Flow

```
User hasn't been active for 12 hours
    ↓
EVA notices pattern: User usually active in morning
    ↓
Current time: 9 AM
    ↓
Decision: Send morning check message
    ↓
AI evaluates: Is this appropriate? Yes
    ↓
Priority: High (7/10)
    ↓
Execution: Send message "Good morning! How are you?"
    ↓
User notification: "🤖 I took the initiative to check on you"
```

## Configuration

### Environment Variables

```bash
# Enable/disable autonomous features
AUTONOMOUS_ENABLED=true

# Thinking frequency (seconds)
AUTONOMOUS_THINKING_INTERVAL=300

# User check times (24-hour format)
MORNING_CHECK_TIME=09:00
EVENING_CHECK_TIME=18:00

# Autonomous decision threshold
AUTONOMOUS_CONFIDENCE_THRESHOLD=0.7
```

### Admin API Key

For security, admin endpoints require an API key:

```bash
ADMIN_API_KEY=your_secure_admin_api_key_here
```

## Admin Endpoints

### Get Autonomous Status
```bash
GET /admin/autonomous-status
Headers: x-admin-token: your_api_key
```

Returns:
- Whether scheduler is running
- Number of pending decisions
- Action history count
- Last thinking time

### Trigger Autonomous Thinking
```bash
POST /admin/trigger-autonomous-think
Headers: x-admin-token: your_api_key
```

Manually triggers EVA's thinking process and returns decisions made.

### Schedule Custom Action
```bash
GET /admin/schedule-custom-action?action_type=user_check&action_data={"message":"test"}&user_id=1&hours_delay=2
Headers: x-admin-token: your_api_key
```

Schedules a custom autonomous action for a specific time.

## Controlling Autonomy

### Disable Autonomous Features

If you want to disable EVA's autonomous behavior:

1. **Environment Variable**:
   ```bash
   AUTONOMOUS_ENABLED=false
   ```

2. **Admin Endpoint**:
   ```bash
   POST /admin/disable-autonomous
   Headers: x-admin-token: your_api_key
   ```

### Adjust Thinking Frequency

Change how often EVA thinks:

```bash
AUTONOMOUS_THINKING_INTERVAL=600  # Every 10 minutes instead of 5
```

### Set User-Specific Preferences

You can configure autonomous behavior per user:

```python
# Disable morning checks for specific user
await storage_service.set_user_preference(
    user_id=1,
    preference="morning_check",
    value=False
)
```

## Safety and Limits

### Built-in Safety Measures

1. **Primary User Only**: Autonomous actions only affect primary users
2. **Confirmation Required**: High-impact actions require user confirmation
3. **Rate Limiting**: Prevents overwhelming users with messages
4. **Time Windows**: Autonomous actions only during reasonable hours
5. **Priority System**: Important actions take precedence

### What EVA Won't Do Autonomously

- Send payments without explicit authorization
- Delete user data
- Share private information
- Execute actions during sleeping hours (unless urgent)
- Make high-stakes decisions without confirmation

## Monitoring and Logs

### View Autonomous Actions

```bash
GET /admin/autonomous-history
Headers: x-admin-token: your_api_key
```

### Check Decision Queue

```bash
GET /admin/decision-queue
Headers: x-admin-token: your_api_key
```

### View Thinking Logs

EVA's thinking process is logged with detailed information:

```
INFO: EVA is thinking about autonomous actions...
INFO: Considering scheduled actions...
INFO: Morning check - user john hasn't been active recently
INFO: Evening check - user jane hasn't been active recently
INFO: EVA decided on 2 autonomous actions
```

## Examples

### Example 1: Proactive Meeting Follow-up

**Scenario**: User had a meeting yesterday at 3 PM

**EVA's Autonomous Action**:
1. **Thinking**: Notices meeting in conversation history
2. **Decision**: Follow up today at 10 AM
3. **Action**: Sends message "How did your meeting with John go yesterday?"
4. **Result**: User appreciates the follow-up

### Example 2: Information Update

**Scenario**: User interested in AI news, last update 3 days ago

**EVA's Autonomous Action**:
1. **Thinking**: Identifies user interest in AI
2. **Decision**: Search for latest AI news
3. **Action**: Uses search agent to gather information
4. **Action**: Sends message "Here are the latest AI developments I found..."

### Example 3: Scheduled Reminder

**Scenario**: User set reminder for "Call mom at 7 PM"

**EVA's Autonomous Action**:
1. **Thinking**: Checks for due reminders
2. **Decision**: Reminder is due now
3. **Action**: Sends message "Reminder: Call mom"
4. **Result**: User doesn't forget important task

## Troubleshooting

### EVA Not Taking Autonomous Actions

1. **Check if autonomous is enabled**:
   ```bash
   GET /admin/autonomous-status
   ```

2. **Check NVIDIA API health**:
   ```bash
   GET /admin/model-health
   ```

3. **Review logs for errors**:
   ```bash
   # Check application logs
   ```

### Too Many Autonomous Messages

1. **Increase thinking interval**:
   ```bash
   AUTONOMOUS_THINKING_INTERVAL=600
   ```

2. **Adjust user preferences**:
   ```python
   await storage_service.set_user_preference(
       user_id=1,
       preference="autonomous_frequency",
       value="low"
   )
   ```

### Autonomous Actions Not Appropriate

1. **Review decision history**:
   ```bash
   GET /admin/autonomous-history
   ```

2. **Adjust confidence threshold**:
   ```bash
   AUTONOMOUS_CONFIDENCE_THRESHOLD=0.8  # Higher threshold = fewer actions
   ```

## Best Practices

### For Users

1. **Give EVA context**: The more EVA knows about you, the better her autonomous decisions
2. **Provide feedback**: Let EVA know when her autonomous actions are helpful or not
3. **Set preferences**: Configure what types of autonomous actions you want
4. **Review regularly**: Check autonomous history to ensure actions are appropriate

### For Developers

1. **Monitor closely**: Watch autonomous actions especially in early deployment
2. **Test thoroughly**: Test autonomous features with safe actions first
3. **Add logging**: Log all autonomous decisions and executions
4. **Implement limits**: Set reasonable limits on autonomous actions
5. **User control**: Always give users ability to disable or adjust autonomy

## Future Enhancements

Planned improvements to EVA's autonomous system:

- **Learning from feedback**: EVA learns which autonomous actions users appreciate
- **Predictive actions**: Anticipate user needs before they arise
- **Collaborative autonomy**: Work with other autonomous systems
- **Context-aware timing**: Better understanding of when to reach out
- **Emotional intelligence**: Consider user's emotional state in decisions

## Security Considerations

1. **API Key Protection**: Admin API key should be kept secret
2. **User Privacy**: Autonomous actions respect user privacy settings
3. **Data Protection**: Autonomous decisions don't expose sensitive data
4. **Access Control**: Only primary users get autonomous actions
5. **Audit Trail**: All autonomous actions are logged and reviewable

## Support

For issues with autonomous features:

1. Check logs for error messages
2. Verify NVIDIA API is working
3. Ensure database connections are healthy
4. Review admin endpoint status
5. Check configuration settings

Remember: EVA's autonomous features are designed to be helpful, not intrusive. You always have control over her autonomy level.
