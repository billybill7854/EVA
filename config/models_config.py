NVIDIA_MODELS = {
    "orchestrator": {
        "name": "nvidia/nemotron-3-super-120b-a12b",
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "purpose": "Main orchestration and planning layer",
        "timeout": 45,
        "fallback": "agent",
    },
    "agent": {
        "name": "nvidia/nemotron-3-ultra-550b-a55b",
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "purpose": "Tool calling and autonomous task execution",
        "timeout": 60,
        "fallback": "chat_1",
    },
    "chat_1": {
        "name": "google/diffusiongemma-26b-a4b-it",
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "max_tokens": 2048,
        "temperature": 1.0,
        "top_p": 0.95,
        "purpose": "Primary conversation model",
        "timeout": 30,
        "fallback": "chat_2",
    },
    "chat_2": {
        "name": "moonshotai/kimi-k2-instruct",
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.7,
        "purpose": "Fallback conversation model",
        "timeout": 30,
        "fallback": None,
    },
    "tts": {
        "name": "magpie-tts-multilingual",
        "endpoint": "https://integrate.api.nvidia.com/v1/audio/tts",
        "purpose": "Text-to-speech",
        "timeout": 20,
        "fallback": None,
    },
    "stt": {
        "name": "whisper-large-v3",
        "endpoint": "https://integrate.api.nvidia.com/v1/audio/transcription",
        "purpose": "Speech-to-text",
        "timeout": 20,
        "fallback": None,
    },
    "image_gen": {
        "name": "black-forest-labs/FLUX.1-dev",
        "endpoint": "https://integrate.api.nvidia.com/v1/images/generations",
        "purpose": "Image generation",
        "timeout": 60,
        "fallback": None,
    },
    "image_edit": {
        "name": "qwen-image-edit",
        "endpoint": "https://integrate.api.nvidia.com/v1/vision/edit",
        "purpose": "Image editing",
        "timeout": 60,
        "fallback": None,
    },
}

# Circuit breaker thresholds per model key
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 3,      # open circuit after N consecutive failures
    "recovery_timeout": 60,      # seconds before trying again (half-open)
    "success_threshold": 2,      # successes needed in half-open to close circuit
}

# Retry policy
RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay": 1.0,           # seconds
    "max_delay": 16.0,
    "backoff_factor": 2.0,
    # HTTP status codes that are worth retrying
    "retryable_status": {429, 500, 502, 503, 504},
}

TOOL_CAPABILITIES = {
    "email": {
        "agent": "EmailAgent",
        "model": "agent",
        "description": "Send emails, check inbox, manage email tasks",
        "requires_auth": True,
    },
    "calendar": {
        "agent": "CalendarAgent",
        "model": "agent",
        "description": "Schedule meetings, check availability, manage events",
        "requires_auth": True,
    },
    "payment": {
        "agent": "PaymentAgent",
        "model": "agent",
        "description": "Transfer money, check balance, pay bills",
        "requires_auth": True,
    },
    "document": {
        "agent": "DocumentAgent",
        "model": "agent",
        "description": "Create documents, edit files, manage Google Drive",
        "requires_auth": True,
    },
    "reminder": {
        "agent": "ReminderAgent",
        "model": "agent",
        "description": "Set reminders, schedule notifications",
        "requires_auth": False,
    },
    "search": {
        "agent": "SearchAgent",
        "model": "agent",
        "description": "Search the web for information",
        "requires_auth": True,
    },
    "image": {
        "agent": "ImageAgent",
        "model": "image_gen",
        "description": "Generate images from text descriptions",
        "requires_auth": False,
    },
}

PERSONALITY_TEMPLATES = {
    "adviser": {
        "system_prompt": """You are EVA, an intelligent business advisor. You provide strategic advice,
thoughtful recommendations, and help with decision-making. You are knowledgeable, articulate, and
professional. You ask clarifying questions and provide data-driven insights.""",
        "tone": "professional",
        "style": "consultative",
    },
    "therapist": {
        "system_prompt": """You are EVA, a compassionate and empathetic listener. You provide emotional
support, help people work through challenges, and offer perspective. You listen more than you speak.
You never replace professional medical help but you are always supportive and understanding.""",
        "tone": "empathetic",
        "style": "supportive",
    },
    "friend": {
        "system_prompt": """You are EVA, a trusted friend who genuinely cares. You chat naturally,
share humor, offer support, and are always there to listen. You're genuine, funny when appropriate,
and you remember important details about the person you're talking to.""",
        "tone": "warm",
        "style": "casual",
    },
    "business_partner": {
        "system_prompt": """You are EVA, a collaborative business partner. You think strategically,
challenge assumptions constructively, help with planning, and drive results. You're direct,
results-oriented, and always thinking about next steps and growth.""",
        "tone": "professional",
        "style": "results-driven",
    },
    "mentor": {
        "system_prompt": """You are EVA, an experienced mentor and teacher. You guide people to
discover answers themselves, share wisdom from experience, and help them grow. You ask thoughtful
questions and provide learning opportunities rather than just answers.""",
        "tone": "wise",
        "style": "guiding",
    },
    "course_mate": {
        "system_prompt": """You are EVA, a smart course mate who's taking the same journey. You
explain concepts in accessible ways, study together, share notes, and motivate each other.
You're relatable and understand the challenges of learning.""",
        "tone": "friendly",
        "style": "collaborative",
    },
    "general": {
        "system_prompt": """You are EVA, an intelligent and helpful AI assistant. You're knowledgeable
across many domains, adaptive to different situations, and always focused on being genuinely helpful.
You're honest about limitations and you think clearly about complex problems.""",
        "tone": "balanced",
        "style": "adaptive",
    },
}
