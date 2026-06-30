from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Type

from pydantic import BaseModel, Field, ValidationError


ToolName = Literal["telegram", "email", "calendar", "reminder", "search", "image", "payment", "document"]


class ToolCall(BaseModel):
    tool: ToolName
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# Telegram
# -----------------------------

class TelegramSendMessageParams(BaseModel):
    phone_or_username: Optional[str] = None
    chat_id: Optional[int] = None
    message: str = ""


class TelegramFindContactParams(BaseModel):
    query: str


class TelegramGetDialogsParams(BaseModel):
    limit: int = 10


class TelegramGetHistoryParams(BaseModel):
    chat_id: int
    limit: int = 20


class TelegramSearchMessagesParams(BaseModel):
    chat_id: int
    query: str = ""


class TelegramGetContactInfoParams(BaseModel):
    user_id: int


class TelegramSendVoiceParams(BaseModel):
    phone_or_username: Optional[str] = None
    chat_id: Optional[int] = None
    text: str = ""


class TelegramSendAudioParams(BaseModel):
    phone_or_username: Optional[str] = None
    chat_id: Optional[int] = None
    text: str = ""
    caption: Optional[str] = None


# -----------------------------
# Email
# -----------------------------

class EmailSendParams(BaseModel):
    to: str
    subject: str
    body: str
    attachments: Optional[list] = None


class EmailReadParams(BaseModel):
    inbox: str = "INBOX"
    limit: int = 10


class EmailSearchParams(BaseModel):
    query: str
    sender: Optional[str] = None


# -----------------------------
# Calendar
# -----------------------------

class CalendarCreateParams(BaseModel):
    title: str
    date: str
    time: str
    duration: int = 60
    description: str = ""
    attendees: Optional[list[str]] = None


class CalendarListParams(BaseModel):
    date: Optional[str] = None
    days_ahead: int = 30


class CalendarUpdateParams(BaseModel):
    event_id: int
    # arbitrary updates allowed
    updates: Dict[str, Any] = Field(default_factory=dict)


class CalendarDeleteParams(BaseModel):
    event_id: int


# -----------------------------
# Reminder
# -----------------------------

class ReminderSetParams(BaseModel):
    title: str
    date: str
    time: str
    description: str = ""
    priority: str = "normal"


class ReminderListParams(BaseModel):
    status: str = "active"


class ReminderUpdateParams(BaseModel):
    reminder_id: int
    updates: Dict[str, Any] = Field(default_factory=dict)


class ReminderCompleteParams(BaseModel):
    reminder_id: int


class ReminderDeleteParams(BaseModel):
    reminder_id: int


# -----------------------------
# Search
# -----------------------------

class SearchWebParams(BaseModel):
    query: str
    num_results: int = 5
    language: str = "en"


class SearchNewsParams(BaseModel):
    topic: str
    num_results: int = 5


class SearchHistoryParams(BaseModel):
    limit: int = 10


# -----------------------------
# Image (best-effort; agent implementation varies)
# -----------------------------

class ImageGenerateParams(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


class ImageEditParams(BaseModel):
    image_id: str
    instruction: str


class ImageUpscaleParams(BaseModel):
    image_id: str


# -----------------------------
# Contracts mapping
# -----------------------------

TOOL_ACTION_SCHEMAS: Dict[str, Dict[str, Type[BaseModel]]] = {
    "telegram": {
        "send_message": TelegramSendMessageParams,
        "find_contact": TelegramFindContactParams,
        "get_dialogs": TelegramGetDialogsParams,
        "get_history": TelegramGetHistoryParams,
        "search_messages": TelegramSearchMessagesParams,
        "get_contact_info": TelegramGetContactInfoParams,
        "send_voice": TelegramSendVoiceParams,
        "send_audio": TelegramSendAudioParams,
    },
    "email": {"send": EmailSendParams, "read": EmailReadParams, "search": EmailSearchParams},
    "calendar": {
        "create": CalendarCreateParams,
        "list": CalendarListParams,
        "update": CalendarUpdateParams,
        "delete": CalendarDeleteParams,
    },
    "reminder": {
        "set": ReminderSetParams,
        "list": ReminderListParams,
        "update": ReminderUpdateParams,
        "complete": ReminderCompleteParams,
        "delete": ReminderDeleteParams,
    },
    "search": {"web": SearchWebParams, "news": SearchNewsParams, "history": SearchHistoryParams},
    "image": {"generate": ImageGenerateParams, "edit": ImageEditParams, "upscale": ImageUpscaleParams},
    # payment/document intentionally omitted until their agents define stable param schemas
}


def validate_and_normalize_tool_call(tool: str, action: str, params: Dict[str, Any]) -> tuple[bool, Dict[str, Any], str]:
    """
    Returns (ok, normalized_params, error_message).
    If no schema exists for tool/action, we allow it through (compat), but return ok=True with original params.
    """
    tool_map = TOOL_ACTION_SCHEMAS.get(tool)
    if not tool_map:
        return True, params, ""
    schema = tool_map.get(action)
    if not schema:
        return False, {}, f"Unknown action '{action}' for tool '{tool}'. Allowed: {sorted(tool_map.keys())}"
    try:
        model = schema(**(params or {}))
        # special case: flatten update-style calls to match existing agent signatures
        if tool == "calendar" and action == "update":
            return True, {"event_id": model.event_id, **model.updates}, ""
        if tool == "reminder" and action == "update":
            return True, {"reminder_id": model.reminder_id, **model.updates}, ""
        return True, model.model_dump(), ""
    except ValidationError as e:
        return False, {}, str(e)

