import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TelegramAgent:
    """
    Gives Eva the ability to act on Telegram as a real user:
    send messages, find contacts, check dialogs, etc.
    Requires the PyrogramService to be injected after startup.
    """

    def __init__(self):
        self._pyrogram = None

    def set_pyrogram(self, pyrogram_service):
        self._pyrogram = pyrogram_service

    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        actions = {
            "send_message":     self._send_message,
            "find_contact":     self._find_contact,
            "get_dialogs":      self._get_dialogs,
            "get_history":      self._get_history,
            "search_messages":  self._search_messages,
            "get_contact_info": self._get_contact_info,
            "send_voice":       self._send_voice,
            "send_audio":       self._send_audio,
        }
        fn = actions.get(action)
        if not fn:
            return {"success": False, "error": f"Unknown telegram action: {action}"}
        try:
            return await fn(**kwargs)
        except Exception as e:
            logger.error(f"TelegramAgent error [{action}]: {e}")
            return {"success": False, "error": str(e)}

    async def _send_message(self, phone_or_username: str = None, chat_id=None,
                             message: str = "", **_) -> Dict[str, Any]:
        if not self._pyrogram:
            return {"success": False, "error": "Telegram user account not connected"}

        target = chat_id or phone_or_username
        if not target:
            return {"success": False, "error": "No recipient specified"}

        # If phone number provided, try to resolve to a user first
        if isinstance(target, str) and target.startswith("+"):
            contacts = await self._pyrogram.get_contacts()
            match = next(
                (c for c in contacts if c.get("phone") and
                 c["phone"].replace(" ", "") == target.replace(" ", "")),
                None,
            )
            if match:
                target = match["id"]
            # If not in contacts, Pyrogram can still attempt by phone import
            # but for now use the phone string directly

        await self._pyrogram.send_message(chat_id=target, text=message)
        return {
            "success": True,
            "message": f"Message sent to {phone_or_username or chat_id}",
            "text_sent": message,
        }

    async def _find_contact(self, query: str = "", **_) -> Dict[str, Any]:
        if not self._pyrogram:
            return {"success": False, "error": "Telegram user account not connected"}

        contacts = await self._pyrogram.get_contacts()
        query_lower = query.lower().replace("+", "").replace(" ", "")

        results = []
        for c in contacts:
            name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip().lower()
            phone = (c.get("phone") or "").replace(" ", "").replace("+", "")
            username = (c.get("username") or "").lower()
            if (query_lower in name or query_lower in phone or query_lower in username):
                results.append(c)

        return {
            "success": True,
            "found": len(results),
            "contacts": results[:5],
        }

    async def _get_dialogs(self, limit: int = 10, **_) -> Dict[str, Any]:
        if not self._pyrogram:
            return {"success": False, "error": "Telegram user account not connected"}
        dialogs = await self._pyrogram.get_dialogs(limit=limit)
        return {"success": True, "dialogs": dialogs}

    async def _get_history(self, chat_id=None, limit: int = 20, **_) -> Dict[str, Any]:
        if not self._pyrogram:
            return {"success": False, "error": "Telegram user account not connected"}
        if not chat_id:
            return {"success": False, "error": "chat_id required"}
        messages = await self._pyrogram.get_chat_history(chat_id, limit=limit)
        return {"success": True, "messages": messages}

    async def _search_messages(self, chat_id=None, query: str = "", **_) -> Dict[str, Any]:
        if not self._pyrogram:
            return {"success": False, "error": "Telegram user account not connected"}
        results = await self._pyrogram.search_messages(chat_id, query)
        return {"success": True, "results": results}

    async def _get_contact_info(self, user_id=None, **_) -> Dict[str, Any]:
        if not self._pyrogram:
            return {"success": False, "error": "Telegram user account not connected"}
        info = await self._pyrogram.get_user_info(user_id)
        return {"success": True, "user": info}

    async def _send_voice(self, phone_or_username: str = None, chat_id=None,
                         text: str = "", **_) -> Dict[str, Any]:
        if not self._pyrogram:
            return {"success": False, "error": "Telegram user account not connected"}
        
        target = chat_id or phone_or_username
        if not target:
            return {"success": False, "error": "No recipient specified"}

        # Generate audio using TTS
        try:
            from services.nvidia_service import NVIDIAService
            import io
            nvidia = NVIDIAService()
            audio_bytes = await nvidia.call_tts_model(text=text)
            
            await self._pyrogram.send_voice(chat_id=target, voice=io.BytesIO(audio_bytes))
            return {
                "success": True,
                "message": f"Voice note sent to {phone_or_username or chat_id}",
                "text_converted": text,
            }
        except Exception as e:
            logger.error(f"Failed to send voice note: {e}")
            return {"success": False, "error": f"TTS or Send failed: {str(e)}"}

    async def _send_audio(self, phone_or_username: str = None, chat_id=None,
                         text: str = "", caption: str = None, **_) -> Dict[str, Any]:
        if not self._pyrogram:
            return {"success": False, "error": "Telegram user account not connected"}
        
        target = chat_id or phone_or_username
        if not target:
            return {"success": False, "error": "No recipient specified"}

        try:
            from services.nvidia_service import NVIDIAService
            import io
            nvidia = NVIDIAService()
            audio_bytes = await nvidia.call_tts_model(text=text)
            
            await self._pyrogram.send_audio(chat_id=target, audio=io.BytesIO(audio_bytes), caption=caption)
            return {
                "success": True,
                "message": f"Audio file sent to {phone_or_username or chat_id}",
                "text_converted": text,
            }
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")
            return {"success": False, "error": f"TTS or Send failed: {str(e)}"}
