import asyncio
import logging
import os
from typing import Optional, Callable, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    # Pyrogram's sync module tries asyncio.get_event_loop() at import time,
    # which raises RuntimeError in Python 3.10+ when no loop exists yet.
    # We set a new event loop before importing to prevent this.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    from pyrogram import Client, filters, enums
    from pyrogram.types import Message, User, Chat
    from pyrogram.errors import (
        FloodWait, UserIsBlocked, PeerIdInvalid,
    )
    HAS_PYROGRAM = True
except ImportError:
    HAS_PYROGRAM = False
    logger.warning("Pyrogram not installed — Eva will run in bot-only mode")


class PyrogramService:
    """
    Eva's real Telegram user account via Pyrogram MTProto.
    Gives Eva full capabilities of a normal Telegram user:
    - Message anyone (not just people who messaged her first)
    - Make and receive voice/video calls
    - Join and manage groups and channels
    - React to messages
    - See online/typing status
    - Forward, pin, delete messages
    - Send any media type
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        phone_number: str,
        session_name: str = "eva_session",
        session_dir: str = ".",
    ):
        if not HAS_PYROGRAM:
            raise RuntimeError(
                "pyrogram and tgcrypto are required. "
                "Install with: pip install pyrogram tgcrypto"
            )

        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.session_name = session_name
        self.session_dir = session_dir

        self._client: Optional[Client] = None
        self._message_handler: Optional[Callable] = None
        self._call_handler: Optional[Callable] = None
        self._is_running = False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self, message_handler: Callable = None, call_handler: Callable = None):
        self._message_handler = message_handler
        self._call_handler = call_handler

        self._client = Client(
            name=os.path.join(self.session_dir, self.session_name),
            api_id=self.api_id,
            api_hash=self.api_hash,
            phone_number=self.phone_number,
        )

        await self._client.start()
        self._is_running = True

        me = await self._client.get_me()
        logger.info(
            f"Eva's Telegram account connected: {me.first_name} "
            f"(@{me.username}) id={me.id}"
        )

        # Register incoming message handler
        if self._message_handler:
            @self._client.on_message(filters.incoming & ~filters.bot)
            async def _on_message(client: Client, message: Message):
                try:
                    await self._message_handler(self._parse_message(message), message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")

        logger.info("Eva is live on Telegram as a real user account")

    async def stop(self):
        if self._client and self._is_running:
            await self._client.stop()
            self._is_running = False
            logger.info("Pyrogram client stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running

    # -----------------------------------------------------------------------
    # Messaging
    # -----------------------------------------------------------------------

    async def send_message(
        self,
        chat_id,
        text: str,
        parse_mode=None,
        reply_to_message_id: int = None,
        disable_notification: bool = False,
    ) -> Optional[Any]:
        try:
            await self._flood_safe(
                self._client.send_message,
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode or enums.ParseMode.MARKDOWN,
                reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification,
            )
        except UserIsBlocked:
            logger.warning(f"User {chat_id} has blocked Eva")
        except PeerIdInvalid:
            logger.error(f"Invalid peer id: {chat_id}")
        except Exception as e:
            logger.error(f"Error sending message to {chat_id}: {e}")
            raise

    async def send_audio(
        self,
        chat_id,
        audio,
        caption: str = None,
        duration: int = None,
    ):
        try:
            await self._flood_safe(
                self._client.send_audio,
                chat_id=chat_id,
                audio=audio,
                caption=caption,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"Error sending audio to {chat_id}: {e}")
            raise

    async def send_voice(self, chat_id, voice, caption: str = None):
        try:
            await self._flood_safe(
                self._client.send_voice,
                chat_id=chat_id,
                voice=voice,
                caption=caption,
            )
        except Exception as e:
            logger.error(f"Error sending voice to {chat_id}: {e}")
            raise

    async def send_photo(self, chat_id, photo, caption: str = None):
        try:
            await self._flood_safe(
                self._client.send_photo,
                chat_id=chat_id,
                photo=photo,
                caption=caption,
            )
        except Exception as e:
            logger.error(f"Error sending photo to {chat_id}: {e}")
            raise

    async def send_video(self, chat_id, video, caption: str = None):
        try:
            await self._flood_safe(
                self._client.send_video,
                chat_id=chat_id,
                video=video,
                caption=caption,
            )
        except Exception as e:
            logger.error(f"Error sending video to {chat_id}: {e}")
            raise

    async def send_document(self, chat_id, document, caption: str = None):
        try:
            await self._flood_safe(
                self._client.send_document,
                chat_id=chat_id,
                document=document,
                caption=caption,
            )
        except Exception as e:
            logger.error(f"Error sending document to {chat_id}: {e}")
            raise

    async def send_sticker(self, chat_id, sticker):
        try:
            await self._flood_safe(
                self._client.send_sticker,
                chat_id=chat_id,
                sticker=sticker,
            )
        except Exception as e:
            logger.error(f"Error sending sticker to {chat_id}: {e}")
            raise

    async def forward_message(self, chat_id, from_chat_id, message_id: int):
        try:
            await self._flood_safe(
                self._client.forward_messages,
                chat_id=chat_id,
                from_chat_id=from_chat_id,
                message_ids=message_id,
            )
        except Exception as e:
            logger.error(f"Error forwarding message: {e}")
            raise

    # -----------------------------------------------------------------------
    # Message actions
    # -----------------------------------------------------------------------

    async def send_typing(self, chat_id):
        try:
            await self._client.send_chat_action(
                chat_id=chat_id,
                action=enums.ChatAction.TYPING,
            )
        except Exception as e:
            logger.debug(f"Could not send typing action: {e}")

    async def send_recording_audio(self, chat_id):
        try:
            await self._client.send_chat_action(
                chat_id=chat_id,
                action=enums.ChatAction.RECORD_AUDIO,
            )
        except Exception as e:
            logger.debug(f"Could not send recording action: {e}")

    async def react_to_message(self, chat_id, message_id: int, emoji: str = "👍"):
        try:
            await self._client.send_reaction(
                chat_id=chat_id,
                message_id=message_id,
                emoji=emoji,
            )
        except Exception as e:
            logger.debug(f"Could not react to message: {e}")

    async def read_messages(self, chat_id):
        try:
            await self._client.read_chat_history(chat_id=chat_id)
        except Exception as e:
            logger.debug(f"Could not mark messages as read: {e}")

    async def delete_message(self, chat_id, message_id: int):
        try:
            await self._client.delete_messages(
                chat_id=chat_id,
                message_ids=message_id,
            )
        except Exception as e:
            logger.error(f"Error deleting message: {e}")

    async def edit_message(self, chat_id, message_id: int, text: str):
        try:
            await self._client.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=enums.ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.error(f"Error editing message: {e}")

    async def pin_message(self, chat_id, message_id: int):
        try:
            await self._client.pin_chat_message(
                chat_id=chat_id,
                message_id=message_id,
            )
        except Exception as e:
            logger.error(f"Error pinning message: {e}")

    # -----------------------------------------------------------------------
    # User & chat info
    # -----------------------------------------------------------------------

    async def get_user_info(self, user_id) -> Optional[Dict[str, Any]]:
        try:
            user = await self._client.get_users(user_id)
            return self._user_to_dict(user)
        except Exception as e:
            logger.error(f"Error getting user info for {user_id}: {e}")
            return None

    async def get_chat_history(
        self,
        chat_id,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        messages = []
        try:
            async for msg in self._client.get_chat_history(chat_id, limit=limit):
                messages.append(self._parse_message(msg))
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
        return messages

    async def get_contacts(self) -> List[Dict[str, Any]]:
        try:
            contacts = await self._client.get_contacts()
            return [self._user_to_dict(c) for c in contacts]
        except Exception as e:
            logger.error(f"Error getting contacts: {e}")
            return []

    async def get_dialogs(self, limit: int = 20) -> List[Dict[str, Any]]:
        dialogs = []
        try:
            async for dialog in self._client.get_dialogs(limit=limit):
                dialogs.append({
                    "chat_id": dialog.chat.id,
                    "name": getattr(dialog.chat, "first_name", None)
                        or getattr(dialog.chat, "title", "Unknown"),
                    "unread_count": dialog.unread_messages_count,
                    "top_message": dialog.top_message.text if dialog.top_message else None,
                    "date": dialog.top_message.date.isoformat()
                        if dialog.top_message and dialog.top_message.date else None,
                })
        except Exception as e:
            logger.error(f"Error getting dialogs: {e}")
        return dialogs

    async def check_online_status(self, user_id) -> Optional[str]:
        try:
            user = await self._client.get_users(user_id)
            status = user.status
            if status is None:
                return "unknown"
            return str(status).replace("UserStatus.", "").lower()
        except Exception as e:
            logger.debug(f"Could not get online status: {e}")
            return None

    # -----------------------------------------------------------------------
    # Groups & channels
    # -----------------------------------------------------------------------

    async def join_chat(self, invite_link_or_username: str):
        try:
            await self._client.join_chat(invite_link_or_username)
            logger.info(f"Eva joined chat: {invite_link_or_username}")
        except Exception as e:
            logger.error(f"Error joining chat: {e}")
            raise

    async def leave_chat(self, chat_id):
        try:
            await self._client.leave_chat(chat_id)
            logger.info(f"Eva left chat: {chat_id}")
        except Exception as e:
            logger.error(f"Error leaving chat: {e}")

    async def get_chat_members(self, chat_id, limit: int = 100) -> List[Dict[str, Any]]:
        members = []
        try:
            async for member in self._client.get_chat_members(chat_id, limit=limit):
                members.append(self._user_to_dict(member.user))
        except Exception as e:
            logger.error(f"Error getting chat members: {e}")
        return members

    # -----------------------------------------------------------------------
    # Media download
    # -----------------------------------------------------------------------

    async def download_media(self, message, file_name: str = None) -> Optional[str]:
        try:
            path = await self._client.download_media(message, file_name=file_name)
            return path
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
            return None

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    async def search_messages(
        self, chat_id, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        results = []
        try:
            async for msg in self._client.search_messages(
                chat_id=chat_id, query=query, limit=limit
            ):
                results.append(self._parse_message(msg))
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
        return results

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _parse_message(self, msg) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "message_id": msg.id,
            "chat_id": msg.chat.id if msg.chat else None,
            "user_id": msg.from_user.id if msg.from_user else None,
            "username": msg.from_user.username if msg.from_user else None,
            "first_name": msg.from_user.first_name if msg.from_user else None,
            "date": msg.date.isoformat() if msg.date else None,
            "type": "text",
            "content": None,
            "raw": msg,
        }

        if msg.text:
            parsed["type"] = "text"
            parsed["content"] = msg.text
        elif msg.voice:
            parsed["type"] = "voice"
            parsed["content"] = msg.voice
            parsed["file_id"] = msg.voice.file_id
        elif msg.audio:
            parsed["type"] = "audio"
            parsed["content"] = msg.audio
            parsed["file_id"] = msg.audio.file_id
        elif msg.photo:
            parsed["type"] = "photo"
            parsed["content"] = msg.photo
            parsed["file_id"] = msg.photo.file_id if hasattr(msg.photo, "file_id") else None
        elif msg.video:
            parsed["type"] = "video"
            parsed["content"] = msg.video
            parsed["file_id"] = msg.video.file_id
        elif msg.document:
            parsed["type"] = "document"
            parsed["content"] = msg.document
            parsed["file_id"] = msg.document.file_id
        elif msg.sticker:
            parsed["type"] = "sticker"
            parsed["content"] = msg.sticker
        elif msg.video_note:
            parsed["type"] = "video_note"
            parsed["content"] = msg.video_note

        return parsed

    def _user_to_dict(self, user) -> Dict[str, Any]:
        return {
            "id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone": getattr(user, "phone_number", None),
            "is_bot": user.is_bot,
        }

    async def _flood_safe(self, coro_func, **kwargs):
        for attempt in range(3):
            try:
                return await coro_func(**kwargs)
            except FloodWait as e:
                logger.warning(f"FloodWait: sleeping {e.value}s")
                await asyncio.sleep(e.value)
            except Exception:
                raise
        raise RuntimeError("Failed after flood wait retries")

    # -----------------------------------------------------------------------
    # Compatibility shim — matches TelegramService interface so the rest of
    # the codebase can call eva.telegram.send_message() regardless of backend
    # -----------------------------------------------------------------------

    async def send_notification(self, chat_id, title: str, message: str):
        await self.send_message(chat_id, f"*{title}*\n{message}")

    def parse_webhook_message(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Pyrogram doesn't use webhooks — this is a no-op shim
        return None
