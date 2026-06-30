"""
Telegram Service
Handles Telegram bot webhook and message routing
"""
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramService:
    """Service for Telegram bot interactions"""
    
    def __init__(self, bot_token: str):
        """
        Initialize Telegram service
        
        Args:
            bot_token: Telegram bot token
        """
        self.bot_token = bot_token
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "Markdown",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send text message to user
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            parse_mode: Markdown or HTML
            reply_to_message_id: Optional message to reply to
        
        Returns:
            API response
        """
        try:
            import httpx
            
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            
            if reply_to_message_id:
                payload["reply_to_message_id"] = reply_to_message_id
            
            url = f"{self.api_url}/sendMessage"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                return response.json()
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            raise
    
    async def send_audio(
        self,
        chat_id: int,
        audio_bytes: bytes,
        caption: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send audio message to user
        
        Args:
            chat_id: Telegram chat ID
            audio_bytes: Audio file bytes
            caption: Optional caption
        
        Returns:
            API response
        """
        try:
            import httpx
            
            files = {
                "audio": ("audio.ogg", audio_bytes, "audio/ogg"),
            }
            
            data = {
                "chat_id": chat_id,
            }
            
            if caption:
                data["caption"] = caption
            
            url = f"{self.api_url}/sendAudio"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data, files=files)
                response.raise_for_status()
                
                return response.json()
            
        except Exception as e:
            logger.error(f"Error sending Telegram audio: {str(e)}")
            raise
    
    async def send_photo(
        self,
        chat_id: int,
        photo_bytes: bytes,
        caption: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send photo to user
        
        Args:
            chat_id: Telegram chat ID
            photo_bytes: Photo file bytes
            caption: Optional caption
        
        Returns:
            API response
        """
        try:
            import httpx
            
            files = {
                "photo": ("photo.png", photo_bytes, "image/png"),
            }
            
            data = {
                "chat_id": chat_id,
            }
            
            if caption:
                data["caption"] = caption
            
            url = f"{self.api_url}/sendPhoto"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data, files=files)
                response.raise_for_status()
                
                return response.json()
            
        except Exception as e:
            logger.error(f"Error sending Telegram photo: {str(e)}")
            raise
    
    async def send_notification(
        self,
        chat_id: int,
        title: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Send notification to user
        
        Args:
            chat_id: Telegram chat ID
            title: Notification title
            message: Notification message
        
        Returns:
            API response
        """
        text = f"*{title}*\n{message}"
        return await self.send_message(chat_id, text)
    
    def parse_webhook_message(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse incoming webhook message
        
        Args:
            payload: Webhook payload from Telegram
        
        Returns:
            Parsed message or None
        """
        try:
            message = None
            message_type = None
            content = None
            
            # Handle text messages
            if "message" in payload:
                msg = payload["message"]
                message = msg
                
                if "text" in msg:
                    message_type = "text"
                    content = msg["text"]
                elif "voice" in msg:
                    message_type = "voice"
                    content = msg["voice"]
                elif "photo" in msg:
                    message_type = "photo"
                    content = msg["photo"]
                elif "audio" in msg:
                    message_type = "audio"
                    content = msg["audio"]
            
            # Handle callback queries
            elif "callback_query" in payload:
                callback = payload["callback_query"]
                message_type = "callback"
                content = callback.get("data")
                message = callback
            
            if not message_type or not content:
                logger.warning(f"Unable to parse message: {payload}")
                return None
            
            return {
                "type": message_type,
                "content": content,
                "message": message,
                "chat_id": message.get("chat", {}).get("id"),
                "user_id": message.get("from", {}).get("id"),
                "username": message.get("from", {}).get("username"),
                "first_name": message.get("from", {}).get("first_name"),
                "timestamp": datetime.fromtimestamp(message.get("date", 0)),
            }
            
        except Exception as e:
            logger.error(f"Error parsing webhook message: {str(e)}")
            return None
    
    def create_keyboard(self, buttons: list) -> Dict[str, Any]:
        """
        Create inline keyboard markup
        
        Args:
            buttons: List of button configs
        
        Returns:
            Keyboard markup
        """
        keyboard = {
            "inline_keyboard": []
        }
        
        row = []
        for button in buttons:
            row.append({
                "text": button.get("text"),
                "callback_data": button.get("callback_data"),
            })
            
            if button.get("new_row", False):
                keyboard["inline_keyboard"].append(row)
                row = []
        
        if row:
            keyboard["inline_keyboard"].append(row)
        
        return keyboard
    
    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = "",
        show_alert: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer callback query (e.g., button click)
        
        Args:
            callback_query_id: Callback query ID
            text: Notification text
            show_alert: Show as alert
        
        Returns:
            API response
        """
        try:
            import httpx
            
            payload = {
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": show_alert,
            }
            
            url = f"{self.api_url}/answerCallbackQuery"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                return response.json()
            
        except Exception as e:
            logger.error(f"Error answering callback query: {str(e)}")
            raise
    
    async def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get info about a user
        
        Args:
            user_id: Telegram user ID
        
        Returns:
            User info or None
        """
        try:
            import httpx
            
            url = f"{self.api_url}/getMe"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                
                return response.json().get("result")
            
        except Exception as e:
            logger.error(f"Error getting user info: {str(e)}")
            return None
