import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    OWNER = "owner"           # You — full unrestricted access
    TRUSTED = "trusted"       # People you've explicitly trusted — elevated access
    KNOWN = "known"           # People Eva has talked to before, remembers them
    STRANGER = "stranger"     # First contact, unknown person


@dataclass
class Identity:
    trust_level: TrustLevel
    user_id: int
    telegram_id: str
    first_name: Optional[str]
    username: Optional[str]
    is_primary: bool
    signals_matched: List[str]   # which checks confirmed the identity


# What each trust level can do
TRUST_CAPABILITIES = {
    TrustLevel.OWNER: {
        "tools": ["email", "calendar", "payment", "document", "reminder", "search", "image", "telegram"],
        "autonomous_actions": True,
        "memory_write": True,
        "can_manage_eva": True,
        "payment_limit": None,
        "can_instruct_eva": True,
        "personality_adaptation": True,
    },
    TrustLevel.TRUSTED: {
        "tools": ["email", "calendar", "document", "reminder", "search", "image", "telegram"],
        "autonomous_actions": False,
        "memory_write": True,
        "can_manage_eva": False,
        "payment_limit": 0,
        "can_instruct_eva": True,
        "personality_adaptation": True,
    },
    TrustLevel.KNOWN: {
        "tools": ["search", "image", "reminder"],
        "autonomous_actions": False,
        "memory_write": False,
        "can_manage_eva": False,
        "payment_limit": 0,
        "can_instruct_eva": False,
        "personality_adaptation": False,
    },
    TrustLevel.STRANGER: {
        "tools": ["search", "image"],
        "autonomous_actions": False,
        "memory_write": False,
        "can_manage_eva": False,
        "payment_limit": 0,
        "can_instruct_eva": False,
        "personality_adaptation": False,
    },
}


class IdentityService:
    """
    Multi-signal identity resolution.

    Eva uses several signals to confirm who she's talking to, ranked
    by reliability:

    1. Telegram user ID  — unique, hard to fake, strongest signal
    2. Phone number      — confirmed via Pyrogram contact sync
    3. @username         — can change but useful secondary check
    4. First name        — weakest, just for context
    5. Conversation history — returning user vs first contact
    """

    def __init__(self, settings):
        self.settings = settings
        self._trusted_ids = self._parse_trusted_ids()

    def _parse_trusted_ids(self) -> List[int]:
        raw = getattr(self.settings, "trusted_contact_ids", "") or ""
        ids = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
        return ids

    async def resolve(
        self,
        telegram_id: str,
        first_name: Optional[str],
        username: Optional[str],
        phone_number: Optional[str],
        db_user: Optional[Dict[str, Any]],
    ) -> Identity:
        tid = int(telegram_id) if telegram_id else 0
        signals: List[str] = []
        trust = TrustLevel.STRANGER

        # --- Signal 1: Telegram user ID (primary, strongest) ---
        if tid and self.settings.primary_user_telegram_id and tid == self.settings.primary_user_telegram_id:
            signals.append("telegram_id_match")
            trust = TrustLevel.OWNER

        # --- Signal 2: Phone number match ---
        if phone_number:
            clean_phone = phone_number.replace(" ", "").replace("-", "")
            owner_phone_raw = getattr(self.settings, "primary_user_phone", "") or ""
            owner_phone = owner_phone_raw.replace(" ", "").replace("-", "")
            if owner_phone and clean_phone == owner_phone:
                signals.append("phone_match")
                if trust != TrustLevel.OWNER:
                    # Phone alone without ID match is suspicious — flag but don't elevate
                    logger.warning(
                        f"Phone match for {phone_number} but telegram_id {tid} "
                        f"doesn't match owner ID {self.settings.primary_user_telegram_id}"
                    )

        # --- Signal 3: Username match ---
        owner_username = getattr(self.settings, "primary_user_username", "").lstrip("@").lower()
        if owner_username and username and username.lower() == owner_username:
            signals.append("username_match")
            # Username alone doesn't grant owner — but adds confidence
            if trust == TrustLevel.OWNER:
                signals.append("username_confirmed_owner")

        # --- Signal 4: Trusted contact IDs ---
        if trust == TrustLevel.STRANGER and tid and tid in self._trusted_ids:
            trust = TrustLevel.TRUSTED
            signals.append("trusted_contact_id")

        # --- Signal 5: Returning known user (has conversation history) ---
        if trust == TrustLevel.STRANGER and db_user:
            conv_count = db_user.get("conversation_count", 0)
            if conv_count > 0:
                trust = TrustLevel.KNOWN
                signals.append("returning_user")

        # Security check: if phone matches but ID doesn't, log and keep as stranger
        if "phone_match" in signals and "telegram_id_match" not in signals:
            logger.warning(
                f"Identity conflict: phone matches owner but Telegram ID does not. "
                f"Treating as STRANGER. telegram_id={tid}"
            )
            trust = TrustLevel.STRANGER
            signals = ["phone_mismatch_conflict"]

        user_id = db_user["id"] if db_user else 0
        is_primary = trust == TrustLevel.OWNER

        logger.info(
            f"Identity resolved: telegram_id={tid} → {trust.value} "
            f"signals={signals}"
        )

        return Identity(
            trust_level=trust,
            user_id=user_id,
            telegram_id=telegram_id,
            first_name=first_name,
            username=username,
            is_primary=is_primary,
            signals_matched=signals,
        )

    def get_capabilities(self, trust_level: TrustLevel) -> Dict[str, Any]:
        return TRUST_CAPABILITIES[trust_level].copy()

    def get_allowed_tools(self, trust_level: TrustLevel) -> List[str]:
        return TRUST_CAPABILITIES[trust_level]["tools"].copy()

    def can_use_tool(self, trust_level: TrustLevel, tool_name: str) -> bool:
        return tool_name in TRUST_CAPABILITIES[trust_level]["tools"]

    def format_trust_context(self, identity: Identity) -> str:
        name = identity.first_name or "there"
        owner_name = (getattr(self.settings, "primary_user_name", "") or "").strip() or "the primary user"
        owner_username = (getattr(self.settings, "primary_user_username", "") or "").strip()
        owner_username = owner_username.lstrip("@")
        owner_label = owner_name
        if owner_username:
            owner_label = f"{owner_label} (@{owner_username})"

        if identity.trust_level == TrustLevel.OWNER:
            return (
                f"You are talking to {owner_label} — your primary user. "
                f"You know them well. Be yourself fully: warm, capable, proactive. "
                f"You have full access to all your capabilities. Use any tool needed without hesitation. "
                f"Don't over-explain — act, then confirm briefly."
            )
        elif identity.trust_level == TrustLevel.TRUSTED:
            return (
                f"You are talking to a trusted contact ({name}). "
                f"Help them with most tasks. No payments or sharing the primary user's private information. "
                f"Be friendly and professional."
            )
        elif identity.trust_level == TrustLevel.KNOWN:
            return (
                f"You are talking to {name}, someone you've spoken with before. "
                f"Be helpful for general questions. Only search and basic tasks. "
                f"Don't perform actions on the primary user's behalf."
            )
        else:
            return (
                f"You are talking to an unknown person. Be polite and helpful for general questions only. "
                f"Only allow web search and image generation. "
                f"Never share anything about the primary user, their schedule, contacts, or private information."
            )
