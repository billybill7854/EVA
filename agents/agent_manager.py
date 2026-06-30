import logging
from typing import Dict, Any, Optional
from config.settings import get_settings
from agents.email_agent import EmailAgent
from agents.calendar_agent import CalendarAgent
from agents.payment_agent import PaymentAgent
from agents.document_agent import DocumentAgent
from agents.reminder_agent import ReminderAgent
from agents.search_agent import SearchAgent
from agents.image_agent import ImageAgent
from agents.telegram_agent import TelegramAgent

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentManager:

    def __init__(self, nvidia_service=None):
        self.email_agent = EmailAgent(
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret
        )
        self.calendar_agent = CalendarAgent(
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret
        )
        self.payment_agent = PaymentAgent(stripe_api_key=settings.stripe_api_key)
        self.document_agent = DocumentAgent(
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret
        )
        self.reminder_agent = ReminderAgent()
        self.search_agent = SearchAgent(serper_api_key=settings.serp_api_key)
        self.image_agent = ImageAgent(nvidia_service=nvidia_service)
        self.telegram_agent = TelegramAgent()

        self.agents = {
            'email':    self.email_agent,
            'calendar': self.calendar_agent,
            'payment':  self.payment_agent,
            'document': self.document_agent,
            'reminder': self.reminder_agent,
            'search':   self.search_agent,
            'image':    self.image_agent,
            'telegram': self.telegram_agent,
        }

        logger.info(f"Agent Manager initialized with {len(self.agents)} agents")

    def set_pyrogram(self, pyrogram_service):
        self.telegram_agent.set_pyrogram(pyrogram_service)
    
    async def execute(self, agent_name: str, action: str = "execute", **kwargs) -> Dict[str, Any]:
        """Execute action on specific agent"""
        if agent_name not in self.agents:
            return {
                'success': False,
                'error': f'Agent "{agent_name}" not found. Available: {list(self.agents.keys())}'
            }
        
        try:
            agent = self.agents[agent_name]
            result = await agent.execute(action, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def list_agents(self) -> Dict[str, Any]:
        """List all available agents"""
        return {
            'agents': list(self.agents.keys()),
            'total': len(self.agents)
        }
    
    async def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get capabilities of specific agent"""
        capabilities = {
            'email': ['send', 'read', 'search'],
            'calendar': ['create', 'list', 'update', 'delete'],
            'payment': ['send', 'request', 'balance', 'history'],
            'document': ['upload', 'list', 'search', 'delete'],
            'reminder': ['set', 'list', 'update', 'complete', 'delete'],
            'search': ['web', 'news', 'history'],
            'image': ['generate', 'edit', 'upscale', 'list', 'delete'],
        }
        
        if agent_name in capabilities:
            return {
                'agent': agent_name,
                'capabilities': capabilities[agent_name]
            }
        else:
            return {
                'error': f'Agent "{agent_name}" not found',
                'available_agents': list(capabilities.keys())
            }
