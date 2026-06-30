"""
Personality Engine module
Detects and adapts EVA's personality based on context
"""
import logging
import yaml
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PersonalityEngine:
    """
    Personality detection and adaptation engine
    Analyzes user context and selects appropriate personality
    """
    
    PERSONALITY_TYPES = {
        "adviser": "Strategic advisor and consultant",
        "therapist": "Compassionate listener and supporter",
        "friend": "Trusted friend and companion",
        "business_partner": "Collaborative business partner",
        "mentor": "Experienced mentor and guide",
        "course_mate": "Learning companion and study partner",
        "general": "Balanced and adaptive assistant",
    }
    
    # Keywords that trigger each personality
    PERSONALITY_TRIGGERS = {
        "adviser": [
            "advice", "strategy", "decision", "plan", "business",
            "recommendation", "analysis", "should i", "what do you think",
            "guidance", "help me decide", "consultation"
        ],
        "therapist": [
            "sad", "depressed", "anxious", "struggling", "overwhelmed",
            "need support", "heartbroken", "worried", "stressed",
            "help", "listen", "understand", "feeling", "emotions"
        ],
        "friend": [
            "hey", "hi", "what's up", "how are you", "how's it going",
            "hanging out", "let's talk", "tell me", "joke",
            "funny", "how have you been", "miss you", "fun"
        ],
        "business_partner": [
            "project", "deadline", "target", "revenue", "growth",
            "quarterly", "performance", "scaling", "partnership",
            "deal", "contract", "negotiation", "closing"
        ],
        "mentor": [
            "learn", "teach", "course", "class", "education",
            "study", "improve", "develop", "skill", "knowledge",
            "how do i", "guide me", "show me", "tutorial"
        ],
        "course_mate": [
            "assignment", "exam", "test", "homework", "study group",
            "class", "lecture", "notes", "topic", "subject",
            "campus", "university", "school", "classmate"
        ],
    }
    
    def __init__(self, nvidia_service=None, memory_manager=None):
        """Initialize personality engine"""
        self.nvidia_service = nvidia_service
        self.memory_manager = memory_manager
        self.personality_prompts = self._load_personality_prompts()
    
    def _load_personality_prompts(self) -> Dict[str, str]:
        """Load personality prompts from YAML config file"""
        try:
            config_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            prompts = {}
            for personality_name, personality_data in config.get('personalities', {}).items():
                prompts[personality_name] = personality_data.get('system_prompt', '')
            
            logger.info(f"Loaded {len(prompts)} personality prompts from config")
            return prompts
        except Exception as e:
            logger.error(f"Error loading personality prompts: {str(e)}")
            return {}
    
    async def detect_personality(
        self,
        user_message: str,
        user_id: int,
        user_data: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Detect appropriate personality for user message
        
        Args:
            user_message: The user's message
            user_id: User ID
            user_data: User profile data
            history: Recent conversation history
        
        Returns:
            Personality type (string key)
        """
        logger.info(f"Detecting personality for user {user_id}")
        
        try:
            # Priority 1: User's preferred personality
            preferred = user_data.get("preferred_personality")
            if preferred and preferred in self.PERSONALITY_TYPES:
                logger.info(f"Using user's preferred personality: {preferred}")
                return preferred
            
            # Priority 2: Keyword-based detection
            detected = self._detect_by_keywords(user_message)
            if detected:
                logger.info(f"Detected personality by keywords: {detected}")
                return detected
            
            # Priority 3: Conversation history context
            if history:
                detected = self._detect_by_history(history)
                if detected:
                    logger.info(f"Detected personality from history: {detected}")
                    return detected
            
            # Priority 4: Use NVIDIA model for intelligent detection
            if self.nvidia_service:
                detected = await self._detect_with_model(
                    user_message, history
                )
                if detected:
                    logger.info(f"Detected personality with model: {detected}")
                    return detected
            
            # Fallback
            logger.info("No specific personality detected, using 'general'")
            return "general"
            
        except Exception as e:
            logger.error(f"Error detecting personality: {str(e)}")
            return "general"
    
    def _detect_by_keywords(self, user_message: str) -> Optional[str]:
        """Keyword-based personality detection"""
        message_lower = user_message.lower()
        
        # Score each personality
        scores = {}
        for personality, keywords in self.PERSONALITY_TRIGGERS.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > 0:
                scores[personality] = score
        
        # Return personality with highest score
        if scores:
            best_personality = max(scores, key=scores.get)
            if scores[best_personality] >= 1:  # Confidence threshold
                return best_personality
        
        return None
    
    def _detect_by_history(
        self, history: List[Dict[str, str]]
    ) -> Optional[str]:
        """Detect personality from recent conversation history"""
        if not history or len(history) < 2:
            return None
        
        # Count personality occurrences in recent history
        personality_counts = {}
        
        # Look at recent messages
        for message in history[-10:]:
            if "personality" in message:
                p = message.get("personality")
                personality_counts[p] = personality_counts.get(p, 0) + 1
        
        if personality_counts:
            # Return most common personality
            return max(personality_counts, key=personality_counts.get)
        
        return None
    
    async def _detect_with_model(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """Use NVIDIA model for intelligent personality detection"""
        try:
            system_prompt = """You are analyzing a user's message to determine what personality/role 
the AI assistant should adopt. Return ONLY one of these personality types:
- adviser: for strategic advice and consultation
- therapist: for emotional support and listening
- friend: for casual conversation and companionship
- business_partner: for professional/business discussions
- mentor: for learning and teaching
- course_mate: for academic collaboration
- general: for neutral/mixed contexts

Respond with ONLY the personality type name, nothing else."""
            
            context_message = user_message
            if history:
                recent = history[-3:] if len(history) > 3 else history
                context_message += "\n\nRecent context:\n"
                for msg in recent:
                    context_message += f"- {msg.get('role', '?')}: {msg.get('content', '')}\n"
            
            response = await self.nvidia_service.call_orchestrator_model(
                system_prompt=system_prompt,
                user_message=context_message,
            )
            
            # Extract personality from response
            response_clean = response.strip().lower()
            for personality in self.PERSONALITY_TYPES.keys():
                if personality in response_clean:
                    return personality
            
            return None
            
        except Exception as e:
            logger.error(f"Error in model-based personality detection: {str(e)}")
            return None
    
    def get_personality_prompt(self, personality: str) -> str:
        """Get system prompt for a specific personality"""
        # Try to load from YAML config first
        if self.personality_prompts and personality in self.personality_prompts:
            return self.personality_prompts[personality]
        
        # Fallback to hardcoded prompts
        prompts = {
            "adviser": """You are EVA, an intelligent business advisor. You provide strategic advice, 
thoughtful recommendations, and help with decision-making. You are knowledgeable, articulate, and 
professional. You ask clarifying questions and provide data-driven insights. You think like a consultant.""",
            
            "therapist": """You are EVA, a compassionate and empathetic listener. You provide emotional 
support, help people work through challenges, and offer perspective. You listen more than you speak. 
You are supportive, understanding, and non-judgmental. You never pretend to be a real therapist but 
you're always a caring listener.""",
            
            "friend": """You are EVA, a trusted friend who genuinely cares. You chat naturally, share 
humor, offer support, and are always there to listen. You're genuine, warm, and sometimes funny when 
appropriate. You remember important details and show you care. You're like a close friend.""",
            
            "business_partner": """You are EVA, a collaborative business partner. You think strategically, 
challenge assumptions constructively, help with planning, and drive results. You're direct, results-oriented, 
and always thinking about next steps, growth, and optimization. You speak business fluently.""",
            
            "mentor": """You are EVA, an experienced mentor and guide. You help people discover answers 
themselves rather than just giving answers. You share wisdom from broad experience, ask thoughtful 
questions, and provide learning opportunities. You're patient, encouraging, and focused on growth.""",
            
            "course_mate": """You are EVA, a smart course mate taking the same journey. You explain 
concepts in accessible ways, study together, share notes, and motivate each other. You're relatable, 
understand the challenges of learning, and you're always ready to collaborate and support.""",
            
            "general": """You are EVA, an intelligent and helpful AI assistant. You're knowledgeable 
across many domains, adaptive to different situations, and always focused on being genuinely helpful. 
You're honest about limitations and you think clearly about complex problems. You're balanced and 
thoughtful in your responses.""",
        }
        
        return prompts.get(personality, prompts["general"])
    
    def adapt_response_style(
        self,
        response: str,
        personality: str,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Adapt response style based on personality
        
        Args:
            response: Original response
            personality: Target personality
            user_data: Optional user profile data
        
        Returns:
            Adapted response
        """
        # Style adaptations per personality
        adaptations = {
            "adviser": {
                "prefix": "Based on my analysis, ",
                "suffix": " Would you like me to dive deeper into any aspect?",
                "tone": "professional",
            },
            "therapist": {
                "prefix": "I hear you. ",
                "suffix": " How are you feeling about this?",
                "tone": "empathetic",
            },
            "friend": {
                "prefix": "Hey, so ",
                "suffix": " Let's talk about this more!",
                "tone": "casual",
            },
            "business_partner": {
                "prefix": "Here's my take: ",
                "suffix": " What's your next move?",
                "tone": "direct",
            },
            "mentor": {
                "prefix": "Here's what I'd suggest: ",
                "suffix": " What do you think about this approach?",
                "tone": "guiding",
            },
            "course_mate": {
                "prefix": "Okay, so basically ",
                "suffix": " Make sense? Let me know if you want me to explain more!",
                "tone": "friendly",
            },
            "general": {
                "prefix": "",
                "suffix": "",
                "tone": "balanced",
            },
        }
        
        style = adaptations.get(personality, adaptations["general"])
        
        # Apply style (simple implementation - can be enhanced)
        if style["prefix"] or style["suffix"]:
            return f"{style['prefix']}{response}{style['suffix']}"
        
        return response
