"""
Tool Registry module
Manages available tools and their execution
"""
import logging
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        pass
    
    @abstractmethod
    async def validate_input(self, **kwargs) -> bool:
        """Validate input parameters"""
        pass


class ToolRegistry:
    """
    Registry for all available tools
    Manages tool discovery, registration, and execution
    """
    
    def __init__(self):
        """Initialize tool registry"""
        self.tools: Dict[str, Tool] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(
        self,
        name: str,
        tool_instance: Tool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a tool
        
        Args:
            name: Tool name
            tool_instance: Tool instance
            metadata: Additional metadata
        
        Returns:
            Success boolean
        """
        try:
            if name in self.tools:
                logger.warning(f"Tool {name} already registered, overwriting")
            
            self.tools[name] = tool_instance
            self.tool_metadata[name] = metadata or {
                "description": tool_instance.description,
            }
            
            logger.info(f"Registered tool: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering tool {name}: {str(e)}")
            return False
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get tool by name
        
        Args:
            name: Tool name
        
        Returns:
            Tool instance or None
        """
        return self.tools.get(name)
    
    async def execute_tool(
        self,
        name: str,
        action: str = "execute",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool
        
        Args:
            name: Tool name
            action: Action to execute (default: "execute")
            **kwargs: Tool arguments
        
        Returns:
            Tool result
        """
        try:
            tool = self.get_tool(name)
            
            if not tool:
                return {
                    "status": "error",
                    "error": f"Tool {name} not found",
                }
            
            logger.info(f"Executing tool: {name} with action: {action}")
            
            # Check if tool has execute method (for agents)
            if hasattr(tool, 'execute'):
                result = await tool.execute(action, **kwargs)
            else:
                # Fallback to standard tool execution
                if not await tool.validate_input(**kwargs):
                    return {
                        "status": "error",
                        "error": f"Invalid input for tool {name}",
                    }
                result = await tool.execute(**kwargs)
            
            logger.info(f"Tool execution completed: {name}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {name}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }
    
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered tools
        
        Returns:
            Dictionary of tool metadata
        """
        return self.tool_metadata
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get info about a specific tool
        
        Args:
            name: Tool name
        
        Returns:
            Tool metadata or None
        """
        return self.tool_metadata.get(name)
    
    def is_tool_available(self, name: str) -> bool:
        """Check if tool is available"""
        return name in self.tools
    
    async def get_available_tools_for_user(
        self,
        user_id: int,
        is_primary_user: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get available tools for a user
        
        Args:
            user_id: User ID
            is_primary_user: Whether primary user
        
        Returns:
            Dictionary of available tools
        """
        available = {}
        
        for name, metadata in self.tool_metadata.items():
            # Restrict some tools for non-primary users
            if not is_primary_user:
                restricted = ["payment", "email"]  # Can't use payment/email for strangers
                if any(r in name.lower() for r in restricted):
                    continue
            
            available[name] = metadata
        
        return available
    
    async def validate_tool_access(
        self,
        user_id: int,
        tool_name: str,
        is_primary_user: bool = True,
    ) -> bool:
        """
        Validate user has access to tool
        
        Args:
            user_id: User ID
            tool_name: Tool name
            is_primary_user: Whether primary user
        
        Returns:
            Access boolean
        """
        # Check if tool exists
        if not self.is_tool_available(tool_name):
            return False
        
        # Restrict for non-primary users
        if not is_primary_user:
            restricted = ["payment", "email", "calendar"]
            if any(r in tool_name.lower() for r in restricted):
                logger.warning(f"User {user_id} denied access to tool {tool_name} (not primary)")
                return False
        
        return True


# ====================== Built-in Tool Implementations ======================

class EmailTool(Tool):
    """Email tool for sending/checking emails"""
    
    def __init__(self, email_service):
        super().__init__("email", "Send and receive emails")
        self.email_service = email_service
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute email tool"""
        action = kwargs.get("action")
        
        if action == "send":
            return await self.email_service.send_email(
                to=kwargs.get("to"),
                subject=kwargs.get("subject"),
                body=kwargs.get("body"),
            )
        elif action == "check":
            return await self.email_service.check_inbox()
        
        return {"status": "error", "error": "Unknown email action"}
    
    async def validate_input(self, **kwargs) -> bool:
        """Validate email input"""
        action = kwargs.get("action")
        return action in ["send", "check"]


class CalendarTool(Tool):
    """Calendar tool for scheduling meetings"""
    
    def __init__(self, calendar_service):
        super().__init__("calendar", "Schedule and manage calendar events")
        self.calendar_service = calendar_service
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute calendar tool"""
        action = kwargs.get("action")
        
        if action == "schedule":
            return await self.calendar_service.schedule_event(
                title=kwargs.get("title"),
                start_time=kwargs.get("start_time"),
                end_time=kwargs.get("end_time"),
                description=kwargs.get("description"),
            )
        elif action == "check":
            return await self.calendar_service.check_availability(
                date=kwargs.get("date"),
            )
        
        return {"status": "error", "error": "Unknown calendar action"}
    
    async def validate_input(self, **kwargs) -> bool:
        """Validate calendar input"""
        action = kwargs.get("action")
        return action in ["schedule", "check"]


class SearchTool(Tool):
    """Web search tool"""
    
    def __init__(self, search_service):
        super().__init__("search", "Search the web for information")
        self.search_service = search_service
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute search tool"""
        query = kwargs.get("query")
        
        return await self.search_service.search(query)
    
    async def validate_input(self, **kwargs) -> bool:
        """Validate search input"""
        return "query" in kwargs and len(kwargs.get("query", "")) > 0


class ImageTool(Tool):
    """Image generation tool"""
    
    def __init__(self, image_service):
        super().__init__("image", "Generate images from text descriptions")
        self.image_service = image_service
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute image tool"""
        prompt = kwargs.get("prompt")
        
        return await self.image_service.generate_image(prompt)
    
    async def validate_input(self, **kwargs) -> bool:
        """Validate image input"""
        return "prompt" in kwargs and len(kwargs.get("prompt", "")) > 0
