"""Integration tests for tool registry, usage tracker, and reward integration.

This test suite verifies the complete tool learning system including:
- Tool registry functionality (registration, discovery, documentation)
- Tool usage tracker functionality (usage recording, success rates, diversity)
- Reward integration (first-success detection, diversity scoring, reward values)

Requirements: 12.14, 12.15
"""

import pytest
import torch
from datetime import datetime

from eva.tools.registry import ToolRegistry, ToolInfo, ParameterInfo
from eva.tools.usage_tracker import ToolUsageTracker, ToolUsage
from eva.curiosity.reward import CuriosityEngine
from eva.core.baby_brain import BabyBrain
from eva.tools.base import Tool


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name: str = "mock_tool", available: bool = True):
        self._name = name
        self._available = available
    
    def execute(self, query: str) -> str:
        return f"[{self._name}] Executed: {query}"
    
    def get_name(self) -> str:
        return self._name
    
    def is_available(self) -> bool:
        return self._available


class TestToolRegistryIntegration:
    """Integration tests for tool registry with multiple tools."""
    
    @pytest.fixture
    def registry_with_tools(self):
        """Create a registry with multiple registered tools."""
        registry = ToolRegistry()
        
        # Register web search tool
        web_tool = MockTool(name="web_search")
        web_info = ToolInfo(
            name="web_search",
            description="Search the web for information",
            parameters=[
                ParameterInfo("query", "str", "Search query", True),
                ParameterInfo("limit", "int", "Max results", False),
            ],
            examples=["web_search python tutorial", "web_search AI research"]
        )
        registry.register(web_tool, web_info)
        
        # Register file handler tool
        file_tool = MockTool(name="file_handler")
        file_info = ToolInfo(
            name="file_handler",
            description="Read and write files",
            parameters=[
                ParameterInfo("operation", "str", "Operation type", True),
                ParameterInfo("path", "str", "File path", True),
            ],
            examples=["file_handler READ test.txt"]
        )
        registry.register(file_tool, file_info)
        
        # Register device control tool
        device_tool = MockTool(name="device_control")
        device_info = ToolInfo(
            name="device_control",
            description="Execute system commands",
            parameters=[
                ParameterInfo("command", "str", "Command to execute", True),
            ],
            examples=["device_control ls -la"]
        )
        registry.register(device_tool, device_info)
        
        return registry
    
    def test_discover_all_available_tools(self, registry_with_tools):
        """Test discovering all available tools in registry."""
        tools = registry_with_tools.discover()
        
        assert len(tools) == 3
        assert "web_search" in tools
        assert "file_handler" in tools
        assert "device_control" in tools
    
    def test_get_documentation_for_each_tool(self, registry_with_tools):
        """Test retrieving documentation for each registered tool."""
        # Get documentation for web_search
        web_doc = registry_with_tools.get_documentation("web_search")
        assert web_doc is not None
        assert web_doc.name == "web_search"
        assert "Search the web" in web_doc.description
        assert len(web_doc.parameters) == 2
        assert len(web_doc.examples) == 2
        
        # Get documentation for file_handler
        file_doc = registry_with_tools.get_documentation("file_handler")
        assert file_doc is not None
        assert file_doc.name == "file_handler"
        assert len(file_doc.parameters) == 2
        
        # Get documentation for device_control
        device_doc = registry_with_tools.get_documentation("device_control")
        assert device_doc is not None
        assert device_doc.name == "device_control"
        assert len(device_doc.parameters) == 1
    
    def test_retrieve_and_execute_tools(self, registry_with_tools):
        """Test retrieving tool instances and executing them."""
        # Retrieve and execute web_search
        web_tool = registry_with_tools.get_tool("web_search")
        assert web_tool is not None
        result = web_tool.execute("test query")
        assert "[web_search]" in result
        assert "test query" in result
        
        # Retrieve and execute file_handler
        file_tool = registry_with_tools.get_tool("file_handler")
        assert file_tool is not None
        result = file_tool.execute("READ test.txt")
        assert "[file_handler]" in result
        
        # Retrieve and execute device_control
        device_tool = registry_with_tools.get_tool("device_control")
        assert device_tool is not None
        result = device_tool.execute("ls")
        assert "[device_control]" in result


class TestUsageTrackerIntegration:
    """Integration tests for usage tracker with multiple tools."""
    
    @pytest.fixture
    def tracker_with_history(self):
        """Create a tracker with usage history for multiple tools."""
        tracker = ToolUsageTracker()
        
        # Web search: 3 successes, 1 failure, diverse queries
        tracker.record_usage("web_search", "python tutorial", True, "result1")
        tracker.record_usage("web_search", "machine learning", True, "result2")
        tracker.record_usage("web_search", "AI research", True, "result3")
        tracker.record_usage("web_search", "invalid query", False, "error")
        
        # File handler: 2 successes, 2 failures, repetitive
        tracker.record_usage("file_handler", "READ test.txt", True, "content")
        tracker.record_usage("file_handler", "READ test.txt", True, "content")
        tracker.record_usage("file_handler", "READ missing.txt", False, "not found")
        tracker.record_usage("file_handler", "READ missing.txt", False, "not found")
        
        # Device control: 1 success, 0 failures
        tracker.record_usage("device_control", "ls -la", True, "files")
        
        return tracker
    
    def test_success_rates_for_multiple_tools(self, tracker_with_history):
        """Test computing success rates for multiple tools."""
        # Web search: 3/4 = 0.75
        assert tracker_with_history.get_success_rate("web_search") == 0.75
        
        # File handler: 2/4 = 0.5
        assert tracker_with_history.get_success_rate("file_handler") == 0.5
        
        # Device control: 1/1 = 1.0
        assert tracker_with_history.get_success_rate("device_control") == 1.0
    
    def test_diversity_scores_for_multiple_tools(self, tracker_with_history):
        """Test computing diversity scores for multiple tools."""
        # Web search: 4 unique patterns out of 4 = 1.0
        assert tracker_with_history.get_usage_diversity("web_search") == 1.0
        
        # File handler: 2 unique patterns out of 4 = 0.5
        assert tracker_with_history.get_usage_diversity("file_handler") == 0.5
        
        # Device control: 1 unique pattern out of 1 = 1.0
        assert tracker_with_history.get_usage_diversity("device_control") == 1.0
    
    def test_first_success_detection(self, tracker_with_history):
        """Test detecting first successful use of tools."""
        # Web search and file handler have more than one usage
        assert tracker_with_history.is_first_success("web_search") is False
        assert tracker_with_history.is_first_success("file_handler") is False
        
        # Device control has only one usage, so it IS first success
        assert tracker_with_history.is_first_success("device_control") is True
        
        # Create new tracker and test first success
        new_tracker = ToolUsageTracker()
        new_tracker.record_usage("new_tool", "query", True, "result")
        assert new_tracker.is_first_success("new_tool") is True
    
    def test_statistics_aggregation(self, tracker_with_history):
        """Test aggregating statistics across all tools."""
        stats = tracker_with_history.get_statistics()
        
        assert stats["total_invocations"] == 9
        assert stats["unique_tools_used"] == 3
        
        # Check success rates
        assert "web_search" in stats["success_rates"]
        assert "file_handler" in stats["success_rates"]
        assert "device_control" in stats["success_rates"]
        
        # Check diversity scores
        assert "web_search" in stats["diversity_scores"]
        assert "file_handler" in stats["diversity_scores"]
        assert "device_control" in stats["diversity_scores"]


class TestRewardIntegration:
    """Integration tests for tool reward computation and integration."""
    
    @pytest.fixture
    def engine_with_tracker(self):
        """Create a curiosity engine with tool tracker."""
        tracker = ToolUsageTracker()
        engine = CuriosityEngine(tool_tracker=tracker)
        return engine, tracker
    
    def test_first_success_bonus_integration(self, engine_with_tracker):
        """Test that first successful tool use gets +0.5 bonus."""
        engine, tracker = engine_with_tracker
        
        # Record first successful use
        tracker.record_usage("web_search", "query1", True, "result")
        
        # Compute reward
        reward = engine._compute_tool_reward("web_search", True)
        
        # Should get first success bonus
        assert reward == 0.5
    
    def test_new_pattern_bonus_integration(self, engine_with_tracker):
        """Test that new usage patterns get +0.2 bonus."""
        engine, tracker = engine_with_tracker
        
        # Record multiple different usages
        tracker.record_usage("web_search", "query1", True, "result")
        tracker.record_usage("web_search", "query2", True, "result")
        
        # Compute reward for the second usage
        reward = engine._compute_tool_reward("web_search", True)
        
        # Should get new pattern bonus but not first success
        assert reward == 0.2
    
    def test_repetition_penalty_integration(self, engine_with_tracker):
        """Test that repetitive usage gets -0.1 penalty."""
        engine, tracker = engine_with_tracker
        
        # Record same usage multiple times
        tracker.record_usage("web_search", "same query", True, "result")
        tracker.record_usage("web_search", "same query", True, "result")
        tracker.record_usage("web_search", "same query", True, "result")
        
        # Compute reward
        reward = engine._compute_tool_reward("web_search", True)
        
        # Diversity is 1/3 = 0.33 < 0.5, so penalty applies
        assert reward == -0.1
    
    def test_combined_rewards_scenario(self, engine_with_tracker):
        """Test a realistic scenario with multiple reward types."""
        engine, tracker = engine_with_tracker
        
        # First use: should get first success bonus
        tracker.record_usage("web_search", "query1", True, "result")
        reward1 = engine._compute_tool_reward("web_search", True)
        assert reward1 == 0.5
        
        # Second use with new pattern: should get new pattern bonus
        tracker.record_usage("web_search", "query2", True, "result")
        reward2 = engine._compute_tool_reward("web_search", True)
        assert reward2 == 0.2
        
        # Third use with new pattern: should get new pattern bonus
        tracker.record_usage("web_search", "query3", True, "result")
        reward3 = engine._compute_tool_reward("web_search", True)
        assert reward3 == 0.2
        
        # Fourth use repeating pattern: diversity still high, no penalty
        tracker.record_usage("web_search", "query1", True, "result")
        reward4 = engine._compute_tool_reward("web_search", True)
        # Diversity is 3/4 = 0.75 > 0.5, so no penalty
        assert reward4 == 0.0
    
    def test_failure_does_not_give_rewards(self, engine_with_tracker):
        """Test that failed tool invocations don't give rewards."""
        engine, tracker = engine_with_tracker
        
        # Record failed usage
        tracker.record_usage("web_search", "query", False, "error")
        
        # Compute reward for failure
        reward = engine._compute_tool_reward("web_search", False)
        
        # Should not get any reward for failure
        assert reward == 0.0
    
    def test_reward_integration_in_compute_reward(self):
        """Test that tool rewards are integrated into total curiosity reward."""
        tracker = ToolUsageTracker()
        engine = CuriosityEngine(tool_tracker=tracker)
        
        # Create minimal brain for testing
        brain = BabyBrain(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=128
        )
        
        # Prepare engine
        engine.prepare(brain)
        
        # Create dummy inputs
        predicted = torch.softmax(torch.randn(100), dim=0)
        actual = 42
        hidden_state = torch.randn(64)
        recent_outcomes = [torch.randn(64) for _ in range(5)]
        
        # Record first successful tool use
        tracker.record_usage("web_search", "query", True, "result")
        
        # Compute reward with tool info
        total_reward, breakdown = engine.compute_reward(
            predicted=predicted,
            actual=actual,
            brain=brain,
            hidden_state=hidden_state,
            recent_outcomes=recent_outcomes,
            tool_name="web_search",
            tool_success=True
        )
        
        # Check that tool_reward is in breakdown
        assert "tool_reward" in breakdown
        assert breakdown["tool_reward"] == 0.5  # First success bonus
        
        # Check that tool reward contributes to total
        assert total_reward > 0.0
    
    def test_reward_without_tool_info(self):
        """Test that compute_reward works without tool information."""
        tracker = ToolUsageTracker()
        engine = CuriosityEngine(tool_tracker=tracker)
        
        # Create minimal brain
        brain = BabyBrain(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=128
        )
        
        # Prepare engine
        engine.prepare(brain)
        
        # Create dummy inputs
        predicted = torch.softmax(torch.randn(100), dim=0)
        actual = 42
        hidden_state = torch.randn(64)
        recent_outcomes = [torch.randn(64) for _ in range(5)]
        
        # Compute reward without tool info
        total_reward, breakdown = engine.compute_reward(
            predicted=predicted,
            actual=actual,
            brain=brain,
            hidden_state=hidden_state,
            recent_outcomes=recent_outcomes
        )
        
        # Check that tool_reward is 0.0
        assert "tool_reward" in breakdown
        assert breakdown["tool_reward"] == 0.0


class TestEndToEndToolLearning:
    """End-to-end integration tests for the complete tool learning system."""
    
    def test_complete_tool_learning_workflow(self):
        """Test the complete workflow: register -> discover -> use -> track -> reward."""
        # 1. Create registry and register tools
        registry = ToolRegistry()
        
        web_tool = MockTool(name="web_search")
        web_info = ToolInfo(
            name="web_search",
            description="Search the web",
            parameters=[ParameterInfo("query", "str", "Search query", True)],
            examples=["web_search python"]
        )
        registry.register(web_tool, web_info)
        
        # 2. Discover available tools
        tools = registry.discover()
        assert "web_search" in tools
        
        # 3. Get documentation
        doc = registry.get_documentation("web_search")
        assert doc is not None
        assert doc.name == "web_search"
        
        # 4. Get tool and execute
        tool = registry.get_tool("web_search")
        assert tool is not None
        result = tool.execute("python tutorial")
        assert "python tutorial" in result
        
        # 5. Track usage
        tracker = ToolUsageTracker()
        tracker.record_usage("web_search", "python tutorial", True, result)
        
        # 6. Verify tracking
        assert tracker.get_success_rate("web_search") == 1.0
        assert tracker.is_first_success("web_search") is True
        
        # 7. Compute reward
        engine = CuriosityEngine(tool_tracker=tracker)
        reward = engine._compute_tool_reward("web_search", True)
        assert reward == 0.5  # First success bonus
    
    def test_multiple_tools_learning_progression(self):
        """Test learning progression with multiple tools over time."""
        # Setup
        registry = ToolRegistry()
        tracker = ToolUsageTracker()
        engine = CuriosityEngine(tool_tracker=tracker)
        
        # Register multiple tools
        for tool_name in ["web_search", "file_handler", "device_control"]:
            tool = MockTool(name=tool_name)
            info = ToolInfo(name=tool_name, description=f"{tool_name} tool")
            registry.register(tool, info)
        
        # Simulate learning progression
        rewards = []
        
        # First use of web_search
        tracker.record_usage("web_search", "query1", True, "result")
        rewards.append(engine._compute_tool_reward("web_search", True))
        
        # Second use of web_search with new pattern
        tracker.record_usage("web_search", "query2", True, "result")
        rewards.append(engine._compute_tool_reward("web_search", True))
        
        # First use of file_handler
        tracker.record_usage("file_handler", "READ file.txt", True, "content")
        rewards.append(engine._compute_tool_reward("file_handler", True))
        
        # Repetitive use of web_search (need enough repetitions to drop diversity below 0.5)
        # After 2 unique patterns and 5 repetitions of same query, diversity = 3/7 = 0.43 < 0.5
        for _ in range(5):
            tracker.record_usage("web_search", "same query", True, "result")
        rewards.append(engine._compute_tool_reward("web_search", True))
        
        # Verify reward progression
        assert rewards[0] == 0.5  # First success
        assert rewards[1] == 0.2  # New pattern
        assert rewards[2] == 0.5  # First success of different tool
        assert rewards[3] == -0.1  # Repetition penalty (diversity < 0.5)
        
        # Verify statistics
        stats = tracker.get_statistics()
        assert stats["unique_tools_used"] == 2
        assert stats["total_invocations"] == 8  # 2 + 1 + 5
    
    def test_tool_discovery_and_documentation_workflow(self):
        """Test the workflow of discovering tools and getting their documentation."""
        registry = ToolRegistry()
        
        # Register tools with detailed documentation
        tools_to_register = [
            ("web_search", "Search the web", [
                ParameterInfo("query", "str", "Search query", True),
                ParameterInfo("limit", "int", "Max results", False),
            ]),
            ("file_handler", "Handle files", [
                ParameterInfo("operation", "str", "Operation type", True),
                ParameterInfo("path", "str", "File path", True),
            ]),
            ("device_control", "Control device", [
                ParameterInfo("command", "str", "Command", True),
            ]),
        ]
        
        for name, desc, params in tools_to_register:
            tool = MockTool(name=name)
            info = ToolInfo(name=name, description=desc, parameters=params)
            registry.register(tool, info)
        
        # Discover all tools
        discovered = registry.discover()
        assert len(discovered) == 3
        
        # Get documentation for each tool
        for tool_name in discovered:
            doc = registry.get_documentation(tool_name)
            assert doc is not None
            assert doc.name == tool_name
            assert len(doc.description) > 0
            assert len(doc.parameters) > 0
            
            # Verify parameter details
            for param in doc.parameters:
                assert param.name
                assert param.type
                assert param.description
                assert isinstance(param.required, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
