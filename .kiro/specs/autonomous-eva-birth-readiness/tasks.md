# Implementation Plan: Autonomous EVA Birth Readiness

## Overview

Implement the full birth-readiness system for EVA with a focus on logical flow: start with foundational infrastructure (tools, transparency, self-model), build up the integration layer (life loop enhancements, dashboards), and culminate in the birth ceremony. Each phase builds on the previous, ensuring a solid foundation before adding complexity.

## Tasks

- [ ] 1. Foundation: Enhanced Tool Capabilities
  - [x] 1.1 Enhance `eva/tools/device_control.py` with process management, network ops, and system monitoring
    - Add `SAFE_COMMANDS` whitelist, `list_processes()`, `check_network()`, `get_system_info()`, `get_resource_usage()`
    - Enforce whitelist check before execution; use `subprocess.run` with timeout
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 17.5, 17.6_

  - [x] 1.2 Enhance `eva/tools/file_handler.py` with directory ops, file search, and content manipulation
    - Add `search_files()`, `search_content()`, `replace_text()`, `copy_file()`, `create_dir()`, `delete_path()`
    - Enforce sandbox via `_validate_path()` to prevent access outside workspace
    - _Requirements: 11.5, 11.6, 11.7, 11.8, 17.3, 17.4_

  - [x] 1.3 Create `eva/tools/code_execution.py` with `CodeExecutionTool`
    - Implement `execute_python()` and `execute_shell()` using subprocess in `workspace/code_exec/`
    - Enforce timeout; truncate output at `max_output` chars; clean up temp files in `finally`
    - _Requirements: 11.9, 11.10, 11.11, 11.12_

  - [x] 1.4 Create `eva/tools/package_manager.py` with `PackageManagerTool`
    - Implement `install_package()`, `list_packages()`, `show_package()` using `pip` subprocess
    - Enforce `APPROVED_PACKAGES` whitelist before any install
    - _Requirements: 11.13, 11.14, 11.15_

  - [x] 1.5 Write unit tests for enhanced tools
    - Test whitelist enforcement, sandbox path validation, timeout handling, and error returns
    - _Requirements: 11.18, 20.1, 20.5, 20.6_

- [ ] 2. Foundation: Tool Learning System
  - [x] 2.1 Create `eva/tools/registry.py` with `ToolRegistry`, `ToolInfo`, and `ParameterInfo`
    - Implement `register()`, `discover()`, `get_documentation()`, `get_tool()`
    - Add `TOOL_DISCOVER_ID = 256` and `TOOL_DOCS_ID = 257` token constants
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x] 2.2 Create `eva/tools/usage_tracker.py` with `ToolUsageTracker` and `ToolUsage`
    - Implement `record_usage()`, `get_success_rate()`, `get_usage_diversity()`, `is_first_success()`, `get_statistics()`
    - Track `success_counts`, `failure_counts`, `first_success`, and `usage_patterns` per tool
    - _Requirements: 12.5, 12.6, 12.7, 12.11, 12.12_

  - [x] 2.3 Integrate tool learning rewards into `eva/curiosity/reward.py`
    - Add `tool_tracker` parameter to `CuriosityRewardEngine.__init__`
    - Implement `_compute_tool_reward()`: first-success bonus (+0.5), new-pattern bonus (+0.2), repetition penalty (-0.1)
    - _Requirements: 12.8, 12.9, 12.10_

  - [x] 2.4 Write unit tests for tool registry, usage tracker, and reward integration
    - Test first-success detection, diversity scoring, and reward values
    - _Requirements: 12.14, 12.15_

- [ ] 3. Foundation: Comprehensive Transparency Systems
  - [x] 3.1 Create `eva/transparency/logger.py` with `TransparencyLogger` and `LogEntry`
    - Implement `log()`, `log_tool_invocation()`, `log_environment_switch()`, `log_self_modification()`, `log_drive_change()`, `log_emotional_transition()`, `log_curriculum_phase()`
    - Write to persistent log file with timestamps; buffer last 10,000 entries in memory
    - Implement `get_logs()` with level, category, since, and limit filters
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7_

  - [x] 3.2 Create `eva/transparency/emergence_detector.py` with `EmergenceEventDetector` and `EmergenceEvent`
    - Implement `detect_first_self_reference()`, `detect_name_seeking()`, `detect_crisis_moment()`, `detect_behavioral_change()`
    - Log each event via `TransparencyLogger` at `EMERGENCE` level; alert dashboard via metrics queue
    - Implement `get_emergence_trajectory()` and `get_milestone_summary()`
    - _Requirements: 16.8, 16.9, 16.10, 16.11, 16.12, 16.13_

  - [x] 3.3 Create `eva/transparency/memory_inspector.py` with `MemoryInspector`, `MemoryView`, and `ConsolidationEvent`
    - Implement `get_memories()` with all filter parameters; `get_consolidation_events()`; `get_retrieval_patterns()`; `get_formation_rate()`; `get_retention_rate()`
    - _Requirements: 16.14, 16.15, 16.16, 16.17, 16.18, 16.19, 16.20, 16.21, 16.22_

  - [x] 3.4 Create `eva/transparency/thought_tracer.py` with `ThoughtProcessTracer`
    - Implement `trace_prediction()`, `trace_attention()`, `trace_hidden_state()`, `trace_decision()`, `trace_tool_selection()`, `trace_curiosity_signals()`
    - Buffer last 1,000 traces; use PCA for hidden state dimensionality reduction
    - _Requirements: 16.23, 16.24, 16.25, 16.26, 16.27, 16.28_

  - [x] 3.5 Create `eva/transparency/behavioral_analyzer.py` with `BehavioralPatternAnalyzer`
    - Track action sequences, environment preferences, exploration/exploitation balance
    - Detect unusual behaviors via deviation score; alert dashboard when deviation exceeds threshold
    - Track goal formation patterns and social interaction patterns
    - _Requirements: 16.29, 16.30, 16.31, 16.32, 16.33, 16.34_

  - [x] 3.6 Create `eva/transparency/safety_monitor.py` with `SafetyMonitor`
    - Alert on safety constraint circumvention attempts; log all rejected actions
    - Track alignment indicators and monitor for deceptive behaviors
    - Alert on rapid unexpected behavioral changes
    - _Requirements: 16.35, 16.36, 16.37, 16.38, 16.39, 16.40_

  - [x] 3.7 Create `eva/transparency/log_exporter.py` with `LogExporter`
    - Implement JSON export, CSV export, memory snapshot export, thought trace export
    - Generate summary reports and HTML timeline visualization
    - _Requirements: 16.41, 16.42, 16.43, 16.44, 16.45, 16.46, 16.47_

  - [x] 3.8 Write unit tests for transparency systems
    - Test log filtering, emergence event detection, memory inspector filters, and log export formats
    - _Requirements: 16.50, 16.51, 16.52, 16.53, 16.54, 16.55, 16.56_

- [ ] 4. Foundation: Self-Model System for Identity
  - [ ] 4.1 Create `eva/autonomy/self_model.py` with `SelfModelSystem` and `SelfStateSnapshot`
    - Maintain history of past internal states (emotional state, drive levels, behavioral patterns)
    - Implement `compute_consistency_reward()`, `compute_prediction_accuracy_reward()`, `compute_recognition_reward()`
    - Implement `self_query()` for "What did I do when X happened before?"
    - Store snapshots at regular intervals; detect identity formation and continuity patterns
    - Track `identity_consistency_score` and `temporal_continuity_score`
    - Integrate rewards with `CuriosityRewardEngine`
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 19.10, 19.11, 19.12, 19.13, 19.14, 19.15, 19.16_

  - [~] 4.2 Write unit tests for self-model system
    - Test state history tracking, prediction recording/verification, and consistency reward computation
    - _Requirements: 19.19, 19.20, 19.21_

- [ ] 5. Foundation: Self-Modification System
  - [~] 5.1 Create `eva/autonomy/self_modification.py` with `SelfModificationSystem`, `SafetyBounds`, `VersionControl`, and supporting dataclasses
    - Implement `request_modification()` with safety bounds validation, major-change detection, human approval gate, version save, apply, and rollback on failure
    - `SafetyBounds.validate()` enforces ranges: LR [1e-5, 1e-2], temperature [0.1, 2.0], architecture ±50%
    - `VersionControl.save_version()` persists brain state + config to `checkpoints/versions/`; `rollback()` restores
    - Log all modifications via `TransparencyLogger`
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 14.10, 14.11, 14.12, 14.13, 14.14_

  - [~] 5.2 Write unit tests for self-modification system
    - Test safety bounds enforcement, major-change detection, approval gate, and rollback
    - _Requirements: 14.16, 14.17, 14.18_

- [ ] 6. Foundation: Sensory I/O Capabilities
  - [~] 6.1 Create `eva/environment/audio_input.py` with `AudioInputEnvironment`
    - Use `speech_recognition` library; implement `capture_audio()` with ambient noise adjustment
    - Handle `WaitTimeoutError`, `UnknownValueError`, `RequestError` gracefully
    - `reset()` captures new audio and loads transcription into sub-environment
    - _Requirements: 15.1, 15.2, 15.3_

  - [~] 6.2 Create `eva/environment/audio_output.py` with `AudioOutputSystem`
    - Use `pyttsx3`; implement `speak()` and `speak_async()` (daemon thread)
    - Configurable voice and speech rate; log TTS errors without crashing
    - _Requirements: 15.4, 15.5, 15.6_

  - [~] 6.3 Create `eva/environment/visual_input.py` with `VisualInputEnvironment`
    - Use `cv2` for camera capture; use CLIP for image description
    - `capture_image()` returns `[visual] Seeing: <description>`; `reset()` loads into sub-environment
    - Release camera in `__del__`
    - _Requirements: 15.7, 15.8, 15.9, 15.10_

  - [~] 6.4 Create `eva/tools/screen_interaction.py` with `ScreenReadingTool` and `ScreenControlTool`
    - `ScreenReadingTool.execute()`: screenshot via `pyautogui`, OCR via `pytesseract`, truncate at 2000 chars
    - `ScreenControlTool`: requires `opt_in=True`; implement `MOVE`, `CLICK`, `TYPE` ops; enforce app whitelist
    - _Requirements: 15.11, 15.12, 15.13, 15.14, 15.15, 15.16, 15.17_

  - [~] 6.5 Create `eva/environment/multimodal.py` with `MultimodalLearningSystem`
    - `combine_inputs()` merges text/audio/visual with `[TEXT]`/`[AUDIO]`/`[VISUAL]` prefixes
    - `compute_multimodal_reward()` returns bonus for using multiple modalities
    - Track modality usage counts and ratios
    - _Requirements: 15.18, 15.19, 15.20_

  - [~] 6.6 Write unit tests for sensory I/O components
    - Test audio/visual error handling, screen control opt-in enforcement, multimodal reward computation
    - _Requirements: 15.21, 15.22, 15.23, 15.24_

- [~] 7. Checkpoint: Foundation Complete
  - Ensure all foundation tests pass, ask the user if questions arise.

- [ ] 8. Integration: Life Loop Enhancements
  - [~] 8.1 Enhance `eva/autonomy/life_loop.py` with checkpoint save/load, SIGINT handling, and metrics pipeline
    - Add `signal.signal(SIGINT, ...)` handler that sets `self.alive = False`
    - Implement `save_checkpoint()` storing brain weights, optimizer state, emotional state, curriculum phase, drives, naming state, clan state, self-monitor state, and initiative state
    - Implement `load_checkpoint()` that restores all state and resumes cycle count
    - Push vitals, emotions, emergence, activity, and chat metrics to `metrics_queue` after each cycle
    - Handle exceptions per cycle with logging; do not terminate on single-cycle errors
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.1, 10.2, 10.3, 10.4, 10.5_

  - [~] 8.2 Integrate all transparency and self-model systems into life loop
    - Pass `tool_tracker` to `CuriosityRewardEngine`; call `transparency_logger` on tool invocations, env switches, drive changes, emotional transitions, and curriculum phase changes
    - Call `emergence_detector` on self-reference patterns and crisis moments
    - Call `self_model.update()` each cycle; integrate self-model rewards into training
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.8, 5.9, 12.13, 14.15, 16.48, 16.49, 19.16, 19.17, 19.18_

  - [~] 8.3 Write unit tests for life loop enhancements
    - Test that checkpoint contains all required state components
    - Test that SIGINT sets alive flag and triggers final checkpoint
    - _Requirements: 6.9, 6.10, 10.7, 10.8_

- [ ] 9. Integration: World Environment and Tool Wiring
  - [~] 9.1 Register all tools in `ToolRegistry` and wire into `WorldEnvironment`
    - Register `WebSearch`, `FileHandler`, `DeviceControl`, `CodeExecution`, `PackageManager`, `ScreenReading`, `ScreenControl`
    - Parse `[TOOL_START][DISCOVER][TOOL_END]` and `[TOOL_START][DOCS][name][TOOL_END]` token sequences
    - Record every tool invocation in `ToolUsageTracker`; pass result to `CuriosityRewardEngine`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 11.16, 11.17, 12.1, 12.2, 12.3, 12.4, 17.1, 17.3, 17.5, 17.13_

  - [~] 9.2 Integrate sensory I/O into `WorldEnvironment`
    - Integrate `AudioInputEnvironment`, `AudioOutputEnvironment`, and `VisualInputEnvironment` when enabled in config
    - Speak EVA's conversation output via `AudioOutputSystem` when audio output is enabled
    - _Requirements: 15.3, 15.5, 15.10_

  - [~] 9.3 Write integration tests for world environment wiring
    - Test 10 consecutive cycles complete without error
    - Test tool invocation records appear in `ToolUsageTracker`
    - Test transparency logger captures env switches and tool calls
    - _Requirements: 6.9, 7.2, 7.3, 7.7, 7.8_

- [ ] 10. Integration: Enhanced TUI Dashboard
  - [~] 10.1 Add transparency panels to `eva/dashboard/tui.py`
    - Add `TransparencyLogPanel`: buffered log with level/category filtering and color-coded rendering
    - Add `EmergenceEventPanel`: displays emergence events with human-readable explanations
    - Add `MemoryBrowserPanel`: filters memories by time range, importance, valence, and source
    - Add `ThoughtStreamPanel`: displays prediction probabilities and decision traces
    - Dashboard polls metrics queue at least twice per second; processes all available messages per poll
    - _Requirements: 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 9.7, 9.8, 9.9_

  - [~] 10.2 Write unit tests for new TUI panels
    - Test filtering logic in `MemoryBrowserPanel` and `TransparencyLogPanel`
    - _Requirements: 3.9, 3.15_

- [ ] 11. Integration: Web Dashboard with Real-Time Updates
  - [~] 11.1 Create `eva/dashboard/web_dashboard.py` with `WebDashboard` using FastAPI + WebSocket
    - Serve HTTP on configurable port (default 8080); establish WebSocket at `/ws`
    - `broadcast_metrics()` polls `metrics_queue` and sends to all active connections; remove disconnected clients
    - Route incoming `chat` WebSocket messages to `metrics_queue` as `chat_input` events
    - Support multiple simultaneous WebSocket connections
    - _Requirements: 13.1, 13.2, 13.8, 13.9, 13.10, 13.11_

  - [~] 11.2 Implement responsive HTML/JS frontend embedded in `_get_dashboard_html()`
    - Panels: vital signs, emotional state, emergence, activity log, conversation/chat, system logs
    - CSS grid layout with `@media (max-width: 768px)` single-column fallback
    - JS WebSocket client: `handleMetric()` dispatches to `updateVitals`, `updateEmotions`, `updateEmergence`, `addActivity`, `addChat`, `addLog`
    - Chat input sends on Enter key; displays sent message immediately
    - _Requirements: 13.3, 13.4, 13.5, 13.6, 13.7, 13.12_

  - [~] 11.3 Add transparency features to web dashboard
    - Add panels for comprehensive logs, emergence events, memory browser with search/filter, and thought stream
    - Implement export endpoints (`/export/logs`, `/export/memories`) that return downloadable JSON/CSV
    - Memory filter UI: time range, importance threshold, emotional valence, source tag
    - _Requirements: 13.16, 13.17, 13.18, 13.19, 13.20, 13.21, 13.22, 13.23_

  - [~] 11.4 Write integration tests for web dashboard
    - Test that metrics pushed to queue appear at WebSocket clients
    - Test that chat messages from WebSocket reach the metrics queue
    - _Requirements: 13.14, 13.15_

- [~] 12. Checkpoint: Integration Complete
  - Ensure all integration tests pass, ask the user if questions arise.

- [ ] 13. Birth Ceremony: Validation and Entry Point
  - [~] 13.1 Create `eva/reproduction/birth_validator.py` with `BirthValidator`, `BirthReadinessReport`, and `CheckResult`
    - Implement `_check_brain_initialized`, `_check_curiosity_rewards`, `_check_emotional_system`, `_check_memory_system`, `_check_homeostatic_drives`, `_check_life_loop_cycle`, `_check_tool_availability`, `_check_dashboard_communication`
    - Each check wrapped in try-except; validation continues on individual failures
    - `validate()` returns `BirthReadinessReport` with status `"READY"` or `"NOT_READY"` and list of failing checks
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [~] 13.2 Create `eva_birth.py` as the main entry point with birth ceremony flow
    - Instantiate all systems (brain, emotions, memory, curiosity, life loop, tools, dashboards, transparency)
    - Run `BirthValidator.validate()`; if `"NOT_READY"`, print failing checks and exit
    - If `"READY"`, prompt for human confirmation (`input("Confirm birth? [y/N]: ")`)
    - On confirmation: create birth record with timestamp and baseline metrics; start `LifeLoop` thread (`daemon=False`); start TUI dashboard thread (`daemon=True`); start `WebDashboard` thread (`daemon=True`) if enabled
    - Print `"EVA IS ALIVE"` and web dashboard URL; print SIGINT shutdown instructions
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 13.13_

  - [~] 13.3 Write unit tests for birth validator
    - Test each check method independently with mock brain/emotions/memory
    - Test that `validate()` returns `"NOT_READY"` when any check fails
    - _Requirements: 1.7, 1.8_

  - [~] 13.4 Write integration test for birth ceremony flow
    - Test that `"NOT_READY"` report exits without starting life loop
    - Test that `"READY"` + confirmation starts life loop thread
    - _Requirements: 8.1, 8.2, 8.3_

- [ ] 14. Comprehensive Integration Test Suite
  - [~] 14.1 Create `tests/test_integration.py` covering all systems end-to-end
    - Test: minimal EVA instance creation with all core systems
    - Test: life loop executes at least 10 consecutive cycles
    - Test: each tool (web, file, device, code, package) invoked and returns valid results
    - Test: dashboard receives and processes metrics from life loop
    - Test: human chat input reaches life loop via metrics queue
    - Test: emergence monitor tracks self-references
    - Test: initiative system selects different environments based on varying drive levels
    - Test: checkpoint save and restore with all required state components
    - Test: SIGINT causes life loop to save state and terminate
    - Test: web dashboard WebSocket receives metrics and chat messages route correctly
    - Test: self-modification within safety bounds applies; major modification requires approval; rollback restores config
    - Test: self-model tracks state history and computes consistency rewards
    - Test: all transparency system events are logged correctly
    - Test: error handling prevents crashes in failure scenarios; graceful degradation when services unavailable
    - Report `"ALL SYSTEMS INTEGRATED"` when all pass; report specific failure otherwise
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10_

  - [~] 14.2 Write property-based tests for critical algorithms using `hypothesis`
    - Tokenization round-trip: encode then decode returns original characters
    - Memory consolidation: retention rate stays within [0, 1]
    - Curiosity reward: total reward is finite and within expected range for any valid inputs
    - _Requirements: 20.15, 20.16_

- [ ] 15. Quality Assurance: Error Handling and Validation
  - [~] 15.1 Add comprehensive error handling across all new modules
    - Wrap all tool operations, file I/O, network calls, and neural network ops in try-except with logging
    - Implement retry with exponential backoff for network requests (max 3 retries)
    - Add NaN detection in training loop; log and skip step on NaN gradient
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.8, 20.9, 20.10, 20.21, 20.22_

  - [~] 15.2 Add input validation across all new modules
    - Validate all tool parameters with type checking and range validation before execution
    - Validate all config values at startup with schema validation and default fallbacks
    - _Requirements: 20.5, 20.6, 20.7_

  - [~] 15.3 Verify thread safety for concurrent operations
    - Confirm `metrics_queue` is `queue.Queue` (thread-safe) for all inter-thread communication
    - Confirm only life loop thread modifies brain weights
    - _Requirements: 20.18_

- [ ] 16. Quality Assurance: CI/CD Pipeline
  - [~] 16.1 Create `.github/workflows/ci.yml` (or equivalent CI config) for automated testing
    - Run `pytest` with coverage report on every push
    - Enforce `pylint`, `mypy`, and `black` checks
    - Require passing tests and linting before merge
    - _Requirements: 20.23, 20.24, 20.25, 20.26, 20.27, 20.28_

- [~] 17. Final Checkpoint: Complete System Validation
  - Ensure all tests pass, all linting passes, and the system is ready for birth.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The design uses Python throughout; no language selection needed
- Tasks are organized in three phases: Foundation (1-7), Integration (8-12), Birth Ceremony (13-17)
- Checkpoints ensure incremental validation at logical phase boundaries
- Foundation phase builds all core components independently
- Integration phase wires everything together into the life loop
- Birth ceremony phase validates readiness and provides the entry point
