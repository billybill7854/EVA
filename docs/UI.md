# Interactive UI

Open with:

```bash
python run.py ui
# default: http://127.0.0.1:8765
```

## Layout

* **Left — Thought Stream.** Raw firing from the brain: every input,
  output, and internal hop. Expect noise; the Insights tab filters.
* **Middle — Conversation.** Voice input by default (click 🎤 to
  toggle). Text input is always available. Click 🔈 to have EVA speak
  its replies (uses the browser's `speechSynthesis`).
* **Right — Inspector** with four tabs:
  * **Insights** — emergent signals above a confidence threshold.
  * **Memory** — episodic replay from the SQLite store.
  * **Evolution** — genome history with growth events.
  * **Tools** — available tools + availability status.

## Voice requirements

EVA's UI uses the browser-native Web Speech API for both input
(`SpeechRecognition`) and output (`speechSynthesis`). Supported in
Chromium-based browsers and Safari. Firefox users should fall back to
the chat textbox.

No server-side speech model is installed — this keeps the 8 GiB RAM
budget focused on the brain.

## Signal vs noise

The *Thoughts* stream shows everything. The *Insights* stream only
shows events flagged by the `EmergenceEventDetector`, with a confidence
badge (low confidence is tinted warm, high confidence cool). Users can
decide whether an insight is a genuine emergent property or just a
plausible artefact.

## Live updates

The UI connects over WebSocket to `/ws`. Every turn, insight, and
evolution event is broadcast to all subscribers — open multiple tabs
to watch from different angles.

## API

If you want to drive EVA programmatically:

| Endpoint | Description |
|---|---|
| `GET /api/status` | Current genome, parameter count, RAM estimate, device. |
| `GET /api/memory/episodes?limit=50` | Persistent episodes. |
| `GET /api/memory/insights?limit=50` | Filtered insight stream. |
| `GET /api/memory/thoughts?limit=200` | Raw thought stream. |
| `GET /api/evolution` | Genome history. |
| `GET /api/tools` | Tool catalogue with availability. |
| `POST /api/chat` | `{"text": "..."}` → `{"response": "..."}`. |
| `WS  /ws` | Multiplexed live stream of all events. |
