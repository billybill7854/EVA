(() => {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const tagDevice = $('#tag-device');
  const tagStep = $('#tag-step');
  const tagParams = $('#tag-params');
  const tagRam = $('#tag-ram');
  const tagLive = $('#tag-live');

  const thoughtStream = $('#thought-stream');
  const insightList = $('#insight-list');
  const memoryList = $('#memory-list');
  const evolutionList = $('#evolution-list');
  const toolList = $('#tool-list');
  const conversation = $('#conversation');
  const input = $('#chat-input');
  const form = $('#chat-form');

  const btnVoice = $('#btn-voice');
  const btnSpeak = $('#btn-speak');

  let speakEnabled = false;
  let voiceOn = false;

  // ---------- status + history bootstrap ----------
  async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
  }

  function renderStatus(s) {
    tagDevice.textContent = s.device;
    tagStep.textContent = `step ${s.step}`;
    tagParams.textContent = `${s.parameter_count.toLocaleString()} params`;
    tagRam.textContent = `${s.memory_estimate_mb} MB`;
  }

  function pushStreamEntry(list, {title, body, kind, confidence, timeStr}) {
    const li = document.createElement('li');
    if (kind) li.classList.add(kind);
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = [kind, timeStr].filter(Boolean).join(' · ');
    if (confidence !== undefined) {
      const c = document.createElement('span');
      c.className = 'confidence' + (confidence < 0.5 ? ' low' : '');
      c.textContent = `· ${(confidence * 100).toFixed(0)}%`;
      meta.appendChild(c);
    }
    li.appendChild(meta);
    if (title) {
      const t = document.createElement('div');
      t.innerHTML = `<strong>${escapeHtml(title)}</strong>`;
      li.appendChild(t);
    }
    if (body) {
      const b = document.createElement('div');
      b.textContent = body;
      li.appendChild(b);
    }
    list.prepend(li);
    while (list.children.length > 200) list.removeChild(list.lastChild);
  }

  function escapeHtml(s) {
    return (s || '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  async function loadInitial() {
    try {
      const status = await fetchJson('/api/status');
      renderStatus(status);
      const [insights, thoughts, ev, tools] = await Promise.all([
        fetchJson('/api/memory/insights?limit=50'),
        fetchJson('/api/memory/thoughts?limit=100'),
        fetchJson('/api/evolution'),
        fetchJson('/api/tools'),
      ]);
      insights.insights.forEach((i) => pushStreamEntry(insightList, {
        kind: 'insight', title: i.kind, body: i.description,
        confidence: i.confidence, timeStr: i.created_at,
      }));
      thoughts.thoughts.forEach((t) => pushStreamEntry(thoughtStream, {
        kind: t.category, body: t.content, timeStr: t.created_at,
      }));
      ev.history.forEach((h) => pushStreamEntry(evolutionList, {
        kind: 'evolution',
        title: `step ${h.step} — ${h.parameter_count.toLocaleString()} params`,
        body: JSON.stringify(h.genes),
        timeStr: h.created_at,
      }));
      tools.tools.forEach((t) => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${escapeHtml(t.name)}</strong>
          <span class="meta">${t.available ? 'available' : 'unavailable'}</span>
          <div>${escapeHtml(t.description)}</div>`;
        toolList.appendChild(li);
      });
    } catch (e) {
      console.error(e);
    }
  }

  function addBubble(role, text) {
    const div = document.createElement('div');
    div.className = `bubble ${role}`;
    div.textContent = text;
    conversation.appendChild(div);
    conversation.scrollTop = conversation.scrollHeight;
    return div;
  }

  // ---------- websocket ----------
  let ws;
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => { tagLive.textContent = 'live'; tagLive.classList.remove('pulse'); };
    ws.onclose = () => {
      tagLive.textContent = 'reconnecting…'; tagLive.classList.add('pulse');
      setTimeout(connectWS, 1500);
    };
    ws.onmessage = (e) => {
      let event;
      try { event = JSON.parse(e.data); } catch { return; }
      handleEvent(event);
    };
  }

  function handleEvent(event) {
    switch (event.type) {
      case 'status':
        renderStatus(event.status);
        break;
      case 'thought':
        pushStreamEntry(thoughtStream, {
          kind: event.category, body: event.content,
          timeStr: `step ${event.step}`,
        });
        break;
      case 'insight':
        pushStreamEntry(insightList, {
          kind: 'insight', title: event.insight.kind,
          body: event.insight.description,
          confidence: event.insight.confidence,
          timeStr: `step ${event.insight.step}`,
        });
        break;
      case 'evolution':
        pushStreamEntry(evolutionList, {
          kind: 'evolution',
          title: `${event.event.kind}: ${event.event.detail}`,
          body: JSON.stringify(event.event.genome_after),
          timeStr: `step ${event.event.step}`,
        });
        break;
      case 'turn':
        tagStep.textContent = `step ${event.step}`;
        break;
    }
  }

  // ---------- chat ----------
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    addBubble('me', text);
    input.value = '';
    try {
      const r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const { response } = await r.json();
      addBubble('eva', response);
      pushStreamEntry(memoryList, {
        kind: 'output', body: `you: ${text} → eva: ${response}`,
      });
      if (speakEnabled && 'speechSynthesis' in window) {
        const utt = new SpeechSynthesisUtterance(response);
        speechSynthesis.speak(utt);
      }
    } catch (e) {
      addBubble('eva', `(error: ${e.message})`);
    }
  });

  // ---------- voice ----------
  btnSpeak.addEventListener('click', () => {
    speakEnabled = !speakEnabled;
    btnSpeak.classList.toggle('active', speakEnabled);
  });

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition;
  if (SR) {
    recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.onresult = (ev) => {
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        if (!ev.results[i].isFinal) continue;
        const text = ev.results[i][0].transcript.trim();
        if (text) {
          input.value = text;
          form.dispatchEvent(new Event('submit'));
        }
      }
    };
    recognition.onerror = () => { voiceOn = false; btnVoice.classList.remove('active'); };
  } else {
    btnVoice.title = 'Voice unavailable (browser lacks SpeechRecognition)';
    btnVoice.disabled = true;
  }

  btnVoice.addEventListener('click', () => {
    if (!recognition) return;
    voiceOn = !voiceOn;
    btnVoice.classList.toggle('active', voiceOn);
    if (voiceOn) recognition.start();
    else recognition.stop();
  });

  // ---------- tabs ----------
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      $$('.tab-content').forEach((c) => {
        c.classList.toggle('hidden', c.id !== `tab-${tab.dataset.tab}`);
      });
    });
  });

  loadInitial();
  connectWS();
})();
