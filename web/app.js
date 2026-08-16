/* jshint esversion: 11 */
'use strict';

const API_BASE = '';  // Empty = same origin; change to 'http://localhost:8000' for local dev

// ──────────────────────────────────────────────────────────
// State
// ──────────────────────────────────────────────────────────
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;
let audioCtx      = null;
let analyser      = null;
let animFrameId   = null;

// ──────────────────────────────────────────────────────────
// DOM References
// ──────────────────────────────────────────────────────────
const recordBtn       = document.getElementById('record-btn');
const recordLabel     = document.getElementById('record-label');
const waveformWrapper = document.getElementById('waveform-canvas-wrapper');
const waveCanvas      = document.getElementById('waveform-canvas');
const waveCtx         = waveCanvas.getContext('2d');
const textQuery       = document.getElementById('text-query');
const textSubmitBtn   = document.getElementById('text-submit-btn');
const langSelect      = document.getElementById('lang-select');
const strategySelect  = document.getElementById('strategy-select');
const statusPill      = document.getElementById('status-pill');
const answerCard      = document.getElementById('answer-card');
const answerText      = document.getElementById('answer-text');
const guardrailBadge  = document.getElementById('guardrail-badge');
const answerStrategy  = document.getElementById('answer-strategy');
const answerLang      = document.getElementById('answer-lang');
const cacheTag        = document.getElementById('cache-tag');
const citationsCard   = document.getElementById('citations-card');
const citationsList   = document.getElementById('citations-list');
const skeleton        = document.getElementById('skeleton');
const emptyState      = document.getElementById('empty-state');
const runBenchBtn     = document.getElementById('run-benchmark-btn');
const benchResults    = document.getElementById('benchmark-results');

// Latency HUD
const hudTotal    = document.getElementById('hud-total');
const hudRetrieval= document.getElementById('hud-retrieval');
const hudGen      = document.getElementById('hud-gen');
const hudStt      = document.getElementById('hud-stt');
const latencyBar  = document.getElementById('latency-bar');

// Benchmark
const p50Val      = document.getElementById('p50-val');
const p70Val      = document.getElementById('p70-val');
const p100Val     = document.getElementById('p100-val');
const complianceVal = document.getElementById('compliance-val');
const benchTbody  = document.getElementById('benchmark-tbody');

// ──────────────────────────────────────────────────────────
// Health Check on Load
// ──────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      statusPill.textContent = `✓ Ready (${data.chunks_indexed.toLocaleString()} chunks)`;
      statusPill.className = 'status-pill ready';
    } else {
      statusPill.textContent = '✗ Backend Error';
      statusPill.className = 'status-pill error';
    }
  } catch (e) {
    statusPill.textContent = '⚠ Connecting…';
    statusPill.className = 'status-pill loading';
  }
});

// ──────────────────────────────────────────────────────────
// Waveform Visualizer
// ──────────────────────────────────────────────────────────
function drawWaveform() {
  if (!analyser) return;
  animFrameId = requestAnimationFrame(drawWaveform);

  const buf = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(buf);

  const W = waveCanvas.width;
  const H = waveCanvas.height;
  waveCtx.clearRect(0, 0, W, H);

  // Background
  waveCtx.fillStyle = '#F7F0DC';
  waveCtx.fillRect(0, 0, W, H);

  // Waveform
  waveCtx.beginPath();
  waveCtx.lineWidth = 2.5;
  waveCtx.strokeStyle = '#2D6A4F';
  waveCtx.shadowColor = '#52A77A';
  waveCtx.shadowBlur = 6;

  const sliceWidth = W / buf.length;
  let x = 0;
  for (let i = 0; i < buf.length; i++) {
    const v = buf[i] / 128.0;
    const y = (v * H) / 2;
    i === 0 ? waveCtx.moveTo(x, y) : waveCtx.lineTo(x, y);
    x += sliceWidth;
  }
  waveCtx.lineTo(W, H / 2);
  waveCtx.stroke();
}

// ──────────────────────────────────────────────────────────
// Recording
// ──────────────────────────────────────────────────────────
recordBtn.addEventListener('click', async () => {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });

    // Audio context for visualization
    audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
    analyser  = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = handleRecordingComplete;
    mediaRecorder.start(100);

    isRecording = true;
    recordBtn.classList.add('active');
    recordBtn.querySelector('.mic-icon').textContent = '⏹';
    recordLabel.textContent = 'Recording…';
    waveformWrapper.classList.remove('hidden');
    document.getElementById('recorder-card').classList.add('recording');
    drawWaveform();
  } catch (e) {
    alert('Microphone access denied. Please allow microphone access and try again.');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  if (animFrameId) cancelAnimationFrame(animFrameId);
  isRecording = false;
  recordBtn.classList.remove('active');
  recordBtn.querySelector('.mic-icon').textContent = '🎤';
  recordLabel.textContent = 'Tap to Record';
  waveformWrapper.classList.add('hidden');
  document.getElementById('recorder-card').classList.remove('recording');
}

async function handleRecordingComplete() {
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const arrayBuffer = await blob.arrayBuffer();
  const base64 = arrayBufferToBase64(arrayBuffer);

  await submitRequest({
    audio_base64: base64,
    language: langSelect.value,
    chunking_strategy: strategySelect.value,
    top_k: 3,
    enable_cache: true,
  });
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return window.btoa(binary);
}

// ──────────────────────────────────────────────────────────
// Text Query
// ──────────────────────────────────────────────────────────
textSubmitBtn.addEventListener('click', handleTextSubmit);
textQuery.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleTextSubmit(); }
});

async function handleTextSubmit() {
  const q = textQuery.value.trim();
  if (!q) return;
  await submitRequest({
    text_query: q,
    language: langSelect.value,
    chunking_strategy: strategySelect.value,
    top_k: 3,
    enable_cache: true,
  });
}

// ──────────────────────────────────────────────────────────
// Sample Chips
// ──────────────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const q    = chip.dataset.q;
    const lang = chip.dataset.lang;
    langSelect.value = lang;
    textQuery.value  = q;
    textQuery.focus();
  });
});

// ──────────────────────────────────────────────────────────
// Submit Request → API
// ──────────────────────────────────────────────────────────
async function submitRequest(payload) {
  showSkeleton();

  try {
    const endpoint = payload.audio_base64 ? '/voice-base64' : '/query';
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResponse(data);
  } catch (e) {
    hideSkeleton();
    renderError(e.message);
  }
}

// ──────────────────────────────────────────────────────────
// Render Response
// ──────────────────────────────────────────────────────────
function renderResponse(data) {
  hideSkeleton();
  emptyState.classList.add('hidden');

  // Latency HUD
  const lat = data.latency || {};
  const totalMs = lat.total_pipeline_ms || 0;
  const isOver  = !lat.under_target_latency;

  updateHUD(hudTotal,    totalMs,             isOver);
  updateHUD(hudRetrieval, lat.retrieval_ms || 0, false);
  updateHUD(hudGen,       lat.generation_ms || 0, false);
  updateHUD(hudStt,       lat.stt_ms || 0, false);

  // Latency bar
  const pct = Math.min((totalMs / 200) * 100, 100);
  latencyBar.style.width = pct + '%';
  latencyBar.classList.toggle('over', isOver);

  // Answer Card
  answerCard.classList.remove('hidden');
  answerText.textContent = data.answer || '—';

  const passed = data.success && (data.guardrail_decision?.passed !== false);
  guardrailBadge.textContent = passed ? '✓ Guardrails Passed' : '⚠ Guardrail Triggered';
  guardrailBadge.className   = 'guardrail-badge ' + (passed ? 'pass' : 'fail');

  answerStrategy.textContent = '🧩 ' + (data.chunking_strategy_used || strategySelect.value).replace(/_/g, ' ');
  answerLang.textContent     = '🌐 ' + (data.language || langSelect.value).toUpperCase();
  cacheTag.classList.toggle('hidden', !lat.is_cache_hit);

  // Citations
  const citations = data.citations || [];
  if (citations.length > 0) {
    citationsCard.classList.remove('hidden');
    citationsList.innerHTML = '';
    citations.slice(0, 3).forEach((c, i) => {
      const div = document.createElement('div');
      div.className = 'citation-item';
      div.innerHTML = `
        <div class="citation-meta">
          <span class="citation-score">Score ${c.score?.toFixed(3) || '—'}</span>
          <span>Passage #${c.passage_index}</span>
          <span>${c.metadata?.query_type || 'GENERAL'}</span>
        </div>
        <div class="citation-excerpt">${escapeHtml(c.excerpt || '')}</div>
      `;
      citationsList.appendChild(div);
    });
  } else {
    citationsCard.classList.add('hidden');
  }
}

function updateHUD(el, val, isOver) {
  el.textContent = val ? val.toFixed(1) : '—';
  el.classList.remove('over-budget', 'under-budget');
  if (val) el.classList.add(isOver ? 'over-budget' : 'under-budget');
}

function renderError(msg) {
  answerCard.classList.remove('hidden');
  emptyState.classList.add('hidden');
  answerText.textContent = `Error: ${msg}`;
  guardrailBadge.textContent = '✗ Error';
  guardrailBadge.className = 'guardrail-badge fail';
  citationsCard.classList.add('hidden');
}

// ──────────────────────────────────────────────────────────
// Benchmark
// ──────────────────────────────────────────────────────────
runBenchBtn.addEventListener('click', async () => {
  runBenchBtn.disabled = true;
  runBenchBtn.textContent = 'Running…';
  benchResults.classList.remove('hidden');

  // Show loading placeholder
  p50Val.textContent = p70Val.textContent = p100Val.textContent = complianceVal.textContent = '…';
  benchTbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-light);padding:20px;">Running benchmark queries…</td></tr>';

  try {
    const strategy = strategySelect.value;
    const res = await fetch(`${API_BASE}/benchmark?strategy=${strategy}&num_queries=10`);
    const data = await res.json();
    const s = data.benchmark_summary;

    const e2e = s.end_to_end_stats;
    p50Val.textContent       = e2e.p50_ms.toFixed(1);
    p70Val.textContent       = e2e.p70_ms.toFixed(1);
    p100Val.textContent      = e2e.p100_max_ms.toFixed(1);
    complianceVal.textContent= s.target_compliance_pct.toFixed(0);

    benchTbody.innerHTML = '';
    (s.detailed_traces || []).forEach((t, i) => {
      const isOver = t.total_ms > 200;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHtml(t.query)}">${escapeHtml(t.query.slice(0, 40))}${t.query.length > 40 ? '…' : ''}</td>
        <td>${t.language.toUpperCase()}</td>
        <td class="${isOver ? 'cell-warn' : 'cell-ok'}">${t.total_ms.toFixed(1)}</td>
        <td>${t.retrieval_ms.toFixed(1)}</td>
        <td>${t.gen_ms.toFixed(1)}</td>
        <td><span class="${t.passed_guardrails ? 'cell-pass' : 'cell-fail'}">${t.passed_guardrails ? 'Pass' : 'Blocked'}</span></td>
        <td class="${isOver ? 'cell-warn' : 'cell-ok'}">${isOver ? '⚠ Over' : '✓ OK'}</td>
      `;
      benchTbody.appendChild(tr);
    });
  } catch (e) {
    benchTbody.innerHTML = `<tr><td colspan="8" style="color:var(--red-400);padding:20px;">Error: ${escapeHtml(e.message)}</td></tr>`;
  }

  runBenchBtn.disabled = false;
  runBenchBtn.textContent = 'Run Again';
});

// ──────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────
function showSkeleton() {
  emptyState.classList.add('hidden');
  answerCard.classList.add('hidden');
  citationsCard.classList.add('hidden');
  skeleton.classList.remove('hidden');
}
function hideSkeleton() {
  skeleton.classList.add('hidden');
}
function escapeHtml(str) {
  return String(str || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
