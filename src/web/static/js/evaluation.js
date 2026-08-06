const socket = io({ transports: ['websocket', 'polling'], reconnection: true });
let driveSequence = 0;
let driveTimer = null;
let latestSensors = null;

const $ = (id) => document.getElementById(id);
const fmtDistance = (value) => Number.isFinite(Number(value)) ? `${Number(value).toFixed(0)} cm` : '—';

socket.on('connect', () => {
  $('connection').innerHTML = '<i></i> Connected';
  $('connection').className = 'pill good';
});
socket.on('disconnect', () => {
  $('connection').innerHTML = '<i></i> Offline';
  $('connection').className = 'pill bad';
});
socket.on('command_error', ({message}) => { $('error').textContent = message; });
socket.on('system_error', ({component, error}) => { $('error').textContent = `${component}: ${error}`; });
socket.on('system_health', renderHealth);
socket.on('enrollment_progress', renderEnrollment);
socket.on('conversation', renderConversation);
socket.on('telemetry', renderTelemetry);

function renderTelemetry(data) {
  $('mode').textContent = data.mode;
  document.querySelectorAll('[data-mode]').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === data.mode));
  latestSensors = data.sensors;
  for (const key of ['fl','f','fr','rl','rr']) $(`s-${key}`).textContent = fmtDistance(data.sensors[key]);
  $('sensor-age').textContent = data.sensors.age_ms == null ? 'stale' : `${data.sensors.age_ms.toFixed(0)} ms`;
  const decision = data.motion;
  $('decision').textContent = decision.reason;
  $('decision').style.color = decision.allowed ? 'var(--green)' : 'var(--red)';
  $('motion-vector').textContent = `Final · ${decision.final.vx.toFixed(2)} / ${decision.final.vy.toFixed(2)} / ${decision.final.omega.toFixed(2)}`;
  const identity = data.vision.identity;
  $('identity-status').textContent = identity.status.toUpperCase();
  $('identity-name').textContent = identity.name || 'Unknown';
  $('identity-confidence').textContent = `Confidence ${Number(identity.confidence || 0).toFixed(2)}`;
  $('vision-backend').textContent = data.vision.backend;
  renderEnrollment(data.enrollment);
  renderHealth(data.health);
  drawRadar(data.sensors);
}

function renderHealth(health) {
  const keys = ['runtime','motors','sensors','camera','microphone','tts','llm','database','mcp_agent','wake_phrase','estop'];
  $('health').innerHTML = keys.map(key => {
    const value = Boolean(health[key]);
    const expectedBad = key === 'estop';
    const good = expectedBad ? !value : value;
    let label = value ? 'READY' : 'OFF';
    if (key === 'tts') label = health.tts_backend || label;
    if (key === 'camera') label = health.vision_backend || label;
    return `<div class="health-item ${good ? '' : 'bad'}"><span class="health-name"><i></i>${key.replaceAll('_',' ')}</span><strong>${label}</strong></div>`;
  }).join('');
}

function renderEnrollment(status) {
  if (!status) return;
  const required = Math.max(1, Number(status.required || 20));
  const accepted = Number(status.accepted || 0);
  $('enroll-progress').style.width = `${Math.min(100, accepted / required * 100)}%`;
  $('enroll-label').textContent = status.complete
    ? `${status.name || 'Person'} enrolled${status.error ? ` (${status.error})` : ''}`
    : status.active ? `Capturing ${accepted}/${required} good face samples…` : (status.error || 'Enrollment idle');
}

function renderConversation(data) {
  if (data.transcript) appendMessage(data.transcript, 'user', data.source);
  if (data.response) appendMessage(data.response, '', `${data.provider}${data.fallback_reason ? ' · fallback' : ''}`);
  const provider = String(data.provider || 'local');
  $('provider').replaceChildren();
  const dot = document.createElement('i');
  $('provider').append(dot, document.createTextNode(` ${provider}`));
}

function appendMessage(text, className, meta) {
  const emptyState = $('chat').querySelector('.chat-empty');
  if (emptyState) emptyState.remove();
  const p = document.createElement('p');
  p.className = className;
  p.textContent = text;
  const small = document.createElement('small');
  small.textContent = meta || '';
  p.appendChild(small);
  $('chat').appendChild(p);
  $('chat').scrollTop = $('chat').scrollHeight;
}

function drawRadar(s) {
  const canvas = $('radar');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height, cx = w / 2, cy = h * .66;
  ctx.clearRect(0, 0, w, h);
  const maxRadius = Math.min(w * 0.42, h * 0.64);
  ctx.strokeStyle = 'rgba(113, 227, 211, 0.12)';
  ctx.lineWidth = 1;
  [0.33, 0.66, 1].forEach(scale => {
    ctx.beginPath();
    ctx.arc(cx, cy, maxRadius * scale, Math.PI, 2 * Math.PI);
    ctx.stroke();
  });
  [-150, -120, -90, -60, -30].forEach(deg => {
    const angle = deg * Math.PI / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * maxRadius, cy + Math.sin(angle) * maxRadius);
    ctx.stroke();
  });

  ctx.save();
  ctx.translate(cx, cy);
  ctx.fillStyle = '#182c2b';
  ctx.strokeStyle = '#71e3d3';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(-24, -17, 48, 34, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#a5f5e9';
  ctx.beginPath();
  ctx.arc(0, -4, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  const readings = [
    ['fl', -135], ['f', -90], ['fr', -45], ['rl', 145], ['rr', 35]
  ];
  readings.forEach(([key, deg]) => {
    const distance = Math.min(200, Number(s[key] || 200));
    const radius = distance / 200 * maxRadius;
    const angle = deg * Math.PI / 180;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    ctx.strokeStyle = distance < 20 ? '#ff6b74' : distance < 35 ? '#eec77b' : '#68dfa6';
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.shadowColor = ctx.strokeStyle;
    ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  });
}

function sendText() {
  const text = $('message').value.trim();
  if (!text) return;
  socket.emit('send_text', {text});
  $('message').value = '';
}
function sendAgentCmd() {
  const text = $('message').value.trim();
  if (!text) return;
  socket.emit('run_mcp_agent', {text});
  $('message').value = '';
}
$('send').onclick = sendText;
if ($('agent-cmd')) $('agent-cmd').onclick = sendAgentCmd;
$('message').addEventListener('keydown', e => { if (e.key === 'Enter') sendAgentCmd(); });
$('listen').onclick = () => socket.emit('start_listening', {});
$('estop').onclick = () => socket.emit('stop', {});
$('clear-estop').onclick = () => socket.emit('clear_estop', {});
$('showcase').onclick = () => socket.emit('perform_showcase', {});
$('enroll').onclick = () => socket.emit('enroll_person', {name: $('enroll-name').value.trim()});
$('remember').onclick = () => socket.emit('remember_fact', {text: $('fact').value.trim()});
document.querySelectorAll('[data-mode]').forEach(btn => btn.onclick = () => socket.emit('set_mode', {mode: btn.dataset.mode}));

let currentSpeed = 1.0;
const speedSlider = $('speed-slider');
const speedValue = $('speed-value');

if (speedSlider && speedValue) {
  speedSlider.addEventListener('input', (e) => {
    currentSpeed = parseFloat(e.target.value);
    speedValue.textContent = currentSpeed.toFixed(2);
    const percentage = ((currentSpeed - 0.2) / 0.8) * 100;
    speedSlider.style.background = `linear-gradient(90deg, var(--accent) 0 ${percentage}%, rgba(255,255,255,.08) ${percentage}% 100%)`;
  });
}

function getDriveVector(name) {
  if (name === 'stop') return [0, 0, 0];
  if (name === 'forward') return [currentSpeed, 0, 0];
  if (name === 'backward') return [-currentSpeed, 0, 0];
  if (name === 'left') return [0, -currentSpeed, 0];
  if (name === 'right') return [0, currentSpeed, 0];
  if (name === 'spin-left') return [0, 0, -currentSpeed];
  if (name === 'spin-right') return [0, 0, currentSpeed];
  return [0, 0, 0];
}

function emitDrive(vector) {
  driveSequence += 1;
  socket.emit('manual_drive', {vx: vector[0], vy: vector[1], omega: vector[2], sequence: driveSequence});
}

function beginDrive(name) {
  endDrive();
  
  // If the robot is in IDLE mode, automatically switch to MANUAL mode
  const currentMode = $('mode').textContent.toLowerCase();
  if (currentMode === 'idle' && name !== 'stop') {
    socket.emit('set_mode', {mode: 'manual'});
  }

  const vector = getDriveVector(name);
  emitDrive(vector);
  if (name !== 'stop') {
    driveTimer = setInterval(() => {
      emitDrive(getDriveVector(name));
    }, 100);
  }
}

function endDrive() {
  if (driveTimer) clearInterval(driveTimer);
  driveTimer = null;
  emitDrive([0, 0, 0]);
}

document.querySelectorAll('[data-drive]').forEach(btn => {
  btn.addEventListener('pointerdown', e => { e.preventDefault(); beginDrive(btn.dataset.drive); });
  btn.addEventListener('pointerup', endDrive);
  btn.addEventListener('pointerleave', endDrive);
  btn.addEventListener('pointercancel', endDrive);
});
