const socket = io({ transports: ['websocket', 'polling'], reconnection: true });
let driveSequence = 0;
let driveTimer = null;
let latestSensors = null;

const $ = (id) => document.getElementById(id);
const fmtDistance = (value) => Number.isFinite(Number(value)) ? `${Number(value).toFixed(0)} cm` : '—';

socket.on('connect', () => {
  $('connection').textContent = 'CONNECTED';
  $('connection').className = 'pill good';
});
socket.on('disconnect', () => {
  $('connection').textContent = 'OFFLINE';
  $('connection').className = 'pill bad';
});
socket.on('command_error', ({message}) => { $('error').textContent = message; });
socket.on('system_error', ({component, error}) => { $('error').textContent = `${component}: ${error}`; });
socket.on('system_health', renderHealth);
socket.on('enrollment_progress', renderEnrollment);
socket.on('conversation', renderConversation);
socket.on('telemetry', renderTelemetry);

function renderTelemetry(data) {
  $('mode').textContent = data.mode.toUpperCase();
  document.querySelectorAll('[data-mode]').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === data.mode));
  latestSensors = data.sensors;
  for (const key of ['fl','f','fr','rl','rr']) $(`s-${key}`).textContent = fmtDistance(data.sensors[key]);
  $('sensor-age').textContent = data.sensors.age_ms == null ? 'stale' : `${data.sensors.age_ms.toFixed(0)} ms`;
  const decision = data.motion;
  $('decision').textContent = decision.reason;
  $('decision').style.color = decision.allowed ? 'var(--green)' : 'var(--red)';
  $('motion-vector').textContent = `final: ${decision.final.vx.toFixed(2)} / ${decision.final.vy.toFixed(2)} / ${decision.final.omega.toFixed(2)}`;
  const identity = data.vision.identity;
  $('identity-status').textContent = identity.status.toUpperCase();
  $('identity-name').textContent = identity.name || 'Unknown';
  $('identity-confidence').textContent = `confidence ${Number(identity.confidence || 0).toFixed(2)}`;
  $('vision-backend').textContent = data.vision.backend;
  renderEnrollment(data.enrollment);
  renderHealth(data.health);
  drawRadar(data.sensors);
}

function renderHealth(health) {
  const keys = ['runtime','motors','sensors','camera','microphone','tts','database','cloud','mcp_agent','wake_phrase','estop'];
  $('health').innerHTML = keys.map(key => {
    const value = Boolean(health[key]);
    const expectedBad = key === 'estop';
    const good = expectedBad ? !value : value;
    let label = value ? 'READY' : 'OFF';
    if (key === 'tts') label = health.tts_backend || label;
    if (key === 'camera') label = health.vision_backend || label;
    return `<div class="health-item ${good ? '' : 'bad'}"><span>${key.replace('_',' ')}</span><strong>${label}</strong></div>`;
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
  $('provider').textContent = String(data.provider || 'local').toUpperCase();
}

function appendMessage(text, className, meta) {
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
  const w = canvas.width, h = canvas.height, cx = w / 2, cy = h * .58;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#163d44'; ctx.lineWidth = 1;
  [60,120,180].forEach(r => { ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI); ctx.stroke(); });
  ctx.fillStyle = '#48d7e8'; ctx.fillRect(cx - 22, cy - 15, 44, 30);
  const readings = [
    ['fl', -135], ['f', -90], ['fr', -45], ['rl', 145], ['rr', 35]
  ];
  readings.forEach(([key, deg]) => {
    const distance = Math.min(200, Number(s[key] || 200));
    const radius = distance / 200 * 180;
    const angle = deg * Math.PI / 180;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    ctx.strokeStyle = distance < 20 ? '#ff4d62' : distance < 35 ? '#ffc857' : '#39d98a';
    ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
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
$('message').addEventListener('keydown', e => { if (e.key === 'Enter') sendText(); });
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
