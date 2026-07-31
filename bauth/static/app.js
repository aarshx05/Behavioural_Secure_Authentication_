'use strict';

/* Keystroke timing capture.
 *
 * `keydown`/`keyup` in the browser give press and release timing directly, on a
 * monotonic clock (performance.now), with none of the problems the terminal
 * version has: no OS keyboard hook to install, no race against that hook going
 * live, no echoing of the password, and the field can be type=password so it
 * masks properly.
 *
 * Auto-repeat must be ignored -- holding a key fires keydown repeatedly but
 * only one physical press happened.
 */
class Recorder {
  constructor(input, onSubmit) {
    this.input = input;
    this.onSubmit = onSubmit;
    this.reset();

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); this.onSubmit(); return; }
      if (e.repeat) return;
      if (e.key === 'Backspace') {
        this.corrections++;
        this.events.pop();
        return;
      }
      if (e.key.length !== 1) return;   // Shift, Tab, arrows, F-keys...
      // `code` is the physical key and is stable regardless of modifier state;
      // `key` is the resulting character and is not. See the keyup handler.
      this.events.push({
        char: e.key,
        code: e.code || ('Key:' + e.key.toLowerCase()),
        down: performance.now(),
        up: null,
      });
    });

    input.addEventListener('keyup', (e) => {
      // Pairing must be by physical key, not by character. `event.key` is
      // resolved against the modifier state at the moment the event fires, so
      // releasing Shift before the letter turns a "D" keydown into a "d"
      // keyup (and "!" into "1"). Matching on the character therefore never
      // finds the press, the release is never recorded, and every capital
      // letter or shifted symbol silently invalidates the sample.
      const code = e.code || (e.key.length === 1 ? 'Key:' + e.key.toLowerCase() : null);
      if (!code) return;

      for (let i = this.events.length - 1; i >= 0; i--) {
        if (this.events[i].code === code && this.events[i].up === null) {
          this.events[i].up = performance.now();
          return;
        }
      }
    });

    // Pasting would produce a perfect-looking sample with no keystrokes at all.
    input.addEventListener('paste', (e) => e.preventDefault());
  }

  reset() {
    this.events = [];
    this.corrections = 0;
    this.input.value = '';
  }

  get text() { return this.events.map((e) => e.char).join(''); }

  payload() {
    return {
      events: this.events.filter((e) => e.up !== null),
      corrections: this.corrections,
    };
  }

  /** Why this sample cannot be used, or null if it is fine. */
  problem(expected) {
    if (this.input.value !== expected) return 'That does not match the password.';
    if (this.text !== expected) return 'Some keystrokes were missed. Please retype it.';
    if (this.events.some((e) => e.up === null)) return 'A key was still held down. Please retype it.';
    return null;
  }
}

/* ------------------------------------------------------------------ utils */
const $ = (id) => document.getElementById(id);

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: res.status, data: await res.json() };
}

function setFeedback(el, msg, kind) {
  el.textContent = msg || '';
  el.className = 'feedback' + (kind ? ' ' + kind : '');
}

function pct(x) { return (x * 100).toFixed(1) + '%'; }

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ));
}

let appConfig = null;

function fillSelect(selectId, values, selected) {
  const el = $(selectId);
  if (!el) return;
  el.innerHTML = values.map((value) =>
    `<option value="${esc(value)}"${value === selected ? ' selected' : ''}>${esc(value)}</option>`
  ).join('');
}

async function loadConfig() {
  const res = await fetch('/api/config');
  appConfig = await res.json();
  const policies = appConfig.adaptation_policies || [];
  const selected = appConfig.default_adaptation_policy || '';
  fillSelect('reg-policy', policies, selected);
  fillSelect('ret-policy', policies, selected);
}

/* -------------------------------------------------------------------- tabs */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $(tab.dataset.panel).classList.add('active');
    if (tab.dataset.panel === 'profiles') loadProfiles();
    if (tab.dataset.panel === 'context') loadContext();
  });
});

/* Render the grouped context attributes as tables. */
function contextTables(ctx) {
  const group = (title, obj) => `
    <h3>${title}</h3>
    <table>${Object.entries(obj)
      .map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`)
      .join('')}</table>`;
  return group('Network', ctx.network) +
         group('Device', ctx.device) +
         group('Clock', ctx.clock) +
         `<p class="hint" style="margin-top:14px">Device fingerprint:
            <code>${esc(ctx.fingerprint)}</code></p>`;
}

async function loadContext() {
  const res = await fetch('/api/context');
  const data = await res.json();
  const box = $('ctx-detail');
  box.innerHTML = contextTables(data.context);
  box.classList.remove('hidden');
}

document.getElementById('ctx-refresh').addEventListener('click', loadContext);

/* ---------------------------------------------------------------- register */
(() => {
  const need = parseInt($('reg-count').parentElement.textContent.split('/')[1], 10) || 10;
  let target = need;
  let samples = [];
  let password = '';

  const rec = new Recorder($('reg-input'), submitSample);

  $('reg-start').addEventListener('click', () => {
    const user = $('reg-user').value.trim();
    password = $('reg-pass').value;
    if (!user || !password) {
      setFeedback($('reg-feedback'), 'User ID and password are required.', 'err');
      $('reg-capture').classList.remove('hidden');
      return;
    }
    samples = [];
    rec.reset();
    $('reg-result').classList.add('hidden');
    $('reg-capture').classList.remove('hidden');
    updateProgress();
    setFeedback($('reg-feedback'), '');
    $('reg-input').focus();
  });

  function updateProgress() {
    $('reg-count').textContent = samples.length;
    $('reg-bar').style.width = (samples.length / target * 100) + '%';
  }

  async function submitSample() {
    const problem = rec.problem(password);
    if (problem) {
      setFeedback($('reg-feedback'), problem, 'err');
      rec.reset();
      return;
    }
    samples.push(rec.payload());
    rec.reset();
    updateProgress();

    if (samples.length < target) {
      setFeedback($('reg-feedback'), `Captured. ${target - samples.length} to go.`, 'ok');
      return;
    }

    setFeedback($('reg-feedback'), 'Training model...', '');
    const { data } = await post('/api/register', {
      user_id: $('reg-user').value.trim(),
      password,
      model_choice: $('reg-model').value,
      adaptation_policy: $('reg-policy').value,
      samples,
    });

    if (!data.ok) {
      setFeedback($('reg-feedback'), data.error, 'err');
      samples = [];
      updateProgress();
      return;
    }

    setFeedback($('reg-feedback'), '');
    $('reg-capture').classList.add('hidden');
    const p = data.profile;
    $('reg-result').innerHTML = `
      <p class="verdict pass">Registered &ldquo;${esc(p.user_id)}&rdquo;</p>
      <table>
        <tr><th>Samples</th><td class="num">${data.info.authentic_samples}</td></tr>
        <tr><th>Features per sample</th><td class="num">${p.features}</td></tr>
        <tr><th>Synthetic negatives</th><td class="num">${data.info.negatives}</td></tr>
        <tr><th>Model preset</th><td class="num">${data.info.model_choice === 1 ? 'Harsh' : 'Easy'}</td></tr>
        <tr><th>Adaptation policy</th><td class="num">${esc(data.profile.adaptation_policy)}</td></tr>
        <tr><th>Starting threshold</th><td class="num">${p.threshold}</td></tr>
      </table>`;
    $('reg-result').classList.remove('hidden');
  }
})();

/* ------------------------------------------------------------------ verify */
(() => {
  const rec = new Recorder($('ver-input'), submit);

  async function submit() {
    const user = $('ver-user').value.trim();
    const password = $('ver-pass').value;
    if (!user || !password) {
      setFeedback($('ver-feedback'), 'User ID and password are required.', 'err');
      return;
    }
    const problem = rec.problem(password);
    if (problem) {
      setFeedback($('ver-feedback'), problem, 'err');
      rec.reset();
      return;
    }

    setFeedback($('ver-feedback'), 'Scoring...', '');
    const payload = Object.assign({ user_id: user, password }, rec.payload());
    const { data } = await post('/api/verify', payload);
    rec.reset();

    if (!data.ok) {
      setFeedback($('ver-feedback'), data.error || 'Verification could not run.', 'err');
      $('ver-result').classList.add('hidden');
      return;
    }

    setFeedback($('ver-feedback'), '');
    const r = data.risk;
    const passed = data.authenticated;
    const fillColor = passed ? 'var(--ok)' : 'var(--bad)';

    let notes = '';
    if (data.adopted) notes += '<li>This sample was added to your profile (adaptive learning).</li>';
    if (data.retrained) notes += '<li>The profile was automatically retrained on your recent typing.</li>';
    if (data.quarantined && !data.adopted) notes += '<li>This sample entered the quarantine buffer for later promotion review.</li>';
    if (data.lockout) notes += `<li>Adaptive learning paused: ${esc(data.lockout)}</li>`;
    if (data.analysis) notes += `<li><strong>${esc(data.analysis.verdict)}:</strong> ${esc(data.analysis.message)}</li>`;
    if (data.quality && data.quality.flags.length) notes += `<li>Quality: ${data.quality.flags.map(esc).join('; ')}</li>`;
    if (r.factors.length) notes += `<li>Context: ${r.factors.map(esc).join('; ')}</li>`;

    const t = data.timing || {};
    $('ver-result').innerHTML = `
      <p class="verdict ${passed ? 'pass' : 'fail'}">
        ${passed ? 'Verified' : 'Rejected'}
      </p>
      <div class="scores">
        <div class="score"><div class="label">Biometric score</div><div class="value">${pct(data.probability)}</div></div>
        <div class="score"><div class="label">Required</div><div class="value">${pct(data.required)}</div></div>
        <div class="score"><div class="label">Disagreement</div><div class="value">${pct(data.disagreement)}</div></div>
        <div class="score"><div class="label">Quality</div><div class="value">${pct(data.quality ? data.quality.score : 0)}</div></div>
        <div class="score"><div class="label">Context risk</div>
          <div class="value"><span class="badge ${r.level}">${r.level}</span></div></div>
      </div>
      <div class="meter">
        <div class="fill" style="width:${data.probability * 100}%;background:${fillColor}"></div>
        <div class="mark" style="left:${data.required * 100}%" title="required"></div>
      </div>
      ${notes ? `<ul class="notes">${notes}</ul>` : ''}
      <details class="drawer">
        <summary>Typing measured &mdash; ${t.characters} keys,
          ${t.total_ms} ms total, ${t.mean_hold_ms} ms mean dwell${t.corrections ? `, ${t.corrections} correction(s)` : ''}</summary>
      </details>
      <details class="drawer">
        <summary>Context scored for this attempt</summary>
        <div class="drawer-body">${contextTables(data.context)}</div>
      </details>`;
    $('ver-result').classList.remove('hidden');
  }
})();

/* ----------------------------------------------------------------- retrain */
(() => {
  let samples = [];
  let password = '';
  let target = 5;

  const rec = new Recorder($('ret-input'), submitSample);

  $('ret-start').addEventListener('click', () => {
    password = $('ret-pass').value;
    target = Math.max(1, parseInt($('ret-n').value, 10) || 5);
    if (!$('ret-user').value.trim() || !password) {
      setFeedback($('ret-feedback'), 'User ID and password are required.', 'err');
      $('ret-capture').classList.remove('hidden');
      return;
    }
    samples = [];
    rec.reset();
    $('ret-total').textContent = target;
    $('ret-result').classList.add('hidden');
    $('ret-capture').classList.remove('hidden');
    update();
    setFeedback($('ret-feedback'), '');
    $('ret-input').focus();
  });

  function update() {
    $('ret-count').textContent = samples.length;
    $('ret-bar').style.width = (samples.length / target * 100) + '%';
  }

  async function submitSample() {
    const problem = rec.problem(password);
    if (problem) {
      setFeedback($('ret-feedback'), problem, 'err');
      rec.reset();
      return;
    }
    samples.push(rec.payload());
    rec.reset();
    update();

    if (samples.length < target) {
      setFeedback($('ret-feedback'), `Captured. ${target - samples.length} to go.`, 'ok');
      return;
    }

    setFeedback($('ret-feedback'), 'Retraining...', '');
    const { data } = await post('/api/retrain', {
      user_id: $('ret-user').value.trim(),
      password,
      adaptation_policy: $('ret-policy').value,
      samples,
    });

    if (!data.ok) {
      setFeedback($('ret-feedback'), data.error, 'err');
      samples = [];
      update();
      return;
    }

    setFeedback($('ret-feedback'), '');
    $('ret-capture').classList.add('hidden');
    $('ret-result').innerHTML = `
      <p class="verdict pass">Retrained</p>
      <table>
        <tr><th>Window size</th><td class="num">${data.profile.samples} samples</td></tr>
        <tr><th>Effective positives</th><td class="num">${data.info.effective_positives} (recency-weighted)</td></tr>
        <tr><th>Synthetic negatives</th><td class="num">${data.info.negatives}</td></tr>
        <tr><th>Adaptation policy</th><td class="num">${esc(data.profile.adaptation_policy)}</td></tr>
        <tr><th>Drift before retrain</th><td class="num">${data.drift_before} sd</td></tr>
      </table>`;
    $('ret-result').classList.remove('hidden');
  }
})();

/* ---------------------------------------------------------------- profiles */
async function loadProfiles() {
  const res = await fetch('/api/users');
  const { users } = await res.json();
  const list = $('prof-list');

  if (!users.length) {
    list.innerHTML = '<p class="hint">No users registered yet.</p>';
    $('prof-detail').classList.add('hidden');
    return;
  }

  list.innerHTML = users.map((u) => `
    <div class="profile-card" data-user="${esc(u.user_id)}">
      <div class="id">${esc(u.user_id)}</div>
      <div class="meta">${u.samples} samples${u.legacy ? ' &middot; legacy v1' : ''}</div>
    </div>`).join('');

  list.querySelectorAll('.profile-card').forEach((card) => {
    card.addEventListener('click', () => showProfile(card.dataset.user));
  });
}

async function showProfile(userId) {
  const res = await fetch('/api/status/' + encodeURIComponent(userId));
  const data = await res.json();
  const box = $('prof-detail');

  if (!data.ok) {
    box.innerHTML = `<p class="verdict fail">${esc(data.error)}</p>`;
    box.classList.remove('hidden');
    return;
  }

  const s = data.profile.status;
  const rows = Object.entries(s)
    .map(([k, v]) => `<tr><th>${esc(k.replace(/_/g, ' '))}</th><td>${esc(v)}</td></tr>`)
    .join('');

  const events = data.events.length
    ? `<h3>Recent events</h3><ul class="notes">${data.events.map((e) =>
        `<li>${esc(e.event)}${e.window ? ` &mdash; window ${e.window}` : ''}</li>`).join('')}</ul>`
    : '';

  box.innerHTML = `<p class="verdict">${esc(userId)}</p><table>${rows}</table>${events}`;
  box.classList.remove('hidden');
}

loadProfiles();
$('prof-refresh').addEventListener('click', loadProfiles);
loadConfig();
