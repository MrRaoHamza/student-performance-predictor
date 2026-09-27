'use strict';

// ─── Toggle buttons (yes/no) ─────────────────────────────────────────────────
document.querySelectorAll('.toggle-btns').forEach(group => {
  const name   = group.dataset.name;
  const hidden = document.querySelector(`input[type="hidden"][name="${name}"]`);
  const btns   = group.querySelectorAll('.toggle-btn');

  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('on'));
      btn.classList.add('on');
      if (hidden) hidden.value = btn.dataset.val;
    });
  });
});

// ─── Scale pickers (1–5) ─────────────────────────────────────────────────────
document.querySelectorAll('.scale-row').forEach(row => {
  const name   = row.dataset.name;
  const hidden = document.querySelector(`input[type="hidden"][name="${name}"]`);
  const dots   = row.querySelectorAll('.scale-dot');

  dots.forEach(dot => {
    dot.addEventListener('click', () => {
      dots.forEach(d => d.classList.remove('active'));
      dot.classList.add('active');
      if (hidden) hidden.value = dot.dataset.val;
    });
  });
});

// ─── Feature importance ───────────────────────────────────────────────────────
async function loadImportances() {
  try {
    const data = await fetch('/model-stats').then(r => r.json());
    renderFeats(data.feature_importances);
  } catch (_) {}
}

function renderFeats(feats) {
  const el = document.getElementById('featList');
  if (!el) return;

  let entries = Array.isArray(feats)
    ? feats
    : Object.entries(feats).map(([name, importance]) => ({ name, importance }));

  entries.sort((a, b) => b.importance - a.importance);
  const top    = entries.slice(0, 6);
  const maxVal = top[0]?.importance || 1;

  el.innerHTML = top.map(f => {
    const barW  = ((f.importance / maxVal) * 100).toFixed(0);
    const pctLbl = f.importance < 1
      ? (f.importance * 100).toFixed(1) + '%'
      : f.importance.toFixed(1) + '%';
    const lbl = String(f.name).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    return `<div class="feat-row">
      <div class="feat-meta">
        <span class="feat-name">${lbl}</span>
        <span class="feat-pct">${pctLbl}</span>
      </div>
      <div class="feat-track"><div class="feat-fill" style="width:${barW}%"></div></div>
    </div>`;
  }).join('');
}

// ─── Form utils ───────────────────────────────────────────────────────────────
function collectForm(form) {
  const fd  = new FormData(form);
  const out = {};
  for (const [k, v] of fd.entries()) {
    out[k] = (v !== '' && !isNaN(v)) ? Number(v) : v;
  }
  return out;
}

function validateForm(form) {
  const fields = form.querySelectorAll('[required]');
  for (const f of fields) {
    if (!f.value) {
      f.focus();
      f.style.borderColor = 'var(--fail)';
      setTimeout(() => { f.style.borderColor = ''; }, 2000);
      return false;
    }
  }
  return true;
}

// ─── Toast ────────────────────────────────────────────────────────────────────
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 4500);
}

// ─── Render result ────────────────────────────────────────────────────────────
function renderResult(data) {
  const isPass = data.pass === 1;
  const prob   = data.probability; // 0–100

  document.getElementById('emptyState').style.display  = 'none';
  document.getElementById('verdictBlock').classList.add('visible');
  document.getElementById('factorsBlock').classList.add('visible');

  // Badge
  const badge = document.getElementById('verdictBadge');
  badge.className = `verdict-badge ${isPass ? 'pass' : 'fail'}`;
  badge.textContent = isPass ? '✓ PASS' : '✗ FAIL';

  // Title + sub
  document.getElementById('verdictTitle').textContent = isPass ? 'Likely to Pass' : 'At Risk of Failing';
  document.getElementById('verdictSub').textContent   = isPass
    ? `${prob}% confidence · Low concern`
    : `${(100 - prob).toFixed(1)}% chance of failing · Needs attention`;

  // SVG circle (circumference = 2π×24 ≈ 150.8)
  const arc      = document.getElementById('circleArc');
  const circLabel= document.getElementById('circleLabel');
  const circ     = 150.8;
  const offset   = circ - (prob / 100) * circ;
  arc.style.stroke = isPass ? 'var(--pass)' : 'var(--fail)';
  setTimeout(() => { arc.style.strokeDashoffset = offset; }, 50);
  circLabel.textContent = `${Math.round(prob)}%`;

  // Bar
  const fill = document.getElementById('probFill');
  fill.className = `prob-fill ${isPass ? 'pass' : 'fail'}`;
  fill.style.width = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${prob}%`; }));
  document.getElementById('probPct').textContent = `${prob}%`;

  // Risk tag
  const riskTag = document.getElementById('riskTag');
  const riskMap = {
    'Low Risk':       'low',
    'Moderate Risk':  'moderate',
    'High Risk':      'high',
    'Very High Risk': 'very-high'
  };
  riskTag.className   = `risk-tag ${riskMap[data.risk_band] || 'moderate'}`;
  riskTag.textContent = data.risk_band;

  // Factors
  const list  = document.getElementById('factorList');
  const count = document.getElementById('factorCount');
  const items = data.insights || [];
  count.textContent = `${items.length} factor${items.length !== 1 ? 's' : ''}`;
  list.innerHTML = items.map(i => `
    <div class="factor-item ${i.type}">
      <div class="factor-pip"></div>
      <span>${i.text}</span>
    </div>`).join('') || '<div class="factor-item strength"><div class="factor-pip"></div><span>No notable risk factors found.</span></div>';

  // Feature importances from response
  if (data.top_features?.length) {
    renderFeats(data.top_features.map(f => ({ name: f.name, importance: f.importance })));
  }

  // Mobile scroll
  if (window.innerWidth < 1100) {
    document.getElementById('resultSidebar').scrollIntoView({ behavior: 'smooth' });
  }
}

// ─── Form submit ──────────────────────────────────────────────────────────────
const form = document.getElementById('predictForm');
const btn  = document.getElementById('predictBtn');

form.addEventListener('submit', async e => {
  e.preventDefault();

  if (!validateForm(form)) {
    toast('Please fill in all required fields before running.');
    return;
  }

  btn.classList.add('loading');

  try {
    const res  = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(collectForm(form))
    });
    const data = await res.json();

    if (!res.ok) {
      toast(data.error || 'Something went wrong. Check your inputs.');
      return;
    }
    renderResult(data);

  } catch (err) {
    toast('Cannot reach the server. Make sure Flask is running on port 5000.');
    console.error(err);
  } finally {
    btn.classList.remove('loading');
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
loadImportances();
