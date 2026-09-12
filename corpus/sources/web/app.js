const STATES = new Set(['healthy', 'degraded', 'down']);

export async function fetchServices({ signal } = {}) {
  const res = await fetch('/api/services', { signal, headers: { accept: 'application/json' } });
  if (!res.ok) throw new Error(`services request failed: ${res.status} ${res.statusText}`);
  const body = await res.json();
  return body.items.map(normalise);
}

function normalise(raw) {
  return {
    name: String(raw.name ?? '').trim(),
    owner: String(raw.owner ?? 'unassigned'),
    state: STATES.has(raw.state) ? raw.state : 'down',
    p99: Number.isFinite(raw.p99_ms) ? Math.round(raw.p99_ms) : null,
  };
}

export function filterServices(items, { q = '', state = '' } = {}) {
  const needle = q.trim().toLowerCase();
  return items.filter((item) => {
    if (state && item.state !== state) return false;
    if (!needle) return true;
    return item.name.toLowerCase().includes(needle) || item.owner.toLowerCase().includes(needle);
  });
}

function row(item) {
  const tr = document.createElement('tr');
  const pill = `<span class="pill ${item.state}">${item.state}</span>`;
  tr.innerHTML = `
    <td>${escapeHtml(item.name)}</td>
    <td>${escapeHtml(item.owner)}</td>
    <td>${pill}</td>
    <td class="num">${item.p99 ?? '—'}</td>`;
  return tr;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

export function render(items, tbody = document.querySelector('#grid tbody')) {
  tbody.replaceChildren(...items.map(row));
}

async function main() {
  const controller = new AbortController();
  let all = [];
  try {
    all = await fetchServices({ signal: controller.signal });
  } catch (err) {
    console.error('could not load services', err);
    return;
  }
  const form = document.getElementById('filters');
  const apply = () => {
    const data = new FormData(form);
    render(filterServices(all, { q: data.get('q') ?? '', state: data.get('state') ?? '' }));
  };
  form.addEventListener('submit', (event) => { event.preventDefault(); apply(); });
  apply();
}

if (typeof document !== 'undefined') main();
