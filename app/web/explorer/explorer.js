// MINERVA · COSMOS Catalog Explorer
// Build-free vanilla ES module. Talks only to the endpoints in app/API.md.
// Contract shapes: /api/fields, /api/catalog/query {total,rows}, /api/object/{field}/{id},
// /api/eazy/{field}/{template}/{id}, /api/cutout, /api/dja/preview/{root}/{basename}.
'use strict';

const PAGE = 100;                 // default rows per page (simple pagination = virtualization)
const NEAR_PAGE = 11;             // nearest-neighbour flow: target + 10 neighbours
const NEAR_RADIUS = 30;           // default cone radius (″) for the nearest flow
const CONE_DEFAULT_R = 3;         // default cone radius (″) when ra+dec given but r blank
const DZ_THRESH = 0.15;           // outlier wedge half-width; mirrors Inspector app.js state.thresh
const SORT_COLS = ['id', 'ra', 'dec', 'z_phot', 'lmass', 'mag', 'sep', 'dist'];

// ── Query-string field wiring ─────────────────────────────────────
// Maps a form control id -> the /api/catalog/query param name it feeds.
const FORM_MAP = {
  'q-ids': 'ids', 'q-ra': 'ra', 'q-dec': 'dec', 'q-radius': 'radius_arcsec',
  'q-zmin': 'z_min', 'q-zmax': 'z_max', 'q-lmassmin': 'lmass_min', 'q-lmassmax': 'lmass_max',
  'q-magband': 'mag_band', 'q-magmin': 'mag_min', 'q-magmax': 'mag_max',
  'q-grademin': 'grade_min', 'q-uvj': 'uvj',
};
const CHECK_MAP = { 'q-hasspec': 'has_spec', 'q-usephot': 'use_phot', 'q-nostar': 'no_star' };

// ── Runtime state ─────────────────────────────────────────────────
const state = {
  fields: [],          // /api/fields list
  field: 'cosmos',
  template: 'sfhz_blue_agn',
  params: {},          // last executed query params (excluding paging/sort)
  sort: 'id',          // active sort key (may be '-'-prefixed)
  offset: 0,
  total: 0,
  rows: [],            // current page rows
  selId: null,
  gridSize: null,      // size (″) the all-bands grid is currently rendered at, or null
  csvConfirm: false,   // two-step CSV-cap confirmation latch (B3)
  pageSize: PAGE,      // rows per page (NEAR_PAGE while in the nearest flow)
  nearId: null,        // target id of an active nearest-neighbour query (URL near=)
  magBand: 'f444w',    // mag band of the LAST EXECUTED query (drives the "mag <band>" header)
};

const $ = (id) => document.getElementById(id);

// ── fetch helper ──────────────────────────────────────────────────
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
}

function toast(msg, isErr) {
  const el = document.createElement('div');
  el.className = 'toast' + (isErr ? ' err' : '');
  el.textContent = msg;
  $('toast-stack').appendChild(el);
  setTimeout(() => el.remove(), 5200);
}

const num = (v, dp) => (v == null || Number.isNaN(v)) ? '—' : Number(v).toFixed(dp);

// Attribute-safe escaping for values interpolated into title="…" tooltips.
function escAttr(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/"/g, '&quot;')
    .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Signed Δz/(1+z) = (z_spec − z_phot)/(1+z_spec). null when either input is
// missing (so it renders muted and never counts as an outlier). Matches the
// Inspector's dzN (app.js), except signed rather than absolute.
function dzNorm(zs, zp) {
  if (zs == null || zp == null || Number.isNaN(zs) || Number.isNaN(zp)) return null;
  if (zs <= -0.99) return null;   // guard the (1+zs) denominator / DJA −1 sentinel
  return (zs - zp) / (1 + zs);
}
// "<class> Δz" cell/tag markup: signed 3dp, danger-coloured past the threshold,
// muted em-dash when null. `label` prefixes the value (e.g. "Δz/(1+z) ").
function dzMarkup(dz, label) {
  if (dz == null) return `<span class="text-muted">${label || ''}—</span>`;
  const cls = Math.abs(dz) > DZ_THRESH ? 'dz-out' : '';
  const sign = dz >= 0 ? '+' : '';
  return `<span class="${cls}">${label || ''}${sign}${dz.toFixed(3)}</span>`;
}

// ── Help overlay ('?' cheat-sheet) ────────────────────────────────
function openHelp() { $('help-backdrop').classList.remove('hidden'); $('help-close').focus(); }
// On close, return focus to the '?' toggle so keyboard users aren't dropped at
// the document root (E2).
function closeHelp() {
  const bd = $('help-backdrop');
  if (bd.classList.contains('hidden')) return;
  bd.classList.add('hidden');
  const btn = $('btn-help');
  if (btn) btn.focus();
}
function toggleHelp() { $('help-backdrop').classList.contains('hidden') ? openHelp() : closeHelp(); }

// Keep Tab focus inside the open help dialog (E2). Esc still closes.
function trapHelpKeydown(e) {
  if (e.key === 'Escape') { e.preventDefault(); closeHelp(); return; }
  if (e.key !== 'Tab') return;
  const f = $('help-backdrop').querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

// ── Lightbox (click-to-enlarge RGB plate, C5c) ────────────────────
function openLightbox(src, alt) {
  let box = $('lightbox');
  if (!box) {
    box = document.createElement('div');
    box.id = 'lightbox';
    box.className = 'lightbox';
    box.innerHTML = '<img class="lightbox-img" alt="">';
    box.addEventListener('click', closeLightbox);
    document.body.appendChild(box);
  }
  const img = box.querySelector('.lightbox-img');
  img.src = src; img.alt = alt || '';
  box.classList.remove('hidden');
}
function closeLightbox() {
  const box = $('lightbox');
  if (box) box.classList.add('hidden');
}
function lightboxOpen() {
  const box = $('lightbox');
  return box && !box.classList.contains('hidden');
}

// First-load cue before any query has run.
function showFirstRunPrompt() {
  const empty = $('results-empty');
  empty.textContent = 'Enter criteria and press Search.';
  empty.classList.remove('hidden');
}

// ══════════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════════
async function init() {
  wireEvents();
  try {
    const data = await fetchJSON('/api/fields');
    state.fields = data.fields || [];
  } catch (e) {
    toast('Could not load /api/fields: ' + e.message, true);
    return;
  }
  if (!state.fields.length) { toast('No fields available.', true); return; }

  // Restore from URL before populating selectors so values stick.
  const url = new URLSearchParams(location.search);
  const wantField = url.get('field');
  state.field = state.fields.some(f => f.name === wantField) ? wantField : state.fields[0].name;

  populateFieldSelect();
  onFieldChange(url);   // fills template + band selectors, restores form, maybe searches
}

function currentField() { return state.fields.find(f => f.name === state.field) || state.fields[0]; }

function populateFieldSelect() {
  const sel = $('sel-field');
  sel.innerHTML = '';
  for (const f of state.fields) {
    const o = document.createElement('option');
    o.value = f.name; o.textContent = f.title || f.name;
    sel.appendChild(o);
  }
  sel.value = state.field;
}

// Fill template + mag-band selectors for the active field, then restore URL state.
function onFieldChange(url) {
  const f = currentField();
  const tsel = $('sel-template');
  tsel.innerHTML = '';
  for (const t of (f.templates || [])) {
    const o = document.createElement('option'); o.value = t; o.textContent = t; tsel.appendChild(o);
  }
  const wantT = url && url.get('template');
  state.template = (f.templates || []).includes(wantT) ? wantT : (f.default_template || (f.templates || [])[0]);
  tsel.value = state.template;

  const bsel = $('q-magband');
  bsel.innerHTML = '';
  for (const b of (f.bands || [])) {
    const o = document.createElement('option'); o.value = b; o.textContent = b; bsel.appendChild(o);
  }
  if ((f.bands || []).includes('f444w')) bsel.value = 'f444w';

  $('nav-inspector').href = `/inspector/?field=${encodeURIComponent(state.field)}&template=${encodeURIComponent(state.template)}`;

  // Brand sub-title reflects the ACTIVE field (was hard-coded "COSMOS").
  const bs = document.querySelector('.brand-sub');
  if (bs) bs.textContent = `· ${f.title || f.name} Catalog Explorer`;

  // Reflect the active template's SPS capability in the filter form + headers.
  updateLmassAvailability();
  updateColumnHeaders();

  // Restore form + run query if the URL carried one.
  if (url) restoreFromURL(url);
}

// ══════════════════════════════════════════════════════════════════
//  URL STATE
// ══════════════════════════════════════════════════════════════════
function restoreFromURL(url) {
  // Nearest-neighbour deep link (e.g. the Inspector back-link
  // /?field=..&template=..&near={mid}) takes precedence: it drives the cone
  // form itself and runs its own query, so skip the generic form restore.
  const near = url.get('near');
  if (near != null && near !== '') {
    const nid = parseInt(near, 10);
    if (!Number.isNaN(nid)) { runNearest(nid); return; }
  }
  let any = false;
  for (const [id, param] of Object.entries(FORM_MAP)) {
    const v = url.get(param);
    if (v != null) { $(id).value = v; any = true; }
  }
  for (const [id, param] of Object.entries(CHECK_MAP)) {
    const v = url.get(param);
    if (v != null) { $(id).checked = v === '1'; any = true; }
  }
  if (url.get('sort')) state.sort = url.get('sort');
  state.offset = parseInt(url.get('offset') || '0', 10) || 0;
  const sel = url.get('sel');
  if (any || url.get('has_spec') != null) {
    runSearch(false).then(() => { if (sel) selectRow(parseInt(sel, 10)); });
  } else {
    // No query in the URL — show the first-run cue.
    showFirstRunPrompt();
    // Only a selection was shared — select without a full query.
    if (sel) selectRow(parseInt(sel, 10));
  }
}

function syncURL() {
  const p = new URLSearchParams();
  p.set('field', state.field);
  p.set('template', state.template);
  for (const [k, v] of Object.entries(state.params)) p.set(k, v);
  if (state.sort && state.sort !== 'id') p.set('sort', state.sort);
  if (state.offset) p.set('offset', String(state.offset));
  if (state.nearId != null) p.set('near', String(state.nearId));
  if (state.selId != null) p.set('sel', String(state.selId));
  history.replaceState(null, '', location.pathname + '?' + p.toString());
}

// ══════════════════════════════════════════════════════════════════
//  QUERY
// ══════════════════════════════════════════════════════════════════
// Read the form into a params object (only non-empty values). Returns null when
// the cone inputs are half-filled (B1) — the caller aborts and a toast explains.
function readForm() {
  const p = {};
  for (const [id, param] of Object.entries(FORM_MAP)) {
    const el = $(id);
    if (el.disabled) continue;   // e.g. log M★ inputs when the template has no SPS
    let v = el.value.trim();
    if (v === '') continue;
    if (param === 'ids') v = v.split(/[\s,]+/).filter(Boolean).join(',');
    if (v !== '') p[param] = v;
  }
  // Cone-search foot-guns (B1): a cone needs BOTH ra and dec. If exactly one is
  // present, abort with an explanatory toast. If both are present but the radius
  // is blank, default it to the placeholder (3″) instead of silently dropping
  // the cone. A stray radius with no center stays meaningless -> discarded.
  const hasRa = 'ra' in p, hasDec = 'dec' in p;
  if (hasRa !== hasDec) {
    toast('Cone search needs both RA and Dec', true);
    return null;
  }
  if (hasRa && hasDec) {
    if (!('radius_arcsec' in p)) p.radius_arcsec = String(CONE_DEFAULT_R);
  } else if ('radius_arcsec' in p) {
    delete p.radius_arcsec;
  }
  if ($('q-hasspec').checked) p.has_spec = '1';
  // use_phot / no_star default to 1 on the server; only send the "off" case.
  if (!$('q-usephot').checked) p.use_phot = '0';
  if (!$('q-nostar').checked) p.no_star = '0';
  return p;
}

// "Clear" button (B2): reset every query control to its default, drop the
// nearest-neighbour context, and shrink the URL back to bare field+template.
// The current table + detail are intentionally left in place until the next
// Search, so a mis-click doesn't wipe results the user is still reading.
function clearForm() {
  for (const id of Object.keys(FORM_MAP)) {
    if (id === 'q-magband') continue;   // band selector has no empty state; keep it
    $(id).value = '';
  }
  $('q-hasspec').checked = false;
  $('q-usephot').checked = true;
  $('q-nostar').checked = true;
  state.nearId = null;
  state.csvConfirm = false;
  const p = new URLSearchParams();
  p.set('field', state.field);
  p.set('template', state.template);
  history.replaceState(null, '', location.pathname + '?' + p.toString());
}

function buildQueryURL() {
  const p = new URLSearchParams(state.params);
  p.set('field', state.field);
  p.set('template', state.template);
  if (state.sort) p.set('sort', state.sort);
  p.set('limit', String(state.pageSize));
  p.set('offset', String(state.offset));
  return '/api/catalog/query?' + p.toString();
}

// fromForm=true -> re-read the form and reset to page 0. A manual form search
// leaves the nearest-neighbour mode (restores full page size, clears near=).
async function runSearch(fromForm) {
  if (fromForm) {
    const p = readForm();
    if (p === null) return;   // half-filled cone: toast already shown, abort (B1)
    state.params = p; state.offset = 0;
    state.pageSize = PAGE; state.nearId = null;
  }
  showSkeleton(true);
  $('btn-search').disabled = true;
  let data;
  try {
    data = await fetchJSON(buildQueryURL());
  } catch (e) {
    showSkeleton(false);
    $('btn-search').disabled = false;
    toast('Query failed: ' + e.message, true);
    return;
  }
  state.total = data.total || 0;
  state.rows = data.rows || [];
  // Freeze the mag-column header to the band this query actually used (the
  // server defaults to f444w when none is sent), so it never live-follows the
  // selector before a re-query.
  state.magBand = state.params.mag_band || 'f444w';
  showSkeleton(false);
  $('btn-search').disabled = false;
  renderTable();
  renderSummary();
  renderPager();
  $('btn-csv').disabled = state.total === 0;
  syncURL();
}

function renderSummary() {
  const s = $('result-summary');
  if (state.total === 0) { s.textContent = 'No matches.'; return; }
  const from = state.offset + 1, to = Math.min(state.offset + state.rows.length, state.total);
  s.textContent = `${state.total.toLocaleString()} match${state.total === 1 ? '' : 'es'} · showing ${from}–${to}`;
}

// ══════════════════════════════════════════════════════════════════
//  NEAREST-NEIGHBOUR CONTEXT (Inspector back-link ?near= / detail button)
// ══════════════════════════════════════════════════════════════════
// Overwrite the query form with a clean cone centred on (ra,dec). Restrictive
// gates (use_phot / no_star / has_spec) are turned OFF and all non-cone criteria
// cleared so the target and its true nearest neighbours are all returned.
function setConeForm(ra, dec, radius) {
  for (const id of Object.keys(FORM_MAP)) {
    if (id === 'q-ra' || id === 'q-dec' || id === 'q-radius' || id === 'q-magband') continue;
    $(id).value = '';
  }
  $('q-ra').value = ra;
  $('q-dec').value = dec;
  $('q-radius').value = String(radius);
  $('q-hasspec').checked = false;
  $('q-usephot').checked = false;
  $('q-nostar').checked = false;
}

// Fetch object coords, seed a 30″ cone sorted by distance (limit 11), then
// auto-select the target (dist 0.00) with its detail pane open.
async function runNearest(id) {
  if (id == null || Number.isNaN(id)) return;
  let obj;
  try {
    obj = await fetchJSON(`/api/object/${encodeURIComponent(state.field)}/${id}`);
  } catch (e) {
    toast(`Could not load object ${id}: ${e.message}`, true);
    return;
  }
  if (obj.ra == null || obj.dec == null) { toast(`Object ${id} has no coordinates`, true); return; }
  setConeForm(obj.ra, obj.dec, NEAR_RADIUS);
  const p = readForm();
  if (p === null) return;   // setConeForm always sets both ra+dec, so this is defensive
  state.params = p;
  state.sort = 'dist';
  state.offset = 0;
  state.pageSize = NEAR_PAGE;
  state.nearId = id;
  await runSearch(false);
  selectRow(id);            // target sorts first at dist 0.00; open its detail
}

// ══════════════════════════════════════════════════════════════════
//  RESULTS TABLE
// ══════════════════════════════════════════════════════════════════
function showSkeleton(on) {
  $('results-skeleton').classList.toggle('hidden', !on);
  $('results-empty').classList.add('hidden');
  if (on) {
    $('results-body').innerHTML = '';
    const box = $('results-skeleton');
    box.innerHTML = '';
    for (let i = 0; i < 12; i++) {
      const d = document.createElement('div');
      d.className = 'skel'; d.style.width = (55 + Math.random() * 45) + '%';
      box.appendChild(d);
    }
  }
}

function gradeDot(g) {
  const cls = g >= 3 ? 'g3' : g === 2 ? 'g2' : g === 1 ? 'g1' : 'g0';
  return `<span class="spec-dot ${cls}"></span>`;
}

function renderTable() {
  const body = $('results-body');
  body.innerHTML = '';
  const empty = $('results-empty');
  empty.textContent = 'No objects match this query.';
  empty.classList.toggle('hidden', state.total !== 0);
  // DIST″ column visible only when the current rows carry cone-center distance.
  const hasDist = state.rows.some(r => r.dist_arcsec != null);
  $('results-table').classList.toggle('has-dist', hasDist);
  updateSortArrows();
  updateColumnHeaders();
  const frag = document.createDocumentFragment();
  for (const r of state.rows) {
    const tr = document.createElement('tr');
    tr.dataset.id = r.id;
    tr.tabIndex = 0;
    tr.setAttribute('role', 'button');
    tr.setAttribute('aria-label', `Object ${r.id} — open detail`);
    if (r.id === state.selId) tr.classList.add('selected');
    // spec cell (A1): grade dot + grade number + z=<zs 4dp>. The visible grade
    // number solves the colour-only legibility of the dot; grating, DJA id and
    // sep move into the cell's title tooltip.
    let specCell = '<span class="text-muted">—</span>';
    if (r.spec) {
      const sp = r.spec, g = sp.grade ?? 0;
      const title = `${sp.dja || 'spectrum'} · ${sp.grating || 'grating ?'} · sep ${num(sp.sep, 2)}″`;
      specCell = `<span class="spec-cell" title="${escAttr(title)}">${gradeDot(g)}` +
        `<span class="spec-grade">${g}</span> ` +
        `<span class="spec-zs">z=${num(sp.zs, 4)}</span></span>`;
    }
    // Δz/(1+z) cell (A2): signed 3dp against z_phot, danger-coloured past the
    // outlier threshold, muted when no spectrum or a null redshift.
    const dzCell = dzMarkup(r.spec ? dzNorm(r.spec.zs, r.z_phot) : null, '');
    // sep column: cone-search separation if present on spec, else the matched-spec sep.
    const sep = r.spec ? r.spec.sep : null;
    // RA/Dec de-emphasised to 5dp + muted (A3); full 6dp stays in the detail copy button.
    tr.innerHTML =
      `<td>${r.id}</td><td class="col-coord">${num(r.ra, 5)}</td><td class="col-coord">${num(r.dec, 5)}</td>` +
      `<td>${num(r.z_phot, 3)}</td><td>${num(r.lmass, 2)}</td>` +
      `<td class="col-uvj">${uvjShort(r.uvj)}</td>` +
      `<td>${num(r.mag, 2)}</td>` +
      `<td>${sep == null ? '—' : num(sep, 2)}</td>` +
      `<td class="col-dist">${r.dist_arcsec == null ? '—' : num(r.dist_arcsec, 2)}</td>` +
      `<td class="col-dz">${dzCell}</td>` +
      `<td>${specCell}</td>`;
    tr.addEventListener('click', () => selectRow(r.id));
    tr.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectRow(r.id); }
    });
    frag.appendChild(tr);
  }
  body.appendChild(frag);
}

function updateSortArrows() {
  const key = state.sort.replace(/^-/, '');
  const desc = state.sort.startsWith('-');
  for (const th of document.querySelectorAll('#results-head th[data-sort]')) {
    const arrow = th.querySelector('.arrow');
    if (arrow) arrow.remove();
    th.tabIndex = 0;
    if (th.dataset.sort === key) {
      th.setAttribute('aria-sort', desc ? 'descending' : 'ascending');
      const s = document.createElement('span'); s.className = 'arrow'; s.textContent = desc ? '▼' : '▲';
      th.appendChild(s);
    } else {
      th.setAttribute('aria-sort', 'none');
    }
  }
}

// Column-header text/flags that follow query + template state:
//  - the "mag" header shows the band of the LAST EXECUTED query (state.magBand),
//    not the live selector, so it always matches the numbers in the column.
//  - the "lmass" header is flagged (class + title) when the active template set
//    computed no stellar masses (all cells are "—").
function updateColumnHeaders() {
  const magTh = document.querySelector('#results-head th[data-sort="mag"]');
  if (magTh && magTh.firstChild) magTh.firstChild.nodeValue = `mag ${state.magBand}`;
  const lmassTh = document.querySelector('#results-head th[data-sort="lmass"]');
  if (lmassTh) {
    const hasSps = activeHasSPS();
    lmassTh.classList.toggle('col-nosps', !hasSps);
    if (!hasSps) lmassTh.title = 'Stellar mass not computed for this template set';
    else lmassTh.removeAttribute('title');
  }
}

// Disable the log M★ filter inputs (keeping any typed values) and show a hint
// when the active template set has no stellar masses; re-enable otherwise.
function updateLmassAvailability() {
  const hasSps = activeHasSPS();
  $('q-lmassmin').disabled = !hasSps;
  $('q-lmassmax').disabled = !hasSps;
  $('lmass-hint').classList.toggle('hidden', hasSps);
}

function onSortClick(col) {
  if (!SORT_COLS.includes(col)) return;
  // lmass sort is meaningless when the active template computed no masses.
  if (col === 'lmass' && !activeHasSPS()) return;
  // toggle asc/desc; default new column ascending (id/ra/…) but desc for magnitude-like? keep asc.
  state.sort = (state.sort === col) ? ('-' + col) : col;
  state.offset = 0;
  runSearch(false);
}

function renderPager() {
  const sz = state.pageSize;
  const totalPages = Math.max(1, Math.ceil(state.total / sz));
  const page = Math.floor(state.offset / sz) + 1;
  $('pg-info').textContent = `Page ${page} / ${totalPages}`;
  $('pg-prev').disabled = state.offset <= 0;
  $('pg-next').disabled = state.offset + sz >= state.total;
}

function gotoPage(delta) {
  const next = state.offset + delta * state.pageSize;
  if (next < 0 || next >= state.total) return;
  state.offset = next;
  runSearch(false).then(() => { $('results-scroll').scrollTop = 0; });
}

// 'i' shortcut (E1): open the Inspector on the selected object's PRIMARY
// spectrum — the same URL (and same tab) as the detail-pane "inspect ↗" link.
// No-op when nothing is selected or the selection has no matched spectrum.
function openInspectorForSelected() {
  if (state.selId == null) return;
  const r = state.rows.find((x) => x.id === state.selId);
  if (!r || !r.spec || !r.spec.dja) return;
  window.location.href =
    `/inspector/?field=${encodeURIComponent(state.field)}` +
    `&template=${encodeURIComponent(state.template)}` +
    `&sel=${encodeURIComponent(r.spec.dja)}`;
}

// j/k keyboard navigation within the current page.
function navRow(delta) {
  if (!state.rows.length) return;
  let idx = state.rows.findIndex(r => r.id === state.selId);
  idx = idx < 0 ? (delta > 0 ? 0 : state.rows.length - 1) : idx + delta;
  if (idx < 0) { if (state.offset > 0) return gotoPage(-1); idx = 0; }
  if (idx >= state.rows.length) { if (state.offset + state.pageSize < state.total) return gotoPage(1); idx = state.rows.length - 1; }
  selectRow(state.rows[idx].id);
  const tr = document.querySelector(`#results-body tr[data-id="${state.rows[idx].id}"]`);
  if (tr) tr.scrollIntoView({ block: 'nearest' });
}

// ══════════════════════════════════════════════════════════════════
//  CSV EXPORT (current result set — walks all pages)
// ══════════════════════════════════════════════════════════════════
async function exportCSV() {
  if (state.total === 0) return;
  // CSV is capped at 5000 rows (matches the server's per-query max). When the
  // result set is larger, require a second click so the truncation is a
  // conscious choice, and suffix the filename to record it (B3).
  const capped = state.total > 5000;
  if (capped && !state.csvConfirm) {
    state.csvConfirm = true;
    toast(`Result has ${state.total.toLocaleString()} rows; export downloads the first 5000. Click again to proceed.`);
    setTimeout(() => { state.csvConfirm = false; }, 5000);
    return;
  }
  state.csvConfirm = false;
  $('btn-csv').disabled = true;
  // Include dist_arcsec only when the active result set carries it (cone / nearest).
  const hasDist = state.rows.some(r => r.dist_arcsec != null);
  const cols = ['id', 'ra', 'dec', ...(hasDist ? ['dist_arcsec'] : []),
    'z_phot', 'z160', 'z840', 'chi2', 'lmass', 'lsfr', 'mag',
    'flux_radius', 'n_bands', 'uvj', 'u_v', 'v_j', 'spec_dja', 'spec_zs', 'spec_grade', 'spec_sep', 'spec_grating'];
  const lines = [cols.join(',')];
  const cap = Math.min(state.total, 5000);
  try {
    for (let off = 0; off < cap; off += 500) {
      const p = new URLSearchParams(state.params);
      p.set('field', state.field); p.set('template', state.template);
      if (state.sort) p.set('sort', state.sort);
      p.set('limit', '500'); p.set('offset', String(off));
      const data = await fetchJSON('/api/catalog/query?' + p.toString());
      for (const r of (data.rows || [])) {
        const sp = r.spec || {};
        const vals = [r.id, r.ra, r.dec, ...(hasDist ? [r.dist_arcsec] : []),
          r.z_phot, r.z160, r.z840, r.chi2, r.lmass, r.lsfr, r.mag,
          r.flux_radius, r.n_bands, r.uvj, r.u_v, r.v_j,
          sp.dja || '', sp.zs, sp.grade, sp.sep, sp.grating || ''];
        lines.push(vals.map(csvCell).join(','));
      }
    }
  } catch (e) {
    toast('CSV export failed: ' + e.message, true);
    $('btn-csv').disabled = false;
    return;
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `minerva_${state.field}_${state.template}_query${capped ? '_first5000' : ''}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
  $('btn-csv').disabled = false;
}

function csvCell(v) {
  if (v == null || (typeof v === 'number' && Number.isNaN(v))) return '';
  const s = String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// ══════════════════════════════════════════════════════════════════
//  DETAIL PANE
// ══════════════════════════════════════════════════════════════════
async function selectRow(id) {
  if (id == null || Number.isNaN(id)) return;
  state.selId = id;
  state.gridSize = null;
  for (const tr of document.querySelectorAll('#results-body tr'))
    tr.classList.toggle('selected', Number(tr.dataset.id) === id);
  $('detail-empty').classList.add('hidden');
  const c = $('detail-content');
  c.classList.remove('hidden');
  c.innerHTML = '<div class="skel" style="height:22px;width:50%"></div>' +
    '<div class="skel sed-loading" style="margin-top:14px"></div>';
  syncURL();

  let obj;
  try {
    obj = await fetchJSON(`/api/object/${encodeURIComponent(state.field)}/${id}`);
  } catch (e) {
    c.innerHTML = `<div class="empty">Could not load object ${id}: ${e.message}</div>`;
    return;
  }
  if (state.selId !== id) return;   // superseded by a newer click
  renderDetail(obj);
}

// UVJ rest-frame colour class: 1=quiescent, 0=star-forming, -1/null=no class.
const UVJ_NOCLASS_TITLE = 'No UVJ classification (eazy rest-frame colors unavailable)';
function uvjNoClass(v) { return v == null || Number.isNaN(v) || v === -1; }
function uvjLabel(v) {
  if (uvjNoClass(v)) return '—';
  return v === 1 ? 'quiescent' : v === 0 ? 'star-forming' : '—';
}
// Compact table-cell markup: Q / SF / — (muted), full word in the title.
function uvjShort(v) {
  if (v === 1) return '<span title="quiescent">Q</span>';
  if (v === 0) return '<span title="star-forming">SF</span>';
  return `<span class="text-muted" title="${UVJ_NOCLASS_TITLE}">—</span>`;
}

// Does the active template carry stellar-population parameters (lmass/lsfr)?
// Reads the /api/fields template_info.{tpl}.has_sps flag; defaults true when the
// flag is absent (older payloads) so nothing is disabled without cause.
function activeHasSPS() {
  const f = currentField();
  const ti = f && f.template_info;
  if (!ti || !(state.template in ti)) return true;
  return ti[state.template].has_sps !== false;
}

function renderDetail(obj) {
  const c = $('detail-content');
  const z = (obj.zout && obj.zout[state.template]) || {};
  const ra = num(obj.ra, 6), dec = num(obj.dec, 6);
  const coordStr = `${ra} ${dec}`;
  c.innerHTML = '';

  // logM / logSFR: render "n/a" (not "—") when the template set has no masses,
  // so an all-NaN LARSON run reads as "not computed" rather than a data bug.
  const hasSps = activeHasSPS();
  const naTitle = `Not computed for the ${state.template.toUpperCase()} template set`;
  const naSpan = `<b class="text-muted" title="${naTitle}">n/a</b>`;
  const logM = hasSps ? `<b>${num(z.lmass, 2)}</b>` : naSpan;
  const logSFR = hasSps ? `<b>${num(z.lsfr, 2)}</b>` : naSpan;
  const uvjTitle = uvjNoClass(z.uvj) ? ` title="${UVJ_NOCLASS_TITLE}"` : '';

  // χ² and band-coverage stats (C1). bands = measured / total-for-field.
  const totalBands = (currentField().bands || []).length;
  const chi2Span = z.chi2 == null
    ? `<b class="text-muted">—</b>` : `<b>${num(z.chi2, 2)}</b>`;
  const bandsSpan = obj.n_bands == null
    ? `<b class="text-muted">—</b>` : `<b>${obj.n_bands}</b>`;

  // Likely-star flag tag (C2), rendered near the id in the header.
  const starTag = (obj.flags && obj.flags.flag_star)
    ? `<span class="star-flag" title="Flagged as a likely star in the MINERVA catalog">★ star flag</span>`
    : '';

  // External-viewer + field-map link row (C4). All open in a new tab.
  const links = [
    `<a href="https://www.legacysurvey.org/viewer?ra=${obj.ra}&dec=${obj.dec}&layer=ls-dr10&zoom=16" target="_blank" rel="noopener">Legacy Survey ↗</a>`,
    `<a href="https://sky.esa.int/esasky/?target=${obj.ra}%20${obj.dec}&fov=0.05" target="_blank" rel="noopener">ESASky ↗</a>`,
    `<a href="https://simbad.cds.unistra.fr/simbad/sim-coo?Coord=${obj.ra}+${obj.dec}&Radius=10&Radius.unit=arcsec" target="_blank" rel="noopener">SIMBAD ↗</a>`,
  ];
  const mapBase = currentField().map_link;
  if (mapBase) {
    const href = mapBase.replace('{ra}', obj.ra).replace('{dec}', obj.dec);
    links.push(`<a href="${escAttr(href)}" target="_blank" rel="noopener">Field map ↗</a>`);
  }

  // Header
  const head = document.createElement('div');
  head.className = 'd-header';
  head.innerHTML =
    `<div class="d-id">#${obj.id}${starTag}` +
      `<button type="button" id="btn-nearest" class="btn btn-secondary btn-mini" ` +
        `title="List this object + its 10 nearest neighbours">Nearest 10</button></div>` +
    `<div class="d-coord">${ra}, ${dec} <button class="copy-btn" data-copy="${coordStr}">copy</button></div>` +
    `<div class="d-links">${links.join('')}</div>` +
    `<div class="d-stats">` +
      `<span>z<sub>phot</sub> <b>${num(z.z_phot, 3)}</b> <span class="text-muted">[${num(z.z160, 2)}–${num(z.z840, 2)}]</span></span>` +
      `<span>logM ${logM}</span>` +
      `<span>logSFR ${logSFR}</span>` +
      `<span${uvjTitle}>UVJ <b>${uvjLabel(z.uvj)}</b></span>` +
      `<span title="EAzY best-fit chi-squared">χ² ${chi2Span}</span>` +
      `<span title="Bands with measured photometry">bands ${bandsSpan}<span class="text-muted">/${totalBands}</span></span>` +
    `</div>`;
  c.appendChild(head);
  const nearBtn = $('btn-nearest');
  if (nearBtn) nearBtn.addEventListener('click', () => runNearest(obj.id));

  // Cutout + size slider
  const cut = document.createElement('div');
  cut.className = 'd-sec';
  cut.innerHTML =
    `<h4>Cutout</h4>` +
    `<div class="cut-row">` +
      `<img id="rgb-cut" class="plate" alt="RGB cutout">` +
      `<div class="cut-ctrl">` +
        `<label>size <span id="size-val">1.5</span>″</label>` +
        `<input type="range" id="size-slider" min="0.5" max="10" step="0.5" value="1.5">` +
        `<button type="button" id="btn-allbands" class="btn btn-secondary btn-mini">All bands ▸</button>` +
      `</div>` +
    `</div>` +
    `<div id="grid-strip"><img id="grid-img" alt="Multiband grid"></div>`;
  c.appendChild(cut);

  // SED quicklook + dedicated p(z) strip (D). The p(z) is its own full-width
  // canvas beneath the SED (was a cramped 108×58 corner inset), with a labelled
  // z axis and the z_phot marker; the SED shrinks slightly to keep the pane
  // height reasonable.
  const sed = document.createElement('div');
  sed.className = 'd-sec';
  sed.innerHTML = `<h4>SED · p(z)</h4>` +
    `<canvas id="sed-canvas" role="img" aria-label="Spectral energy distribution for object #${obj.id}, z_phot ${num(z.z_phot, 3)}"></canvas>` +
    `<canvas id="pz-canvas" role="img" aria-label="Redshift probability p(z) for object #${obj.id}, z_phot ${num(z.z_phot, 3)}"></canvas>`;
  c.appendChild(sed);

  // Spectra list
  const spec = document.createElement('div');
  spec.className = 'd-sec';
  spec.id = 'spec-sec';
  c.appendChild(spec);
  renderSpectra(obj);

  // wire cutout
  const slider = $('size-slider');
  const loadRGB = () => {
    $('size-val').textContent = Number(slider.value).toFixed(1);
    $('rgb-cut').src = `/api/cutout?mode=rgb&ra=${obj.ra}&dec=${obj.dec}&size=${slider.value}`;
  };
  loadRGB();
  slider.addEventListener('input', () => { $('size-val').textContent = Number(slider.value).toFixed(1); });
  // On commit, refresh the RGB plate AND — if the all-bands grid is currently
  // shown — re-render it at the new size too (C5b: the old gridLoaded latch
  // froze the grid at its first size).
  slider.addEventListener('change', () => {
    loadRGB();
    const strip = $('grid-strip');
    if (strip && strip.style.display === 'block') loadGrid(obj, slider.value);
  });
  $('rgb-cut').addEventListener('error', () => $('rgb-cut').setAttribute('alt', 'cutout unavailable'));
  // Click-to-enlarge (C5c): the /api/cutout size param is an angular field of
  // view, not a render resolution, so a bigger request would reframe rather than
  // magnify — the lightbox scales the existing plate instead.
  $('rgb-cut').addEventListener('click', () => {
    const src = $('rgb-cut').getAttribute('src');
    if (src) openLightbox(src, `RGB cutout for object #${obj.id}, enlarged`);
  });
  $('btn-allbands').addEventListener('click', () => loadGrid(obj, slider.value));

  loadSED(obj.id);
}

// Show the all-bands grid, (re)loading its image whenever the requested size
// differs from what's on screen (keyed by size rather than a one-shot latch).
function loadGrid(obj, size) {
  const strip = $('grid-strip');
  strip.style.display = 'block';
  const key = String(size);
  if (state.gridSize === key) return;
  state.gridSize = key;
  const img = $('grid-img');
  img.src = `/api/cutout?mode=grid&ra=${obj.ra}&dec=${obj.dec}&size=${size}`;
  img.addEventListener('error', () => img.setAttribute('alt', 'grid unavailable'), { once: true });
}

// ── Spectra list ──────────────────────────────────────────────────
function renderSpectra(obj) {
  const sec = $('spec-sec');
  const specs = obj.spectra || [];
  if (!specs.length) {
    sec.innerHTML = `<h4>Spectra</h4><div class="text-muted" style="font-size:13px">No matched DJA spectra.</div>`;
    return;
  }
  sec.innerHTML = `<h4>Spectra (${specs.length})</h4>`;
  const mapBase = currentField().map_link || null;   // forward-compatible if /api/fields adds it
  // Active-template z_phot for the per-spectrum Δz/(1+z) tag (C3).
  const zTpl = (obj.zout && obj.zout[state.template]) || {};
  const zp = zTpl.z_phot;
  for (const s of specs) {
    const el = document.createElement('div');
    el.className = 'spec-item';
    const inspHref = `/inspector/?field=${encodeURIComponent(state.field)}&template=${encodeURIComponent(state.template)}&sel=${encodeURIComponent(s.dja)}`;
    const preview = `/api/dja/preview/${encodeURIComponent(s.root)}/${encodeURIComponent(s.dja + '.fnu.png')}`;
    let extLink = '';
    if (mapBase) {
      const href = mapBase.replace('{ra}', s.dja_ra ?? obj.ra).replace('{dec}', s.dja_dec ?? obj.dec)
        .replace('{root}', s.root).replace('{dja}', s.dja);
      extLink = `<a href="${href}" target="_blank" rel="noopener">map ↗</a>`;
    }
    el.innerHTML =
      `<img class="spec-thumb" src="${preview}" alt="preview" loading="lazy">` +
      `<div class="spec-meta">` +
        `<span class="grating">${s.grating || 'spectrum'} ${gradeDot(s.grade)}</span>` +
        `<div class="spec-tags">` +
          `<span>zs <b>${num(s.zs, 4)}</b></span>` +
          `<span>grade ${s.grade ?? 0}</span>` +
          `<span>sep ${num(s.sep, 2)}″</span>` +
          (s.sn != null ? `<span>S/N ${num(s.sn, 1)}</span>` : '') +
          (dzNorm(s.zs, zp) != null ? `<span>${dzMarkup(dzNorm(s.zs, zp), 'Δz/(1+z) ')}</span>` : '') +
        `</div>` +
        `<div class="spec-links"><a href="${inspHref}">inspect ↗</a>${extLink}</div>` +
      `</div>`;
    el.querySelector('.spec-thumb').addEventListener('error', function () { this.style.visibility = 'hidden'; }, { once: true });
    sec.appendChild(el);
  }
}

// ══════════════════════════════════════════════════════════════════
//  SED QUICKLOOK CANVAS  (log-x µm 0.3–6, y = AB mag; inset p(z))
// ══════════════════════════════════════════════════════════════════
async function loadSED(id) {
  const cv = $('sed-canvas');
  const pzcv = $('pz-canvas');
  if (!cv) return;
  drawSEDMessage(cv, 'loading SED…');
  if (pzcv) drawSEDMessage(pzcv, '');
  let d;
  try {
    d = await fetchJSON(`/api/eazy/${encodeURIComponent(state.field)}/${encodeURIComponent(state.template)}/${id}`);
  } catch (e) {
    drawSEDMessage(cv, 'SED unavailable: ' + e.message);
    if (pzcv) drawSEDMessage(pzcv, 'p(z) unavailable');
    return;
  }
  if (state.selId !== id) return;
  drawSED(cv, d);
  if (pzcv) drawPZStrip(pzcv, d);
}

function themeInk() {
  const cs = getComputedStyle(document.body);
  return {
    ink: cs.getPropertyValue('--color-text').trim() || '#201f1d',
    accent: cs.getPropertyValue('--color-accent').trim() || '#b68235',
    grid: 'color-mix(in srgb, ' + (cs.getPropertyValue('--color-text').trim() || '#201') + ' 12%, transparent)',
  };
}

function drawSEDMessage(cv, msg) {
  const ctx = setupCanvas(cv);
  const { w, h } = cvSize(cv);
  ctx.fillStyle = 'rgba(128,128,128,0.8)';
  ctx.font = '13px serif';
  ctx.textAlign = 'center';
  ctx.fillText(msg, w / 2, h / 2);
}

function cvSize(cv) { return { w: cv.clientWidth, h: cv.clientHeight }; }

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return ctx;
}

function drawSED(cv, d) {
  const ctx = setupCanvas(cv);
  const { w, h } = cvSize(cv);
  const ink = themeInk();
  const padL = 40, padR = 12, padT = 12, padB = 26;
  const x0 = padL, x1 = w - padR, y0 = padT, y1 = h - padB;

  const LAM_MIN = 0.3, LAM_MAX = 6.0;
  const lx = (lam) => x0 + (Math.log10(lam) - Math.log10(LAM_MIN)) / (Math.log10(LAM_MAX) - Math.log10(LAM_MIN)) * (x1 - x0);

  // Gather AB mags from photometry (fnu in µJy -> AB = 23.9 - 2.5 log10(µJy)).
  const P = d.phot || {};
  const pts = [];
  const bands = P.band || [], plam = P.lam_um || [], pf = P.fnu_uJy || [], pe = P.efnu_uJy || [], pok = P.ok || [];
  const mags = [];
  for (let i = 0; i < plam.length; i++) {
    if (pok[i] === false) continue;
    const lam = plam[i], fnu = pf[i];
    if (lam == null || lam < LAM_MIN || lam > LAM_MAX) continue;
    if (fnu == null || fnu <= 0) continue;
    const mag = 23.9 - 2.5 * Math.log10(fnu);
    const e = pe[i];
    const mHi = (e != null && fnu - e > 0) ? 23.9 - 2.5 * Math.log10(fnu - e) : mag;   // fainter bound? invert
    const mLo = (e != null) ? 23.9 - 2.5 * Math.log10(fnu + e) : mag;
    pts.push({ lam, mag, eLo: mLo, eHi: mHi, band: bands[i] });
    mags.push(mag);
  }
  // Template mags
  const T = d.templ || {};
  const tl = T.lam_um || [], tf = T.fnu_uJy || [];
  const tpts = [];
  for (let i = 0; i < tl.length; i++) {
    const lam = tl[i], fnu = tf[i];
    if (lam == null || lam < LAM_MIN || lam > LAM_MAX || fnu == null || fnu <= 0) continue;
    const mag = 23.9 - 2.5 * Math.log10(fnu);
    tpts.push({ lam, mag }); mags.push(mag);
  }
  if (!mags.length) { drawSEDMessage(cv, 'No positive photometry to plot'); return; }

  let mMin = Math.min(...mags), mMax = Math.max(...mags);
  const padM = (mMax - mMin) * 0.08 || 0.5;
  mMin -= padM; mMax += padM;
  // y inverted: brighter (small mag) at top.
  const my = (m) => y0 + (m - mMin) / (mMax - mMin) * (y1 - y0);

  // Axes
  ctx.strokeStyle = ink.ink; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.globalAlpha = 1;

  // x ticks (µm)
  ctx.fillStyle = ink.ink; ctx.font = '10px serif'; ctx.textAlign = 'center';
  for (const t of [0.3, 0.5, 1, 2, 3, 5]) {
    const x = lx(t);
    ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y1 + 3); ctx.stroke();
    ctx.globalAlpha = 0.75; ctx.fillText(String(t), x, y1 + 14);
  }
  ctx.globalAlpha = 0.6; ctx.fillText('µm', x1 - 6, y1 + 14);
  // y ticks (mag)
  ctx.textAlign = 'right';
  const yticks = niceTicks(mMin, mMax, 4);
  for (const m of yticks) {
    const y = my(m);
    ctx.globalAlpha = 0.5; ctx.fillText(m.toFixed(1), x0 - 4, y + 3);
    ctx.globalAlpha = 0.12; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  }
  ctx.save(); ctx.translate(11, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2);
  ctx.globalAlpha = 0.6; ctx.textAlign = 'center'; ctx.fillText('AB mag', 0, 0); ctx.restore();
  ctx.globalAlpha = 1;

  // Template line
  if (tpts.length) {
    ctx.strokeStyle = ink.accent; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.85;
    ctx.beginPath();
    tpts.forEach((p, i) => { const x = lx(p.lam), y = my(p.mag); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.globalAlpha = 1;
  }

  // Photometry points + error bars
  ctx.strokeStyle = ink.ink; ctx.fillStyle = ink.ink;
  for (const p of pts) {
    const x = lx(p.lam), y = my(p.mag);
    ctx.globalAlpha = 0.7; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, my(p.eLo)); ctx.lineTo(x, my(p.eHi)); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI); ctx.fill();
  }
}

// Full-width p(z) strip drawn on its own canvas beneath the SED (D). Same data
// (d.zgrid / d.pz — what the old corner inset used), same tokens; now with a
// labelled z axis + ticks and the z_phot marker.
function drawPZStrip(cv, d) {
  const ctx = setupCanvas(cv);
  const { w, h } = cvSize(cv);
  const ink = themeInk();
  const zg = d.zgrid || [], pz = d.pz || [];
  const padL = 40, padR = 12, padT = 8, padB = 18;
  const x0 = padL, x1 = w - padR, y0 = padT, y1 = h - padB;
  if (!zg.length || !pz.length) { drawSEDMessage(cv, 'p(z) unavailable'); return; }

  let pmax = 0, izpk = 0;
  for (let i = 0; i < pz.length; i++) if (pz[i] > pmax) { pmax = pz[i]; izpk = i; }
  if (pmax <= 0) { drawSEDMessage(cv, 'p(z) unavailable'); return; }

  const zmin = zg[0], zmax = zg[zg.length - 1];
  const zx = (z) => x0 + (z - zmin) / (zmax - zmin) * (x1 - x0);
  const py = (p) => y1 - (p / pmax) * (y1 - y0);

  // baseline axis
  ctx.strokeStyle = ink.ink; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.globalAlpha = 1;

  // filled curve + outline
  ctx.fillStyle = ink.accent; ctx.globalAlpha = 0.30;
  ctx.beginPath(); ctx.moveTo(zx(zmin), y1);
  for (let i = 0; i < pz.length; i++) ctx.lineTo(zx(zg[i]), py(pz[i]));
  ctx.lineTo(zx(zmax), y1); ctx.closePath(); ctx.fill();
  ctx.globalAlpha = 0.9; ctx.strokeStyle = ink.accent; ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < pz.length; i++) { const x = zx(zg[i]), y = py(pz[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.stroke(); ctx.globalAlpha = 1;

  // z-axis ticks + labels
  ctx.fillStyle = ink.ink; ctx.font = '10px serif'; ctx.textAlign = 'center';
  for (const zt of zTicks(zmin, zmax, 6)) {
    if (zt < zmin || zt > zmax) continue;
    const x = zx(zt);
    ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y1 + 3); ctx.stroke();
    ctx.globalAlpha = 0.75; ctx.fillText(String(zt), x, y1 + 13);
  }
  ctx.globalAlpha = 0.6; ctx.textAlign = 'right'; ctx.fillText('z', x1, y1 + 13);
  // rotated p(z) axis label
  ctx.save(); ctx.translate(11, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2);
  ctx.globalAlpha = 0.6; ctx.textAlign = 'center'; ctx.fillText('p(z)', 0, 0); ctx.restore();
  ctx.globalAlpha = 1;

  // z_phot marker (dashed) + label
  const zmark = d.z_phot ?? zg[izpk];
  if (zmark != null) {
    const mx = zx(zmark);
    ctx.strokeStyle = ink.ink; ctx.globalAlpha = 0.55; ctx.setLineDash([3, 2]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(mx, y0); ctx.lineTo(mx, y1); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 0.9; ctx.fillStyle = ink.accent; ctx.font = '9px serif'; ctx.textAlign = 'left';
    const tx = Math.min(mx + 3, x1 - 34);
    ctx.fillText(`z=${zmark.toFixed(2)}`, tx, y0 + 8);
  }
  ctx.globalAlpha = 1;
}

// 1/2/5×10^k "nice" ticks across [lo,hi] — cleaner than niceTicks over the wide
// log-z range (e.g. 0–20 -> 0,5,10,15,20 rather than 3.7,7.4,…).
function zTicks(lo, hi, n) {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    out.push(Math.round(v * 100) / 100);
  }
  return out;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo; if (span <= 0) return [lo];
  const step = Math.max(0.1, Math.round((span / n) * 10) / 10);
  const out = []; const start = Math.ceil(lo / step) * step;
  for (let v = start; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

// ══════════════════════════════════════════════════════════════════
//  EVENT WIRING
// ══════════════════════════════════════════════════════════════════
function wireEvents() {
  $('query-form').addEventListener('submit', (e) => { e.preventDefault(); runSearch(true); });
  $('btn-csv').addEventListener('click', exportCSV);
  $('sel-field').addEventListener('change', (e) => {
    state.field = e.target.value; state.selId = null; state.total = 0; state.rows = [];
    $('detail-content').classList.add('hidden'); $('detail-empty').classList.remove('hidden');
    $('results-body').innerHTML = ''; $('result-summary').textContent = '—'; $('btn-csv').disabled = true;
    showFirstRunPrompt();
    onFieldChange(null);
  });
  $('sel-template').addEventListener('change', (e) => {
    state.template = e.target.value;
    $('nav-inspector').href = `/inspector/?field=${encodeURIComponent(state.field)}&template=${encodeURIComponent(state.template)}`;
    updateLmassAvailability();   // enable/disable log M★ inputs for the new template
    updateColumnHeaders();       // flag the lmass header if the new template has no SPS
    if (state.total) runSearch(false);       // re-query so per-template columns refresh
    else syncURL();
  });
  for (const th of document.querySelectorAll('#results-head th[data-sort]')) {
    th.tabIndex = 0;
    th.setAttribute('aria-sort', 'none');
    th.addEventListener('click', () => onSortClick(th.dataset.sort));
    th.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSortClick(th.dataset.sort); }
    });
  }
  $('pg-prev').addEventListener('click', () => gotoPage(-1));
  $('pg-next').addEventListener('click', () => gotoPage(1));

  document.addEventListener('click', (e) => {
    const b = e.target.closest('.copy-btn');
    if (b) {
      navigator.clipboard?.writeText(b.dataset.copy).then(
        () => toast('Copied: ' + b.dataset.copy),
        () => toast('Copy failed', true));
    }
  });

  $('btn-clear').addEventListener('click', clearForm);

  $('btn-help').addEventListener('click', openHelp);
  $('help-close').addEventListener('click', closeHelp);
  $('help-backdrop').addEventListener('click', (e) => { if (e.target === $('help-backdrop')) closeHelp(); });
  $('help-backdrop').addEventListener('keydown', trapHelpKeydown);   // Tab trap + Esc (E2)

  document.addEventListener('keydown', (e) => {
    // Esc closes the lightbox first, then the help dialog (C5c / E2).
    if (e.key === 'Escape' && lightboxOpen()) { closeLightbox(); return; }
    if (e.key === 'Escape' && !$('help-backdrop').classList.contains('hidden')) { closeHelp(); return; }
    const tag = (e.target.tagName || '').toUpperCase();
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.metaKey || e.ctrlKey) return;
    if (e.key === '?') { e.preventDefault(); toggleHelp(); }
    else if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); navRow(1); }
    else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); navRow(-1); }
    else if (e.key === 'i') { e.preventDefault(); openInspectorForSelected(); }
  });

  window.addEventListener('resize', debounce(() => {
    const cv = $('sed-canvas');
    if (cv && state.selId != null) loadSED(state.selId);
  }, 250));
}

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

init();
