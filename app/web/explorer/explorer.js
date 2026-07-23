// MINERVA · COSMOS Catalog Explorer
// Build-free vanilla ES module. Talks only to the endpoints in app/API.md.
// Contract shapes: /api/fields, /api/catalog/query {total,rows}, /api/object/{field}/{id},
// /api/eazy/{field}/{template}/{id}, /api/cutout, /api/dja/preview/{root}/{basename}.
'use strict';

const PAGE = 100;                 // default rows per page (simple pagination = virtualization)
const NEAR_PAGE = 11;             // nearest-neighbour flow: target + 10 neighbours
const NEAR_RADIUS = 30;           // default cone radius (″) for the nearest flow
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
  gridLoaded: false,
  pageSize: PAGE,      // rows per page (NEAR_PAGE while in the nearest flow)
  nearId: null,        // target id of an active nearest-neighbour query (URL near=)
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

// ── Help overlay ('?' cheat-sheet) ────────────────────────────────
function openHelp() { $('help-backdrop').classList.remove('hidden'); $('help-close').focus(); }
function closeHelp() { $('help-backdrop').classList.add('hidden'); }
function toggleHelp() { $('help-backdrop').classList.contains('hidden') ? openHelp() : closeHelp(); }

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
// Read the form into a params object (only non-empty values).
function readForm() {
  const p = {};
  for (const [id, param] of Object.entries(FORM_MAP)) {
    let v = $(id).value.trim();
    if (v === '') continue;
    if (param === 'ids') v = v.split(/[\s,]+/).filter(Boolean).join(',');
    if (v !== '') p[param] = v;
  }
  // radius only meaningful with ra & dec
  if ((p.ra == null || p.dec == null) && p.radius_arcsec != null) delete p.radius_arcsec;
  if ($('q-hasspec').checked) p.has_spec = '1';
  // use_phot / no_star default to 1 on the server; only send the "off" case.
  if (!$('q-usephot').checked) p.use_phot = '0';
  if (!$('q-nostar').checked) p.no_star = '0';
  return p;
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
    state.params = readForm(); state.offset = 0;
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
  state.params = readForm();
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
  const frag = document.createDocumentFragment();
  for (const r of state.rows) {
    const tr = document.createElement('tr');
    tr.dataset.id = r.id;
    tr.tabIndex = 0;
    tr.setAttribute('role', 'button');
    tr.setAttribute('aria-label', `Object ${r.id} — open detail`);
    if (r.id === state.selId) tr.classList.add('selected');
    let specCell = '<span class="text-muted">—</span>';
    if (r.spec) {
      specCell = `${gradeDot(r.spec.grade)}<span title="${r.spec.grating || ''} · zs=${num(r.spec.zs, 3)}">${r.spec.grating || 'spec'}</span>`;
    }
    // sep column: cone-search separation if present on spec, else the matched-spec sep.
    const sep = r.spec ? r.spec.sep : null;
    tr.innerHTML =
      `<td>${r.id}</td><td>${num(r.ra, 6)}</td><td>${num(r.dec, 6)}</td>` +
      `<td>${num(r.z_phot, 3)}</td><td>${num(r.lmass, 2)}</td><td>${num(r.mag, 2)}</td>` +
      `<td>${sep == null ? '—' : num(sep, 2)}</td>` +
      `<td class="col-dist">${r.dist_arcsec == null ? '—' : num(r.dist_arcsec, 2)}</td>` +
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

function onSortClick(col) {
  if (!SORT_COLS.includes(col)) return;
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
  a.download = `minerva_${state.field}_${state.template}_query.csv`;
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
  state.gridLoaded = false;
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

function uvjLabel(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return v === 1 ? 'quiescent' : v === 0 ? 'star-forming' : String(v);
}

function renderDetail(obj) {
  const c = $('detail-content');
  const z = (obj.zout && obj.zout[state.template]) || {};
  const ra = num(obj.ra, 6), dec = num(obj.dec, 6);
  const coordStr = `${ra} ${dec}`;
  c.innerHTML = '';

  // Header
  const head = document.createElement('div');
  head.className = 'd-header';
  head.innerHTML =
    `<div class="d-id">#${obj.id}` +
      `<button type="button" id="btn-nearest" class="btn btn-secondary btn-mini" ` +
        `title="List this object + its 10 nearest neighbours">Nearest 10</button></div>` +
    `<div class="d-coord">${ra}, ${dec} <button class="copy-btn" data-copy="${coordStr}">copy</button></div>` +
    `<div class="d-stats">` +
      `<span>z<sub>phot</sub> <b>${num(z.z_phot, 3)}</b> <span class="text-muted">[${num(z.z160, 2)}–${num(z.z840, 2)}]</span></span>` +
      `<span>logM <b>${num(z.lmass, 2)}</b></span>` +
      `<span>logSFR <b>${num(z.lsfr, 2)}</b></span>` +
      `<span>UVJ <b>${uvjLabel(z.uvj)}</b></span>` +
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

  // SED quicklook
  const sed = document.createElement('div');
  sed.className = 'd-sec';
  sed.innerHTML = `<h4>SED · p(z)</h4>` +
    `<canvas id="sed-canvas" role="img" aria-label="Spectral energy distribution and p(z) for object #${obj.id}, z_phot ${num(z.z_phot, 3)}"></canvas>`;
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
  slider.addEventListener('change', loadRGB);
  $('rgb-cut').addEventListener('error', () => $('rgb-cut').setAttribute('alt', 'cutout unavailable'));
  $('btn-allbands').addEventListener('click', () => loadGrid(obj, slider.value));

  loadSED(obj.id);
}

function loadGrid(obj, size) {
  const strip = $('grid-strip');
  strip.style.display = 'block';
  if (state.gridLoaded) return;
  state.gridLoaded = true;
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
  if (!cv) return;
  drawSEDMessage(cv, 'loading SED…');
  let d;
  try {
    d = await fetchJSON(`/api/eazy/${encodeURIComponent(state.field)}/${encodeURIComponent(state.template)}/${id}`);
  } catch (e) {
    drawSEDMessage(cv, 'SED unavailable: ' + e.message);
    return;
  }
  if (state.selId !== id) return;
  drawSED(cv, d);
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

  drawPZInset(ctx, d, x0, y0, x1, ink);
}

// small p(z) inset in the upper region of the SED panel
function drawPZInset(ctx, d, x0, y0, x1, ink) {
  const zg = d.zgrid || [], pz = d.pz || [];
  if (!zg.length || !pz.length) return;
  const iw = 108, ih = 58, ix = x1 - iw - 6, iy = y0 + 4;
  // frame
  ctx.globalAlpha = 0.85; ctx.fillStyle = 'rgba(255,255,255,0.06)';
  ctx.strokeStyle = ink.ink; ctx.globalAlpha = 0.25; ctx.lineWidth = 1;
  ctx.strokeRect(ix, iy, iw, ih); ctx.globalAlpha = 1;
  // determine z window around the peak for readability
  let pmax = 0, izpk = 0;
  for (let i = 0; i < pz.length; i++) if (pz[i] > pmax) { pmax = pz[i]; izpk = i; }
  if (pmax <= 0) return;
  const zmin = zg[0], zmax = zg[zg.length - 1];
  const zx = (z) => ix + (z - zmin) / (zmax - zmin) * iw;
  const py = (p) => iy + ih - (p / pmax) * (ih - 6) - 3;
  ctx.fillStyle = ink.accent; ctx.globalAlpha = 0.35;
  ctx.beginPath(); ctx.moveTo(zx(zmin), iy + ih);
  for (let i = 0; i < pz.length; i++) ctx.lineTo(zx(zg[i]), py(pz[i]));
  ctx.lineTo(zx(zmax), iy + ih); ctx.closePath(); ctx.fill();
  ctx.globalAlpha = 0.9; ctx.strokeStyle = ink.accent; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < pz.length; i++) { const x = zx(zg[i]), y = py(pz[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.stroke();
  // z_phot marker
  if (d.z_phot != null) {
    ctx.strokeStyle = ink.ink; ctx.globalAlpha = 0.5; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(zx(d.z_phot), iy); ctx.lineTo(zx(d.z_phot), iy + ih); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.globalAlpha = 0.7; ctx.fillStyle = ink.ink; ctx.font = '9px serif'; ctx.textAlign = 'left';
  ctx.fillText(`p(z)  z=${(d.z_phot ?? zg[izpk]).toFixed(2)}`, ix + 3, iy + 10);
  ctx.globalAlpha = 1;
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

  $('btn-help').addEventListener('click', openHelp);
  $('help-close').addEventListener('click', closeHelp);
  $('help-backdrop').addEventListener('click', (e) => { if (e.target === $('help-backdrop')) closeHelp(); });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('help-backdrop').classList.contains('hidden')) { closeHelp(); return; }
    const tag = (e.target.tagName || '').toUpperCase();
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.metaKey || e.ctrlKey) return;
    if (e.key === '?') { e.preventDefault(); toggleHelp(); }
    else if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); navRow(1); }
    else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); navRow(-1); }
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
