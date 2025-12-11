"""
MINERVA Galaxy Browser - Panel Web Application

Run with: 
    panel serve zgui_panel.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=*

For remote access (e.g. via ngrok):
    1. Start the app as above.
    2. Run ngrok: ngrok http 5006
    3. Access the ngrok URL.

Requirements:
    pip install panel numpy pandas matplotlib astropy eazy-photoz pillow requests
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import panel as pn
from io import BytesIO
from PIL import Image
import requests
import param
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from pathlib import Path

# Astropy imports
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

# EAzY imports
from eazy import hdf5

# Enable Panel extension
pn.extension('tabulator')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("minerva.galaxy_browser")

# =============================================================================
# LRU CACHE CLASS
# =============================================================================
class LRUCache:
    """Thread-safe LRU cache with max size"""
    
    def __init__(self, maxsize=200):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                self.cache[key] = value
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
    
    def contains(self, key):
        with self.lock:
            return key in self.cache
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {len(self.cache)}/{self.maxsize} items, {hit_rate:.1f}% hit rate"


# Global caches (persist across sessions)
CUTOUT_CACHE = LRUCache(maxsize=300)  # ~300 cutouts, ~1-5MB each = ~1.5GB max
SED_CACHE = LRUCache(maxsize=300)      # ~300 SED images
SPECTRUM_CACHE = LRUCache(maxsize=300) # ~300 spectrum images

# Background prefetch executor
PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# =============================================================================
# CONFIGURATION - Update these paths for your setup
# =============================================================================
DATA_DIR = "../data"
CATALOG_PATH = f"{DATA_DIR}/catalogs/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG.fits"
SPS_CATALOG_PATH = f"{DATA_DIR}/catalogs/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG_SPScatalog_0.0.fits"
EAZY_H5_PATH = f"{DATA_DIR}/EAzY/SFHZ/SUPER/ZPiter/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_zpiter_CATALOG.sfhz.eazypy.h5"
EAZY_ZOUT_PATH = f"{DATA_DIR}/EAzY/SFHZ/SUPER/ZPiter/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_zpiter_CATALOG_sfhz.zout.fits"

SPECTRUM_LOCAL_DIR = Path(os.environ.get("MINERVA_SPECTRA_DIR", f"{DATA_DIR}/spectra"))
SPEC_TABLE_LOCAL_PATH = Path(os.environ.get("MINERVA_SPEC_TABLE", f"{DATA_DIR}/catalogs/dja_msaexp_emission_lines_v4.4.csv.gz"))
SPEC_TABLE_TIMEOUT = int(os.environ.get("MINERVA_SPEC_TIMEOUT", "120"))

FILTERS = "f090w-clear,f115w-clear,f140m-clear,f150w-clear,f182m-clear,f200w-clear,f210m-clear,f250w-clear,f277w-clear,f300m-clear,f356w-clear,f360m-clear,f410m-clear,f444w-clear,f460m-clear"
SPEC_TABLE_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions/dja_msaexp_emission_lines_v4.4.csv.gz"


# =============================================================================
# DATA LOADING - Using pn.cache to persist across reloads
# =============================================================================

@pn.cache
def load_photometric_catalog():
    """Load the main photometric catalog"""
    logger.info("Loading photometric catalog...")
    df = Table.read(CATALOG_PATH).to_pandas()
    df.set_index('id', inplace=True)
    
    # Calculate S/N columns
    for f_col in df.filter(like='f_').columns:
        e_col = f_col.replace('f_', 'e_')
        if e_col in df.columns:
            sn_col = f_col.replace('f_', 'sn_')
            df[sn_col] = df[f_col] / df[e_col]
    
    df_detected = df[(df.filter(regex='sn_.*m') > 5).sum(axis=1) >= 4]
    logger.info("Loaded %s sources, %s detected", len(df), len(df_detected))
    return df, df_detected


@pn.cache
def load_spectroscopic_catalog():
    """Load spectroscopic catalog"""
    logger.info("Loading spectroscopic catalog...")
    tab = None
    
    # Prefer local cached CSV if present
    if SPEC_TABLE_LOCAL_PATH.is_file():
        try:
            tab_df = pd.read_csv(SPEC_TABLE_LOCAL_PATH)
            tab = Table.from_pandas(tab_df)
            logger.info("Loaded spectroscopic catalog from %s", SPEC_TABLE_LOCAL_PATH)
        except Exception as exc:
            logger.warning("Local spectroscopic catalog read failed: %s", exc)
    
    # Fall back to remote download with timeout
    if tab is None:
        try:
            logger.info("Downloading spectroscopic catalog from %s (timeout=%ss)", SPEC_TABLE_URL, SPEC_TABLE_TIMEOUT)
            response = requests.get(SPEC_TABLE_URL, timeout=SPEC_TABLE_TIMEOUT)
            response.raise_for_status()
            tab_df = pd.read_csv(BytesIO(response.content), compression='gzip')
            tab = Table.from_pandas(tab_df)
            logger.info("Downloaded spectroscopic catalog from %s", SPEC_TABLE_URL)
            try:
                SPEC_TABLE_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                tab_df.to_csv(SPEC_TABLE_LOCAL_PATH, index=False)
            except Exception as exc:
                logger.warning("Could not cache spectroscopic catalog locally: %s", exc)
        except Exception as exc:
            logger.error("Spectroscopic catalog download failed: %s", exc)
    
    # If still nothing, return empty arrays so the app can run photo-only
    if tab is None:
        logger.warning("Proceeding without spectroscopic catalog (no spectra available).")
        return {
            'ra': np.array([], dtype=float),
            'dec': np.array([], dtype=float),
            'z_best': np.array([], dtype=float),
            'file': np.array([], dtype=str),
            'root': np.array([], dtype=str),
        }
    
    tab['metafile'] = [m.split('_')[0] for m in tab['msamet']]
    
    return {
        'ra': np.array(tab['ra'], dtype=float),
        'dec': np.array(tab['dec'], dtype=float),
        'z_best': np.array(tab['z_best'], dtype=float),
        'file': np.array(tab['file'], dtype=str),
        'root': np.array(tab['root'], dtype=str),
    }


@pn.cache
def get_matched_data():
    """Load and match catalogs (cached)"""
    logger.info("Loading and matching catalogs...")
    df, df_detected = load_photometric_catalog()
    spec_tab = load_spectroscopic_catalog()
    
    if len(spec_tab['ra']) == 0:
        logger.warning("No spectroscopic matches available (empty catalog).")
        df_matched = df.iloc[0:0].copy()
        return df, df_detected, df_matched
    
    logger.info("Cross-matching catalogs...")
    photo_coords = SkyCoord(ra=df['ra'].values * u.degree, dec=df['dec'].values * u.degree)
    spec_coords = SkyCoord(ra=spec_tab['ra'] * u.degree, dec=spec_tab['dec'] * u.degree)
    
    idx, d2d, _ = photo_coords.match_to_catalog_sky(spec_coords)
    match_mask = d2d < 0.5 * u.arcsec
    
    # Create matched dataframe
    matched_ids = df.index[match_mask]
    matched_idx = idx[match_mask]
    
    df_matched = df.loc[matched_ids].copy()
    df_matched['spec_z'] = spec_tab['z_best'][matched_idx]
    df_matched['spec_file'] = spec_tab['file'][matched_idx]
    df_matched['spec_root'] = spec_tab['root'][matched_idx]
    
    print(f"Found {len(df_matched)} spec-matched sources")
    return df, df_detected, df_matched


@pn.cache
def load_eazy_photoz():
    """Load EAzY photoz instance"""
    logger.info("Loading EAzY photoz (this can take ~90s)...")
    photoz = hdf5.initialize_from_hdf5(h5file=EAZY_H5_PATH)
    photoz.cat['zphoto'] = photoz.zbest
    logger.info("EAzY loaded!")
    return photoz


@pn.cache
def load_eazy_zout():
    """Load EAzY zout table"""
    print("Loading EAzY zout...")
    return Table.read(EAZY_ZOUT_PATH)


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================
class MinervaGalaxyBrowser(pn.viewable.Viewer):
    """
    Interactive Panel-based Galaxy Browser
    
    Parameters
    ----------
    custom_ids : list, optional
        Custom list of galaxy IDs to browse. If provided, "Custom" will be 
        added to the source list options.
    
    Examples
    --------
    # Default usage (spec-matched sources)
    app = MinervaGalaxyBrowser()
    
    # With custom ID list
    my_ids = [1160703, 1234567, 9876543]
    app = MinervaGalaxyBrowser(custom_ids=my_ids)
    
    # With outliers or any filtered list
    outlier_ids = df_matched[df_matched['delta_z'] > 0.15].index.tolist()
    app = MinervaGalaxyBrowser(custom_ids=outlier_ids)
    """
    
    galaxy_index = param.Integer(default=0, bounds=(0, 1000))
    source_list = param.Selector(default='Spec-matched', objects=['Spec-matched', 'All detected', 'High-z (z>3)', 'Custom'])
    cutout_size = param.Number(default=1.5, bounds=(0.5, 5.0))
    cutout_scale = param.Number(default=4.0, bounds=(1.0, 10.0))
    
    def __init__(self, custom_ids=None, **params):
        # Store custom IDs before super().__init__
        self._custom_ids = custom_ids
        super().__init__(**params)
        
        # Load global data (cached)
        self.df, self.df_detected, self.df_matched = get_matched_data()
        
        self.photoz = load_eazy_photoz()
        self.zout = load_eazy_zout()
        
        # Prefetch tracking
        self._prefetch_lock = threading.Lock()
        self._prefetching = set()
        self._display_token = 0
        
        # Initialize galaxy list
        self._update_galaxy_list()
        
        # Create widgets
        self._create_widgets()
        
        # Start initial prefetch
        self._schedule_prefetch()
    
    def _update_galaxy_list(self):
        """Update galaxy ID list based on selection"""
        has_custom = self._custom_ids is not None and len(self._custom_ids) > 0
        
        if self.source_list == 'Custom' and has_custom:
            # Filter to only IDs that exist in catalog
            valid_ids = [i for i in self._custom_ids if i in self.df.index]
            self.galaxy_ids = valid_ids if valid_ids else self.df_matched.index.tolist()[:10]
        elif self.source_list == 'Spec-matched':
            if len(self.df_matched) == 0:
                # Fallback to detected sample if no spectroscopic catalog available
                self.galaxy_ids = self.df_detected.index.tolist()[:500]
            else:
                self.galaxy_ids = self.df_matched.index.tolist()
        elif self.source_list == 'All detected':
            self.galaxy_ids = self.df_detected.index.tolist()[:500]
        elif self.source_list == 'High-z (z>3)':
            high_z = self.df_matched[self.df_matched['spec_z'] > 3.0]
            self.galaxy_ids = high_z.index.tolist() if len(high_z) > 0 else self.df_matched.index.tolist()[:10]
        else:
            self.galaxy_ids = self.df_matched.index.tolist()
        
        max_idx = max(0, len(self.galaxy_ids) - 1)
        self.param.galaxy_index.bounds = (0, max_idx)
        if self.galaxy_index > max_idx:
            self.galaxy_index = 0
    
    @property
    def current_id(self):
        if len(self.galaxy_ids) == 0:
            return None
        idx = min(self.galaxy_index, len(self.galaxy_ids) - 1)
        return self.galaxy_ids[idx]
    
    def _get_galaxy_info(self):
        gal_id = self.current_id
        if gal_id is None:
            return None
        
        try:
            ix = np.where(self.photoz.cat['id'] == gal_id)[0][0]
        except IndexError:
            return None
        
        info = {
            'id': gal_id,
            'ix': ix,
            'ra': float(self.photoz.RA[ix]),
            'dec': float(self.photoz.DEC[ix]),
            'z_phot': float(self.photoz.zbest[ix]),
            'has_spec': gal_id in self.df_matched.index,
        }
        
        if info['has_spec']:
            info['z_spec'] = float(self.df_matched.loc[gal_id, 'spec_z'])
            info['spec_file'] = str(self.df_matched.loc[gal_id, 'spec_file'])
            info['spec_root'] = str(self.df_matched.loc[gal_id, 'spec_root'])
            info['delta_z'] = abs(info['z_phot'] - info['z_spec']) / (1 + info['z_spec'])
        
        return info
    
    # =========================================================================
    # CACHING AND PREFETCHING
    # =========================================================================
    def _get_cache_key(self, gal_id, data_type, **kwargs):
        """Generate cache key for a galaxy and data type"""
        if data_type == 'cutout':
            return f"cutout_{gal_id}_{kwargs.get('size', 1.5)}_{kwargs.get('scale', 4.0)}"
        elif data_type == 'sed':
            return f"sed_{gal_id}"
        elif data_type == 'spectrum':
            return f"spectrum_{gal_id}"
        return f"{data_type}_{gal_id}"

    def _next_display_token(self):
        """Increment display token to keep async updates in sync with current selection"""
        self._display_token = time.time()
        return self._display_token

    def _safe_ui_update(self, func):
        """Update Panel document from background thread"""
        doc = pn.state.curdoc
        if doc:
            doc.add_next_tick_callback(lambda: func())
        else:
            func()

    def _async_update_pane(self, pane, fetch_fn, token):
        """
        Fetch data in background and update pane on the main thread.

        Parameters
        ----------
        pane : pn.pane.PaneBase
            The pane to update.
        fetch_fn : callable
            Function that returns the data to set on the pane.
        token : any
            Token identifying the current display state; prevents stale updates.
        """
        pane.loading = True
        logger.info("Fetching %s ...", pane.name if hasattr(pane, "name") else pane)

        def task():
            try:
                result = fetch_fn()
            except Exception as exc:
                logger.error("Background fetch error: %s", exc)
                result = None

            def update():
                if token != self._display_token:
                    return  # Ignore stale update
                pane.object = result
                pane.loading = False
                logger.info("Finished %s", pane.name if hasattr(pane, "name") else pane)

            self._safe_ui_update(update)

        PREFETCH_EXECUTOR.submit(task)
    
    def _fetch_cutout_data(self, info, size=1.5, scale=4.0):
        """Fetch cutout and return as PIL Image (cacheable)"""
        cache_key = self._get_cache_key(info['id'], 'cutout', size=size, scale=scale)
        
        # Check cache first
        cached = CUTOUT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            rd = f"{info['ra']:.6f},{info['dec']:.6f}"
            url = f"https://grizli-cutout.herokuapp.com/thumb?coord={rd}&all_filters=True&size={size}&scl={scale}&asinh=True&filters={FILTERS}&rgb_scl=1.0,0.95,1.2&pl=2"
            
            logger.info("Fetching cutout for id=%s from %s", info['id'], url)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Fix black pixels
            data = np.array(img)
            black = data.max(axis=2) == 0
            for ioff in range(-2, 3):
                black &= np.roll(black, ioff, axis=0)
            for i in range(3):
                data[:, :, i][black] = 255
            
            result = Image.fromarray(data)
            CUTOUT_CACHE.put(cache_key, result)
            return result
        except Exception as e:
            print(f"Cutout fetch error for {info['id']}: {e}")
            return None
    
    def _fetch_sed_data(self, info):
        """Generate SED and return as PNG bytes (cacheable)"""
        cache_key = self._get_cache_key(info['id'], 'sed')
        
        # Check cache first
        cached = SED_CACHE.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            fig, sed_data = self.photoz.show_fit(id=info['id'], add_label=False)
            title = f"ID {info['id']} | z_phot={info['z_phot']:.3f}"
            if info['has_spec']:
                title += f" | z_spec={info['z_spec']:.3f}"
            fig.suptitle(title, fontsize=11, fontweight='bold')
            
            # Convert to PNG bytes for caching
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            result = buf.getvalue()
            plt.close(fig)
            
            SED_CACHE.put(cache_key, result)
            return result
        except Exception as e:
            print(f"SED generation error for {info['id']}: {e}")
            return None
    
    def _fetch_spectrum_data(self, info):
        """Fetch spectrum and return as PIL Image (cacheable)"""
        if not info['has_spec']:
            return None
        
        cache_key = self._get_cache_key(info['id'], 'spectrum')
        
        # Check cache first
        cached = SPECTRUM_CACHE.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            png_name = info['spec_file'].replace('.spec.fits', '.flam.png')
            local_candidates = [
                SPECTRUM_LOCAL_DIR / info['spec_root'] / png_name,
                SPECTRUM_LOCAL_DIR / png_name,
            ]
            
            for local_path in local_candidates:
                if local_path.is_file():
                    result = Image.open(local_path)
                    SPECTRUM_CACHE.put(cache_key, result)
                    return result
            
            url = f"https://s3.amazonaws.com/msaexp-nirspec/extractions/{info['spec_root']}/{info['spec_file']}".replace('.spec.fits', '.flam.png')
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            result = Image.open(BytesIO(response.content))
            
            SPECTRUM_CACHE.put(cache_key, result)
            return result
        except Exception as e:
            print(f"Spectrum fetch error for {info['id']}: {e}")
            return None
    
    def _prefetch_galaxy(self, gal_id):
        """Prefetch all data for a single galaxy (runs in background thread)"""
        with self._prefetch_lock:
            if gal_id in self._prefetching:
                return
            self._prefetching.add(gal_id)
        
        try:
            info = self._get_galaxy_info_by_id(gal_id)
            if info is None:
                return
            
            # Prefetch all three data types
            self._fetch_cutout_data(info, self.cutout_size, self.cutout_scale)
            self._fetch_sed_data(info)
            self._fetch_spectrum_data(info)
        finally:
            with self._prefetch_lock:
                self._prefetching.discard(gal_id)
    
    def _get_galaxy_info_by_id(self, gal_id):
        """Get galaxy info by ID (not current index)"""
        if gal_id is None:
            return None
        
        try:
            ix = np.where(self.photoz.cat['id'] == gal_id)[0][0]
        except IndexError:
            return None
        
        info = {
            'id': gal_id,
            'ix': ix,
            'ra': float(self.photoz.RA[ix]),
            'dec': float(self.photoz.DEC[ix]),
            'z_phot': float(self.photoz.zbest[ix]),
            'has_spec': gal_id in self.df_matched.index,
        }
        
        if info['has_spec']:
            info['z_spec'] = float(self.df_matched.loc[gal_id, 'spec_z'])
            info['spec_file'] = str(self.df_matched.loc[gal_id, 'spec_file'])
            info['spec_root'] = str(self.df_matched.loc[gal_id, 'spec_root'])
            info['delta_z'] = abs(info['z_phot'] - info['z_spec']) / (1 + info['z_spec'])
        
        return info
    
    def _schedule_prefetch(self):
        """Schedule prefetch for nearby galaxies"""
        current_idx = self.galaxy_index
        n_galaxies = len(self.galaxy_ids)
        
        # Prefetch next 3 and previous 1 galaxies
        indices_to_prefetch = []
        for offset in [1, 2, 3, -1]:
            idx = current_idx + offset
            if 0 <= idx < n_galaxies:
                indices_to_prefetch.append(idx)
        
        for idx in indices_to_prefetch:
            gal_id = self.galaxy_ids[idx]
            # Check if already cached
            cache_key = self._get_cache_key(gal_id, 'cutout', size=self.cutout_size, scale=self.cutout_scale)
            if not CUTOUT_CACHE.contains(cache_key):
                PREFETCH_EXECUTOR.submit(self._prefetch_galaxy, gal_id)
    
    def _create_widgets(self):
        """Create UI widgets"""
        # Navigation
        self.prev_btn = pn.widgets.Button(name='◀ Prev', button_type='primary', width=80)
        self.next_btn = pn.widgets.Button(name='Next ▶', button_type='primary', width=80)
        self.random_btn = pn.widgets.Button(name='🎲 Random', button_type='warning', width=100)
        
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)
        self.random_btn.on_click(self._on_random)
        
        self.jump_input = pn.widgets.IntInput(name='Jump to ID', value=self.current_id or 0, width=120)
        self.jump_btn = pn.widgets.Button(name='Go', button_type='success', width=50)
        self.jump_btn.on_click(self._on_jump)
        
        # Source list - include Custom option if custom_ids provided
        source_options = ['Spec-matched', 'All detected', 'High-z (z>3)', 'Custom']
        has_custom = self._custom_ids is not None and len(self._custom_ids) > 0
        default_source = 'Custom' if has_custom else 'Spec-matched'
        
        self.source_selector = pn.widgets.Select(
            name='Source List',
            options=source_options,
            value=default_source,
            width=180
        )
        self.source_list = default_source
        self.source_selector.param.watch(self._on_source_change, 'value')
        
        self.index_slider = pn.widgets.IntSlider(
            name='Index', start=0, end=max(1, len(self.galaxy_ids) - 1),
            value=self.galaxy_index, width=250
        )
        self.index_slider.param.watch(self._on_slider_change, 'value')
        
        # Custom ID input
        self.custom_ids_input = pn.widgets.TextAreaInput(
            name='Custom IDs (comma-separated)',
            placeholder='e.g., 1160703, 1234567, 9876543',
            height=60, width=250
        )
        self.load_custom_btn = pn.widgets.Button(name='Load IDs', button_type='primary', width=100)
        self.load_custom_btn.on_click(self._on_load_custom_ids)
        
        # Preload button for batch prefetching
        self.preload_btn = pn.widgets.Button(name='⚡ Preload All', button_type='warning', width=100)
        self.preload_btn.on_click(self._on_preload_all)
        
        # Cutout controls
        self.size_input = pn.widgets.FloatInput(name='Size (arcsec)', value=self.cutout_size, step=0.5, width=100)
        self.scale_input = pn.widgets.FloatInput(name='Scale', value=self.cutout_scale, step=0.5, width=100)
        self.refresh_btn = pn.widgets.Button(name='🔄 Refresh', button_type='light', width=100)
        self.refresh_btn.on_click(self._on_refresh)
        
        # Output panes - initialized with loading state
        self.info_pane = pn.pane.Markdown("Loading...", width=280)
        self.progress_pane = pn.pane.Markdown("0 / 0")
        self.cache_stats_pane = pn.pane.Markdown("Cache: initializing...")
        self.cutout_pane = pn.pane.PNG(None, width=1050)
        self.sed_pane = pn.pane.PNG(None, width=1050)  # Changed to PNG for caching
        self.spectrum_pane = pn.pane.PNG(None, width=1050)
        
        # Title panes for each panel
        self.cutout_title = pn.pane.Markdown("### 📷 Multi-band Cutout", styles={'font-weight': 'bold'})
        self.sed_title = pn.pane.Markdown("### 📊 EAzY SED Fit", styles={'font-weight': 'bold'})
        self.spectrum_title = pn.pane.Markdown("### 🌈 NIRSpec Spectrum", styles={'font-weight': 'bold'})
    
    def _on_load_custom_ids(self, event):
        """Load custom IDs from text input"""
        text = self.custom_ids_input.value
        if not text:
            return
        
        try:
            # Parse comma or whitespace separated IDs
            ids = [int(x.strip()) for x in text.replace(',', ' ').split() if x.strip()]
            if len(ids) > 0:
                self._custom_ids = ids
                self.source_selector.value = 'Custom'
                self.source_list = 'Custom'
                self._update_galaxy_list()
                self.index_slider.end = max(1, len(self.galaxy_ids) - 1)
                self._sync_widgets()
                self._update_display()
                self._schedule_prefetch()
                print(f"Loaded {len(self.galaxy_ids)} valid IDs")
        except ValueError as e:
            print(f"Error parsing IDs: {e}")
    
    def _on_preload_all(self, event):
        """Preload all galaxies in current list (up to 100)"""
        max_preload = min(100, len(self.galaxy_ids))
        print(f"Starting preload of {max_preload} galaxies...")
        self.preload_btn.name = f"⏳ Loading 0/{max_preload}..."
        self.preload_btn.disabled = True
        
        def preload_batch():
            for i, gal_id in enumerate(self.galaxy_ids[:max_preload]):
                self._prefetch_galaxy(gal_id)
                # Update button text periodically
                if (i + 1) % 5 == 0:
                    count = i + 1
                    self._safe_ui_update(lambda c=count: setattr(self.preload_btn, "name", f"⏳ Loading {c}/{max_preload}..."))
            
            def finish():
                self.preload_btn.name = f"✅ Loaded {max_preload}"
                self.preload_btn.disabled = False
                self.cache_stats_pane.object = f"📊 {CUTOUT_CACHE.stats()}"
                print(f"Preload complete! {CUTOUT_CACHE.stats()}")
            self._safe_ui_update(finish)
        
        # Run in background
        PREFETCH_EXECUTOR.submit(preload_batch)
    
    # Event handlers
    def _on_prev(self, event):
        if self.galaxy_index > 0:
            self.galaxy_index -= 1
            self._sync_widgets()
            self._update_display()
            self._schedule_prefetch()
    
    def _on_next(self, event):
        if self.galaxy_index < len(self.galaxy_ids) - 1:
            self.galaxy_index += 1
            self._sync_widgets()
            self._update_display()
            self._schedule_prefetch()
    
    def _on_random(self, event):
        self.galaxy_index = np.random.randint(0, len(self.galaxy_ids))
        self._sync_widgets()
        self._update_display()
        self._schedule_prefetch()
    
    def _on_jump(self, event):
        target_id = self.jump_input.value
        if target_id in self.galaxy_ids:
            self.galaxy_index = self.galaxy_ids.index(target_id)
            self._sync_widgets()
            self._update_display()
            self._schedule_prefetch()
    
    def _on_slider_change(self, event):
        self.galaxy_index = event.new
        self.jump_input.value = self.current_id or 0
        self._update_display()
        self._schedule_prefetch()
    
    def _on_source_change(self, event):
        self.source_list = event.new
        self._update_galaxy_list()
        self.index_slider.end = max(1, len(self.galaxy_ids) - 1)
        self._sync_widgets()
        self._update_display()
        self._schedule_prefetch()
    
    def _on_refresh(self, event):
        self.cutout_size = self.size_input.value
        self.cutout_scale = self.scale_input.value
        token = self._next_display_token()
        self._update_cutout(token)
        self._schedule_prefetch()  # Re-prefetch with new settings
    
    def _sync_widgets(self):
        self.index_slider.value = self.galaxy_index
        self.jump_input.value = self.current_id or 0
    
    # Display updates
    def _update_display(self):
        """Update all display panes"""
        info = self._get_galaxy_info()
        if info is None:
            return
        token = self._next_display_token()
        
        # Update info
        self.progress_pane.object = f"**{self.galaxy_index + 1} / {len(self.galaxy_ids)}**"
        
        # Update cache stats
        self.cache_stats_pane.object = f"📊 {CUTOUT_CACHE.stats()}"
        
        md = f"""### Galaxy ID: {info['id']}
**RA:** {info['ra']:.6f}  
**Dec:** {info['dec']:.6f}  
**z_phot:** {info['z_phot']:.4f}  
"""
        if info['has_spec']:
            md += f"""**z_spec:** {info['z_spec']:.4f}  
**Δz/(1+z):** {info['delta_z']:.4f}
"""
        else:
            md += "**z_spec:** *N/A*\n"
        md += f"\n[🔗 MINERVA Viewer](https://minerva.colorado.edu/uds/?ra={info['ra']:.6f}&dec={info['dec']:.6f}&zoom=11)"
        self.info_pane.object = md
        
        # Update panel titles with object ID
        self.cutout_title.object = f"### 📷 Multi-band Cutout — **ID: {info['id']}**"
        self.sed_title.object = f"### 📊 EAzY SED Fit — **ID: {info['id']}**"
        if info['has_spec']:
            self.spectrum_title.object = f"### 🌈 NIRSpec Spectrum — **ID: {info['id']}**"
        else:
            self.spectrum_title.object = f"### 🌈 NIRSpec Spectrum — **ID: {info['id']}** (No spectrum available)"
        
        # Update panels
        self._update_cutout(token)
        self._update_sed(token)
        self._update_spectrum(token)
    
    def _update_cutout(self, token=None):
        info = self._get_galaxy_info()
        if info is None:
            self.cutout_pane.object = None
            self.cutout_pane.loading = False
            return
        
        token = token or self._display_token
        
        def fetch():
            return self._fetch_cutout_data(info, self.cutout_size, self.cutout_scale)
        
        self._async_update_pane(self.cutout_pane, fetch, token)
    
    def _update_sed(self, token=None):
        info = self._get_galaxy_info()
        if info is None:
            self.sed_pane.object = None
            self.sed_pane.loading = False
            return
        
        token = token or self._display_token
        
        def fetch():
            sed_bytes = self._fetch_sed_data(info)
            return Image.open(BytesIO(sed_bytes)) if sed_bytes is not None else None
        
        self._async_update_pane(self.sed_pane, fetch, token)
    
    def _update_spectrum(self, token=None):
        info = self._get_galaxy_info()
        if info is None or not info['has_spec']:
            self.spectrum_pane.object = None
            self.spectrum_pane.loading = False
            return
        
        token = token or self._display_token
        
        def fetch():
            return self._fetch_spectrum_data(info)
        
        self._async_update_pane(self.spectrum_pane, fetch, token)
    
    def __panel__(self):
        """Return the Panel layout"""
        # Initial display update
        self._update_display()
        
        # Start prefetching
        self._schedule_prefetch()
        
        # Navigation row
        nav_row = pn.Row(
            self.prev_btn, self.next_btn, self.random_btn,
            pn.Spacer(width=15),
            self.jump_input, self.jump_btn,
            pn.Spacer(width=15),
            self.progress_pane,
            height=50
        )
        
        # Sidebar
        sidebar = pn.Column(
            "## Controls",
            self.source_selector,
            self.index_slider,
            pn.layout.Divider(),
            "### Custom IDs",
            self.custom_ids_input,
            pn.Row(self.load_custom_btn, self.preload_btn),
            pn.layout.Divider(),
            "### Cutout",
            self.size_input,
            self.scale_input,
            self.refresh_btn,
            pn.layout.Divider(),
            self.info_pane,
            pn.layout.Divider(),
            self.cache_stats_pane,
            width=300
        )
        
        # Tabs
        tabs = pn.Tabs(
            ('📷 Cutout', pn.Column(self.cutout_title, self.cutout_pane)),
            ('📊 SED', pn.Column(self.sed_title, self.sed_pane)),
            ('🌈 Spectrum', pn.Column(self.spectrum_title, self.spectrum_pane)),
            dynamic=True
        )
        
        return pn.Column(
            pn.pane.Markdown("# 🔭 MINERVA Galaxy Browser", styles={'text-align': 'center'}),
            nav_row,
            pn.Row(sidebar, tabs),
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

# Helper function for programmatic use
def create_browser(custom_ids=None):
    """
    Create a galaxy browser with optional custom ID list.
    
    Parameters
    ----------
    custom_ids : list, optional
        List of galaxy IDs to browse
    
    Returns
    -------
    MinervaGalaxyBrowser
        The browser app instance
    
    Examples
    --------
    # In a notebook:
    import panel as pn
    pn.extension()
    
    # Default
    app = create_browser()
    app.servable()
    
    # With custom IDs
    app = create_browser(custom_ids=[1160703, 1234567])
    app.servable()
    
    # With outliers
    outliers = df_matched[df_matched['spec_z'] > 5].index.tolist()
    app = create_browser(custom_ids=outliers)
    app.servable()
    """
    return MinervaGalaxyBrowser(custom_ids=custom_ids)


# Default app (uncomment and modify to pass custom_ids)
# Example with custom list:
# my_ids = [1160703, 1234567, 9876543]
# app = MinervaGalaxyBrowser(custom_ids=my_ids)

# Default: no custom list
app = MinervaGalaxyBrowser()
app.servable()
