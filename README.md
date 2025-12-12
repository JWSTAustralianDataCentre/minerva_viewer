# MINERVA Galaxy Viewer

An interactive web application for exploring galaxies from the JWST MINERVA survey, featuring multi-band NIRCam cutouts, photometric redshift (photo-z) SED fits, and NIRSpec spectroscopy where available.

## About MINERVA Survey

**MINERVA (Medium-band Imaging of NIRCam for Extragalactic Research and Vision Advantage)** is a JWST NIRCam imaging survey conducting ultra-deep observations over the Hubble legacy fields. The survey uses medium-band filters to achieve high-precision photometric redshifts and detailed characterization of galaxy properties.

See https://ui.adsabs.harvard.edu/abs/2025arXiv250719706M for details. 

## Repository Overview

This repository contains tools for visualizing and analyzing MINERVA survey data:

1. **`minerva_viewer.py`** - Analysis scripts for photo-z performance evaluation and catalog matching
2. **`zgui_panel.py`** - Interactive Panel web application for browsing galaxies with real-time cutout generation

## Features

### Interactive Galaxy Browser (`zgui_panel.py`)
- 🖼️ **Multi-band cutouts** from Grizli cutout service (15 NIRCam filters)
- 📊 **EAzY SED fitting** with photo-z estimates
- 🌈 **NIRSpec spectra** display (when available)
- 🎲 **Navigation tools**: Next/Previous, Random, Jump to ID
- 📋 **Multiple source lists**: All detected, Spec-matched, High-z (z>3), Custom ID lists
- ⚡ **Smart caching** with background prefetching for smooth browsing
- 🔄 **Customizable cutouts**: Adjustable size and scale

### Analysis Tools (`minerva_viewer.py`)
- Photo-z vs spec-z comparison plots
- NMAD (Normalized Median Absolute Deviation) statistics
- Outlier fraction calculations
- Cross-matching between photometric and spectroscopic catalogs
- Support for multiple template sets (SFHZ, SFHZ_BLUE_AGN, LARSON, LARSON_MIRI)

## Installation

### Prerequisites
- Python 3.8 or later
- Access to MINERVA data files (catalogs, EAzY outputs)

### Method 1: Using setup.py (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd minerva_viewer

# Install in development mode (editable)
pip install -e .

# Or install normally
pip install .
```

### Method 2: Using requirements.txt

```bash
# Install dependencies
pip install -r requirements.txt
```

### Method 3: Manual Installation

```bash
pip install numpy pandas matplotlib astropy eazy-py panel param pillow requests scipy grizli
```

## Configuration

The viewer uses a flexible configuration system that reads paths from multiple sources (in priority order):

1. **Environment variables** (highest priority)
2. **Configuration file** (`~/.minerva/config.ini` or `minerva_viewer/config.ini`)
3. **Default values** (lowest priority)

### Quick Setup (Recommended)

Run the interactive configuration wizard:

```bash
python setup_paths.py
```

This will guide you through configuring all data paths and save them to a configuration file.

### Configuration Methods

#### Method 1: Interactive Setup Script

```bash
# Run the wizard
python setup_paths.py

# Check current configuration
python setup_paths.py --check
```

#### Method 2: Environment Variables

```bash
# Set environment variables (add to ~/.bashrc or ~/.zshrc for persistence)
export MINERVA_DATA_DIR="/path/to/your/data"
export MINERVA_CATALOG_PATH="/path/to/catalog.fits"
export MINERVA_EAZY_H5_PATH="/path/to/eazy.h5"
# ... etc

# Then run the viewer
python minerva_viewer.py
```

#### Method 3: Configuration File

Create `~/.minerva/config.ini` or `minerva_viewer/config.ini`:

```ini
[paths]
data_dir = /path/to/your/data
catalog_path = /path/to/catalog.fits
sps_catalog_path = /path/to/sps_catalog.fits
eazy_h5_path = /path/to/eazy.h5
eazy_zout_path = /path/to/eazy_zout.fits
spec_table_path = /path/to/spec_table.csv.gz
spectra_dir = /path/to/spectra
```

#### Method 4: Non-Interactive Setup

```bash
python setup_paths.py \
  --data-dir /path/to/data \
  --catalog /path/to/catalog.fits \
  --eazy-h5 /path/to/eazy.h5
```

### Configuration Priority

If you set the same path in multiple places:
- Environment variable > Config file > Default value
- This allows you to temporarily override settings without editing files

## Usage

### Running the Interactive Browser

The Panel application can be launched in several ways:

#### Option 1: Command Line (Production)

```bash
panel serve zgui_panel.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=*
```

Then open your browser to `http://localhost:5006`

#### Option 2: Remote Access (via ngrok)

```bash
# Terminal 1: Start the Panel server
panel serve zgui_panel.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=*

# Terminal 2: Create ngrok tunnel
ngrok http 5006
```

Access the ngrok URL from anywhere!

#### Option 3: Jupyter Notebook

```python
import panel as pn
from zgui_panel import create_browser

pn.extension()

# Default browser with spec-matched sources
app = create_browser()
app.servable()

# Or with custom ID list
my_galaxy_ids = [1160703, 1234567, 9876543]
app = create_browser(custom_ids=my_galaxy_ids)
app.servable()
```

### Using the Analysis Scripts

```python
# Run the photo-z analysis
python minerva_viewer.py
```

This will:
1. Load the photometric and spectroscopic catalogs
2. Cross-match sources within 0.5 arcsec
3. Generate photo-z vs spec-z comparison plots
4. Calculate NMAD and outlier statistics
5. Display results for multiple template sets

### Programmatic Access

```python
from zgui_panel import MinervaGalaxyBrowser
import panel as pn

pn.extension()

# Create browser with outliers only
outlier_ids = df_matched[df_matched['spec_z'] - df_matched['z_phot'] > 0.5].index.tolist()
app = MinervaGalaxyBrowser(custom_ids=outlier_ids)
app.servable()
```

## Data Requirements

The application expects the following data structure:

```
minerva_viewer/
├── minerva_viewer.py
├── zgui_panel.py
├── templates/                 # EAzY templates
└── ../data/
    ├── catalogs/
    │   ├── MINERVA-UDS_*.fits
    │   ├── *_SPScatalog_*.fits
    │   └── dja_msaexp_emission_lines_v4.4.csv.gz
    ├── EAzY/
    │   ├── SFHZ/
    │   ├── LARSON/
    │   └── LARSON_MIRI/
    └── spectra/               # Optional: Local NIRSpec spectra
```

### Required Data Files:

1. **Main photometric catalog** (`MINERVA-UDS_*_SUPER_CATALOG.fits`)
2. **SPS catalog** (Stellar Population Synthesis results)
3. **EAzY outputs**: 
   - `.h5` file (photo-z HDF5 archive)
   - `.zout.fits` file (redshift outputs)
4. **Spectroscopic catalog** (optional, can download from URL)
5. **EAzY templates** in `templates/` directory

**Note:** After obtaining the data, run `python setup_paths.py` to configure the paths.

## Browser Interface Guide

### Navigation
- **◀ Prev / Next ▶**: Navigate through galaxies sequentially
- **🎲 Random**: Jump to a random galaxy
- **Jump to ID**: Enter a specific galaxy ID
- **Index Slider**: Scroll through the current list
- **Source List**: Choose between different galaxy samples
  - *Spec-matched*: Galaxies with spectroscopy
  - *All detected*: All detected sources (4+ medium-band detections)
  - *High-z (z>3)*: High-redshift candidates
  - *Custom*: Your own ID list

### Display Panels
1. **Multi-band Cutout**: RGB composite from Grizli cutout service
2. **EAzY SED Fit**: Photo-z fit with observed photometry and best-fit template
3. **NIRSpec Spectrum**: 2D and 1D spectrum (when available)

### Cutout Controls
- **Size**: Cutout size in arcseconds (0.5-5.0)
- **Scale**: Image scaling factor (1.0-10.0)
- **🔄 Refresh**: Regenerate cutout with new parameters

### Advanced Features
- **⚡ Preload All**: Cache up to 100 galaxies for instant viewing
- **Custom IDs**: Load your own galaxy ID lists via text input
- **Cache Statistics**: Monitor cache performance

## Performance Tips

1. **Use Preload**: Click "⚡ Preload All" to cache galaxies before browsing
2. **Local Spectra**: Set `SPECTRUM_LOCAL_DIR` to local path for faster spectrum loading
3. **Adjust Cache Size**: Modify `LRUCache(maxsize=300)` in code for your memory constraints
4. **Background Prefetch**: The app automatically prefetches nearby galaxies

## Troubleshooting

### Panel server won't start
- Check that port 5006 is not in use: `lsof -i :5006`
- Try a different port: `panel serve zgui_panel.py --port 5007`

### Data files not found
- Run `python setup_paths.py --check` to verify your configuration
- Run `python setup_paths.py` to reconfigure paths
- Check that all required FITS files exist in the configured locations
- Ensure EAzY .h5 file is present (can take ~90s to load)

### Configuration issues
- Check configuration: `python setup_paths.py --check`
- View current paths: `python -c "from config import Config; print(Config())"`
- Reset configuration: Delete `~/.minerva/config.ini` and run setup again

### Slow performance
- Enable local spectrum caching
- Reduce `maxsize` for caches if memory constrained
- Use "Preload All" for batch viewing
- Check internet connection for remote cutout service

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Install eazy-py: `pip install eazy-py`
- Install grizli: `pip install grizli`

## Technical Details

### Photo-z Methods
- **EAzY (Easy and Accurate Zphot from Yale)**: Template fitting code
- **Templates**: SFHZ (Star Formation History-z), LARSON (dusty), custom sets
- **Metrics**: NMAD (σ_NMAD), outlier fraction (η), bias

### Spectroscopy
- **NIRSpec MSA**: Multi-shutter array spectroscopy
- **Data source**: MSAexp emission line catalog (v4.4)
- **Wavelength range**: 0.6-5.3 μm (depending on grating)

### Cutout Service
- **Provider**: Grizli cutout service (https://grizli-cutout.herokuapp.com)
- **Filters**: 15 NIRCam bands
- **Imaging**: ACS/WFC3 (HST) + NIRCam (JWST)

## Citation

If you use MINERVA data or this viewer in your research, please cite:

```
@ARTICLE{2024MINERVA,
       author = {{[MINERVA Team]}},
        title = "{MINERVA: Medium-band Imaging of NIRCam for Extragalactic Research}",
      journal = {In prep},
         year = 2024,
}
```

And for the tools used:
- **EAzY**: Brammer et al. 2008 ([ApJ 686, 1503](https://ui.adsabs.harvard.edu/abs/2008ApJ...686.1503B))
- **Grizli**: [GitHub repository](https://github.com/gbrammer/grizli)
- **Panel**: [https://panel.holoviz.org/](https://panel.holoviz.org/)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For questions or issues:
- Open an issue on GitHub
- Contact the MINERVA team

## License

[Specify your license here, e.g., MIT, GPL-3.0, etc.]

## Acknowledgments

- **JWST**: NASA, ESA, CSA
- **STScI**: Data archive and pipeline
- **Grizli cutout service**: G. Brammer
- **MSAexp catalog**: MSAexp collaboration
- **EAzY**: G. Brammer & the EAzY team

---

*Last updated: December 2025*
