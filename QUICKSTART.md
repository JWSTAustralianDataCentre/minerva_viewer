# MINERVA Viewer - Quick Start Guide

## What Was Done

✅ **Added `.gitignore`** - Excludes `__pycache__` and other unnecessary files from git
✅ **Removed `__pycache__` from git tracking** - Cleaned up repository
✅ **Created comprehensive `README.md`** - Full documentation with MINERVA survey context
✅ **Created `setup.py`** - Easy installation via pip
✅ **Created `requirements.txt`** - Dependency list for pip install

## Installation (Choose One Method)

### Method 1: Using setup.py (Recommended)
```bash
cd minerva_viewer
pip install -e .  # Development/editable mode
```

### Method 2: Using requirements.txt
```bash
cd minerva_viewer
pip install -r requirements.txt
```

### Method 3: Manual
```bash
pip install numpy pandas matplotlib astropy eazy-py panel param pillow requests scipy grizli
```

## Quick Test

After installation, test the imports:

```python
python -c "import panel; import eazy; import grizli; print('✓ All packages installed!')"
```

## Running the Browser

```bash
# From minerva_viewer directory
panel serve zgui_panel.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=*
```

Then open: http://localhost:5006

## Configuration Required

Before first run, configure your data paths:

```bash
# Interactive setup (recommended)
python setup_paths.py

# Or check current configuration
python setup_paths.py --check
```

The script will guide you through configuring all required paths.

## File Structure

```
minerva_viewer/
├── .gitignore              # Git ignore rules
├── README.md               # Full documentation
├── QUICKSTART.md           # This file
├── setup.py                # Package installer
├── setup_paths.py          # Path configuration wizard
├── requirements.txt        # Dependencies list
├── config.py               # Configuration module
├── config.ini.example      # Example config file
├── minerva_viewer.py       # Analysis scripts
├── zgui_panel.py          # Interactive browser app
└── templates/             # EAzY templates
```

## What the Tools Do

### `zgui_panel.py` - Interactive Browser
- View galaxy cutouts (15 NIRCam bands)
- Display photo-z SED fits
- Show NIRSpec spectra (when available)
- Navigate with buttons, sliders, or custom ID lists
- Smart caching for smooth browsing

### `minerva_viewer.py` - Analysis Script
- Photo-z vs spec-z comparison plots
- Calculate NMAD statistics
- Evaluate outlier fractions
- Support multiple template sets

## Git Commits Made

1. **"Add .gitignore and remove __pycache__ from tracking"**
   - Added comprehensive .gitignore
   - Removed cached bytecode files

2. **"Add comprehensive README, setup.py, and requirements.txt"**
   - Full documentation
   - Easy installation setup
   - Dependency management

## Next Steps

1. **Configure data paths** in both Python files
2. **Install dependencies**: `pip install -e .`
3. **Test the browser**: `panel serve zgui_panel.py ...`
4. **Customize**: Add your own galaxy ID lists or filters

## Support

See README.md for:
- Detailed usage instructions
- Configuration guide
- Troubleshooting tips
- Citation information

---
Repository: https://github.com/JWSTAustralianDataCentre/minerva_viewer
Last updated: December 11, 2025
