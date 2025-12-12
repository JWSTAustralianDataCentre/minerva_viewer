#!/usr/bin/env python
"""
MINERVA Viewer - Interactive Path Configuration Setup

This script helps you configure the paths for MINERVA data files.
It creates a configuration file that will be used by all MINERVA viewer tools.

Usage:
    python setup_paths.py
    
Or for non-interactive mode:
    python setup_paths.py --data-dir /path/to/data
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Optional

# Add current directory to path to import config
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config


def print_header():
    """Print welcome header"""
    print("=" * 70)
    print("🔭 MINERVA Viewer - Path Configuration Setup")
    print("=" * 70)
    print()


def prompt_path(
    prompt_text: str,
    default: Optional[Path] = None,
    must_exist: bool = False,
    is_file: bool = False
) -> Path:
    """
    Prompt user for a path with validation.
    
    Parameters
    ----------
    prompt_text : str
        Text to display to user
    default : Path, optional
        Default path if user presses Enter
    must_exist : bool
        If True, path must exist
    is_file : bool
        If True, check for file; if False, check for directory
    
    Returns
    -------
    Path
        Validated path
    """
    while True:
        if default:
            prompt = f"{prompt_text}\n  Default: {default}\n  Path: "
        else:
            prompt = f"{prompt_text}\n  Path: "
        
        user_input = input(prompt).strip()
        
        if not user_input and default:
            path = default
        elif not user_input:
            print("  ❌ Path cannot be empty. Please try again.\n")
            continue
        else:
            path = Path(user_input).expanduser().resolve()
        
        # Validation
        if must_exist:
            if is_file and not path.is_file():
                print(f"  ❌ File not found: {path}")
                retry = input("  Try again? (y/n): ").lower()
                if retry != 'y':
                    return path
                continue
            elif not is_file and not path.is_dir():
                print(f"  ❌ Directory not found: {path}")
                retry = input("  Try again? (y/n): ").lower()
                if retry != 'y':
                    return path
                continue
        
        print(f"  ✓ Using: {path}\n")
        return path


def prompt_yes_no(prompt_text: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer"""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt_text} ({default_str}): ").strip().lower()
        if not response:
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("  Please answer 'y' or 'n'")


def interactive_setup():
    """Run interactive configuration setup"""
    print_header()
    
    print("This wizard will help you configure paths to your MINERVA data files.")
    print("You can press Enter to accept default values.\n")
    
    config_dict = {}
    
    # 1. Base data directory
    print("📁 Step 1: Base Data Directory")
    print("-" * 70)
    print("This is the root directory containing all MINERVA data.")
    print("Expected structure: data/catalogs/, data/EAzY/, data/spectra/, etc.\n")
    
    default_data_dir = Path('../data').resolve()
    data_dir = prompt_path(
        "Enter the path to your MINERVA data directory:",
        default=default_data_dir,
        must_exist=False
    )
    config_dict['data_dir'] = str(data_dir)
    
    # 2. Check if using standard structure
    print("\n📦 Step 2: Directory Structure")
    print("-" * 70)
    
    catalogs_dir = data_dir / 'catalogs'
    eazy_dir = data_dir / 'EAzY' / 'SFHZ' / 'SUPER' / 'ZPiter'
    
    if catalogs_dir.exists() and eazy_dir.exists():
        print(f"✓ Found standard directory structure in {data_dir}")
        print(f"  - Catalogs: {catalogs_dir}")
        print(f"  - EAzY: {eazy_dir}")
        
        use_standard = prompt_yes_no(
            "\nUse standard structure with default filenames?",
            default=True
        )
        
        if use_standard:
            print("\n✓ Using standard structure. Setup complete!")
            return config_dict
    else:
        print(f"⚠️  Standard structure not found in {data_dir}")
        print("You'll need to specify individual file paths.\n")
    
    # 3. Individual file paths
    print("\n📄 Step 3: Catalog Files")
    print("-" * 70)
    
    # Main catalog
    default_catalog = data_dir / 'catalogs' / 'MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG.fits'
    catalog_path = prompt_path(
        "Main photometric catalog (FITS file):",
        default=default_catalog,
        must_exist=True,
        is_file=True
    )
    config_dict['catalog_path'] = str(catalog_path)
    
    # SPS catalog
    default_sps = data_dir / 'catalogs' / 'MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG_SPScatalog_0.0.fits'
    sps_path = prompt_path(
        "SPS catalog (FITS file):",
        default=default_sps,
        must_exist=True,
        is_file=True
    )
    config_dict['sps_catalog_path'] = str(sps_path)
    
    # 4. EAzY files
    print("\n📊 Step 4: EAzY Photo-z Files")
    print("-" * 70)
    
    # EAzY H5
    default_h5 = data_dir / 'EAzY' / 'SFHZ' / 'SUPER' / 'ZPiter' / 'MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_zpiter_CATALOG.sfhz.eazypy.h5'
    h5_path = prompt_path(
        "EAzY HDF5 file (.h5):",
        default=default_h5,
        must_exist=True,
        is_file=True
    )
    config_dict['eazy_h5_path'] = str(h5_path)
    
    # EAzY zout
    default_zout = data_dir / 'EAzY' / 'SFHZ' / 'SUPER' / 'ZPiter' / 'MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_zpiter_CATALOG_sfhz.zout.fits'
    zout_path = prompt_path(
        "EAzY zout file (.zout.fits):",
        default=default_zout,
        must_exist=True,
        is_file=True
    )
    config_dict['eazy_zout_path'] = str(zout_path)
    
    # 5. Optional files
    print("\n🌟 Step 5: Optional Files (press Enter to skip)")
    print("-" * 70)
    
    # Spectroscopic catalog (optional)
    has_spec = prompt_yes_no(
        "Do you have a local copy of the spectroscopic catalog?",
        default=False
    )
    
    if has_spec:
        default_spec = data_dir / 'catalogs' / 'dja_msaexp_emission_lines_v4.4.csv.gz'
        spec_path = prompt_path(
            "Spectroscopic catalog (.csv.gz):",
            default=default_spec,
            must_exist=True,
            is_file=True
        )
        config_dict['spec_table_path'] = str(spec_path)
    
    # Spectra directory (optional)
    has_spectra = prompt_yes_no(
        "Do you have local NIRSpec spectra files?",
        default=False
    )
    
    if has_spectra:
        default_spectra = data_dir / 'spectra'
        spectra_path = prompt_path(
            "Spectra directory:",
            default=default_spectra,
            must_exist=True,
            is_file=False
        )
        config_dict['spectra_dir'] = str(spectra_path)
    
    return config_dict


def save_configuration(config_dict: dict):
    """Save configuration to file"""
    print("\n💾 Saving Configuration")
    print("-" * 70)
    
    # Create a Config object with the new settings
    config = Config()
    
    # Update internal config
    config._config.update(config_dict)
    
    # Choose save location
    default_path = Path.home() / '.minerva' / 'config.ini'
    local_path = Path(__file__).parent / 'config.ini'
    
    print("\nWhere would you like to save the configuration?")
    print(f"  1. User directory (recommended): {default_path}")
    print(f"  2. Package directory: {local_path}")
    print("  3. Custom location")
    
    while True:
        choice = input("\nChoice (1-3): ").strip()
        
        if choice == '1':
            output_path = default_path
            break
        elif choice == '2':
            output_path = local_path
            break
        elif choice == '3':
            output_path = Path(input("Enter path: ").strip()).expanduser().resolve()
            break
        else:
            print("Please enter 1, 2, or 3")
    
    # Save
    saved_path = config.save_config(str(output_path))
    
    print(f"\n✓ Configuration saved to: {saved_path}")
    
    # Verify
    print("\n🔍 Verifying configuration...")
    new_config = Config(str(saved_path))
    print(new_config)
    
    return saved_path


def non_interactive_setup(args):
    """Setup configuration from command line arguments"""
    config_dict = {}
    
    if args.data_dir:
        config_dict['data_dir'] = args.data_dir
    
    if args.catalog:
        config_dict['catalog_path'] = args.catalog
    
    if args.sps_catalog:
        config_dict['sps_catalog_path'] = args.sps_catalog
    
    if args.eazy_h5:
        config_dict['eazy_h5_path'] = args.eazy_h5
    
    if args.eazy_zout:
        config_dict['eazy_zout_path'] = args.eazy_zout
    
    if args.spec_table:
        config_dict['spec_table_path'] = args.spec_table
    
    if args.spectra_dir:
        config_dict['spectra_dir'] = args.spectra_dir
    
    return config_dict


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Configure MINERVA Viewer data paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python setup_paths.py
  
  # Non-interactive mode
  python setup_paths.py --data-dir /path/to/data
  
  # Specify all paths
  python setup_paths.py \\
    --data-dir /data/minerva \\
    --catalog /data/minerva/catalogs/MINERVA-UDS_*.fits \\
    --eazy-h5 /data/minerva/EAzY/SFHZ/*.h5
        """
    )
    
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--catalog', help='Main photometric catalog path')
    parser.add_argument('--sps-catalog', help='SPS catalog path')
    parser.add_argument('--eazy-h5', help='EAzY HDF5 file path')
    parser.add_argument('--eazy-zout', help='EAzY zout file path')
    parser.add_argument('--spec-table', help='Spectroscopic catalog path (optional)')
    parser.add_argument('--spectra-dir', help='Local spectra directory (optional)')
    parser.add_argument('--output', '-o', help='Output config file path')
    parser.add_argument('--check', action='store_true', help='Check current configuration')
    
    args = parser.parse_args()
    
    # Check mode
    if args.check:
        print_header()
        config = get_config()
        print(config)
        print("\n" + "=" * 70)
        return 0
    
    # Setup mode
    if any([args.data_dir, args.catalog, args.eazy_h5]):
        # Non-interactive mode
        print("Running in non-interactive mode...")
        config_dict = non_interactive_setup(args)
    else:
        # Interactive mode
        try:
            config_dict = interactive_setup()
        except KeyboardInterrupt:
            print("\n\n❌ Setup cancelled by user.")
            return 1
    
    # Save configuration
    if config_dict:
        try:
            save_configuration(config_dict)
            
            print("\n" + "=" * 70)
            print("✅ Setup complete!")
            print("\nYou can now run:")
            print("  - python minerva_viewer.py")
            print("  - panel serve zgui_panel.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=*")
            print("\nTo reconfigure, run this script again.")
            print("=" * 70)
            
            return 0
        except Exception as e:
            print(f"\n❌ Error saving configuration: {e}")
            return 1
    else:
        print("\n⚠️  No configuration changes made.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
