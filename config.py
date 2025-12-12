"""
MINERVA Viewer Configuration Module

This module handles all configuration for the MINERVA viewer applications.
Paths can be configured via:
1. Environment variables (highest priority)
2. config.ini file in the package directory
3. User's home directory ~/.minerva/config.ini
4. Default values (lowest priority)

Environment Variables:
    MINERVA_DATA_DIR: Base data directory
    MINERVA_CATALOG_PATH: Main photometric catalog
    MINERVA_SPS_CATALOG_PATH: SPS catalog
    MINERVA_EAZY_H5_PATH: EAzY HDF5 file
    MINERVA_EAZY_ZOUT_PATH: EAzY zout FITS file
    MINERVA_SPEC_TABLE: Spectroscopic catalog (local)
    MINERVA_SPECTRA_DIR: Directory with local spectra
    MINERVA_SPEC_TIMEOUT: Timeout for downloading spec catalog

Usage:
    from config import Config
    
    config = Config()
    catalog_path = config.catalog_path
    
    # Or get all paths
    paths = config.get_all_paths()
"""

import os
import sys
from pathlib import Path
from configparser import ConfigParser
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


class Config:
    """
    Configuration manager for MINERVA viewer applications.
    
    Attributes
    ----------
    data_dir : Path
        Base data directory
    catalog_path : Path
        Main photometric catalog path
    sps_catalog_path : Path
        SPS catalog path
    eazy_h5_path : Path
        EAzY HDF5 file path
    eazy_zout_path : Path
        EAzY zout FITS file path
    spec_table_path : Path
        Spectroscopic table path (optional)
    spectra_dir : Path
        Local spectra directory (optional)
    templates_dir : Path
        EAzY templates directory
    spec_table_url : str
        URL for downloading spectroscopic catalog
    spec_table_timeout : int
        Timeout for downloading spec catalog
    filters : str
        NIRCam filters for cutouts
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'data_dir': '../data',
        'catalog_filename': 'MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG.fits',
        'sps_catalog_filename': 'MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG_SPScatalog_0.0.fits',
        'eazy_h5_filename': 'MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_zpiter_CATALOG.sfhz.eazypy.h5',
        'eazy_zout_filename': 'MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_zpiter_CATALOG_sfhz.zout.fits',
        'spec_table_filename': 'dja_msaexp_emission_lines_v4.4.csv.gz',
        'eazy_subdir': 'EAzY/SFHZ/SUPER/ZPiter',
        'catalogs_subdir': 'catalogs',
        'spectra_subdir': 'spectra',
        'spec_table_url': 'https://s3.amazonaws.com/msaexp-nirspec/extractions/dja_msaexp_emission_lines_v4.4.csv.gz',
        'spec_table_timeout': '120',
        'filters': 'f090w-clear,f115w-clear,f140m-clear,f150w-clear,f182m-clear,f200w-clear,f210m-clear,f250w-clear,f277w-clear,f300m-clear,f356w-clear,f360m-clear,f410m-clear,f444w-clear,f460m-clear',
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Parameters
        ----------
        config_file : str, optional
            Path to custom config file. If None, searches default locations.
        """
        self._config = self._load_config(config_file)
        self._validate_paths()
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, str]:
        """Load configuration from multiple sources with priority order"""
        config = self.DEFAULT_CONFIG.copy()
        
        # 1. Load from config file if exists
        config_path = self._find_config_file(config_file)
        if config_path:
            logger.info(f"Loading configuration from {config_path}")
            config.update(self._read_config_file(config_path))
        
        # 2. Override with environment variables (highest priority)
        config.update(self._read_env_vars())
        
        return config
    
    def _find_config_file(self, config_file: Optional[str] = None) -> Optional[Path]:
        """Find config file in standard locations"""
        if config_file:
            path = Path(config_file)
            if path.exists():
                return path
            logger.warning(f"Specified config file not found: {config_file}")
        
        # Search order: package dir, user home dir
        search_paths = [
            Path(__file__).parent / 'config.ini',
            Path.home() / '.minerva' / 'config.ini',
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _read_config_file(self, config_path: Path) -> Dict[str, str]:
        """Read configuration from INI file"""
        parser = ConfigParser()
        try:
            parser.read(config_path)
            if 'paths' in parser:
                return dict(parser['paths'])
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
        return {}
    
    def _read_env_vars(self) -> Dict[str, str]:
        """Read configuration from environment variables"""
        env_mapping = {
            'MINERVA_DATA_DIR': 'data_dir',
            'MINERVA_CATALOG_PATH': 'catalog_path',
            'MINERVA_SPS_CATALOG_PATH': 'sps_catalog_path',
            'MINERVA_EAZY_H5_PATH': 'eazy_h5_path',
            'MINERVA_EAZY_ZOUT_PATH': 'eazy_zout_path',
            'MINERVA_SPEC_TABLE': 'spec_table_path',
            'MINERVA_SPECTRA_DIR': 'spectra_dir',
            'MINERVA_SPEC_TIMEOUT': 'spec_table_timeout',
            'MINERVA_FILTERS': 'filters',
        }
        
        config = {}
        for env_var, config_key in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                config[config_key] = value
                logger.debug(f"Using environment variable {env_var}")
        
        return config
    
    def _validate_paths(self):
        """Validate that required paths exist or can be created"""
        # Check if data directory exists
        if not self.data_dir.exists():
            logger.warning(
                f"Data directory not found: {self.data_dir}\n"
                f"Please run 'python setup_paths.py' to configure paths."
            )
    
    @property
    def data_dir(self) -> Path:
        """Base data directory"""
        return Path(self._config['data_dir']).expanduser().resolve()
    
    @property
    def catalog_path(self) -> Path:
        """Main photometric catalog path"""
        if 'catalog_path' in self._config:
            return Path(self._config['catalog_path']).expanduser().resolve()
        return self.data_dir / self._config['catalogs_subdir'] / self._config['catalog_filename']
    
    @property
    def sps_catalog_path(self) -> Path:
        """SPS catalog path"""
        if 'sps_catalog_path' in self._config:
            return Path(self._config['sps_catalog_path']).expanduser().resolve()
        return self.data_dir / self._config['catalogs_subdir'] / self._config['sps_catalog_filename']
    
    @property
    def eazy_h5_path(self) -> Path:
        """EAzY HDF5 file path"""
        if 'eazy_h5_path' in self._config:
            return Path(self._config['eazy_h5_path']).expanduser().resolve()
        return self.data_dir / self._config['eazy_subdir'] / self._config['eazy_h5_filename']
    
    @property
    def eazy_zout_path(self) -> Path:
        """EAzY zout FITS file path"""
        if 'eazy_zout_path' in self._config:
            return Path(self._config['eazy_zout_path']).expanduser().resolve()
        return self.data_dir / self._config['eazy_subdir'] / self._config['eazy_zout_filename']
    
    @property
    def spec_table_path(self) -> Path:
        """Spectroscopic table path (optional)"""
        if 'spec_table_path' in self._config:
            return Path(self._config['spec_table_path']).expanduser().resolve()
        return self.data_dir / self._config['catalogs_subdir'] / self._config['spec_table_filename']
    
    @property
    def spectra_dir(self) -> Path:
        """Local spectra directory (optional)"""
        if 'spectra_dir' in self._config:
            return Path(self._config['spectra_dir']).expanduser().resolve()
        return self.data_dir / self._config['spectra_subdir']
    
    @property
    def templates_dir(self) -> Path:
        """EAzY templates directory"""
        return Path(__file__).parent / 'templates'
    
    @property
    def spec_table_url(self) -> str:
        """URL for downloading spectroscopic catalog"""
        return self._config['spec_table_url']
    
    @property
    def spec_table_timeout(self) -> int:
        """Timeout for downloading spec catalog"""
        return int(self._config['spec_table_timeout'])
    
    @property
    def filters(self) -> str:
        """NIRCam filters for cutouts"""
        return self._config['filters']
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Get all configured paths as a dictionary"""
        return {
            'data_dir': self.data_dir,
            'catalog_path': self.catalog_path,
            'sps_catalog_path': self.sps_catalog_path,
            'eazy_h5_path': self.eazy_h5_path,
            'eazy_zout_path': self.eazy_zout_path,
            'spec_table_path': self.spec_table_path,
            'spectra_dir': self.spectra_dir,
            'templates_dir': self.templates_dir,
        }
    
    def check_paths_exist(self) -> Dict[str, bool]:
        """Check which paths exist"""
        return {
            name: path.exists()
            for name, path in self.get_all_paths().items()
        }
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to INI file.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save config. If None, saves to ~/.minerva/config.ini
        """
        if output_path is None:
            output_path = Path.home() / '.minerva' / 'config.ini'
        else:
            output_path = Path(output_path)
        
        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        parser = ConfigParser()
        parser['paths'] = self._config
        
        with open(output_path, 'w') as f:
            parser.write(f)
        
        logger.info(f"Configuration saved to {output_path}")
        return output_path
    
    def __repr__(self):
        """String representation"""
        paths = self.get_all_paths()
        exists = self.check_paths_exist()
        
        lines = ["MINERVA Viewer Configuration:"]
        for name, path in paths.items():
            status = "✓" if exists[name] else "✗"
            lines.append(f"  {status} {name}: {path}")
        
        return "\n".join(lines)


# Global configuration instance (singleton pattern)
_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get the global configuration instance.
    
    Parameters
    ----------
    reload : bool
        If True, reload configuration from disk
    
    Returns
    -------
    Config
        Global configuration instance
    """
    global _config_instance
    
    if _config_instance is None or reload:
        _config_instance = Config()
    
    return _config_instance


def set_config(config: Config):
    """
    Set the global configuration instance.
    
    Parameters
    ----------
    config : Config
        Configuration instance to use globally
    """
    global _config_instance
    _config_instance = config


if __name__ == '__main__':
    # Print configuration when run as script
    config = Config()
    print(config)
    print("\nPath existence:")
    for name, exists in config.check_paths_exist().items():
        print(f"  {name}: {'EXISTS' if exists else 'NOT FOUND'}")
