# Standard library imports
import os
from io import BytesIO

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy import stats

# Astropy imports
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

# EAzY imports
from eazy import filters, utils, param, photoz, hdf5, visualization

# IPython/Jupyter imports
from IPython.display import display, Markdown, Latex

# PIL for image handling
from PIL import Image
import requests


df = Table.read('../data/catalogs/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG.fits').to_pandas()
df.set_index('id', inplace=True)
sps_catalogues = Table.read('../data/catalogs/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG_SPScatalog_0.0.fits').to_pandas()
sps_catalogues.set_index('id', inplace=True)
for f_col in df.filter(like='f_').columns:
    e_col = f_col.replace('f_', 'e_')
    if e_col in df.columns:
        sn_col = f_col.replace('f_', 'sn_')
        df[sn_col] = df[f_col] / df[e_col]
        
df_detected = df[(df.filter(regex='sn_.*m') > 5).sum(axis=1) >= 4] #f098m not covered yet?
df_detected = df_detected.query('n_bands_mb > 4')   
     
print('The stats', 'Full catalog:', len(df), 'Detected sources:', len(df_detected))

### load the msaexp tables

from grizli import utils
from astropy.utils.data import download_file

table_url = "https://s3.amazonaws.com/msaexp-nirspec/extractions/dja_msaexp_emission_lines_v4.4.csv.gz"
tab = utils.read_catalog(download_file(table_url, cache=True), format='csv')

# Column descriptions
columns_url = "https://s3.amazonaws.com/msaexp-nirspec/extractions/dja_msaexp_emission_lines_v4.4.columns.csv"
tab_columns = utils.read_catalog(download_file(columns_url, cache=True), format='csv')

# Set column metadata
for row in tab_columns:
    c = row['column']
    if row['unit'] != '--':
        tab[c].unit = row['unit']
    if row['format'] != '--':
        tab[c].format = row['format']
    if row['description'] != '--':
        tab[c].description = row['description']

RGB_URL = "https://grizli-cutout.herokuapp.com/thumb?size=1.5&scl=2.0&asinh=True&filters=f115w-clear%2Cf277w-clear%2Cf444w-clear&rgb_scl=1.5%2C0.74%2C1.3&pl=2&coord={ra}%2C{dec}"
tab['metafile'] = [m.split('_')[0] for m in tab['msamet']]
SLIT_URL = "https://grizli-cutout.herokuapp.com/thumb?size=1.5&scl=4.0&invert=True&filters=f444w-clear&rgb_scl=1.5%2C0.74%2C1.3&pl=2&coord={ra}%2C{dec}&nirspec=True&dpi_scale=6&nrs_lw=0.5&nrs_alpha=0.8&metafile={metafile}"
FITS_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}"

tab['Thumb'] = [
    "<img src=\"{0}\" height=200px>".format(
        RGB_URL.format(**row['ra','dec'])
    )
    for row in tab
]

tab['Slit_Thumb'] = [
    "<img src=\"{0}\" height=200px>".format(
        SLIT_URL.format(**row['ra','dec','metafile'])
    )
    for row in tab
]

tab['Spectrum_fnu'] = [
    "<img src=\"{0}\" height=200px>".format(
        FITS_URL.format(**row['root','file']).replace('.spec.fits', '.fnu.png')
    )
    for row in tab
]

tab['Spectrum_flam'] = [
    "<img src=\"{0}\" height=200px>".format(
        FITS_URL.format(**row['root','file']).replace('.spec.fits', '.flam.png')
    )
    for row in tab
]


photo_coords = SkyCoord(ra=df['ra'].values*u.degree, dec=df['dec'].values*u.degree)
spec_coords = SkyCoord(ra=tab['ra'], dec=tab['dec'])

idx, d2d, d3d = photo_coords.match_to_catalog_sky(spec_coords)

max_sep = 0.5 * u.arcsec

# Create a boolean mask for sources that are within the search radius
match_mask = d2d < max_sep

num_matches = match_mask.sum()
print(f"Found {num_matches} matches within {max_sep.value} arcseconds.")

if num_matches > 0:
    # Create a new dataframe with only the matched photometric sources
    df_photo_spec_matched = df[match_mask].copy()
    
    # Get the indices from the spectroscopic catalog for the matched sources
    matched_spec_indices = idx[match_mask]
    
    # Add columns from the spectroscopic catalog to the new dataframe
    df_photo_spec_matched['spec_match_sep_arcsec'] = d2d[match_mask].to(u.arcsec).value
    df_photo_spec_matched['spec_z'] = tab['z_best'][matched_spec_indices]
    df_photo_spec_matched['spec_ra'] = tab['ra'][matched_spec_indices]
    df_photo_spec_matched['spec_dec'] = tab['dec'][matched_spec_indices]
    df_photo_spec_matched['file'] = tab['file'][matched_spec_indices]
    
    print("\nHead of the new matched dataframe:")
    display(df_photo_spec_matched.head())
else:
    print("No matches found.")

template_names = ['SFHZ', 'SFHZ_BLUE_AGN', 'LARSON', 'LARSON_MIRI']
zout_dfs = []

for template_name in template_names:

    if template_name=='LARSON_MIRI':
        eazy_file_path = f'../data/EAzY/{template_name}/SUPER/noZPiter'
        zout_fits_table = Table.read(f'{eazy_file_path}/MINERVA-UDS_n2.2_m2.0_v1.0.1_ext_LW_Kf444w_SUPER_CATALOG_wMIRI.eazypy.zout.fits')

    else: 
        eazy_file_path = f'../data/EAzY/{template_name}/SUPER/ZPiter'
        zout_fits_table = Table.read(f'{eazy_file_path}/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_zpiter_CATALOG_{template_name.lower()}.zout.fits')

        print('opening' , f'{eazy_file_path}/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_zpiter_CATALOG_{template_name.lower()}.zout.fits')

    names_1d = [name for name in zout_fits_table.colnames if len(zout_fits_table[name].shape) <= 1]

    minevra_eazy_sfhz_zout = zout_fits_table[names_1d].to_pandas()
    minevra_eazy_sfhz_zout.set_index('id', inplace=True)
    
    # Add suffix and append to list
    zout_dfs.append(minevra_eazy_sfhz_zout.add_suffix(f'_{template_name}'))

# Concatenate all dataframes
merged_zout_df = pd.concat(zout_dfs, axis=1)

merged_zout_df['z_phot_prospector'] = sps_catalogues['z_50']
merged_zout_df['z_phot_prospector16'] = sps_catalogues['z_16']
merged_zout_df['z_phot_prospector84'] = sps_catalogues['z_84']


mask = ((df.n_bands_mb==8) & ((df.sn_f814w>0) |  (df.sn_f775w>0)))


# Filter for galaxies with a positive z_spec
spec_galaxies = merged_zout_df[(merged_zout_df['z_spec_LARSON_MIRI'] > 0) & mask].copy(deep=True)

# Define template names and colors for plotting - add Prospector
template_names =['LARSON_MIRI'] #  ['SFHZ', 'SFHZ_BLUE_AGN', 'LARSON', 'LARSON_MIRI', 'PROSPECTOR']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', 'magenta']  # Blue, Orange, Green, Purple
markers = ['o', '^', 's', 'D', '*']  # Changed to filled markers for better visibility

# Add prospector columns with consistent naming
spec_galaxies['z_phot_PROSPECTOR'] = spec_galaxies['z_phot_prospector']
spec_galaxies['z_spec_PROSPECTOR'] = spec_galaxies['z_spec_LARSON_MIRI']  # Use same z_spec for all


# Function to calculate NMAD (Normalized Median Absolute Deviation)
def calculate_nmad(z_spec, z_phot):
    """Calculate NMAD = 1.48 * median(|Δz|/(1+z_spec))"""
    residuals = np.abs(z_phot - z_spec) / (1 + z_spec)
    return 1.48 * np.nanmedian(residuals)

# Function to calculate outlier fraction
def calculate_outlier_fraction(z_spec, z_phot, threshold=0.15):
    """Calculate fraction of outliers with |Δz|/(1+z_spec) > threshold"""
    residuals = np.abs(z_phot - z_spec) / (1 + z_spec)
    return np.nansum(residuals > threshold) / len(residuals)

# Create the figure with refined layout
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.02)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

# --- Main Plot: z_phot vs z_spec ---

# Plot the one-to-one line
z_range = [0, spec_galaxies['z_spec_LARSON_MIRI'].max() * 1.05]
ax1.plot(z_range, z_range, 'k--', lw=1.5, alpha=0.7, zorder=1)

# Calculate statistics for each template
stats_text = []
for i, template in enumerate(template_names):
    z_phot_col = f'z_phot_{template}'
    z_spec_col = f'z_spec_{template}'
    
    # Calculate statistics
    nmad = calculate_nmad(spec_galaxies[z_spec_col], spec_galaxies[z_phot_col])
    outlier_frac = calculate_outlier_fraction(spec_galaxies[z_spec_col], 
                                              spec_galaxies[z_phot_col])
    n_sources = len(spec_galaxies)
    
    # Plot with reduced alpha and smaller markers for better visibility
    ax1.scatter(spec_galaxies[z_spec_col], spec_galaxies[z_phot_col], 
                alpha=1, s=15, color=colors[i], marker=markers[i], 
                edgecolors='none', rasterized=True,  # Rasterize for smaller file size
                label=f'{template}: σ_NMAD={nmad:.3f}, η={outlier_frac:.2%}')

# ## add connecting lines
# for idx, row in spec_galaxies.iterrows():
#     z_spec = row['z_spec_SFHZ']
#     z_phots = [row[f'z_phot_{t}'] for t in template_names]
#     ax1.plot([z_spec]*len(template_names), z_phots, 'k-', alpha=1, lw=0.3)

# Axes labels and formatting
ax1.set_ylabel(r'$z_{\mathrm{phot}}$', fontsize=14)
ax1.set_xlim(z_range)
ax1.set_ylim(z_range)
ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.tick_params(labelbottom=False)  # Remove x-axis labels from top plot

# Legend with statistics
legend = ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, 
                   title=f'N = {len(spec_galaxies)} galaxies')
legend.get_title().set_fontsize(11)
legend.get_title().set_fontweight('normal')

# --- Residual Plot ---

# Plot horizontal reference lines
ax2.axhline(0, color='k', linestyle='-', lw=1, alpha=0.5)
ax2.axhline(0.15, color='k', linestyle=':', lw=1, alpha=0.3)
ax2.axhline(-0.15, color='k', linestyle=':', lw=1, alpha=0.3)

# Plot the residuals for each template
for i, template in enumerate(template_names):
    z_phot_col = f'z_phot_{template}'
    z_spec_col = f'z_spec_{template}'
    
    # Calculate residual (with sign)
    residual = (spec_galaxies[z_phot_col] - spec_galaxies[z_spec_col]) / (1 + spec_galaxies[z_spec_col])
    
    ax2.scatter(spec_galaxies[z_spec_col], residual, 
                alpha=0.4, s=15, color=colors[i], marker=markers[i],
                edgecolors='none', rasterized=True)

# Add connecting lines for residuals
for idx, row in spec_galaxies.iterrows():
    z_spec = row['z_spec_LARSON_MIRI']
    residuals = []
    for template in template_names:
        z_phot_col = f'z_phot_{template}'
        z_spec_col = f'z_spec_{template}' # In case z_spec differs per template in future
        res = (row[z_phot_col] - row[z_spec_col]) / (1 + row[z_spec_col])
        residuals.append(res)
    ax2.plot([z_spec] * len(residuals), residuals, 'k-', alpha=0.8, lw=0.3)


# Axes labels and formatting
ax2.set_xlabel(r'$z_{\mathrm{spec}}$', fontsize=14)
ax2.set_ylabel(r'$\Delta z/(1+z_{\mathrm{spec}})$', fontsize=14)
ax2.set_xlim(z_range)
ax2.set_ylim([-0.5, 0.5])  # Adjust based on your data
ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', which='major', labelsize=12)

# Add text indicating outlier threshold
ax2.text(0.02, 0.95, '|Δz/(1+z)| = 0.15', transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', style='italic', alpha=0.6)

# Save the figure in high quality
plt.tight_layout()
plt.show()

#####################################################################################
# Print statistics table
print("\nPhotometric Redshift Performance Statistics")
print("=" * 60)
print(f"{'Template':<15} {'N':<8} {'σ_NMAD':<10} {'η (>0.15)':<12} {'<Δz>':<10}")
print("-" * 60)

for template in template_names:
    z_phot_col = f'z_phot_{template}'
    z_spec_col = f'z_spec_{template}'
    
    nmad = calculate_nmad(spec_galaxies[z_spec_col], spec_galaxies[z_phot_col])
    outlier_frac = calculate_outlier_fraction(spec_galaxies[z_spec_col], 
                                              spec_galaxies[z_phot_col])
    bias = np.nanmedian((spec_galaxies[z_phot_col] - spec_galaxies[z_spec_col]) / 
                     (1 + spec_galaxies[z_spec_col]))
    
    print(f"{template:<15} {len(spec_galaxies):<8} {nmad:<10.4f} "
          f"{outlier_frac:<12.3%} {bias:<10.4f}")

