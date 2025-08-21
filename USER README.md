
# skirt-fits-photometry

Tools and documentation for performing region-based photometry on SKIRT simulation FITS cubes and comparing model fluxes to observed data. Includes an interactive Python script to extract fluxes in circular, annular, or rectangular regions, compute chi-square and correlation metrics, and produce plots and summaries.

## Introduction

This repository contains `general_fits_photometry.py`, a Python script designed to analyze FITS cubes produced by SKIRT dust radiative transfer simulations. Each slice of the cube corresponds to a specific wavelength in the simulation grid. Users can compare model fluxes extracted from a user-defined region (circle, annulus, rectangle) with observed fluxes at corresponding wavelengths, calculating chi-square, Pearson correlation, and Spearman correlation metrics.

## Features

- Interactive prompts for uploading a FITS cube and observed data file.
- Supports circular, annular, and rectangular photometric apertures.
- Automatic unit conversion for wavelength (Å to µm) and optional custom calibration constants.
- Computes chi-square, Pearson’s r, and Spearman’s ρ between model and observed fluxes.
- Generates diagnostic plots for model vs. observed flux and residuals.
- Detailed usage guide for SKIRT users.

## Requirements

- Python 3.6+
- astropy (for FITS handling)
- numpy
- matplotlib

Install these via pip:

```bash
pip install numpy astropy matplotlib
```

(Optional: SciPy for Spearman correlation; the script falls back to a built-in implementation if SciPy is absent.)

## Observed Data Format

Provide a text file containing at least three whitespace-separated columns:

1. Wavelength (Å)
2. Observed flux (e.g. photon cm⁻² s⁻¹ Å⁻¹ sr⁻¹)
3. 1‑σ error in the flux

Example:

```
1004    6.54E+04    0.22E+04
1056    1.06E+05    0.16E+04
...
```

Ensure the wavelengths match the wavelengths in SKIRT’s `sed.dat` . Use the first column in `sed.dat` to find the slice index for each wavelength.

## Usage

After completing your SKIRT run and preparing the observed data:

1. Run the script:

   ```bash
   python general_fits_photometry.py
   ```

2. Follow the interactive prompts:

   - Enter the path to the SKIRT FITS cube (e.g., `MyModel_i00_total.fits`).
   - Enter the path to your observed data file.
   - Choose a photometric region (circle, annulus, or rectangle) and provide its parameters.
   - Confirm or enter calibration constants.
   - Enter each observed wavelength and its corresponding cube slice index (found in `sed.dat`).

3. The script will compute model fluxes, compare them to your observed fluxes, output summary statistics, and generate plots.

## Repository Structure

- `general_fits_photometry.py` — main analysis script.
- `USER README.md` — this document.
