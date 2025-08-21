#!/usr/bin/env python3
"""
general_fits_photometry.py
---------------------------

This script performs aperture or annular photometry on a three‑dimensional FITS
data cube and compares the resulting model fluxes with an observational data
set.  It prompts the user for two input files (a FITS cube and an observation
table) and for the geometry of the region of interest.  Supported shapes
include circular apertures, annuli (rings) and rectangles.  After summing the
model flux within the chosen region for each requested spectral plane, the
script applies a user‑supplied calibration to convert counts into physical
units.  It then computes goodness‑of‑fit statistics (χ², Pearson and Spearman
correlations) between the model and observational fluxes, prints a summary
table and displays two diagnostic plots.

All of the key parameters are collected interactively, so the script can be
reused with different files and geometries without modification.  Comments
throughout the code explain the purpose of every step.  To run the script,
execute it with Python 3 and follow the prompts.
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def annulus_mask(ny: int, nx: int, cx: float, cy: float, r_in_pix: float, r_out_pix: float) -> np.ndarray:
    """Return a boolean mask selecting pixels within an annulus.

    The annulus is defined by an inner radius ``r_in_pix`` and an outer radius
    ``r_out_pix`` (both in pixel units) centred on (``cx``, ``cy``).  Pixels
    whose centre lies between these radii (inclusive on the outer edge) are
    marked as ``True``; all others are ``False``.  The function uses a vectorized
    approach via ``numpy.ogrid`` for efficiency.

    Parameters
    ----------
    ny, nx : int
        Dimensions of the image (number of rows ``ny`` and columns ``nx``).
    cx, cy : float
        The x and y coordinates (in pixel units) of the annulus centre.  Note
        that these coordinates use NumPy's 0‑based indexing convention.
    r_in_pix, r_out_pix : float
        Inner and outer radii of the annulus in pixel units.

    Returns
    -------
    mask : ndarray of bool
        A two‑dimensional boolean array of shape ``(ny, nx)`` where ``True``
        indicates pixels inside the annulus and ``False`` elsewhere.
    """
    # Create a coordinate grid of pixel indices.  ``Y`` and ``X`` are arrays of
    # shape (ny, 1) and (1, nx), respectively, so operations broadcast
    # efficiently across the image.
    Y, X = np.ogrid[:ny, :nx]
    # Compute radial distance of every pixel centre from the given centre.
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    # Select pixels whose radius falls between the inner and outer bounds.
    return (r >= r_in_pix) & (r <= r_out_pix)


def circular_mask(ny: int, nx: int, cx: float, cy: float, radius_pix: float) -> np.ndarray:
    """Return a boolean mask selecting pixels inside a circular aperture.

    Parameters are as in :func:`annulus_mask`, except that only a single
    ``radius_pix`` is required.  The circle is centred on (``cx``, ``cy``) and
    includes all pixels whose centre lies within the specified radius.
    """
    Y, X = np.ogrid[:ny, :nx]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return r <= radius_pix


def rectangular_mask(ny: int, nx: int, cx: float, cy: float, width_pix: float, height_pix: float) -> np.ndarray:
    """Return a boolean mask selecting pixels inside a rectangle.

    The rectangle is axis‑aligned (not rotated) and centred on (``cx``, ``cy``).
    Pixels are included if their x coordinate lies within half the width and
    their y coordinate lies within half the height from the centre.
    """
    Y, X = np.ogrid[:ny, :nx]
    return (np.abs(X - cx) <= width_pix / 2.0) & (np.abs(Y - cy) <= height_pix / 2.0)


def chi_squared(observed: np.ndarray, model: np.ndarray, errors: np.ndarray) -> tuple:
    """Compute the χ² statistic, degrees of freedom and reduced χ².

    The χ² statistic measures the squared deviation of the model from the
    observations, weighted by the observational uncertainties.  The degrees of
    freedom are (N − 1) for N data points.  The reduced χ² is χ² divided by the
    degrees of freedom.  A value close to unity indicates an acceptable fit
    within the stated errors.

    Returns
    -------
    chi2 : float
        The χ² statistic.
    dof : int
        Number of degrees of freedom (length of ``observed`` minus one).
    reduced : float
        The reduced χ² (χ² divided by the degrees of freedom).  If the
        degrees of freedom is zero the function returns ``np.nan``.
    """
    # Residuals between observed and model fluxes
    residuals = observed - model
    # Sum over squared residuals normalized by variance
    chi2 = np.sum((residuals) ** 2 / (errors ** 2))
    # Degrees of freedom: number of points minus one
    dof = len(observed) - 1
    # Reduced χ²: χ² divided by DOF; guard against division by zero
    reduced = chi2 / dof if dof != 0 else np.nan
    return chi2, dof, reduced


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient between two arrays.

    Pearson's r measures linear correlation on a scale from −1 (perfect
    anticorrelation) to +1 (perfect correlation).  If either array has fewer
    than two elements the function returns ``np.nan``.
    """
    if len(x) < 2:
        return float('nan')
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    # np.corrcoef returns the correlation matrix; [0,1] extracts r
    return np.corrcoef(x, y)[0, 1]


def rankdata(a: np.ndarray) -> np.ndarray:
    """Assign ranks to data, dealing gracefully with ties.

    Each element of ``a`` is replaced by its rank in ascending order, starting
    from 1.  Tied values are assigned the average of their ranks, which is the
    standard behaviour of Spearman's rank correlation.  This helper function
    avoids the need for SciPy if it is unavailable.
    """
    a = np.asarray(a, dtype=float)
    # argsort returns the indices that would sort the array
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    # Initial ranking: position in sorted order plus one
    ranks[order] = np.arange(len(a), dtype=float) + 1.0
    # Identify unique values and their inverse mapping
    uniques, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    # Sum ranks for each unique value
    sums = np.bincount(inverse, weights=ranks)
    # Average rank for each unique value
    average_ranks = sums / counts
    # Map the average ranks back to the original positions
    return average_ranks[inverse]


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman's rank correlation coefficient between two arrays.

    Spearman's rho is Pearson's correlation applied to the ranks of the data.
    It measures monotonic relationships irrespective of linearity.  If SciPy
    is available, the function uses ``scipy.stats.spearmanr``; otherwise it
    falls back on a simple rank implementation using :func:`rankdata`.
    """
    try:
        # Import inside the function to avoid a hard dependency on SciPy
        from scipy.stats import spearmanr  # type: ignore
        rho, _ = spearmanr(x, y)
        return rho
    except Exception:
        # If SciPy is not available, compute ranks manually and then Pearson
        rx = rankdata(np.asarray(x, dtype=float))
        ry = rankdata(np.asarray(y, dtype=float))
        return pearson_correlation(rx, ry)


def prompt_float(message: str) -> float:
    """Prompt the user for a floating‑point number with validation.

    The function repeatedly asks the user until a valid float is entered.
    Comments in the code emphasise clarity and reuse.
    """
    while True:
        try:
            return float(input(message).strip())
        except ValueError:
            print("Please enter a valid number.")


def main() -> None:
    """Main entry point for interactive photometry comparison.

    This function orchestrates user interaction, file loading, region mask
    construction, flux computation, calibration, statistical analysis and
    plotting.  It avoids hard‑coded constants by prompting the user for
    everything necessary to repeat the analysis on different data sets.
    """
    print("\nGeneralized FITS photometry comparison tool")
    print("This script will compare model fluxes extracted from a FITS cube with observational data.\n")

    # Ask the user for the path to the model FITS cube.  ``strip()`` removes
    # leading/trailing whitespace to ensure clean paths.
    model_path = input("Enter path to model FITS file: ").strip()
    # Ask for the path to the observational data file.  The observational file
    # should contain at least three columns: wavelength, observed flux and
    # 1‑sigma error.
    obs_path = input("Enter path to observation file: ").strip()

    # Load the observational data table.  ``np.loadtxt`` reads a whitespace
    # separated text file into a NumPy array.  If the file has a single row,
    # reshape it into a 2D array with one row to unify indexing later.
    obs_data = np.loadtxt(obs_path)
    if obs_data.ndim == 1:
        obs_data = obs_data.reshape(1, -1)
    # Basic validation: require at least three columns (wavelength, flux, error).
    if obs_data.shape[1] < 3:
        raise ValueError("Observation file must have at least three columns: wavelength, flux and error.")
    # Split the columns into wavelength, observed flux and uncertainties.
    wavelength_obs = obs_data[:, 0]
    obs_flux = obs_data[:, 1]
    obs_err = obs_data[:, 2]

    # Ask the user for the units of the wavelength column.  These units will
    # determine how we convert the wavelengths into microns for the model
    # conversion below.  Provide explicit options to reduce ambiguity.
    print("\nWhat are the units of the wavelength column?")
    print("1) microns (µm)")
    print("2) Angstroms (Å)")
    # Accept either 1 or 2; default to microns if input is invalid
    w_unit_choice = input("Enter 1 or 2: ").strip()
    if w_unit_choice == '2':
        # Convert Angstroms to microns (1 µm = 10 000 Å)
        wavelength_micron = wavelength_obs / 1.0e4
    else:
        wavelength_micron = wavelength_obs.astype(float)

    # Request the z‑plane (spectral index) corresponding to each observation.  We
    # ask the user to provide a comma‑separated list of integers; any extra or
    # missing indices relative to the number of observations will be truncated.
    print("\nEnter the z-plane indices (0-based) corresponding to each observation, separated by commas.")
    zbin_input = input("Indices: ").strip()
    # Convert the comma‑separated string into a list of integers.  ``split`` and
    # ``strip`` remove whitespace around each element.
    z_bins = [int(s.strip()) for s in zbin_input.split(',') if s.strip() != '']

    # Warn if the number of z‑bins does not match the number of observations.
    if len(z_bins) != len(wavelength_micron):
        print(f"Warning: You provided {len(z_bins)} z indices for {len(wavelength_micron)} observations. "
              f"Only the minimum of the two lengths will be used.")

    # Open the FITS cube and read the data and header.  ``memmap=True`` tells
    # astropy to memory‑map the file instead of loading it all at once, which
    # can be efficient for large cubes.  We assume the data cube is stored
    # in the primary HDU (index 0).
    with fits.open(model_path, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    # Ensure that the FITS contains a 3D cube (spectral, y, x).  If not,
    # raise an error so the user knows the file is incompatible.
    if data.ndim != 3:
        raise ValueError("Model FITS file does not contain a 3D data cube.")
    # Record the dimensions: nz = number of spectral planes, ny = rows, nx = columns
    nz, ny, nx = data.shape
    # Pixel scale (arcsec per pixel).  Astropy stores this as the absolute
    # value of CDELT1; we take the magnitude because the sign encodes axis
    # orientation.
    cdelt = float(abs(header.get('CDELT1')))
    # Reference pixel coordinates (1‑based in FITS).  Subtract 1 to convert
    # to 0‑based indices for NumPy indexing.
    crpix1 = float(header.get('CRPIX1')) - 1.0
    crpix2 = float(header.get('CRPIX2')) - 1.0

    # Ask the user to choose the geometry of the region of interest.  The
    # geometry determines which mask function will be used and what parameters
    # will be requested.
    print("\nSelect region geometry:")
    print("1) Circular aperture")
    print("2) Annulus (ring)")
    print("3) Rectangle")
    shape_choice = input("Enter 1, 2 or 3: ").strip()

    # The region centre is specified in arcseconds relative to the FITS
    # reference coordinates.  We convert these offsets to pixel units using
    # the pixel scale and add them to the FITS reference pixel coordinates.
    center_x_arcsec = prompt_float("Enter centre X coordinate (arcsec): ")
    center_y_arcsec = prompt_float("Enter centre Y coordinate (arcsec): ")
    # Convert arcsec offsets to pixel coordinates.  Adding to ``crpix*``
    # converts relative offsets into absolute pixel positions.  Because FITS
    # stores CRPIX as 1‑based, we already subtracted 1 above.
    cx = crpix1 + (center_x_arcsec / cdelt)
    cy = crpix2 + (center_y_arcsec / cdelt)

    # Depending on the user's selection, prompt for the appropriate shape
    # parameters (radii for circles/annuli, width/height for rectangles) and
    # convert them from arcsec to pixel units.  Then create the mask.
    if shape_choice == '1':
        radius_as = prompt_float("Enter radius of circle (arcsec): ")
        radius_pix = radius_as / cdelt
        mask = circular_mask(ny, nx, cx, cy, radius_pix)
    elif shape_choice == '2':
        r_in_as = prompt_float("Enter inner radius of annulus (arcsec): ")
        r_out_as = prompt_float("Enter outer radius of annulus (arcsec): ")
        r_in_pix = r_in_as / cdelt
        r_out_pix = r_out_as / cdelt
        mask = annulus_mask(ny, nx, cx, cy, r_in_pix, r_out_pix)
    elif shape_choice == '3':
        width_as = prompt_float("Enter rectangle width (arcsec): ")
        height_as = prompt_float("Enter rectangle height (arcsec): ")
        width_pix = width_as / cdelt
        height_pix = height_as / cdelt
        mask = rectangular_mask(ny, nx, cx, cy, width_pix, height_pix)
    else:
        raise ValueError("Invalid shape selection; please choose 1, 2 or 3.")

    # Compute the sum of model counts within the selected region for each
    # requested spectral plane.  If fewer z indices than observations were
    # provided, only that many planes will be processed; if more, the list
    # will be truncated to the number of observations.  We use ``np.nansum``
    # to safely ignore any NaN values in the data.
    ann_sums = []
    # Determine how many comparisons to make.  This avoids index errors if
    # there is a mismatch between the number of observations and provided z
    # indices.
    ncompare = min(len(z_bins), len(obs_flux))
    for i in range(ncompare):
        z = z_bins[i]
        # Validate the z index to ensure it falls within the data cube
        if z < 0 or z >= nz:
            raise IndexError(f"Z index {z} outside valid range 0..{nz - 1}")
        # Convert the selected plane to float64 to reduce accumulation error
        plane = np.asarray(data[z], dtype=np.float64)
        # Sum all pixels inside the mask; NaNs are ignored
        ann_sums.append(np.nansum(plane[mask], dtype=np.float64))
    ann_sums = np.array(ann_sums, dtype=np.float64)

    # Prompt the user for calibration constants.  These constants convert
    # raw summed counts into physical flux units.  The script offers the
    # default values from the original example or allows the user to specify
    # their own.  The factor ``K2`` is an intermediate normalization and
    # ``factor`` combines the remaining constants.  See the original code for
    # the derivation of these constants.
    use_default = input("\nUse default calibration constants? (y/n): ").strip().lower()
    if use_default.startswith('y'):
        divisor = 40441.662
        numerator = 5.0341125e14
        denom1 = 2.3504431e-11
        denom2 = 1e4
    else:
        divisor = prompt_float("Enter divisor constant: ")
        numerator = prompt_float("Enter numerator constant: ")
        denom1 = prompt_float("Enter denominator constant 1: ")
        denom2 = prompt_float("Enter denominator constant 2: ")

    # Apply the calibration to convert counts into flux units.  ``K2``
    # normalizes the summed counts by the divisor; ``factor`` accounts for
    # instrument throughput and unit conversion.  Multiplying by the
    # wavelength in microns (converted earlier) is specific to the original
    # example; you may need to adjust this term if your calibration differs.
    K2 = ann_sums / divisor
    factor = numerator / (denom1 * denom2)
    model_flux = factor * wavelength_micron[:ncompare] * K2

    # Trim observational arrays to the number of comparisons.  This ensures
    # consistent shapes for statistical computations.
    obs_flux_trim = obs_flux[:ncompare]
    obs_err_trim = obs_err[:ncompare]
    wavelength_trim = wavelength_micron[:ncompare]

    # Compute χ² and degrees of freedom.  We use the trimmed arrays to
    # guarantee matching lengths.  Reduced χ² gives a goodness‑of‑fit measure.
    chi2, dof, red_chi2 = chi_squared(obs_flux_trim, model_flux, obs_err_trim)
    # Compute Pearson's linear correlation coefficient
    pearson_r = pearson_correlation(obs_flux_trim, model_flux)
    # Compute Spearman's rank correlation coefficient
    spearman_rho = spearman_correlation(obs_flux_trim, model_flux)

    # Print a formatted table of the comparison.  Each row lists the index,
    # wavelength in microns, observed flux, model flux, observational error and
    # the residual (obs − model).  Scientific notation is used for clarity.
    print("\nComparison Table:")
    header = "Index   Wavelength[µm]     Observed       Model         Error         Residual"
    print(header)
    print("-" * len(header))
    for i in range(ncompare):
        resid = obs_flux_trim[i] - model_flux[i]
        print(f"{i:>5d}   {wavelength_trim[i]:>14.6f}   "
              f"{obs_flux_trim[i]:>12.6e}  {model_flux[i]:>12.6e}  "
              f"{obs_err_trim[i]:>12.6e}  {resid:>12.6e}")

    # Print the summary statistics with descriptive labels
    print("\nStatistics:")
    print(f"Chi-square         : {chi2:.6f}")
    print(f"Degrees of freedom : {dof}")
    print(f"Reduced chi-square : {red_chi2:.6f}")
    print(f"Pearson r          : {pearson_r:.6f}")
    print(f"Spearman rho       : {spearman_rho:.6f}\n")

    # Plot 1: Observed and model flux versus wavelength.  Error bars show
    # observational uncertainties.  We do not specify colours explicitly to
    # allow matplotlib to choose sensible defaults.
    plt.figure()
    plt.errorbar(wavelength_trim, obs_flux_trim, yerr=obs_err_trim, fmt='o', label='Observed', capsize=3)
    plt.plot(wavelength_trim, model_flux, 's--', label='Model')
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Flux')
    plt.title('Observed and Model Flux versus Wavelength')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Scatter plot of model versus observed flux with a 1:1 reference
    # line.  This visualizes the correlation; points on the dashed line
    # represent perfect agreement.  We again avoid specifying colours.
    plt.figure()
    min_val = min(np.min(obs_flux_trim), np.min(model_flux))
    max_val = max(np.max(obs_flux_trim), np.max(model_flux))
    plt.plot([min_val, max_val], [min_val, max_val], '--', label='1:1 line')
    plt.scatter(model_flux, obs_flux_trim, marker='o', label='Data points')
    plt.xlabel('Model Flux')
    plt.ylabel('Observed Flux')
    plt.title('Observed versus Model Flux Correlation')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Execute the main function only when the script is run directly.  This
    # allows the functions to be imported without side effects.
    main()