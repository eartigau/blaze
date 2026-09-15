# blaze

A physical model of the blaze function of an échelle spectrograph, fitted to a
flat-lamp exposure. Instead of smoothing the observed flat, the code builds the
blaze out of the pieces that actually produce it (lamp spectrum, instrument
transmission, grating diffraction, detector sampling) and fits the few free
parameters of that description.

Nothing is hard-coded for a particular instrument. The diffraction orders are
read off the wavelength solution, and the grating constants are fitted. The
example data shipped here is SPIRou (CFHT), reduced with APERO.

---

## The model

For diffraction order `m` and pixel `i`:

```
model(m, i) = BB_photon(lambda, Teff) x Trans(lambda) x sinc^2(blaze) x dlambda/dpixel
```

| term | what it is | why it matters |
|---|---|---|
| `BB_photon` | blackbody of the calibration lamp, in **photon** density: `1 / (lambda^4 (exp(hc/lambda.k.T) - 1))` | the detector counts photons, not energy. Using `B_lambda` instead shifts the peak of the lamp spectrum by hundreds of nm |
| `Trans` | `exp(P(lambda))`, i.e. `log(transmission)` is a polynomial. Its constant term carries the global amplitude | filters, coatings, fibre and detector QE. It is by far the largest term, a factor of ~20 across the SPIRou range |
| `sinc^2` | single-groove diffraction envelope of the grating | the blaze proper, the only term that knows about `m` |
| `dlambda/dpixel` | width of a pixel in wavelength | the wavelength solution is not linear, so this varies by ~28% along a SPIRou order and tilts every one of them |

### The grating envelope

The natural variable of the blaze function is the distance to the blaze peak
counted **in orders**,

```
x = m (lambda - lambda_blaze) / lambda_blaze      with    m . lambda_blaze = C
```

`C = 2 d sin(theta_b) cos(gamma)` is the grating spacing as the beam sees it:
the optical path difference between two adjacent grooves at the blaze peak. It
is the same number for every order, which is what makes a single fit possible.
The code uses the un-linearised form of the argument, which is slightly
asymmetric in wavelength and brings in the blaze angle `theta_b`.

`beta` scales the width of the envelope. `beta = 1` means a fully illuminated
groove, i.e. the first zeros of the `sinc^2` fall exactly one free spectral
range away from the peak.

### Free parameters

Three non-linear parameters, `C`, `beta` and `theta_b`, plus the coefficients of
the transmission polynomial. Because `log(transmission)` enters linearly, it is
solved exactly by least squares at every step, so the non-linear solver only
ever works in three dimensions. The fit runs in seconds on a 49 x 4088 array and
is followed by two passes of sigma clipping.

---

## Finding the diffraction orders

`mkblaze.py` works out the real diffraction order of each spectral order, which
is normally not stored in extracted data. Two independent methods, which agree:

1. **Order overlap**, wavelength solution only, no blaze needed. At a given
   pixel all orders share the same diffraction angle, so `m.lambda` is the same
   for all of them and `m = lambda / |dlambda/dorder|` directly.
2. **Flatness of `m.lambda_blaze`**. Scan the integer offset and keep the value
   that makes the product across orders the flattest.

On the included data both give orders **79 (bluest) down to 31 (reddest)**, the
published SPIRou numbering. The flatness criterion is unambiguous: relative
scatter `1.8e-3` for the best offset against `4.0e-3` for the runner-up.

`fit_blaze.py` has this built in as `diffraction_orders()`. It handles orders
stored either way round (blue to red or red to blue) and a subset of the array.

---

## Quick start

```bash
pip install numpy scipy astropy matplotlib
python mkblaze.py                 # identify the diffraction orders
python fit_blaze.py               # fit the model
python fit_blaze.py myblaze.fits mywave.fits    # any other instrument
```

Both files are 2D arrays of the same shape, `(n_orders, n_pixels)`: the
extracted flat-lamp spectrum and its wavelength solution. NaNs are ignored.

## What comes out

| file | content |
|---|---|
| `blaze_model.fits` | the model spectrum, same shape as the input, fitted parameters in the header (`MODCST`, `MODBETA`, `MODTHETA`, `MODTR*`, ...) |
| `blaze_model_debug.pdf` | 4 pages: observed vs model over the whole range, the model components separated, a panel per order, and the peak-drift check |
| `blaze_orders.pdf` | the order identification from `mkblaze.py` |

The model is evaluated on **every pixel of every order**, including where the
pipeline threw the blaze away, so it can be used to extend or replace a
thresholded blaze.

---

## Tuning

Everything lives in the constants at the top of `fit_blaze.py`:

| constant | default | meaning |
|---|---|---|
| `TEFF` | 5000 | blackbody temperature of the calibration lamp, in K |
| `NPOLY` | 11 | order of the `log(transmission)` polynomial |
| `WAVE_FIT_MAX` | 2500 | red limit of the **fitted** range. The model is still evaluated beyond it, as an extrapolation |
| `SIGMA_CLIP` | 5 | rejection threshold, in units of the residual rms |
| `THETA_B0` | 45 deg | starting blaze angle. It is fitted, this is only a neutral start |

`NPOLY` is the knob that matters most, see the limitations below.

## Results on the included SPIRou data

```
diffraction orders 79 to 31
C = 76742.3 nm from the blaze peaks, 76850.2 nm after the fit (+0.141%)
blaze width beta   = 0.8495
blaze angle        = 65.26 deg (R2.2 grating)
groove spacing     = 23.64 grooves/mm
obs/model - 1      : median 3.06%, rms 5.69%
```

Two sanity checks worth pointing at:

* The fitted groove spacing, **23.64 grooves/mm**, lands within 2% of the 23.2
  grooves/mm of the real SPIRou grating, which was never given to the code.
* `m.lambda_peak` is not quite constant across the array, it drifts by ~800 nm.
  That drift is not a failure of the constant `C`, it is the slope of the lamp
  spectrum pulling the observed maximum off the true blaze peak. A model with a
  strictly constant `C` reproduces it, see page 4 of the debug PDF.

## Limitations

* **The transmission polynomial is the limiting term**, not the grating physics.
  Real transmission has structure a polynomial cannot follow (filter edges,
  coatings). On this data the residual goes 9.5% rms at `NPOLY = 4`, 8.8% at 6,
  5.7% at 11. A free multiplicative constant per order gets to ~4%, which is
  roughly the floor set by the `sinc^2` shape itself.
* **The blue and red ends are the worst**, because the blocking filters cut far
  more sharply than any polynomial. `WAVE_FIT_MAX` exists to keep the red cut-off
  from dragging the whole fit with it.
* **Beyond the fitted range the polynomial is deliberately left free**, not
  clamped, so that the model stays smooth. A high-order polynomial extrapolates
  violently: check the model before using it outside the fitted range.
* `theta_b` is **weakly constrained**. It only controls a subtle asymmetry of the
  envelope, and the residual is nearly flat between 25 and 70 degrees. Do not
  read the fitted value as a measurement of the grating geometry.
* APERO thresholds its blaze at 25% of the peak, so only the top of the `sinc^2`
  is ever constrained on this data.

## Structure

`fit_blaze.py` is split in two by a banner comment. Everything above it is the
model and the fit, and depends only on numpy, scipy and astropy. Everything
below is the debug plotting, in a single `debug_plots()` function that imports
matplotlib itself. The first half can be lifted into a pipeline as is.

## Requirements

Python 3, numpy, scipy, astropy. matplotlib for the debug plots only.
