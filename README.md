# blaze

A physical model of the blaze function of an échelle spectrograph, fitted to a
flat-lamp exposure. Instead of smoothing the observed flat, the code builds the
blaze out of the pieces that actually produce it (lamp spectrum, instrument
transmission, grating diffraction, detector sampling) and fits the few free
parameters of that description.

Nothing is hard-coded for a particular instrument. The diffraction orders are
read off the wavelength solution, and the grating constants are fitted. The
example data shipped here is SPIRou (CFHT), reduced with APERO.

![observed blaze and fitted model](docs/fit.png)

*49 orders of a flat lamp, the model on top of them, and what is left over. One
set of parameters for the whole array: no per-order normalisation, no smoothing.*

---

## The model

For diffraction order `m` and pixel `i`:

```
model(m, i) = BB_photon(lambda, Teff) x Trans(lambda) x sinc^2(blaze) x dlambda/dpixel
```

| term | what it is | why it matters |
|---|---|---|
| `BB_photon` | blackbody of the calibration lamp, in **photon** density: `1 / (lambda^4 (exp(hc/lambda.k.T) - 1))` | the detector counts photons, not energy. Using `B_lambda` instead shifts the peak of the lamp spectrum by hundreds of nm |
| `Trans` | `log(transmission)` as a linear spline through the peaks of the orders (default), or as a polynomial. It carries the global amplitude | filters, coatings, fibre and detector QE. It is by far the largest term, a factor of ~30 across the SPIRou range |
| `sinc^2` | single-groove diffraction envelope of the grating | the blaze proper, the only term that knows about `m` |
| `dlambda/dpixel` | width of a pixel in wavelength | the wavelength solution is not linear, so this varies by ~28% along a SPIRou order and tilts every one of them |

![the four components of the model](docs/components.png)

*The four terms, on the same wavelength axis. Only the third one knows about the
diffraction order. Colour runs from the bluest order to the reddest.*

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

### The transmission

Each order contributes one spline knot, placed at its observed peak. The knot is
the median, over +-50 pixels around the peak, of

```
log(observed) - log(BB_photon x sinc^2 x dlambda/dpixel)
```

that is, the observed peak with the other three terms taken out first. They have
to come out: the peak flux also carries the lamp, the blaze and the pixel width,
and would count them twice otherwise. The knots are joined by straight lines in
`log(transmission)`, and the end segments carry on beyond the first and last
knot.

A polynomial is still available (`TRANSMISSION = 'poly'`). The spline is the
default because it is better where it counts, see [Spline or
polynomial](#spline-or-polynomial) below.

### Free parameters

Three non-linear parameters, `C`, `beta` and `theta_b`. The transmission never
goes through the non-linear solver: the spline knots are read straight off the
data for the current grating parameters, and a polynomial would be solved by
linear least squares, since `log(transmission)` enters linearly. The fit takes
about a second on a 49 x 4088 array and is followed by two passes of sigma
clipping.

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

![order identification and peak drift](docs/orders.png)

*Left: the scatter of `m.lambda` against the trial order number, more than a
decade deep at the right answer. Right: `m.lambda_peak` is not constant, and a
model with a strictly constant `C` reproduces that drift, because it comes from
the slope of the lamp spectrum pulling the observed maximum off the true blaze
peak.*

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

The figures in this README are built separately by `docs/make_figures.py`, which
rebuilds the model from the header of `blaze_model.fits` rather than re-fitting,
and checks the two agree before plotting.

The model is evaluated on **every pixel of every order**, including where the
pipeline threw the blaze away, so it can be used to extend or replace a
thresholded blaze.

---

## Tuning

Everything lives in the constants at the top of `fit_blaze.py`:

| constant | default | meaning |
|---|---|---|
| `TEFF` | 5000 | blackbody temperature of the calibration lamp, in K |
| `TRANSMISSION` | `'spline'` | model of `log(transmission)`, `'spline'` or `'poly'` |
| `SPLINE_K` | 1 | degree of the spline, 1 is linear between order peaks |
| `PEAK_HALF_WIDTH` | 50 | half width, in pixels, of the window each knot is the median of |
| `NPOLY` | 21 | order of the polynomial, only with `TRANSMISSION = 'poly'` |
| `WAVE_FIT_MAX` | 2500 | red limit of the **fitted** range. The model is still evaluated beyond it, as an extrapolation |
| `SIGMA_CLIP` | 5 | rejection threshold, in units of the residual rms |
| `THETA_B0` | 45 deg | starting blaze angle. It is fitted, this is only a neutral start |

With the polynomial, `NPOLY` is the knob that matters most. The knots are stored
in a `TRANS_KNOTS` table extension of `blaze_model.fits`, so the model can be
rebuilt from that file and the wavelength solution, without re-fitting.

## Results on the included SPIRou data

```
diffraction orders 79 to 31
C = 76742.3 nm from the blaze peaks, 76847.2 nm after the fit (+0.137%)
blaze width beta   = 0.8546
blaze angle        = 63.39 deg (R2.0 grating)
groove spacing     = 23.27 grooves/mm
obs/model - 1      : median 2.36%, rms 6.67%, median per-order rms 4.48%
```

![three orders close up](docs/orders_zoom.png)

*Three orders at full scale. The asymmetry of the observed profile is real and
the un-linearised `sinc^2` follows most of it.*

Two sanity checks worth pointing at:

* **The grating comes out right.** SPIRou uses an R2 échelle (tan(theta_b) = 2,
  63.43 deg) ruled at 23.2 grooves/mm. The fit gives 63.39 deg and 23.27
  grooves/mm, within 0.3%, and neither number was ever given to the code.
* `m.lambda_peak` is not quite constant across the array, it drifts by ~800 nm.
  That drift is not a failure of the constant `C`, it is the slope of the lamp
  spectrum pulling the observed maximum off the true blaze peak. A model with a
  strictly constant `C` reproduces it, see page 4 of the debug PDF.

### A second instrument

The same script, unchanged, on a NIRPS flat (75 orders, 966 to 1954 nm, with the
gap between J and H):

```
diffraction orders 149 to 75
blaze angle        = 76.09 deg (R4.0 grating)
groove spacing     = 13.38 grooves/mm
obs/model - 1      : median 1.67%, median per-order rms 3.10%
```

NIRPS uses an R4 échelle with a published blaze angle of 76 deg
([Bach Research contract](https://www.laserfocusworld.com/test-measurement/spectroscopy/article/16569072/nirps-consortium-awards-bach-research-echelle-grating-contract-for-exoplanet-research)),
again recovered without being told. The NIRPS files are not in this repository.

### Spline or polynomial

Same data, same settings, only the transmission model changes:

| transmission | median `obs/model - 1` | rms | `theta_b` | grooves/mm |
|---|---|---|---|---|
| **SPIRou**, polynomial, order 11 | 3.06% | 5.69% | 65.26 deg | 23.64 |
| polynomial, order 21 | 2.70% | 5.13% | 65.38 deg | 23.66 |
| **linear spline** | **2.36%** | 6.67% | **63.39 deg** | **23.27** |
| cubic spline | 2.39% | 6.41% | 63.10 deg | 23.21 |
| **NIRPS**, polynomial, order 11 | 4.96% | 24.2% | 64.37 deg | 12.43 |
| polynomial, order 21 | 4.51% | 26.2% | 75.90 deg | 13.37 |
| **linear spline** | **1.67%** | 27.2% | **76.09 deg** | 13.38 |
| cubic spline | 1.68% | 27.7% | 75.71 deg | 13.36 |

The spline wins on the typical pixel, by a factor of three on NIRPS, and it is
the one that gets the grating right on SPIRou: a polynomial stiff enough to
miss the structure of the transmission leaves tilts inside the orders, and the
blaze angle absorbs them. The polynomial keeps a lower global rms on SPIRou,
because it spreads its error more evenly into the wings, where the spline only
answers to the peak.

The decisive argument is outside the fitted range. With `WAVE_FIT_MAX = 2400` on
SPIRou, the linear spline is off by a median 31% on the pixels it did not see,
the cubic one by 12%, and the order 21 polynomial by 2e14%. The linear and
cubic splines differ by 0.15% (median) inside the fitted range; the linear one
is the default because it cannot overshoot. Its price is a small break of slope
at every knot: carried over one free spectral range, the change of slope is worth
1.6% in flux in a typical order, 6% at worst.

## Limitations

* **The global rms is set by a handful of orders** where the transmission
  changes within a single free spectral range, which one knot per order cannot
  follow: the K-band cut-off on SPIRou (orders 31 to 38), the red end (75 to 78)
  and the J/H gap (104, 105) on NIRPS. Leave those six orders out and both
  instruments sit at 4.6% rms. The script prints the median of the per-order
  rms, which is the fairer summary.
* **Inside an order, the limit is the `sinc^2` shape itself**, about 4 to 5%
  rms. The spline pins every peak, so what is left is the fall-off towards the
  edges of the order.
* **Beyond the fitted range the transmission is deliberately left free**, not
  clamped, so that the model stays smooth. The spline carries on with its end
  segments; a high-order polynomial extrapolates violently. Check the model
  before using it outside the fitted range.
* `theta_b` is **only loosely constrained on the low side**. With the spline, the
  minimum sits on the true value for both instruments (63.4 deg on SPIRou, 76 deg
  on NIRPS), but at 45 deg the cost is only 3% (SPIRou) to 6% (NIRPS) higher,
  while it climbs steeply above the true value. Treat the fitted angle as a
  consistency check, not a measurement.
* APERO thresholds its blaze at 25% of the peak, so only the top of the `sinc^2`
  is ever constrained on this data.

## Structure

`fit_blaze.py` is split in two by a banner comment. Everything above it is the
model and the fit, and depends only on numpy, scipy and astropy. Everything
below is the debug plotting, in a single `debug_plots()` function that imports
matplotlib itself. The first half can be lifted into a pipeline as is.

## Requirements

Python 3, numpy, scipy, astropy. matplotlib for the debug plots only.
