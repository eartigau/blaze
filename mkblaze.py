from astropy.io import fits
import matplotlib.pyplot as plt 
import numpy as np

flat = fits.getdata('FDFCD50E5Ff_pp_blaze_AB.fits')
wave = fits.getdata('FEF450D367a_pp_e2dsff_AB_wave_night_AB.fits')

index = np.arange(wave.shape[0])
wmax = wave[index,np.nanargmax(flat,axis=1)]

# The index above is just the position of the order in the e2ds file, not the
# physical diffraction order. For an echelle grating every order is blazed at
# the same angle, so m * lambda_blaze = 2*d*sin(theta_b) is the same constant
# for all of them. The e2ds runs blue to red, so the diffraction order goes
# down by one per index step: m = m_first - index. m_first is an integer, so we
# scan it and keep the value that makes m * wmax the flattest.
trial = np.arange(index.size + 1, 200)
cst = trial[:, None] - index[None, :]
cst = cst * wmax[None, :]
scatter = np.std(cst, axis=1) / np.mean(cst, axis=1)

m_first = trial[np.argmin(scatter)]
morder = m_first - index

# Independent check that does not use the blaze at all. At a given pixel all
# orders see the same diffraction angle, so m*lambda is constant there too and
# m_k = lambda_(k+1) / (lambda_(k+1) - lambda_k) straight from the overlap of
# consecutive orders.
lam = wave[:, wave.shape[1] // 2]
m_overlap = lam[1:] / (lam[1:] - lam[:-1])
m_overlap = np.append(m_overlap, m_overlap[-1] - 1)

print('first order = {}, last order = {}'.format(morder[0], morder[-1]))
print('relative scatter of m*lambda_blaze : {:.2e} (runner-up {:.2e})'.format(
    np.min(scatter), np.sort(scatter)[1]))
print('2*d*sin(theta_b) = {:.1f} nm'.format(np.mean(morder * wmax)))
print('')
print(' idx   m   lambda_blaze   m*lambda   m from order overlap')
for i in index:
    print('{:4d} {:4d} {:12.2f} {:11.1f} {:14.2f}'.format(
        i, morder[i], wmax[i], morder[i] * wmax[i], m_overlap[i]))

fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for i in index:
    ax[0].plot(wave[i], flat[i], alpha=.5, lw=.5)
ax[0].plot(wmax, np.nanmax(flat, axis=1), 'ko', ms=3, label='blaze peak')
for i in index[::4]:
    ax[0].annotate(str(morder[i]), (wmax[i], np.nanmax(flat[i])),
                   textcoords='offset points', xytext=(0, 5), ha='center', fontsize=7)
ax[0].set(ylabel='blaze', title='diffraction order {} to {}'.format(morder[0], morder[-1]))
ax[0].legend(loc='lower right')
ax[1].plot(wmax, morder * wmax, 'o-', ms=3)
ax[1].set(xlabel='wavelength [nm]', ylabel=r'$m\,\lambda_{blaze}$ [nm]')
plt.tight_layout()
fig.savefig('blaze_orders.pdf')
plt.show()
