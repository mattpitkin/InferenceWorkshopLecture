#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as pl
from matplotlib import gridspec
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

# show a plot of a credible interval

# create a bimodal Gaussian pdf
mus = [1., 4.5]
sigmas = [1., 3.5]
x = np.linspace(-5, 17, 1000)

pdf = np.zeros(len(x))
for mu, sigma in zip(mus, sigmas):
    pdf += (1./(np.sqrt(2.*np.pi) * sigma)) * np.exp(-0.5*(x - mu)**2/sigma**2)

# normalise the pdf
area = np.trapz(pdf, x)
pdf /= area

# calculate the cdf
cdf = cumtrapz(pdf, x, initial=0.)

# calculate the upper and lower credible bounds about the mean
cent = 0.5
credint = 0.9  # 90% interval

intcdf = interp1d(cdf, x)
median = intcdf(cent)
lower = intcdf(cent - credint / 2)
upper = intcdf(cent + credint / 2)

# plot the pdf and cdf and intervals
pl.rc('text', usetex=True)
pl.rc('font', family='serif')
pl.rc('font', size=14)
pl.rc('text.latex', preamble=r'\usepackage{color}')

fig = pl.figure(figsize=(9, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])
ax0 = pl.subplot(gs[0])
ax1 = pl.subplot(gs[1])

ax0.plot(x, pdf)
ax0.set_xlim([x[0], x[-1]])
ax0.set_ylim([0., 1.05 * np.max(pdf)])
ax1.plot(x, cdf)
ax1.set_xlim([x[0], x[-1]])
ax1.set_ylim([0., 1.])

idxs = (x >= lower) & (x <= upper)

ax0.axvline(lower, color='k', ls='--')
ax0.axvline(upper, color='k', ls='--')
ax0.axvline(median, color='k', ls='-', linewidth=0.5)
ax0.fill_between(x[idxs], 0, pdf[idxs], color='b', alpha=0.2)

ax1.axvline(lower, color='k', ls='--')
ax1.axvline(upper, color='k', ls='--')
ax1.axhline(cent - credint / 2, color='k', ls='--')
ax1.axhline(cent + credint / 2, color='k', ls='--')
ax1.axhline(cent, color='k', ls='-', linewidth=0.5)

ax1.arrow(1, cent, dx=0., dy=0.95*(credint / 2), head_width=0.2, head_length=0.02, linewidth=2, color='k', length_includes_head=True)
ax1.arrow(1, cent, dx=0., dy=-0.95*(credint / 2), head_width=0.2, head_length=0.02, linewidth=2, color='k', length_includes_head=True)
ax1.text(2, cent, 'symmetric 90\%', rotation=270, horizontalalignment='center', verticalalignment='center')

#mid = 0.5 * 1.05 * np.max(pdf)
#ax0.text(lower - 0.75, mid, 'lower bound = {0:.2f}'.format(lower), rotation=90, horizontalalignment='center', verticalalignment='center')
#ax0.text(upper + 0.75, mid, 'upper bound = {0:.2f}'.format(upper), rotation=270, horizontalalignment='center', verticalalignment='center')

ax0.set_ylabel('PDF')
ax0.set_xlabel('$x$')
ax1.set_ylabel('CDF')
ax1.set_xlabel('$x$')

# find minimal interval
low = 0.
high = credint
minrange = [0, np.inf]
intrange = [low, high]
dcdf = 0.001
while 1:
    lowermin = intcdf(low)
    uppermin = intcdf(high)
    if (uppermin - lowermin) < np.diff(minrange)[0]:
        minrange = [lowermin, uppermin]
        intrange = [low, high]

    if high >= 1.:
        break

    low += dcdf
    high += dcdf

idxs = (x >= minrange[0]) & (x <= minrange[1])

ax0.axvline(minrange[0], color='m', ls='--')
ax0.axvline(minrange[1], color='m', ls='--')
ax0.fill_between(x[idxs], 0, pdf[idxs], color='r', alpha=0.1, hatch='//')

ax1.axvline(minrange[0], color='m', ls='--')
ax1.axvline(minrange[1], color='m', ls='--')
ax1.axhline(intrange[0], color='m', ls='--')
ax1.axhline(intrange[1], color='m', ls='--')

ax1.arrow(11, np.mean(intrange), dx=0., dy=0.95*(np.diff(intrange)[0] / 2), head_width=0.2, head_length=0.02, linewidth=2, color='m', length_includes_head=True)
ax1.arrow(11, np.mean(intrange), dx=0., dy=-0.95*(np.diff(intrange)[0] / 2), head_width=0.2, head_length=0.02, linewidth=2, color='m', length_includes_head=True)
ax1.text(12, np.mean(intrange), 'minimal 90\%', rotation=270, horizontalalignment='center', verticalalignment='center')

# put median with intervals in title
ax0.text(10.5, 0.21, '$x = {0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'.format(median, (upper - median), (median - lower)))
ax0.text(10.5, 0.19, '$x = {0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'.format(median, (minrange[1] - median), (median - minrange[0])), fontdict={'color': 'magenta'})

gs.tight_layout(fig)
fig.savefig('../credint.pdf')
