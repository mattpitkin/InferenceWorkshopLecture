#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as pl
from matplotlib import gridspec
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

# show a plot of a credible interval

# create a bimodal Gaussian pdf
mus = [1., 5.]
sigmas = [1., 2.]
x = np.linspace(-5, 15, 1000)

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

lower = interp1d(cdf, x)(cent - credint / 2)
upper = interp1d(cdf, x)(cent + credint / 2)

# plot the pdf and cdf and intervals
pl.rc('text', usetex=True)
pl.rc('font', family='serif')
pl.rc('font', size=14)

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
ax0.fill_between(x[idxs], 0, pdf[idxs], color='b', alpha=0.2)

ax1.axvline(lower, color='k', ls='--')
ax1.axvline(upper, color='k', ls='--')
ax1.axhline(cent - credint / 2, color='k', ls='--')
ax1.axhline(cent + credint / 2, color='k', ls='--')
ax1.axhline(cent, color='k', ls='-', linewidth=0.5)

ax1.arrow(10, cent, dx=0., dy=0.95*(credint / 2), head_width=0.2, head_length=0.02, linewidth=2, color='k', length_includes_head=True)
ax1.arrow(10, cent, dx=0., dy=-0.95*(credint / 2), head_width=0.2, head_length=0.02, linewidth=2, color='k', length_includes_head=True)
ax1.text(11, cent, '90\%', rotation=270, horizontalalignment='center', verticalalignment='center')

mid = 0.5 * 1.05 * np.max(pdf)
ax0.text(lower - 0.75, mid, 'lower bound = {0:.2f}'.format(lower), rotation=90, horizontalalignment='center', verticalalignment='center')
ax0.text(upper + 0.75, mid, 'upper bound = {0:.2f}'.format(upper), rotation=270, horizontalalignment='center', verticalalignment='center')

ax0.set_ylabel('PDF')
ax0.set_xlabel('$x$')
ax1.set_ylabel('CDF')
ax1.set_xlabel('$x$')

gs.tight_layout(fig)
fig.savefig('../credint.pdf')
