#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

(r, data) = wavfile.read(sys.argv[1])

(fig, axes) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
spec = axes[0]
td = axes[1]

(spectrum, freqs, t, im) = spec.specgram(data, NFFT = 8096, Fs = r)

spec.set_ylim(0, 600)
spec.set_xlim(0, .2)
fig.colorbar(im, ax = spec, orientation = 'horizontal')

td.plot(np.arange(data.size) / r, data)

plt.show()
