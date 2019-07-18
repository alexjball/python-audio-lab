#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal
import scipy.interpolate
import scipy.stats

def plot_signal_analysis(signal, sample_rate):
    (fig, axes) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    analysis_axis = axes[0]
    signal_axis = axes[1]
    analysis_axis.set_ylim(0, 600)
    #analysis_axis.set_xlim(0, .2)
    
    signal_t = np.arange(signal.size) / sample_rate
    signal_axis.plot(signal_t, signal)
    return (fig, analysis_axis)

def plot_ft_analysis(f, t, ft_analysis, signal, sample_rate):
    """Plots frequency-time analysis data.
    f - (M + 1,) boundary of each frequency bin
    t - (N + 1,) boundary of each time bin
    ft_analysis - (M, N) analysis output
    signal_t - (K,) original time-domain signal time bins
    signal - (K,) original time-domain signal
    """
    (fig, analysis_axis) = plot_signal_analysis(signal, sample_rate)
    
    mesh = analysis_axis.pcolormesh(t, f, ft_analysis)
    fig.colorbar(mesh, ax = analysis_axis, orientation = 'horizontal')
    plt.show() 

def spectrogram(signal, sample_rate):
    (f, t, Sxx) = scipy.signal.spectrogram(signal, sample_rate, nperseg = 8096)
    # t is the right boundary of each bin
    t = np.concatenate(([0], t))
    # f is the left boundary of each bin
    f = np.concatenate((f, [f[-1] + f[-1] - f[-2]]))
    return (f, t, Sxx)

def harmonic_power(power, f_0, f_step, std_dev):
    """Computes the power of the harmonics of a fundamental
    
    This picks out fundamentals.

    power - (M,) power spectrum
    f_0 - starting frequency
    f_step - frequency step size
    std_dev - standard deviation of the window Gaussian in Hz
    """

    window_f = np.concatenate((np.arange(0, -3 * std_dev, -f_step)[-1:0:-1], np.arange(0, 3 * std_dev, f_step)))
    window = scipy.stats.norm.pdf(window_f, loc = 0, scale = std_dev)

    f = f_0 + f_step * np.arange(power.size)
    # Return 0 for frequencies outside the interpolation range, so they
    # don't contribute to the harmonic power.
    power = scipy.interpolate.interp1d(f, power, bounds_error = False, fill_value = 0)

    harmonics = 1 + np.arange(6)
    sampling_f = f[:, np.newaxis, np.newaxis] * harmonics[np.newaxis, np.newaxis, :] + window_f[np.newaxis, :, np.newaxis]
    harmonic_power = power(sampling_f).sum(axis = 2).dot(window[:, np.newaxis]).flatten()

    return (f, harmonic_power)

# Commands

I_FIRST_CMD_ARG = 3

def trim_signal(signal, sample_rate, start_time = None, end_time = None):
    """Trims the signal according to CLI args"""
    start_time = float(sys.argv[I_FIRST_CMD_ARG]) if start_time is None else start_time
    end_time = float(sys.argv[I_FIRST_CMD_ARG + 1]) if end_time is None else end_time

    return signal[int(start_time * sample_rate):int(end_time*sample_rate)]

def plot_autocorrelation(signal, sample_rate):
    plt.plot(scipy.signal.correlate(signal, signal))
    plt.show()

def plot_harmonic_power(signal, sample_rate): 
    """Plot the harmonic power spectrum for an audio snippet.

    start_time - start of analysis window
    end_time - end of analysis window
    """

    (f, Pxx) = scipy.signal.periodogram(signal, sample_rate)
    (f, Pxx) = harmonic_power(Pxx, f[0], f[1] - f[0], float(sys.argv[I_FIRST_CMD_ARG + 2]))
    plt.plot(f[1:], np.log(Pxx[1:]))
    plt.show()

def plot_periodogram_autocorrelation(signal, sample_rate): 
    """Plot the power spectrum for an audio snippet.

    start_time - start of analysis window
    end_time - end of analysis window
    """
    (f, p) = scipy.signal.periodogram(signal, sample_rate)
    f = f[1:]
    p = np.log(p[1:])
    peak_indices, _ = scipy.signal.find_peaks(p, prominence = 1)
    f_peak = f[peak_indices]
    
    p = p[peak_indices[0]:peak_indices[-1]]
    plt.plot(p)
    plt.plot(scipy.signal.correlate(p, p))
    plt.show()

def plot_periodogram(signal, sample_rate): 
    """Plot the power spectrum for an audio snippet.

    start_time - start of analysis window
    end_time - end of analysis window
    """
    (f, p) = scipy.signal.periodogram(signal, sample_rate)
    f = f[1:]
    p = np.log(p[1:])
    plt.plot(f, p)

    peak_indices, _ = scipy.signal.find_peaks(p, prominence = 1)
    f_peak = f[peak_indices]
    plt.plot(f_peak, p[peak_indices], 'x')
    plt.xlim(f_peak[0] * .9, f_peak[-1] * 1.1) 
    plt.show()

def plot_cepstral(signal, sample_rate):
    spectrum = np.log(np.abs(np.fft.fft(signal, n = signal.size)))
    plt.plot(spectrum)
    spectrum = spectrum[0:150]
    ceps = np.abs(np.fft.ifft(spectrum - np.mean(spectrum)))
    plt.plot(spectrum)
    plt.plot(ceps)
    plt.show()

def plot_spectrogram(signal, sample_rate): 
    (f, t, Sxx) = spectrogram(signal, sample_rate)
    plot_ft_analysis(f, t, Sxx, signal, sample_rate)

def plot_log_spectrogram(signal, sample_rate):
    (f, t, Sxx) = spectrogram(signal, sample_rate)
    plot_ft_analysis(f, t, np.log(Sxx), signal, sample_rate)

def plot_cwt(signal, sample_rate):
    widths = np.arange(1, 1000)
    cwt = scipy.signal.cwt(signal, scipy.signal.ricker, widths)
    plt.imshow(cwt, extent = [0, signal.size / sample_rate, widths[0], widths[-1]], cmap='PRGn', aspect='auto', vmax=abs(cwt).max(), vmin=-abs(cwt).max())
    plt.show()

def compute_fundamentals(signal, sample_rate):
    f, p = scipy.signal.periodogram(signal, sample_rate)
    f = f[1:]
    p = np.log(p[1:])

    peaks, properties = scipy.signal.find_peaks(p, prominence = 1)
    f_peaks = f[peaks]

    return peaks, properties, f_peaks



def analyze(signal, sample_rate):
    signal = trim_signal(signal, sample_rate)
    return globals()[sys.argv[2]](signal, sample_rate)

# Usage: analysis.py file analysis_name
# analysis_name corresponds to a function name defined in this file.

(sample_rate, audio) = wavfile.read(sys.argv[1])

analyze(audio, sample_rate)

# Also consider https://docs.scipy.org/doc/scipy-0.16.1/reference/signal.html
# Math: https://github.com/matplotlib/matplotlib/blob/fc8638e25f299a66be90d192adc3c44f533b0907/lib/matplotlib/mlab.py#L525

