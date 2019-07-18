import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact
import logging
import librosa
import librosa.display
import time
import scipy.signal
import os

import audio_lab.jupyter_utils as jupyter_utils
from audio_lab.io import *

logger = logging.getLogger(__name__)


class SystemMonitor(AudioSink):
    """Monitors input/output system data"""

    def __init__(self):

        # Input signal, callers should add this as a sink
        self.system_input = AudioStream()
        system_input_sink = AudioSink()
        system_input_sink.write = self.write_system_input
        self.system_input.add_sink(system_input_sink)

        # The input channel (data column) to use
        self.input_channel_index = 0

        # Output signal, callers should add this as a sink
        self.system_output = AudioStream()
        system_output_sink = AudioSink()
        system_output_sink.write = self.write_system_output
        self.system_output.add_sink(system_output_sink)

        # The input channel (data column) to use
        self.output_channel_index = 0

        # Latency in frames between the system input and output signals.
        self.system_latency_frames = 0

        self.reset()

    def write_system_input(self, data):
        self.input_signal.append(data[:, self.input_channel_index])

    def write_system_output(self, data):
        self.output_signal.append(data[:, self.output_channel_index])

    def write_available(self):
        return AudioConstants.unlimited_frames

    def reset(self):
        self.input_signal = [np.zeros((0, ))]
        self.output_signal = [np.zeros((0, ))]

    def get_aligned_signals(self):
        """Returns (input_signal, output_signal) arrays, aligned and trimmed
        so that each signal is the same size and has the same time base."""
        # time = index
        in_signal = np.concatenate(self.input_signal)
        # time = index + self.system_latency_frames
        out_signal = np.concatenate(self.output_signal)
        start_time = self.system_latency_frames
        end_time = min(in_signal.size,
                       out_signal.size + self.system_latency_frames)
        return (in_signal[start_time:end_time],
                out_signal[0:end_time - self.system_latency_frames])


class SystemPlotter:
    def __init__(self,
                 signal_fn,
                 n_fft=2048,
                 sampling_rate=AudioConstants.sampling_rate,
                 update_interval=1,
                 label="Plot System"):
        """"Creates a new plotter and associated figure.

        signal_fn: () -> (in_data, out_data), in_data.shape == out_data.shape
        """

        self.signal_fn = signal_fn
        self.n_fft = 2048
        self.sampling_rate = sampling_rate
        self.update_interval = update_interval
        self._init_ui(label)

    def _init_ui(self, label):
        fig, (ax_sig, ax_freq, ax_H) = jupyter_utils.start_figure(label,
                                                                  nrows=3,
                                                                  ncols=1,
                                                                  figsize=(8,
                                                                           5))

        ax_H.set_ylim((-20, 20))
        ax_H.set_xlim((10, 1e4))
        ax_H.set_xscale('log')
        ax_H.set_xlabel('Hz')
        ax_H.set_ylabel('H (dB)')
        ax_H.xaxis.set_major_formatter(librosa.display.LogHzFormatter())
        ax_H.set_title('Transfer Function')

        ax_sig.set_ylim((-2, 2))
        ax_sig.set_xlim((0, 1.1 * self.update_interval))
        ax_sig.set_title('Signal t domain')

        ax_freq.set_xlim((10, 1e4))
        ax_freq.set_xscale('log')
        ax_freq.set_xlabel('Hz')
        ax_freq.set_ylabel('dB')
        ax_freq.xaxis.set_major_formatter(librosa.display.LogHzFormatter())
        ax_freq.set_title('Signal f domain')

        (self.fig, self.ax_sig, self.ax_freq,
         self.ax_H) = fig, ax_sig, ax_freq, ax_H
        self.animation = None
        self.lines = []

    def _update_ui(self, *args):
        try:
            (in_data, out_data) = self.signal_fn()

            logger.info(f'{in_data.shape} {out_data.shape}')

            if not in_data.size or not out_data.size:
                return

            [line.remove() for line in self.lines]
            self.lines = []

            to_db = lambda x: librosa.amplitude_to_db(abs(x))

            out_fft = librosa.stft(out_data, n_fft=self.n_fft)
            in_fft = librosa.stft(in_data, n_fft=self.n_fft)
            t = np.arange(in_data.size) / self.sampling_rate
            f = librosa.fft_frequencies(self.sampling_rate, self.n_fft)

            self.lines.extend(self.ax_sig.plot(t, in_data, 'r-'))
            self.lines.extend(self.ax_sig.plot(t, out_data, 'b-'))

            X = to_db(in_fft).mean(axis=1)
            Y = to_db(out_fft).mean(axis=1)
            self.lines.extend(self.ax_freq.plot(f, X, 'r-'))
            self.lines.extend(self.ax_freq.plot(f, Y, 'b-'))

            # Plot the transfer function
            H = Y - X
            self.lines.extend(self.ax_H.plot(f, H, 'r-'))
        except:
            logger.error('Error in _update_ui', exc_info=True)

    def start(self):
        if self.animation is not None:
            return

        self.animation = FuncAnimation(self.fig,
                                       self._update_ui,
                                       interval=self.update_interval * 1e3)
        self.fig.canvas.draw_idle()

    def stop(self):
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None


def freqs_logspace(start_freq=30, end_freq=15e3, n_freqs=100):
    """np.linspace but logarithmic"""
    return start_freq * (end_freq / start_freq)**(np.arange(n_freqs) /
                                                  (n_freqs - 1))


def get_freq_alignment(sr, n_fft, freqs):
    """Shifts input frequencies to align with the nearest fft bins.
    
    This hopefully improves the accuracy of measurements without having to integrate energy.
    """
    fft_freqs = librosa.fft_frequencies(sr, n_fft)
    # index of the first element in fft_freqs larger than freqs[i], or N if none exist
    least_upper_bound = np.searchsorted(fft_freqs, freqs)

    aligned_freqs = np.zeros(freqs.shape)
    aligned_fft_freq_idx = np.zeros(freqs.shape, dtype=int)

    # Compute whether each frequency is closer to its least upper bound or greatest lower bound
    is_min_value = least_upper_bound == 0
    is_max_value = least_upper_bound == fft_freqs.size
    is_center_value = np.logical_and(~is_min_value, ~is_max_value)

    center_lub_idx = least_upper_bound[is_center_value]
    center_lub = fft_freqs[center_lub_idx]
    center_glb = fft_freqs[center_lub_idx - 1]
    is_glb = np.round((center_lub - freqs[is_center_value]) /
                      (center_lub - center_glb)).astype(int)

    aligned_fft_freq_idx[is_center_value] = center_lub_idx - is_glb
    aligned_fft_freq_idx[is_min_value] = 0
    aligned_fft_freq_idx[is_max_value] = fft_freqs.size - 1

    aligned_freqs = fft_freqs[aligned_fft_freq_idx]
    return aligned_freqs, fft_freqs, aligned_fft_freq_idx


def align_freqs(sr, n_fft, freqs):
    return get_freq_alignment(sr, n_fft, freqs)[0]


def generate_1_f_signal(freqs=None, sr=22050, duration=20, gain=None):
    if freqs is None:
        freqs = freqs_logspace()
    if gain is None:
        gain = freqs[0]
    length = sr * duration
    signal = np.zeros(length)
    for freq in freqs:
        signal += gain / freq * librosa.tone(freq, sr=sr, length=length)

    # Remove any DC bias
    signal -= signal.mean()
    # Normalize peak to +- 1
    signal *= 1 / max(signal)
    return signal


class FrequencyResponsePlotter:
    def __init__(self,
                 buffer,
                 channel_index,
                 n_fft,
                 sr,
                 analysis_signal_duration,
                 label="Plot Frequency Response"):

        self.buffer = buffer
        self.channel_index = channel_index
        self.n_fft = n_fft
        self.sr = sr
        self.analysis_signal_duration = analysis_signal_duration
        self.fig, self.ax = jupyter_utils.start_figure(label, figsize=(8, 5))
        self.animation = None
        self.lines = []

    def _update_ui(self, *args):
        try:
            frames = self.buffer.read(self.buffer.read_available())
            frames = frames[:min(frames.
                                 shape[0], self.analysis_signal_duration *
                                 self.sr), self.channel_index]

            if not frames.size:
                return

            [line.remove() for line in self.lines]
            self.lines = plot_frequency_response(frames,
                                                 sr=self.sr,
                                                 n_fft=self.n_fft,
                                                 ax=self.ax)

        except:
            logger.error('Error in _update_ui', exc_info=True)

    def start(self):
        if self.animation is not None:
            return

        self.animation = FuncAnimation(self.fig,
                                       self._update_ui,
                                       interval=.5e3 +
                                       self.analysis_signal_duration * 1e3)
        self.fig.canvas.draw_idle()

    def stop(self):
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None


def plot_frequency_response(signal,
                            sr=22050,
                            n_fft=4096,
                            ylim=(-20, 2),
                            ax=None):
    """Plots the power spectrum of the given signal"""
    ax = plt.gca() if ax is None else ax
    freqs, Pxx = scipy.signal.welch(signal, fs=sr, nperseg=n_fft)
    # Power relative to peak response. This allows comparing linear systems.
    power = librosa.power_to_db(Pxx, max(Pxx))
    #     peaks, _ = scipy.signal.find_peaks(db_fft, prominence=20)
    #     peak_freqs, peak_values = fft_freqs[peaks], db_fft[peaks]
    #     cs = scipy.interpolate.CubicSpline(peak_freqs, peak_values, bc_type='natural')

    ax.set_ylim(ylim)
    ax.set_xlim((10, 1e4))
    ax.set_xscale('log')
    ax.set_xlabel('Hz')
    ax.set_ylabel('dB (10dB = 10x Power, 3.16 Amplitude)')
    ax.set_title(f'Frequency response, {n_fft} bins')
    ax.xaxis.set_major_formatter(librosa.display.LogHzFormatter())
    return ax.plot(freqs, power, 'b-')


#     ax.plot(fft_freqs, cs(fft_freqs), 'r-')


def measure_pickup_frequency_response(test_signal_channel_index=0,
                                      response_signal_channel_index=0,
                                      num_channels=AudioConstants.num_channels,
                                      frames_per_buffer=4096,
                                      n_fft=4096,
                                      sr=22050,
                                      signal_duration=1):
    """Measures the transfer function of an electric guitar using a test signal."""
    soundcard = Soundcard(frames_per_buffer=frames_per_buffer,
                          sampling_rate=sr,
                          channels=num_channels)
    assert soundcard.sampling_rate == sr

    response_buffer = AudioBuffer(num_channels=num_channels,
                                  num_frames=sr * signal_duration * 3)
    soundcard.output.add_sink(response_buffer)

    # Use an audio transform to provide system input.
    test_freqs = align_freqs(sr, n_fft, freqs_logspace(n_freqs=100))
    test_signal = generate_1_f_signal(test_freqs,
                                      sr=sr,
                                      duration=max(15, signal_duration * 1.2))
    test_signal_generator = AudioTransform()
    trigger = AudioTransform.Input(test_signal_generator)
    soundcard.output.add_sink(trigger)

    i = 0

    def write_test_signal():
        nonlocal i
        n_frames, n_channels = trigger.take().shape
        frames = np.zeros((n_frames, n_channels))
        frames[:, test_signal_channel_index] = test_signal[np.arange(
            i, i + n_frames) % test_signal.size]
        i = (i + n_frames) % test_signal.size
        soundcard.input.write(frames)

    test_signal_generator.apply = write_test_signal

    run_button = widgets.Button(description="Start")
    save_path = widgets.Text(value='.',
                             description='save dir:',
                             disabled=False)
    save_name = widgets.Text(value='pickup-frequency-response',
                             description='save filename:',
                             disabled=False)
    save_button = widgets.Button(description="Save Plot")
    debug_view = widgets.Output(layout={'border': '1px solid white'})
    plotter = FrequencyResponsePlotter(response_buffer,
                                       response_signal_channel_index, n_fft,
                                       sr, signal_duration,
                                       'Pickup Frequency Response')

    @debug_view.capture(clear_output=True)
    def measure(b):
        if soundcard.is_running():
            logger.info("Stopping")
            plotter.stop()
            soundcard.stop()

            # Next action is start
            run_button.description = "Start"
        elif soundcard.is_idle():
            logger.info("Starting")
            plotter.start()
            soundcard.start()

            # Next action is stop
            run_button.description = "Stop"

    @debug_view.capture(clear_output=True)
    def save_plot(b):
        # Set the current figure
        plt.figure(plotter.fig.number)
        plt.gcf().suptitle(save_name.value)
        plt.savefig(os.path.join(save_path.value, save_name.value + '.png'))

    run_button.on_click(measure)
    save_button.on_click(save_plot)

    display(run_button)
    display(save_path)
    display(save_name)
    display(save_button)
    jupyter_utils.get_cell_logger().show_logs()
    display(debug_view)

    return soundcard


def plot_transfer_function(input_channel_index,
                           output_channel_index,
                           num_channels=AudioConstants.num_channels,
                           frames_per_buffer=4096,
                           n_fft=4096,
                           update_interval=1):
    soundcard = Soundcard(frames_per_buffer=frames_per_buffer,
                          channels=num_channels)

    monitor = SystemMonitor()
    monitor.input_channel_index = input_channel_index
    monitor.output_channel_index = output_channel_index
    monitor.system_latency_frames = frames_per_buffer

    soundcard.output.add_sink(monitor.system_input)
    soundcard.output.add_sink(monitor.system_output)

    # Play back the input, passing it through a gain node.
    playback_volume = Gain()
    soundcard.output.add_sink(playback_volume.input)
    playback_volume.output.add_sink(soundcard.input)

    def update_volume(level):
        playback_volume.level = level

    def take_monitor_signal():
        out = monitor.get_aligned_signals()
        monitor.reset()
        return out

    run_button = widgets.Button(description="Start")
    debug_view = widgets.Output(layout={'border': '1px solid white'})
    plotter = SystemPlotter(take_monitor_signal,
                            n_fft=n_fft,
                            sampling_rate=soundcard.sampling_rate,
                            update_interval=update_interval)

    @debug_view.capture(clear_output=True)
    def toggle_playback(b):
        if soundcard.is_running():
            logger.info("Stopping")
            plotter.stop()
            soundcard.stop()

            # Next action is start
            run_button.description = "Start"
        elif soundcard.is_idle():
            logger.info("Starting")
            plotter.start()
            soundcard.start()

            # Next action is stop
            run_button.description = "Stop"

    run_button.on_click(toggle_playback)
    display(run_button)
    interact(update_volume,
             level=widgets.FloatSlider(min=0,
                                       max=1,
                                       step=.01,
                                       value=1,
                                       description='Volume'))
    jupyter_utils.get_cell_logger().show_logs()
    display(debug_view)

    return soundcard