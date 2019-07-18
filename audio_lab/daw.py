import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import display
import ipywidgets as widgets
import logging
import librosa
import librosa.display
import time

import audio_lab.jupyter_utils as jupyter_utils
from audio_lab.io import *

logger = logging.getLogger(__name__)


class SignalMonitor:
    """Subscribes to signals and provides callbacks for plotting and analysis.
    
    def transfer_function(input, output):
        def plot_transfer(in_data, out_data):
            ...plot
            
        monitor = SignalMonitor()
        monitor.signals = [input, output]
        monitor.callback = plot_transfer
        monitor.start()
    """

    def __init__(self):
        # List of AudioStreams to monitor
        self.signals = []
        self.update_rate = 1
        self.callback = self.log_basic
        self.fig = None

        self._animation = None

    def start(self):
        if self._animation:
            self.stop()
        if not self.fig:
            self.fig = plt.gcf()

        self.buffers = []
        for signal in self.signals:
            buffer_size = round(3 * AudioConstants.sampling_rate /
                                self.update_rate)
            buffer = AudioBuffer(num_frames=buffer_size)
            signal.add_sink(buffer)
            self.buffers.append(buffer)

        self._animation = FuncAnimation(self.fig,
                                        self._update,
                                        interval=1e3 / self.update_rate)
        self.fig.canvas.draw_idle()

    def stop(self):
        self._animation.event_source.stop()
        self._animation = None
        for i in range(len(self.signals)):
            self.signals[i].remove_sink(self.buffers[i])

    def _update(self, *args):
        try:
            datas = [
                buffer.read(buffer.read_available()) for buffer in self.buffers
            ]
            self.callback(*datas)
        except:
            logger.error('Error in monitor callback', exc_info=True)

    def log_basic(self, *datas):
        logger.info(f'Monitor of {len(datas)} signals:')
        for data in datas:
            logger.info(f'     shape: {data.shape}')


class Daw:
    """A simple interface for recording, processing, and playing audio."""

    def __init__(self, name='Daw', fig=None, ax=None):
        self.name = name
        self.ax = ax if ax is not None else plt.gca()
        self.soundcard = Soundcard(frames_per_buffer=4096)
        self.soundcard.output.add_sink(self.soundcard.input)

        self.monitor = SignalMonitor()
        self.monitor.update_rate = 1
        self.monitor.signals = [self.soundcard.output]
        self.monitor.callback = self._update_monitor
        self.monitor.fig = fig if fig is not None else plt.gcf()

        self._update_timer = PerfTimer('Daw._update_monitor',
                                       n=3,
                                       period=1 / self.monitor.update_rate)

        self._run()

    def _update_monitor(self, soundcard_output):
        assert AudioConstants.num_channels == 1
        if not soundcard_output.shape[0]:
            logger.warning('No new soundcard data')
            return

        self._update_timer.start()
        fft = librosa.stft(soundcard_output.flatten())
        db = librosa.amplitude_to_db(abs(fft))
        librosa.display.specshow(db,
                                 sr=AudioConstants.sampling_rate,
                                 x_axis='time',
                                 y_axis='log',
                                 ax=self.ax)
        plt.title('Log-frequency power spectrogram')
        self._update_timer.end()

    def _run(self):
        soundcard = self.soundcard
        monitor = self.monitor

        run_button = widgets.Button(description="Start")
        terminate_button = widgets.Button(description="Terminate")
        debug_view = widgets.Output(layout={'border': '1px solid white'})

        @debug_view.capture(clear_output=False)
        def toggle_playback(b):
            if soundcard.is_running():
                logger.info("Stopping")
                monitor.stop()
                soundcard.stop()

                # Next action is start
                run_button.description = "Start"
            elif soundcard.is_idle():
                logger.info("Starting")
                monitor.start()
                soundcard.start()

                # Next action is stop
                run_button.description = "Stop"

        def terminate_cell(b):
            logger.info("Terminating")
            soundcard.terminate()
            run_button.description = "Terminated"
            run_button.disabled = True

        run_button.on_click(toggle_playback)
        terminate_button.on_click(terminate_cell)
        display(run_button)
        display(terminate_button)
        jupyter_utils.get_cell_logger().show_logs()
        display(debug_view)

    def terminate(self):
        self.soundcard.terminate()
