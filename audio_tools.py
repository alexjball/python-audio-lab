import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import display
import ipywidgets as widgets
from pprint import pprint
import logging
import threading
import librosa
import librosa.display
import time

import jupyter_utils

# Set up the root logger and display its output in this cell.
logger = jupyter_utils.get_cell_logger(__name__)
logger.setLevel(logging.DEBUG)

def print_pyaudio_devices():
    """Print info about host API's, devices, and defaults"""

    # Initialize PyAudio
    pa = pyaudio.PyAudio()
    default_host_index = pa.get_default_host_api_info()['index']
    default_input_device_index = pa.get_default_input_device_info()['index']
    default_output_device_index = pa.get_default_output_device_info()['index']

    # TODO: Resolve default pulse device
    print('Host API\'s')
    for i in range(pa.get_host_api_count()):
        if i == default_host_index: print("Default:")
        pprint(pa.get_host_api_info_by_index(i))

    print('\nDevices')
    for i in range(pa.get_device_count()):
        if i == default_input_device_index: print("Default Input:")
        if i == default_output_device_index: print("Default Output:")
        pprint(pa.get_device_info_by_index(i))

    # Clean up PyAudio
    pa.terminate()


class AudioConstants:
    """Constants used by framework types"""
    # Sampling rate and the number of channels must be consistent throughout the graph, so
    # these are chosen to support stereo at a sampling rate supported by the Scarlett.
    # TODO: move these into framework object configuration, or configuration owned by
    # clients.
    sampling_rate = 44100
    num_channels = 1
    dtype = np.float64

    # The max value of a numpy index, which is a safe upper bound for the max number of
    # frames that can be read or written.
    unlimited_frames = np.iinfo(np.intp).max


class AudioSource:
    """Simple interface for sources"""

    def read(self, num_frames_or_output):
        """Reads data into a provided or create array, potentially blocking."""
        raise NotImplementedError()

    def read_available(self):
        """Returns the number of frames that can be read without potentially blocking."""
        raise NotImplementedError()


class AudioSink:
    """Simple interface for sinks"""

    def write(self, data):
        """Writes (num_frames, num_channels) data, potentially blocking."""
        raise NotImplementedError()

    def write_available(self):
        """Returns the number of frames that can be written without potentially blocking."""
        raise NotImplementedError()


class AudioStream(AudioSink):
    # TODO: Consider using a generic message as the payload rather than raw data.
    #       this would also allow sending control signals (stream start/end) over
    #       the graph and attaching more info to raw data.
    # TODO: Test

    def __init__(self):
        self.sinks = []
        self.is_writing = False

    def add_sink(self, sink):
        """Starts sending audio data written to this stream to this sink.
        
        sink: an AudioSink
        """
        assert isinstance(sink, AudioSink)
        if sink not in self.sinks:
            self.sinks.append(sink)

    def remove_sink(self, sink):
        self.sinks.remove(sink)

    def write(self, data):
        """Writes data to the channel, blocking until all sinks have handled the data."""
        if not self.sinks:
            raise RuntimeError("No sinks connected.")

        if self.is_writing:
            raise RuntimeError("Write cycle detected.")

        self.is_writing = True
        try:
            for sink in self.sinks:
                # This uses recursion to pass control to sinks, resulting in a depth-first traversal.
                # So memory usage is O(N) for a maximum chain of N streams. It could be more efficient
                # to evaluate the graph not in depth first order or without recursion, but this is
                # the simplest.
                sink.write(data)
        finally:
            self.is_writing = False

    def write_available(self):
        """Returns the number of frames that can be written without potentially blocking.
        
        If all sinks are queue's, returns the minimum write_available. Otherwise, if any
        sinks are transforms or streams, returns 0.
        
        Also returns 0 if no sinks are connected, even though writing to such a stream 
        doesn't block. This is treated as a null-state, and the rationale for this over
        -1 or something is if write_available == 0 you don't want to write, and you 
        generally don't want to write if no sinks are connected.
        """

        if not self.sinks:
            return 0

        min_write_available = float("inf")
        for sink in self.sinks:
            if isinstance(sink, AudioQueue):
                min_write_available = min(min_write_available,
                                          sink.write_available())
            else:
                return 0

        return min_write_available


class AudioTransform:

    class Input(AudioSink):
        """Interface for the input to a transform."""

        next_input_id = 0

        def __init__(self, owning_transform=None):
            self.owner = None
            self.data = None
            self.id = AudioTransform.Input.next_input_id
            AudioTransform.Input.next_input_id += 1

            if owning_transform:
                assert isinstance(owning_transform, AudioTransform)
                owning_transform.add_input(self)

        def write(self, data):
            if self.data:
                raise RuntimeError("Input is already set")
            self.data = data
            self.owner._handle_write(self)

        def write_available(self):
            # For transforms with multiple inputs, only the write to the last input will
            # block. Return a conservative value here, which is fine since transforms are
            # used in a blocking context anyway.
            return 0

        def take(self):
            assert self.data is not None
            data = self.data
            self.data = None
            return data

        def reset(self):
            self.data = None

    def __init__(self):
        self.inputs = []
        self.unset_inputs = set()

    def apply():
        """Read from inputs and write to outputs or perform other operations.
        
        This is called syncronously with setting the last input. After this is
        called, inputs are cleared.
        """
        raise NotImplementedError()

    def add_input(self, input):
        assert input.owner is None and input not in self.inputs

        input.owner = self
        input.reset()
        self.inputs.append(input)
        self.unset_inputs.add(input.id)

    def remove_input(self, input):
        assert input.owner == self and input in self.inputs

        self.inputs.remove(input)
        if input.id in self.unse_inputs:
            self.unset_inputs.remove(input.id)

    def _handle_write(self, input):
        logger.info(
            f'_handle_write input.id {input.id} unset_inputs {self.unset_inputs}'
        )
        assert input in self.inputs and input.id in self.unset_inputs

        self.unset_inputs.remove(input.id)
        if not self.unset_inputs:
            self.apply()
            self.reset()

    def reset(self):
        """resets all inputs"""
        self.unset_inputs = set()
        for input in self.inputs:
            self.input.reset()
            self.unset_inputs.add(input.id)


class AudioQueue(AudioSource, AudioSink):
    """A blocking queue used to pass audio data between threads."""

    def __init__(self,
                 num_channels=AudioConstants.num_channels,
                 num_frames=1024,
                 dtype=AudioConstants.dtype):
        self.num_channels = num_channels
        self.num_frames = num_frames

        # The writing and reading threads block on each other using conditions.
        # Whenever the queue is read/written to, on_read/write is notified.
        # Before reading or writing data, the queue waits for there to be available
        # space by waiting on on_write/on_read.
        #
        # Note that notify wakes up any single thread that is waiting on the condition,
        # which leads to undefined behavior if multiple threads are reading or writing
        # concurrently. Handling multiple readers would significantly complicate the
        # logic in this class, and higher-level locks can be used to coordinate access.
        self.lock = threading.RLock()
        self.on_read = threading.Condition(self.lock)
        self.on_write = threading.Condition(self.lock)

        # buffer backing the queue, guarded by lock
        self.data = np.zeros((num_frames, num_channels), dtype=dtype)
        # guarded by lock
        self.write_marker = 0
        # guarded by lock
        self.read_marker = 0

    def reset(self):
        with self.lock:
            self.write_marker = 0
            self.read_marker = 0

    def write(self, data):
        """Writes data to the queue, blocking until everything is transfered into the queue's buffer.
        
        data: (frames, num_channels) array of audio data
        """
        assert data.ndim == 2
        assert data.shape[1] == self.num_channels

        with self.on_write:
            while data.shape[0]:
                self.on_read.wait_for(self.write_available)
                data = data[self._write_data(data):, :]
                self.on_write.notify()

    def _write_data(self, data):
        """Inserts as much data as possible into the buffer and returns the number of frames written.
        
        with r = read_marker and w = write_marker:
        [ r  w  ]: write is ahead of read, neither have wrapped around
        
        [ w  r  ]: write is ahead of read, write has wrapped around
        
        [ w    r] -> [ r  w  ] write is ahead of read, read has wrapped around
                               read/write markers are decremented by num_frames.
                               So r is in [0, n-1] and w is in [0, 2n-1] and the
                               indices available to write are, in order, 
                               (w:(r + num_frames)) % num_frames. 
        """
        data_placement = np.arange(
            self.write_marker,
            min(self.read_marker + self.num_frames,
                self.write_marker + data.shape[0])) % self.num_frames
        to_write = min(self.write_available(), data.shape[0])

        assert data_placement.size == to_write
#         logger.info(f'stats {data_placement.size} {to_write}, ' + 
#                        f'{self.write_marker}, {self.read_marker}, {data.shape}, ' +
#                        f'{self.num_frames}')

        self.data[data_placement, :] = data[0:to_write, :]
        self.write_marker += to_write
        return to_write

    def write_available(self):
        """Returns the number of frames that can be written without blocking."""
        return self.num_frames - self.read_available()

    def read_available(self):
        """Returns the number of frames that can be written without blocking."""
        return self.write_marker - self.read_marker

    def read(self, num_frames_or_output):
        """Reads data from the queue, blocking until complete.
        
        num_frames_or_output: the number of frames to read into a new array, 
        or an existing array to fill.
        """

        if isinstance(num_frames_or_output, np.ndarray):
            assert num_frames_or_output.ndim == 2
            assert num_frames_or_output.shape[1] == self.num_channels
            output = num_frames_or_output
            num_frames = output.shape[0]
        else:
            num_frames = num_frames_or_output
            output = np.zeros((num_frames_or_output, self.num_channels),
                              dtype=self.data.dtype)

        output_marker = 0
        with self.on_read:
            while output_marker < num_frames:
                self.on_write.wait_for(self.read_available)
                to_read = min(num_frames - output_marker, self.read_available())
                if self.read_marker + to_read >= self.num_frames:
                    # The read wraps around. Read the rest of the array, then shift
                    # marker indices back by a buffer.
                    to_end = self.num_frames - self.read_marker
                    output[:to_end, :] = self.data[self.read_marker:, :]
                    output_marker += to_end
                    to_read -= to_end
                    self.read_marker = 0
                    self.write_marker -= self.num_frames

                output[output_marker:output_marker +
                       to_read, :] = self.data[self.
                                               read_marker:self.read_marker +
                                               to_read, :]
                self.read_marker += to_read
                output_marker += to_read

                self.on_read.notify()

        return output


class AudioBuffer(AudioQueue):
    """Drops frames instead of blocking on write"""

    def write(self, data):
        frames_to_write = data.shape[0]
        write_available = super().write_available()
        if frames_to_write > write_available:
            logger.debug(f'Dropping {frames_to_write - write_available} frames')
            frames_to_write = write_available
        if frames_to_write:
            super().write(data[:frames_to_write, :])

    def write_available(self):
        return AudioConstants.unlimited_frames


class Gain(AudioTransform):

    def __init__(self):
        super(Gain, self).__init__()
        self.input = AudioTransform.Input(self)
        self.output = AudioStream()
        self.level = 1

    def apply(self):
        self.output.write(self.level * self.input.take())


class AudioFileLoader:
    """Loads audio data from a file and writes to an AudioStream.
    
    `output` is an AudioStream to which sinks can be attached. Callers should not write
    to it.
    """

    def __init__(self):
        self.output = AudioStream()
        self._load_thread = None

    def load_async(self, source):
        if self._load_thread:
            raise RuntimeError("Already loading async")

        def load():
            try:
                self.load(source)
                self._load_thread = None
            except:
                logger.error("Error loading file", exc_info=True)

        self._load_thread = threading.Thread(target=load)
        self._load_thread.start()

    def wait_for_load(self, timeout=None):
        load_thread = self._load_thread
        if load_thread:
            load_thread.join(timeout)

    def is_loading(self):
        return self._load_thread is not None

    def load(self, source):
        """Loads the source"""

        if AudioConstants.num_channels > 2:
            raise NotImplementedError("Only mono or stereo supported for files")

        # Load the source, either as (n,) mono or (2, n) stereo
        source_data, _ = librosa.load(source,
                                      sr=AudioConstants.sampling_rate,
                                      mono=False,
                                      dtype=AudioConstants.dtype)
        if source_data.ndim == 1:
            # source_data.shape == (n,)
            source_data = np.tile(source_data[:, np.newaxis],
                                  AudioConstants.num_channels)
        else:
            # source_data.shape == (2, n)
            source_data = source_data.transpose()

        self.output.write(source_data)

    def reset(self):
        if isinstance(self._source, AudioStream):
            self._source.remove_sink(self._stream_sink)
            self._stream_sink = None

        self._buffer = None
        self._source = None

    def read(self, num_frames_or_output):
        if not self._buffer:
            raise RuntimeError("No source loaded")
        return self._buffer.read(num_frames_or_output)

    def read_available(self):
        if not self._buffer:
            raise RuntimeError("No source loaded")
        return self._buffer.read_available()


class PerfTimer:
    def __init__(self, name, n = 100, period = 1):
        self.name = name
        self.n = n
        self.period = period
        self.dt = []
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def end(self):
        self.dt.append(time.perf_counter() - self.start_time)
        if len(self.dt) >= self.n:
            self.log_stats(np.array(self.dt))
            self.dt = []
    
    def log_stats(self, dt):
        load = dt / self.period
        logger.debug(f'{self.name} load stats: min {min(load)} max {max(load)} mean {load.mean()}')
        
        
class Soundcard:
    """Represents a sound card with an input and output."""

    # numpy number formats and byte widths indexed by the corresponding pyaudio format.
    numpy_by_pyaudio_formats = {
        pyaudio.paFloat32: (np.float32, 4),
        pyaudio.paInt32: (np.int32, 4)
    }

    def __init__(self,
                 device_index=None,
                 sampling_rate=AudioConstants.sampling_rate,
                 channels=AudioConstants.num_channels,
                 device_sample_format=pyaudio.paFloat32,
                 processor_sample_format=AudioConstants.dtype,
                 frames_per_buffer=1024):
        """Initializes a sound card connection.
        
        On my machine frames_per_buffer needs to be at least 256 to avoid dropping frames.
        I don't think this latency is CPU bound, since increasing frames_per_buffer to 512
        drops reported CPU load to around .1. So I think the issue is in one or more of the
        audio hardware, kernel configuration, or IPC.
        """
        self.io = pyaudio.PyAudio()

        # These are inputs and outputs from the perspective of other nodes, not from the
        # hardware device. This class reads from input and writes to output.
        self.input = AudioQueue(num_frames=frames_per_buffer)
        self.output = AudioStream()

        # Use one device index for a sound card.
        self.device_index = device_index
        if self.device_index is None:
            self.device_index = self.io.get_default_input_device_info()['index']

        self.sampling_rate = sampling_rate
        self.channels = channels
        self.device_sample_format = device_sample_format
        self.processor_sample_format = processor_sample_format
        self.frames_per_buffer = frames_per_buffer

        self.state = 'idle'

    def is_idle(self):
        return self.state == 'idle'

    def is_running(self):
        return self.state == 'running'

    def is_terminated(self):
        return self.state == 'terminated'

    def start(self):
        """Starts reading and writing audio data.
        
        processor is a function from (frames, channels) inputs to (frames, channels) outputs
        """

        if not self.is_idle():
            raise RuntimeError(f'Invalid state {self.state}')

        sample_format, sample_width = self.numpy_by_pyaudio_formats[
            self.device_sample_format]

        perf_timer = PerfTimer('stream', period = self.frames_per_buffer / self.sampling_rate)
        def stream_callback(in_data, frame_count, time_info, status_flags):
            if not self.is_running():
                logger.info('stopping')
                return (b'\x00' * len(in_data), pyaudio.paComplete)

            if self.frames_per_buffer != 0 and frame_count != self.frames_per_buffer:
                logger.warning('Unexpected frame count %d', frame_count)
            if status_flags != 0:
                logger.warning('Nonzero status %d', status_flags)

            cpu_load = self.stream.get_cpu_load()
            if cpu_load > 0.9:
                logger.warning(f'High CPU load {cpu_load}')

            in_data = np.frombuffer(in_data, dtype=sample_format).astype(
                self.processor_sample_format).reshape(frame_count,
                                                      self.channels)
            
            try:
                perf_timer.start()
                self.output.write(in_data)
                if self.input.read_available() != frame_count:
                    logger.warning(
                        f'Unexpected frame count at input {self.input.read_available()}'
                    )
                out_data = self.input.read(frame_count)
                perf_timer.end()
            except:
                logger.error("Error in soundcard callback", exc_info=True)
                return (b'\x00' * len(in_data), pyaudio.paComplete)
                
            return (out_data.astype(sample_format).tobytes(),
                    pyaudio.paContinue)

        self.stream = self.io.open(rate=self.sampling_rate,
                                   channels=self.channels,
                                   format=self.device_sample_format,
                                   input=True,
                                   output=True,
                                   input_device_index=self.device_index,
                                   output_device_index=self.device_index,
                                   frames_per_buffer=self.frames_per_buffer,
                                   stream_callback=stream_callback)
        self.state = 'running'

    def stop(self):
        if self.is_running():
            self.stream.stop_stream()
            self.stream.close()
            self.state = 'idle'

    def terminate(self):
        if self.is_running() or self.is_idle():
            self.stop()
            self.io.terminate()
            self.state = 'terminated'

        
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
            buffer_size = round(3 * AudioConstants.sampling_rate / self.update_rate)
            buffer = AudioBuffer(num_frames=buffer_size)
            signal.add_sink(buffer)
            self.buffers.append(buffer)
            
        self._animation = FuncAnimation(self.fig, self._update, interval = 1e3 / self.update_rate)
        self.fig.canvas.draw_idle()
    
    def stop(self):
        self._animation.event_source.stop()
        self._animation = None 
        for i in range(len(self.signals)):
            self.signals[i].remove_sink(self.buffers[i])
        
    def _update(self, *args):
        try:
            datas = [buffer.read(buffer.read_available()) for buffer in self.buffers]
            self.callback(*datas)
        except:
            logger.error('Error in monitor callback', exc_info=True)
        
    def log_basic(self, *datas):
        logger.info(f'Monitor of {len(datas)} signals:')
        for data in datas:
            logger.info(f'     shape: {data.shape}')
            
    
class Daw:

    def __init__(self, name='Daw', fig = None, ax = None):
        self.name = name
        self.ax = ax if ax is not None else plt.gca()
        self.soundcard = Soundcard(frames_per_buffer=4096)
        self.soundcard.output.add_sink(self.soundcard.input)
        
        self.monitor = SignalMonitor()
        self.monitor.update_rate = 1
        self.monitor.signals = [self.soundcard.output]
        self.monitor.callback = self._update_monitor
        self.monitor.fig = fig if fig is not None else plt.gcf()
        
        self._update_timer = PerfTimer('Daw._update_monitor', n = 3, period = 1 / self.monitor.update_rate)
        
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
#         plt.colorbar(format='%+2.0f dB')
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
        logger.show_logs()
        display(debug_view)

    def terminate(self):
        self.soundcard.terminate()
