import pyaudio
import numpy as np
from pprint import pprint
import logging
import threading
import librosa
import time

logger = logging.getLogger(__name__)

class AudioConstants:
    """Constants used by framework types"""
    # Sampling rate and the number of channels must be consistent throughout the graph, so
    # these should be chosen to be compatible with any hardware devices.
    sampling_rate = 44100
    num_channels = 1
    dtype = np.float64

    # The max value of a numpy index, which is a safe upper bound for the max number of
    # frames that can be read or written.
    unlimited_frames = np.iinfo(np.intp).max


class AudioSource:
    """Simple interface for sources of audio data"""

    def read(self, num_frames_or_output):
        """Reads data into a provided or create array, potentially blocking."""
        raise NotImplementedError()

    def read_available(self):
        """Returns the number of frames that can be read without potentially blocking."""
        raise NotImplementedError()


class AudioSink:
    """Simple interface for sinks of audio data"""

    def write(self, data, ):
        """Writes (num_frames, num_channels) data, potentially blocking."""
        raise NotImplementedError()

    def write_available(self):
        """Returns the number of frames that can be written without potentially blocking."""
        raise NotImplementedError()


class AudioStream(AudioSink):
    """Proxies audio data to downstream audio sinks.
    
    Audio streams provide a transport mechanism for audio data, implementing the
    edges of the audio graph.

    Audio streams pass data synchronously on the writing thread. To receive data
    on a separate thread, attach an AudioQueue to the stream and read from that.
    """

    # TODO: Consider using a generic message as the payload rather than raw data.
    #       this would also allow sending control signals (stream start/end) over
    #       the graph and attaching more info to raw data.

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
    """Combines multiple inputs using a callback, hiding synchronization logic.

    Subgraphs of AudioTransforms can be computed synchronously, and are 
    compatible with real-time use cases. This is in contrast to using buffers
    to pass data between threads. AudioTransforms should be used whenever the
    computation is flexible wrt scheduling and window size. 
    """

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
        """Add an input to the transform"""
        assert input.owner is None and input not in self.inputs

        input.owner = self
        input.reset()
        self.inputs.append(input)
        self.unset_inputs.add(input.id)

    def remove_input(self, input):
        """Remove an input from the transform"""
        assert input.owner == self and input in self.inputs

        self.inputs.remove(input)
        if input.id in self.unse_inputs:
            self.unset_inputs.remove(input.id)

    def _handle_write(self, input):
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


class Gain(AudioTransform):
    """A simple gain transform illustrating AudioTransform usage."""

    def __init__(self):
        super(Gain, self).__init__()
        self.input = AudioTransform.Input(self)
        self.output = AudioStream()
        self.level = 1

    def apply(self):
        self.output.write(self.level * self.input.take())


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
                to_read = min(num_frames - output_marker,
                              self.read_available())
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
            logger.debug(
                f'Dropping {frames_to_write - write_available} frames')
            frames_to_write = write_available
        if frames_to_write:
            super().write(data[:frames_to_write, :])

    def write_available(self):
        return AudioConstants.unlimited_frames


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
            raise NotImplementedError(
                "Only mono or stereo supported for files")

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
    """Measures time intervals and periodically logs stats."""

    def __init__(self, name, n=100, period=1):
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
        logger.debug(
            f'{self.name} dt stats: min {min(dt)} max {max(dt)} mean {dt.mean()}'
        )
        logger.debug(
            f'{self.name} load stats: min {min(load)} max {max(load)} mean {load.mean()}'
        )

def print_pyaudio_devices():
    """Print info about host API's, devices, and defaults"""

    # Initialize PyAudio
    pa = pyaudio.PyAudio()
    default_host_index = pa.get_default_host_api_info()['index']
    default_input_device_index = pa.get_default_input_device_info()['index']
    default_output_device_index = pa.get_default_output_device_info()['index']

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

class Soundcard:
    """
    Represents a sound card with an input and output.
    
    Latency is has two flavors here: there is communication latency between the
    python process and hardware device, and relative latency between the input
    and output audio streams. The relative latency equals
    frames_per_buffer / sampling_rate, and the communication latency is system
    dependent but ideally approaches 0. 
    """

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
            self.device_index = self.io.get_default_input_device_info(
            )['index']

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

        perf_timer = PerfTimer('stream',
                               period=self.frames_per_buffer /
                               self.sampling_rate)

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