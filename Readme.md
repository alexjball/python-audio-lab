# Interactive Audio Processing With Python And Jupyter Notebooks

This project brings the interactivity of Jupyter notebooks to audio
algorithm development. It provides a modular graph-based system for
processing audio, hiding the complexities of I/O from your code.

This was originally motivated by wanting to do some analysis of my
electric guitar using my soundcard and PyAudio (results in `pickup-testing`). I 
am also interested in DSP applications, and Jupyter sounded great for 
prototyping, so the scope grew to be a general purpose audio-processing solution.

A stretch goal for this was to develop DAW-like UI for cells and 
use notebooks to organize compositions, recordings, practice sessions,
and educational content. I didn't pursue this due to the GIL limitation
below.

Features:

- Modular structure
- PyAudio integration for hardware I/O
- Jupyter notebook integration for logging and display
- Real-time and offline processing

Limitations:

- The python GIL means that main thread operations can block real-time
  callbacks, even though they run in separate threads. This makes it 
  hard to guarantee reliable real-time performance if e.g. cell UI is
  updated while audio is open. I was able to udpate line graphics at 1 Hz
  without issue, but was unable to update a spectrogram at 1 Hz.

Overall, this library provides useful glue that can be used to when developing
audio algorithms. I wouldn't build a DAW out of this, but it works well to put
the human in the processing loop. I plan to iterate on these ideas with a React
app that uses WebAudio, and focus more on the stretch goals. I expect that to be
more widely accessible.

## Requirements/Structure

1. Processing/analyzing an audio stream
2. Command line for running analyses on files or audio inputs
3. UX prototype GUI. Displays analysis in an interactive or live fassion.

## Design

We want to prototype in UI and algorithms. Python and Scipy provide the best signal processing prototyping environment. Mixing in UI means using either a native framework or extending Jupyter notebook display capabilities. Jupyter notebooks have other good features, and UI representations are more flexible (HTML or Python) so we'll go with Jupyter notebooks.

Jupyter notebooks are composed of cells of code or markdown. Code cells execute in the notebook kernel, which may be on a remote machine. Code cells can use Jupyter API's to pipe rich output back to the notebook UI, and API's exist for reacting to user input.
