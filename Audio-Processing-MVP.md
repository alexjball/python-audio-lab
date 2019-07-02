# Audio Processing MVP

# Goals
Explore/prototype approaches for automatic guitar transcription.

# Non-goals
Create something tha works outside a dev/rapid-eval environment

# Requirements/Structure
1. Processing/analyzing an audio stream
2. Command line for running analyses on files or audio inputs
3. UX prototype GUI. Displays analysis in an interactive or live fassion.

# Design

We want to prototype in UI and algorithms. Python and Scipy provide the best signal processing prototyping environment. Mixing in UI means using either a native framework or extending Jupyter notebook display capabilities. Jupyter notebooks have other good features, and UI representations are more flexible (HTML or Python) so we'll go with Jupyter notebooks.

Jupyter notebooks are composed of cells of code or markdown. Code cells execute in the notebook kernel, which may be on a remote machine. Code cells can use Jupyter API's to pipe rich output back to the notebook UI, and API's exist for reacting to user input.

The page will have a global input/output buffer that is initialized with the notebook. The user uses this to set up their source, and cells use this to read and output processed audio. Ideally, the user can use the same interface to process audio files/existing sound, and the filtering cells access a common abstraction.

# Release
I will initially use the library for my own blog posts, and advertise the library after it's matured. But I should make the library public asap, for transparency. Some blog post ideas:
- Guitar pickup transfer function -- How is this different from the existing articles?
- Effect simulation
- Pitch estimation
- Guitar modeling -- especially body response.

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTY0MzUyMDA0MSwxMDQ2NzE5OTIwXX0=
-->