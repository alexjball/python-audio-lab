import matplotlib
import matplotlib.pyplot as plt
import logging
from logging.handlers import QueueHandler, QueueListener
import ipywidgets as widgets
import queue
from IPython.display import display

matplotlib.rcParams['figure.figsize'] = [3, 2]

def start_figure(label, **kwargs):
    """Creates a figure and axes, closing an existing one"""
    plt.close(label)
    subplots = plt.subplots(**kwargs, num = label)
    plt.gcf().suptitle(label)
    return subplots

def fig_name():
    """Returns the name passed to start_figure"""
    return plt.gcf().get_label()

class OutputWidgetHandler(logging.Handler):
    """Custom logging handler sending logs to an output widget
    
    Copied from https://ipywidgets.readthedocs.io/en/stable/examples/Output%20Widget.html
    """

    def __init__(self, max_logs_to_show, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'overflow': 'auto',
            'height': '300px',
            'border': '1px solid white'
        }
        self.out = widgets.Output(layout=layout)
        self.max_logs_to_show = max_logs_to_show
        
    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record+'\n'
        }
        self.out.outputs = (new_output, ) + self.out.outputs[0:min(len(self.out.outputs), self.max_logs_to_show - 1)]

    def show_logs(self):
        """ Show the logs below a button to clear them """
        def clear(b):
            self.clear_logs()
        button = widgets.Button(description="Clear logs")
        button.on_click(clear)

        display(button)
        display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()

def get_cell_logger(name = None, level = logging.INFO, format='%(asctime)s  - [%(levelname)s] %(message)s', max_logs_to_show = 100):
    """
    Configures a logger to output to a specific cell
    
    The logger should be accessed with logging.getLogger if the config doesn't need to be changed.
    """
    logger = logging.getLogger(name)

    if hasattr(logger, '_cell_logger_handler'):
        handler = logger._cell_logger_handler
    else:
        handler = OutputWidgetHandler(max_logs_to_show)
        
        que = queue.Queue(-1)  # no limit on size
        queue_handler = QueueHandler(que)
        listener = QueueListener(que, handler)
        logger.addHandler(queue_handler)    
        listener.start()
        
        # TODO: stop the listener somewhere
        
        logger.show_logs = handler.show_logs
        logger.clear_logs = handler.clear_logs
        logger.capture = handler.out.capture
        
        logger._cell_logger_handler = handler

    logger.setLevel(level)
    handler.max_logs_to_show = max_logs_to_show
    handler.setFormatter(logging.Formatter(format))
    
    return logger