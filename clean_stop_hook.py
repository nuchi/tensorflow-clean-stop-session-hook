# Copyright 2018 Haggai Nuchi
# MIT License

""" A hook for the purpose of safely interrupting a Monitored training session.
I couldn't another way to interrupt an Estimator in the process of training.
It works by installing a custom SIGINT handler that politely requests that the
run context stop, instead of rudely raising a KeyboardInterrupt.

This way, the current checkpoint can be saved instead of being lost.

Usage:

```python
estimator.train(
    my_input_fn,
    hooks=[CleanStopHook()],
    )
```
"""

import signal

from tensorflow.python.training.session_run_hook import SessionRunHook

class CleanStopHook(SessionRunHook):
    def __init__(self):
        self._should_stop = False
        self._already_installed_handler = False

    def before_run(self, run_context):
        """The interrupt handler is installed as late as possible before the
        session is run; it could go in __init__(), or begin(), or
        after_create_session(), but we might want to preserve the default
        KeyboardInterrupt behavior during those phases.
        """
        if not self._already_installed_handler:
            self._already_installed_handler = True
            self._previous_handler = signal.signal(
                signal.SIGINT, self._handle_interrupt)

    def after_run(self, run_context, run_values):
        """Check the internal flag for whether to request a clean stop.
        """
        if self._should_stop:
            run_context.request_stop()

    def _handle_interrupt(self, sig, frame):
        """Catch interrupt: set internal flag which will get checked after
        one run step, and then restore the previous interrupt handler.
        """
        self._should_stop = True
        signal.signal(signal.SIGINT, self._previous_handler)
