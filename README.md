# Clean stop session hook for Tensorflow

## What is this?

When training an estimator with Tensorflow, interrupting training via ctrl-C breaks off training abruptly. This is annoying if you'd prefer to keep the most recent checkpoint; especially annoying if you want to train in the cloud with a preemptible instance and want to keep your work when your instance is interrupted.

There are ways around this if you're doing something simple and using low-level primitives (e.g. calling `session.run` directly) but it's not obvious how to accomplish this if you're using a high-level object like an Estimator.

This solution is designed to cleanly exit from a SIGINT (called from a `ctrl-C` or via `kill -SIGINT $PID` or equivalent). It will finish the current call to the active session's `run` method and do any associated cleanup (like save checkpoints).

## How do I use it?

```python
from clean_stop_hook import CleanStopHook
```
And then, when calling your estimator's `train` method:
```python
estimator.train(my_input_fn, hooks=[CleanStopHook()])
```
