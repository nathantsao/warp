Add `wp.Stream`, `wp.Event`, `wp.ScopedStream` and event timing for SYCL devices, and accept `stream=` in `wp.launch()`.
Support array allocations during SYCL graph capture, retaining storage until both the graph and array references are released.
Support nested event timing during SYCL graph capture/replay using stock Level Zero v2 native recording.
Use CPU-resident host USM for captured timestamp samples to avoid migration faults when Python's `faulthandler` is enabled.
Expose `Device.is_graph_capture_event_supported` and honor oneAPI adapter selection without patching Intel libraries.
