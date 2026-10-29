Add `Device.supports_graph_capture` and `Device.is_conditional_graph_supported` to query ordinary capture/replay and conditional control flow independently of `Device.is_graph_capture_event_supported`.
Preserve the existing CUDA-specific behavior of `wp.is_conditional_graph_supported()`.
