def replay_windows(stream, window_steps):
    """Yield (window_index, window_df) in event-time order.
    Within a window, rows are sorted by record_id so runs are deterministic."""
    stream = stream.sort_values(["step", "record_id"], kind="stable")
    w = (stream["step"] - stream["step"].min()) // window_steps
    for k, g in stream.groupby(w, sort=True):
        yield int(k), g.sort_values("record_id")