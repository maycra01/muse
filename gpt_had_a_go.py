from os import times
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_streams
from collections import deque
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi
import time

# Constants
NUM_CHANNELS = 4
SAMPLE_RATE = 256
BUFFER_LENGTH = 5
PLOT_INTERVAL = 0.05
BATCH_SIZE = 10
FRAME_SKIP = 5  # plot every 5th frame

# Frequency bands
bands = {
    'alpha': (8, 12),
    'beta':  (12, 30),
    'theta': (4,  8),
    'delta': (0.5,4),
    'gamma': (30,100)
}

# Pre-allocate deques
timestamp_buffer = deque(maxlen=SAMPLE_RATE * BUFFER_LENGTH)
eeg_buffers = [deque(maxlen=SAMPLE_RATE * BUFFER_LENGTH) for _ in range(NUM_CHANNELS)]
band_buffers = {name: deque(maxlen=SAMPLE_RATE * BUFFER_LENGTH) for name in bands}

# Design SOS filters and states for streaming
sos_filters = {}
filter_states = {}
for name, (low, high) in bands.items():
    sos = butter(4, [low/SAMPLE_RATE*2, high/SAMPLE_RATE*2], btype='band', fs=SAMPLE_RATE, output='sos')
    sos_filters[name] = sos
    # one zi per channel
    filter_states[name] = [sosfilt_zi(sos) for _ in range(NUM_CHANNELS)]

# Plot setup (similar to yours)...
# ... create fig, axes, lines ...

# Prefill buffers for 5 s
print("Warming up buffers...")
t0 = None
while len(timestamp_buffer) < SAMPLE_RATE * BUFFER_LENGTH:
    samples, ts = [], []
    for _ in range(BATCH_SIZE):
        s, time_stamp = eeg_stream.pull_sample(timeout=0.01)
        if time_stamp:
            samples.append(s)
            ts.append(time_stamp)
    if not samples:
        continue
    if t0 is None:
        t0 = ts[0]
    avg = np.mean(samples, axis=0)
    rel = ts[-1] - t0
    timestamp_buffer.append(rel)
    for i in range(NUM_CHANNELS):
        eeg_buffers[i].append(avg[i])
print("Buffers warmed.")

# Main loop
frame_count = 0
while True:
    # Pull one batch
    samples, ts = [], []
    for _ in range(BATCH_SIZE):
        s, time_stamp = eeg_stream.pull_sample()
        if time_stamp:
            samples.append(s)
            ts.append(time_stamp)
    if not samples:
        continue
    avg = np.mean(samples, axis=0)
    rel = ts[-1] - t0
    timestamp_buffer.append(rel)
    for i in range(NUM_CHANNELS):
        eeg_buffers[i].append(avg[i])

    # Streaming filter: one sample at a time per channel
    for name, sos in sos_filters.items():
        buf = band_buffers[name]
        states = filter_states[name]
        # new sample per channel
        ch_outs = []
        for ch_idx in range(NUM_CHANNELS):
            out, states[ch_idx] = sosfilt(sos, [avg[ch_idx]], zi=states[ch_idx])
            ch_outs.append(out[0])
        # append channel-average of band
        buf.append(np.mean(ch_outs))

    # Plot every few frames
    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
        # raw EEG: downsample for speed
        ds = 8
        t_ds = list(timestamp_buffer)[::ds]
        for ch_idx, line in enumerate(raw_lines):
            y_ds = np.array(eeg_buffers[ch_idx])[::ds]
            line.set_xdata(t_ds)
            line.set_ydata(y_ds)
        # band plots
        for name, line in band_lines.items():
            tb = band_buffers[name]
            t_ds = list(timestamp_buffer)[-len(tb):][::ds]
            y_ds = list(tb)[::ds]
            line.set_xdata(t_ds)
            line.set_ydata(y_ds)
        # refresh axes
        for a in all_axes:
            a.relim()
            a.autoscale_view(scalex=False, scaley=True)
        plt.draw()
        plt.pause(0.001)
