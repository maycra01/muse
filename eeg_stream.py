from os import times

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import Stream
from numpy.distutils.mingw32ccompiler import rc_name
from pylsl import  StreamInlet, resolve_streams
from collections import deque
from scipy.signal import butter, welch, sosfiltfilt
from scipy.integrate import trapezoid

import time

NUM_CHANNELS = 4
SAMPLE_RATE = 256  # Muse 2 default
BUFFER_LENGTH = 5  # seconds of data to show
PLOT_INTERVAL = 0.05  # update every 50 ms
BATCH_SIZE = 10  # Pull 10 samples at a time
FILTER_ORDER = 4
DOWNSAMPLE = 2
PLOT_SKIP = 5
PADLEN = 3 * (2*FILTER_ORDER + 1)
WINDOW_LEN = 2 # seconds for each Welch segment
frame_count = 0
t0 = None  # For relative time

# Frequency ranges for each band

bands = {
    'alpha': (8,  12),
    'beta':  (12, 30),
    'theta': (4,   8),
    'delta': (0.5, 4),
    'gamma': (30, 100),
}

sos_filters = {
    name: butter(
        N=FILTER_ORDER,
        Wn=[low, high],
        btype='band',
        fs=SAMPLE_RATE,
        output='sos'
    )
    for name, (low, high) in bands.items()
}

power_deques = {
    name: deque(maxlen=SAMPLE_RATE * BUFFER_LENGTH)
    for name in bands
}


# band_deques = {
#     name: deque(maxlen=SAMPLE_RATE * BUFFER_LENGTH)
#                 for name in bands}


def get_eeg_stream():
    print("looking for EEG streams...")
    streams = resolve_streams(wait_time=1.0)
    print(streams)
    print(f"Found {len(streams)} EEG streams.")

    if not streams:
        print("No streams found")
    else:
        for stream in streams:
            name = stream.name()
            type = stream.type()
            print(f"{name} | {type}")
            if type == "EEG":
                eeg_resolve = stream

        eeg_stream = StreamInlet(eeg_resolve)
        stream_info = eeg_stream.info()
        print(stream_info)

    return eeg_stream


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
      Apply a Butterworth bandpass filter to the EEG data.

      Parameters:
      - data: The EEG signal (a list or NumPy array)
      - lowcut: Low frequency cutoff (e.g. 8 for alpha)
      - highcut: High frequency cutoff (e.g. 12 for alpha)
      - fs: Sampling rate (e.g. 256 for Muse)
      - order: Filter order (higher = sharper but slower)

      Returns:
      - Filtered signal (same length as input)
      """

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(N=order,Wn=[low, high],btype='band', output='sos', fs=fs)
    return sosfiltfilt(sos, data)

def update_plot(eeg_deque, timestamp_deque, lines, ax):
    if timestamp_deque:
        ax.set_xlim(max(0, timestamp_deque[-1] - BUFFER_LENGTH), timestamp_deque[-1])

    for i, line in enumerate(lines):
        line.set_xdata(timestamp_deque)
        line.set_ydata(eeg_deque[i])

        # print(f"Line data (x): {lines[i].get_xdata()}")
        # print(f"Line data (y): {lines[i].get_ydata()}")
        # print(f"Line color: {lines[i].get_color()}")
        # print(f"Line label: {lines[i].get_label()}")

    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

def bandpower(x, band, window_len=WINDOW_LEN):
    no_per_seg = int(window_len * SAMPLE_RATE)
    freqs, Pxx = welch(x, fs=SAMPLE_RATE, nperseg=no_per_seg, noverlap=0)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    return trapezoid(Pxx[mask], freqs[mask])



eeg_stream = get_eeg_stream()

# Print details about the stream
# print(f"Stream name: {stream_info.name()}")
# print(f"Stream type: {stream_info.type()}")
# print(f"Number of channels: {stream_info.channel_count()}")
# print(f"Sampling rate: {stream_info.nominal_srate()}")

# for all 5 channels
# sample, timestamp = eeg_stream.pull_sample
# print(f"Sample data: {sample}")
# print(f"Channel labels: {eeg_stream.info().get_channel_labels()}")
# for i in range(stream_info.channel_count()):
#     print(f"Channel {i}: {stream_info.desc()}")

# for the 4 eeg channels
# print(f"Channel labels: {eeg_stream.info().get_channel_labels()[:4]}")
#
# for i in range(3):
#     sample, timestamp = get_eeg_sample(eeg_stream)
#     print(f"Sample data: {sample}")
#     print(f"Sample timestamp: {timestamp}")

# plt.ion()
timestamp_deque = deque(maxlen=SAMPLE_RATE * BUFFER_LENGTH)
eeg_deque = [deque(maxlen=SAMPLE_RATE * BUFFER_LENGTH) for _ in range(NUM_CHANNELS)]

fig, axarr = plt.subplots(4, 2, figsize=(10, 14))
axes = axarr.flatten() # Flatten and grab all the axes
ax = axes[:7]
ax_power = axes[7]

plt.ion()
fig.show()            # make the figure window appear now

# create lines for each channel of raw eeg
lines = [ax[0].plot([], [])[0] for _ in range(NUM_CHANNELS)]
colors = ['r', 'g', 'b', 'm']

for line, color in zip(lines, colors):
    line.set_color(color)

# Raw eeg data
ax[0].set_xlim(0, BUFFER_LENGTH)  # x-axis: 0 to 5 seconds (time window)
ax[0].set_ylim(-300, 300)  # y-axis: range of EEG amplitude (microvolts)
ax[0].set_title("Live EEG (Muse 2)")  # Title of the plot
ax[0].set_xlabel("Time (s)")  # x-axis label
ax[0].set_ylabel("Amplitude (μV)")  # y-axis label

# All Frequency Bands Combined Plot (second plot)
line_alpha_combined = ax[1].plot([], [])[0]  # Alpha line
line_beta_combined = ax[1].plot([], [])[0]   # Beta line
line_theta_combined = ax[1].plot([], [])[0]  # Theta line
line_delta_combined = ax[1].plot([], [])[0]  # Delta line
line_gamma_combined = ax[1].plot([], [])[0]  # Gamma line

# Assign a color to each line for visual distinction
line_alpha_combined.set_color('b')   # Blue for Alpha
line_beta_combined.set_color('g')    # Green for Beta
line_theta_combined.set_color('r')   # Red for Theta
line_delta_combined.set_color('c')   # Cyan for Delta
line_gamma_combined.set_color('m')   # Magenta for Gamma

ax[1].set_xlim(0, BUFFER_LENGTH)
ax[1].set_title("All Frequency Bands Combined")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude (μV)")

# Individual Frequency Band Plots (ax[2] - ax[6])
line_alpha = ax[2].plot([], [])[0]
line_beta = ax[3].plot([], [])[0]
line_theta = ax[4].plot([], [])[0]
line_delta = ax[5].plot([], [])[0]
line_gamma = ax[6].plot([], [])[0]

# Assign a color to each line for visual distinction
line_alpha.set_color('b')   # Blue for Alpha
line_beta.set_color('g')    # Green for Beta
line_theta.set_color('r')   # Red for Theta
line_delta.set_color('c')   # Cyan for Delta
line_gamma.set_color('m')   # Magenta for Gamma

# Set the titles, limits, and labels for each band
ax[2].set_title("Alpha Band")
ax[3].set_title("Beta Band")
ax[4].set_title("Theta Band")
ax[5].set_title("Delta Band")
ax[6].set_title("Gamma Band")

# ax[2].set_ylim(-60000, -50000)
# ax[3].set_ylim(-1000000, 1)
# ax[4].set_ylim(-1000000, 1)
# ax[5].set_ylim(-1000000, 1)
# ax[6].set_ylim(-120000, 70000)

for axis in ax[2:]:
    axis.set_xlim(0, BUFFER_LENGTH)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude (μV)")

power_lines = []
for name, colour in zip(bands, ['b','g','r','c','m']):
    line = ax_power.plot([], [], label=name)[0]
    line.set_color(colour)
    power_lines.append(line)

ax_power.set_title("Band Power (Welch)")
ax_power.set_xlabel("Time (s)")
ax_power.set_ylabel("Time (s)")
ax_power.legend(loc="upper right")


try:
    while True:
        print(len(eeg_deque[0]), len(timestamp_deque)) # Sanity check

        samples = []
        timestamps = []

        # Pull samples in a batch
        for _ in range(BATCH_SIZE):
            sample, timestamp = eeg_stream.pull_sample(timeout=0.01)
            if timestamp:
                samples.append(sample)
                timestamps.append(timestamp)

        # Skip if no samples were pulled
        if not samples:
            continue

        if t0 is None:
            t0 = timestamps[0]

        # Average the samples in the batch across all channels
        average_sample = np.mean(samples, axis=0)

        # Add the averaged sample to the buffers
        relative_time = timestamps[-1] - t0 # use last timestamp in batch
        timestamp_deque.append(relative_time)

        # Add samples to buffers
        for ch in range(NUM_CHANNELS):  # Only use first 4 channels
            eeg_deque[ch].append(average_sample[ch])

       # # Apply the bandpass filters for each frequency band **inside the loop**
       #  filtered_data = {'alpha': [], 'beta': [], 'theta': [], 'delta': [], 'gamma': []}
       #
       #  # Filter each channel's data for the specific frequency bands
       #  for channel_deque in eeg_buffers:
       #      channel_data = np.array(channel_deque) # turn deque into array for processing
       #      alpha_data = bandpass_filter(channel_data, *alpha_range, fs=SAMPLE_RATE)
       #      beta_data = bandpass_filter(channel_data, *beta_range, fs=SAMPLE_RATE)
       #      theta_data = bandpass_filter(channel_data, *theta_range, fs=SAMPLE_RATE)
       #      delta_data = bandpass_filter(channel_data, *delta_range, fs=SAMPLE_RATE)
       #      gamma_data = bandpass_filter(channel_data, *gamma_range, fs=SAMPLE_RATE)
       #
       #      # Store the filtered data for each band in the dictionary
       #      filtered_data['alpha'].append(alpha_data)
       #      filtered_data['beta'].append(beta_data)
       #      filtered_data['theta'].append(theta_data)
       #      filtered_data['delta'].append(delta_data)
       #      filtered_data['gamma'].append(gamma_data)
       #
       #  # Now compute the average for each frequency band **across channels** immediately
       #  alpha_buffer = np.mean(filtered_data['alpha'], axis=0)
       #  beta_buffer = np.mean(filtered_data['beta'], axis=0)
       #  theta_buffer = np.mean(filtered_data['theta'], axis=0)
       #  delta_buffer = np.mean(filtered_data['delta'], axis=0)
       #  gamma_buffer = np.mean(filtered_data['gamma'], axis=0)

        # for name in bands:
        #     sos = sos_filters[name]
        #     states = filter_states[name]
        #     outs = []
        #
        #     for ch in range(NUM_CHANNELS):
        #         y, states[ch] = sosfilt(sos,
        #                                 [average_sample[ch]],
        #                                 zi=states[ch])
        #         outs.append(y[0])
        #
        #     band_value = np.mean(outs)
        #     band_deques[name].append(band_value)

        # Plot every N frames to reduce CPU usage
        frame_count += 1
        if frame_count % PLOT_SKIP == 0:  # Adjust to control FPS
            ts_full = np.array(timestamp_deque)  # shape (N,)
            data = np.vstack(eeg_deque)  # shape (4, N)
            N = data.shape[1]

            # only filtifilt once we have > PADLEN samples
            if N <= PADLEN:
                # skip filtering (you could fall back to sosfilt streaming here)
                continue

            # 3a) Raw EEG (ax[0])
            for ch, line in enumerate(lines):
                line.set_xdata(ts_full)
                line.set_ydata(data[ch])
            ax[0].set_xlim(max(0, ts_full[-1] - BUFFER_LENGTH),
                           ts_full[-1])

            # 3b) Combined bands (ax[1])
            combined_lines = [
                line_alpha_combined, line_beta_combined,
                line_theta_combined, line_delta_combined,
                line_gamma_combined
            ]
            for name, line in zip(bands, combined_lines):
                filt = sosfiltfilt(sos_filters[name], data, axis=1)
                mean_ts = filt.mean(axis=0)
                xs = ts_full[::DOWNSAMPLE]
                ys = mean_ts[::DOWNSAMPLE]
                line.set_xdata(xs)
                line.set_ydata(ys)
            ax[1].set_xlim(max(0, ts_full[-1] - BUFFER_LENGTH),
                           ts_full[-1])

            # 3c) Individual band plots (ax[2]–ax[6])
            indiv_lines = [line_alpha, line_beta, line_theta,
                           line_delta, line_gamma]
            for idx, (name, line) in enumerate(zip(bands, indiv_lines), start=2):
                filt = sosfiltfilt(sos_filters[name], data, axis=1)
                mean_ts = filt.mean(axis=0)
                xs = ts_full[::DOWNSAMPLE]
                ys = mean_ts[::DOWNSAMPLE]
                line.set_xdata(xs)
                line.set_ydata(ys)
                ax[idx].set_xlim(max(0, ts_full[-1] - BUFFER_LENGTH),
                                 ts_full[-1])

            for name in bands:
                filt = sosfiltfilt(sos_filters[name], data, axis=1)
                mean_filt = filt.mean(axis=0)
                p = bandpower(mean_filt, bands[name], window_len=WINDOW_LEN)
                power_deques[name].append(p)

            for (name, line, dq) in zip(bands, power_lines, power_deques.values()):
                buf = np.array(dq)
                if buf.size:
                    xs = ts_full[-buf.size:]
                    line.set_xdata(xs)
                    line.set_ydata(buf)
            ax_power.set_xlim(max(0, ts_full[-1] - BUFFER_LENGTH), ts_full[-1])

            if 'beta' in power_deques:
                beta_hist = power_deques['beta']
                baseline = float(np.mean(beta_hist)) if beta_hist else 0.0
                curr_beta = beta_hist[-1] if beta_hist else 0.0

                if curr_beta < 0.8 * baseline:
                    line_beta_combined.set_linewidth(3)
                else:
                    line_beta_combined.set_linewidth(1)

            # 3d) Redraw everything
            for a in ax:
                a.relim()
                a.autoscale_view(scalex=False)

            ax_power.relim()
            ax_power.autoscale_view(scalex=False)

            plt.draw()
            plt.pause(PLOT_INTERVAL)

        # Append the sample and timestamp to the buffers
        # if not timestamp_buffer:
        #     t0 = timestamp
        #
        # relative_time = timestamp - t0
        #
        # for i in range(NUM_CHANNELS):
        #     eeg_buffers[i].append(sample[i])
        # print("EEG Buffers:", eeg_buffers)

        # timestamp_buffer.append(relative_time)
        # print("Timestamp Buffer:", timestamp_buffer)

        # Update plot
        # update_plot(eeg_buffers, timestamp_buffer, lines, ax)

        # To stop computer from taking off, take a nap
        # time.sleep(0.1)

except KeyboardInterrupt:
    print("plot stopped.")
    plt.ioff()


