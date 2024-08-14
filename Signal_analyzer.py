import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib
import warnings
import time
import pywt
import os, sys
from scipy.signal import savgol_filter, find_peaks
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
from matplotlib.widgets import Button as py_btn, Slider
import gc

blocksize = 1e6  # Adjust the chunk size if necessary
max_y_limit = 500  # Y-axis limit
# target_data_points_per_second = 500000  # Processing rate

# pdf_size_check = window_size - 50000

win = Tk()
win.geometry("750x250")
win.resizable(TRUE, TRUE)


def chooseFile():
    global csv_filename
    file_path = askopenfilename()
    csv_filename = file_path
    print(csv_filename)
    Label(win, text=csv_filename, font=("Century 10 bold")).pack(pady=4)

    return csv_filename


plt.ion()


def callback(visualize):
    # Get the current working directory
    script_directory = os.getcwd()

    jpg_file_name = "output_plot.pdf"

    jpg_file_path = os.path.join(script_directory, jpg_file_name)
    print(csv_filename)
    # Dask DataFrame
    df = dd.read_csv(csv_filename, blocksize=blocksize)

    # Figure size
    fig, ax = plt.subplots(figsize=(24, 6))
    if visualize == False:
        plt.close()

    # Initialize the first chunk
    chunk = df.get_partition(0)
    y = chunk.compute()["adc2"]

    # Create a buffer for the data to display
    y_display = []
    y_scroll = []

    # Second y_scroll to keep track of peak location after scrolling starts
    y_scroll_for_unadjusted = []

    start_time = time.time()
    processed_data_count = 0
    window_start = 0
    processed_data = []
    original_data = []
    sleep_amount = slider_delay.get()
    window_size = int(slider_data_window.get())  # Size of window

    # Create lists to store peak information
    all_peak_indices = []
    all_peak_values = []
    actual_peak_indices = []
    all_peak_values_unadjusted = []
    peak_indices_unadjusted_list = []

    def remove_baseline(data):
        baseline = np.mean(data)
        return data - baseline

    # iteration
    for partition_number in range(1, len(df.divisions) - 1):
        # Get the next partition
        next_chunk = df.get_partition(partition_number)
        next_y = next_chunk.compute()["adc2"]
        original_data.append(next_y)

        # Calculate the number of data points to process based on target_data_points_per_second
        # elapsed_time = time.time() - start_time
        # target_data_points = int(elapsed_time * target_data_points_per_second)

        # if target_data_points > processed_data_count:
        # data_to_process = next_y[: target_data_points - processed_data_count]
        # else:
        # data_to_process = []

        # Due to data loss problems the data is uncapped but slowed down with time.sleep()
        data_to_process = next_y
        time.sleep(sleep_amount)
        data_to_process_detrended = remove_baseline(data_to_process)

        # Wavelet denoising for baseline removal
        wavelet = "db4"  # Choose 'db4' b/c its better performance in both denoising and preserving signal features
        level = 4  # Adjust the level of decomposition based on your data
        coeffs = pywt.wavedec(data_to_process_detrended, wavelet, level=level)

        # Set a threshold for wavelet coefficients
        threshold = 0.1  # Adjust the threshold based on your data
        coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]

        smoothed_data = pywt.waverec(coeffs, wavelet)
        processed_data.append(smoothed_data)

        # Append smoothed data to the scrolling buffer
        y_scroll.extend(smoothed_data)
        y_scroll_for_unadjusted.extend(smoothed_data)

        # Determine the maximum value in the current data
        max_y_value = max(y_scroll)

        # Calculate the new Y-axis limit
        y_axis_limit = (0, max_y_value + max_y_limit)

        ax.clear()

        if visualize == True:
            if len(y_scroll) < window_size:
                # dynamic Y-axis
                ax.plot(range(len(y_scroll)), y_scroll)
                ax.set_ylim(y_axis_limit)
            # dynamic Y-axis

        # Use scipy's find_peaks to detect peaks in the smoothed data
        peak_indices, _ = find_peaks(smoothed_data, height=80, distance=20)

        peaks_in_full_signal = [
            index + len(y_scroll) + window_size * partition_number
            for index in peak_indices
        ]
        actual_peak_indices.extend(peaks_in_full_signal)

        # Adjust peak indices based on the length of y_scroll and the window_start
        peak_indices_adjusted = [
            index + len(y_scroll) - len(smoothed_data) - window_start
            for index in peak_indices
        ]

        # Ensure the adjusted indices are within the valid range
        peak_indices_adjusted = [max(0, idx) for idx in peak_indices_adjusted]

        # Store peak information
        all_peak_indices.extend(peak_indices_adjusted)
        all_peak_values.extend([y_scroll[i] for i in peak_indices_adjusted])

        processed_data_count += len(data_to_process)

        # Calculate Data per second
        elapsed_time = time.time() - start_time
        dps = processed_data_count / elapsed_time

        # Unadjusted peak indices for plotting ---------------
        peak_indices_unadjusted = [
            index + processed_data_count - len(smoothed_data) - window_start
            for index in peak_indices
        ]
        peak_indices_unadjusted_list.extend(peak_indices_unadjusted)
        all_peak_values_unadjusted.extend(
            [y_scroll_for_unadjusted[i] for i in peak_indices_unadjusted]
        )
        # ----------------------------------------------------

        # Remove older data points
        if len(y_scroll) > window_size:
            # Record the length of y_scroll before removing data
            y_scroll_before_removal = len(y_scroll)

            # y_scroll = y_scroll[-window_size:]
            y_scroll = y_scroll[-window_size:]

            # Record the length of y_scroll after removing data
            y_scroll_after_removal = len(y_scroll)

            # Calculate the number of data points removed
            removed_data_count = y_scroll_before_removal - y_scroll_after_removal

            if visualize == True:
                print(f"Removed {removed_data_count} data points.")
                ax.plot(range(len(y_scroll)), y_scroll)
                ax.set_ylim(y_axis_limit)

            # Store peak information
            valid_peak_indices = [i for i in peak_indices if 0 <= i < len(y_scroll)]

            all_peak_indices.extend(valid_peak_indices)
            all_peak_values.extend([y_scroll[i] for i in valid_peak_indices])

            # Remove y_scroll_diff from every entry in all_peak_indices
            all_peak_indices = [
                value - removed_data_count for value in all_peak_indices
            ]

            # Filter out entries with negative values
            all_peak_indices = [value for value in all_peak_indices if value >= 0]

            # Update all_peak_values
            all_peak_values = [y_scroll[i] for i in all_peak_indices]

            time.sleep(sleep_amount)

        if visualize == True:
            # Plot values
            ax.text(
                0.02,
                0.95,
                f"Data Processed: {processed_data_count}",
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.85,
                f"Delay amount: {sleep_amount}",
                transform=ax.transAxes,
            )
            ax.text(0.02, 0.90, f"DPS: {dps:.2f}", transform=ax.transAxes)
            ax.text(
                0.02,
                0.80,
                f"Data window size: {window_size:.2f}",
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.75,
                f"Peaks found: {int(len(peak_indices_unadjusted_list)/50):.2f}",
                transform=ax.transAxes,
            )
            ax.plot(all_peak_indices, all_peak_values, "ro", markersize=2)
            ax.set_xlabel("Amount of datapoints")
            ax.set_ylabel("Signal [arb]")
            plt.pause(0.05)  # Pause for animation

        print(f"Data Processed: {processed_data_count}")
        print(f"DPS: {dps:.2f}")
        print(f"Peaks found: {int(len(peak_indices_unadjusted_list)/50):.2f}")

    plt.ioff()
    plt.close()

    plt.ion()
    print("Analyzing completed")

    processed_data = np.concatenate(processed_data)
    original_data = np.concatenate(original_data)
    # peaks = []
    # peaks = find_peaks(processed_data, height=800, distance=2)
    # plt.figure(1, figsize=(20,6))
    fig, axs = plt.subplots(2, figsize=(16, 10), layout="constrained")
    axs[0].text(0.02, 0.85, f"File: {csv_filename}", transform=axs[0].transAxes)
    axs[0].plot(original_data, label="Original", color="blue")
    axs[0].set_title("Original signal")
    axs[0].text(
        0.02,
        0.75,
        f"Estimated peak amount: {int(len(peak_indices_unadjusted_list) / 50):.2f}",
        transform=axs[0].transAxes,
    )
    axs[1].plot(processed_data, label="Processed", color="green")
    axs[1].set_title("Processed signal")
    axs[1].plot(
        peak_indices_unadjusted_list, all_peak_values_unadjusted, "ro", markersize=2
    )

    plt.tight_layout()
    plt.show(block=True)

    def save_pdf():
        # Save the window as a PDF
        plt.savefig(jpg_file_path, bbox_inches="tight")

        print(f"Last window saved as PDF: {jpg_file_path}")

    # axes = plt.axes([0.81, 0.000001, 0.1, 0.075])
    # bnext = py_btn(axes, 'Add',color="yellow")
    # bnext.on_clicked(save_pdf)


def analyze():
    callback(visualize=False)


def visualize():
    callback(visualize=True)


btn = Button(win, text="Choose file", command=chooseFile)
btn.pack(ipadx=10)
btn_start = Button(win, text="Start visualization", command=visualize)
btn_analyze = Button(win, text=("Analyze signal"), command=analyze)
btn_start.pack(ipadx=12)
btn_analyze.pack(ipadx=14)
slider_delay = Scale(
    win,
    from_=0,
    to=0.5,
    orient="horizontal",
    resolution=0.01,
    label="Faster                                           Slower",
    troughcolor="white",
    length=200,
    showvalue=0,
)
slider_delay.pack()
slider_data_window = Scale(
    win,
    from_=100000,
    to=2000000,
    orient="horizontal",
    resolution=1,
    label="Data in a window",
)
slider_data_window.pack()
win.bind("<Return>", lambda event: callback())
win.mainloop()
