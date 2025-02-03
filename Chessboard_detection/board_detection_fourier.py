import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

def show_img(*imgs):
    if len(imgs) == 0 or len(imgs) > 6:
        raise ValueError("Number of images should be between 1 and 6")

    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    if len(imgs) == 1:
        axes = [axes]
    
    for ax, img in zip(axes, imgs):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    
    plt.show()

# img_path = Path("Chessboard_detection", "TestImages", "17_12_2024", "4", "1.jpg")
# img_path = Path("Chessboard_detection", "TestImages", "16_12_2024", "6", "1.jpg")
img_path = Path("Chessboard_detection", "TestImages", "Temp", "manual_saved.jpg")
img_path_str = str(img_path)
img = cv2.imread(img_path_str)

canny_edges = cv2.Canny(img, 75, 150)
kernel = np.ones((11, 11), np.uint8)
canny_edges_dilated = cv2.dilate(canny_edges, kernel, iterations=1)
canny_edges_eroded = cv2.erode(canny_edges_dilated, kernel, iterations=1)

def detect_horizontal_lines_period(img):
    # Compute the Fourier Transform of the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # Analyze the frequency domain to find the dominant frequencies
    rows, cols = img.shape
    print(f"Rows: {rows}, Cols: {cols}")
    crow, ccol = rows // 2 , cols // 2

    col_dist = 10

    # Mask to keep only vertical frequencies
    low_pass_mask = np.zeros((rows, cols), np.uint8)
    low_pass_mask[crow-20:crow+20, ccol-col_dist:ccol+col_dist] = 1
   

    high_pass_mask = np.zeros((rows, cols), np.uint8)
    high_pass_mask[:crow-20, ccol-col_dist:ccol+col_dist] = 1
    high_pass_mask[crow+20:, ccol-col_dist:ccol+col_dist] = 1

    fshift_masked_low = fshift * low_pass_mask
    magnitude_spectrum_masked_low = 20 * np.log(np.abs(fshift_masked_low))

    fshift_masked_high = fshift * high_pass_mask
    magnitude_spectrum_masked_high = 20 * np.log(np.abs(fshift_masked_high))
    
    # Inverse Fourier Transform to visualize the detected frequencies
    f_ishift_low = np.fft.ifftshift(fshift_masked_low)
    img_back_low = np.fft.ifft2(f_ishift_low)
    img_back_low = np.abs(img_back_low)

    f_ishift_high = np.fft.ifftshift(fshift_masked_high)
    img_back_high = np.fft.ifft2(f_ishift_high)
    img_back_high = np.abs(img_back_high)

    return magnitude_spectrum, magnitude_spectrum_masked_low, magnitude_spectrum_masked_high, img_back_low, img_back_high

def detect_vertical_lines_period(img):
    # Compute the Fourier Transform of the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Analyze the frequency domain to find the dominant frequencies
    rows, cols = img.shape
    print(f"Rows: {rows}, Cols: {cols}")
    crow, ccol = rows // 2 , cols // 2

    row_dist = 15

    # Mask to keep only vertical frequencies
    low_pass_mask = np.zeros((rows, cols), np.uint8)
    low_pass_mask[crow-row_dist:crow+row_dist, ccol-20:ccol+20] = 1

    high_pass_mask = np.zeros((rows, cols), np.uint8)
    high_pass_mask[crow-row_dist:crow+row_dist, :ccol-20] = 1
    high_pass_mask[crow-row_dist:crow+row_dist, ccol+20:] = 1

    fshift_masked_low = fshift * low_pass_mask
    magnitude_spectrum_masked_low = 20 * np.log(np.abs(fshift_masked_low))

    fshift_masked_high = fshift * high_pass_mask
    magnitude_spectrum_masked_high = 20 * np.log(np.abs(fshift_masked_high))
    
    # Inverse Fourier Transform to visualize the detected frequencies
    f_ishift_low = np.fft.ifftshift(fshift_masked_low)
    img_back_low = np.fft.ifft2(f_ishift_low)
    img_back_low = np.abs(img_back_low)

    f_ishift_high = np.fft.ifftshift(fshift_masked_high)
    img_back_high = np.fft.ifft2(f_ishift_high)
    img_back_high = np.abs(img_back_high)

    return magnitude_spectrum, magnitude_spectrum_masked_low, magnitude_spectrum_masked_high, img_back_low, img_back_high

def detect_rotation(img):
    # Compute the Fourier Transform of the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    max_val = 0
    max_angle = 0
    for i in range(-40, 40):
        angle = 0.5*i * np.pi / 180

        value = objective_function(angle, magnitude_spectrum)

        if value > max_val:
            max_val = value
            max_angle = angle

    return max_angle*180/np.pi, max_val


def objective_function(angle, magnitude_spectrum):
    """ Add the values along the line that passes through the image center at the given angle """
    line_width = 5
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2

    # Compute the line equation
    x = np.arange(cols)
    y = np.tan(angle) * (x - ccol) + crow

    # Remove points that are outside the image
    mask = (y >= 0) & (y < rows)
    x = x[mask].astype(int)
    y = y[mask].astype(int)

    # Sum the values along the line with a certain width
    values = 0
    for i in range(-line_width // 2, line_width // 2 + 1):
        y_offset = y + i
        mask = (y_offset >= 0) & (y_offset < rows)
        values += magnitude_spectrum[y_offset[mask], x[mask]]

    return np.sum(values)

max_angle, max_val = detect_rotation(canny_edges_eroded)
print(f"Max angle: {max_angle}, Max value: {max_val}")

# Rotate the image by the detected angle
(h, w) = canny_edges_eroded.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, max_angle, 1.0)
rotated_canny_edges_eroded = cv2.warpAffine(canny_edges_eroded, M, (w, h))
# Detect the period of vertical lines using Fourier Transform
magnitude_spectrum, magnitude_spectrum_masked_low, magnitude_spectrum_masked_high, img_back_low, img_back_high = detect_horizontal_lines_period(rotated_canny_edges_eroded)


board = cv2.findChessboardCorners(rotated_canny_edges_eroded, (9, 6))
# show_img(img, canny_edges)

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(2, 4, 2)
plt.title('Canny Edges')
plt.imshow(rotated_canny_edges_eroded, cmap='gray')

plt.subplot(2, 4, 3)
plt.title('Filtered Magnitude Spectrum Low')
plt.imshow(magnitude_spectrum_masked_low, cmap='gray')

plt.subplot(2, 4, 4)
plt.title('Reverse Fourier Transform Low')
plt.imshow(img_back_low, cmap='hot')

plt.subplot(2, 4, 5)
plt.title('Filtered Magnitude Spectrum HIgh')
plt.imshow(magnitude_spectrum_masked_high, cmap='gray')

plt.subplot(2, 4, 6)
plt.title('Reverse Fourier Transform High')
plt.imshow(img_back_high, cmap='hot')

difference = 2*img_back_high - img_back_low
plt.subplot(2, 4, 7)
plt.title('Filtered Magnitude Spectrum diff')
plt.imshow(difference, cmap='hot')

# Sum img_back_high - img_back_low in the column direction
column_sum = np.sum(difference, axis=1)

# Normalize the column sum
column_sum_normalized = (column_sum - np.min(column_sum)) / (np.max(column_sum) - np.min(column_sum))
# column_sum_normalized[np.where(column_sum_normalized < 0.5)] = 0

# Plot the normalized column sum
plt.subplot(2, 4, 8)
plt.title('Normalized Column Sum')
plt.plot(column_sum_normalized)

def moving_average(data, window_size):
    return data

window_size = 21
smoothed_column_sum = moving_average(column_sum_normalized, window_size)

peaks, _ = find_peaks(smoothed_column_sum, prominence=0.1, distance=640/(9*2))
print("Peaks found at:", peaks)

plt.subplot(2, 4, 8)
plt.title('Column Sum with Peaks')
plt.plot(smoothed_column_sum)
plt.plot(peaks, smoothed_column_sum[peaks], "x")

plt.subplot(2, 4, 8)
plt.title('Column Sum')
plt.plot(smoothed_column_sum)

plt.tight_layout()
plt.show()

# Detect the period of vertical lines using Fourier Transform
magnitude_spectrum, magnitude_spectrum_masked_low, magnitude_spectrum_masked_high, img_back_low, img_back_high = detect_vertical_lines_period(rotated_canny_edges_eroded)

# show_img(img, canny_edges)

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(2, 4, 2)
plt.title('Canny Edges')
plt.imshow(rotated_canny_edges_eroded, cmap='gray')

plt.subplot(2, 4, 3)
plt.title('Filtered Magnitude Spectrum Low')
plt.imshow(magnitude_spectrum_masked_low, cmap='gray')

plt.subplot(2, 4, 4)
plt.title('Reverse Fourier Transform Low')
plt.imshow(img_back_low, cmap='hot')

plt.subplot(2, 4, 5)
plt.title('Filtered Magnitude Spectrum HIgh')
plt.imshow(magnitude_spectrum_masked_high, cmap='gray')

plt.subplot(2, 4, 6)
plt.title('Reverse Fourier Transform High')
plt.imshow(img_back_high, cmap='hot')

difference = 2*img_back_high - img_back_low
plt.subplot(2, 4, 7)
plt.title('Filtered Magnitude Spectrum diff')
plt.imshow(difference, cmap='hot')

# Sum img_back_high - img_back_low in the row direction
row_sum = np.sum(difference, axis=0)

# Normalize the row sum
row_sum_normalized = (row_sum - np.min(row_sum)) / (np.max(row_sum) - np.min(row_sum))
# row_sum_normalized[np.where(row_sum_normalized < 0.5)] = 0

# Plot the normalized row sum
plt.subplot(2, 4, 8)
plt.title('Normalized row Sum')
plt.plot(row_sum_normalized)

def moving_average(data, window_size):
    return data

window_size = 21
smoothed_row_sum = moving_average(row_sum_normalized, window_size)

peaks, _ = find_peaks(smoothed_row_sum, prominence=0.1, distance=640/(9*2))
print("Peaks found at:", peaks)

plt.subplot(2, 4, 8)
plt.title('row Sum with Peaks')
plt.plot(smoothed_row_sum)
plt.plot(peaks, smoothed_row_sum[peaks], "x")

plt.subplot(2, 4, 8)
plt.title('row Sum')
plt.plot(smoothed_row_sum)

plt.tight_layout()
plt.show()