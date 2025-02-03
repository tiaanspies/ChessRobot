import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

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

def detect_lines(img, threshold_angle=15):
    # Apply Canny edge detector
    edges = cv2.Canny(img, 75, 150)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 2, np.pi / 180, 200)

    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if abs(angle - 90) < threshold_angle:
                horizontal_lines.append((rho, theta))
            elif abs(angle) < threshold_angle:
                vertical_lines.append((rho, theta))
            elif abs(angle - 180) < threshold_angle:
                theta = theta - np.pi
                rho = -rho
                vertical_lines.append((rho, theta))

    return vertical_lines, horizontal_lines, edges

def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])

    if np.linalg.det(A) == 0:
        return None, None  # Lines are parallel and do not intersect

    x, y = np.linalg.solve(A, b)
    return int(np.round(x)), int(np.round(y))

def in_range(x, y, x_min, x_max, y_min, y_max):
    return x_min <= x <= x_max and y_min <= y <= y_max

def group_lines(lines):
    if not lines:
        return []

    grouped_lines = []
    group_id = [0] * len(lines)
    current_group_id = 0
    groups = {}
    for i, line1 in enumerate(lines[:-1]):
        for j, line2 in enumerate(lines[i+1:]):
            
            x, y = intersection(line1, line2)

            # Check if the intersection point is within the image bounds
            if x is not None and in_range(x, y, -100, 600, -100, 800):
                if group_id[i] == 0 and group_id[j+i+1] == 0:
                    #Neither line is in a group
                    
                    current_group_id += 1
                    group_id[i] = current_group_id
                    group_id[j+i+1] = current_group_id

                    groups[current_group_id] = [line1, line2]

                elif group_id[i] != 0 and group_id[j+i+1] == 0:
                    #First line is in a group, second line is not
                    group_id[j+i+1] = group_id[i]

                    groups[group_id[i]].append(line2)
                elif group_id[i] == 0 and group_id[j+i+1] != 0:
                    #First line is not in a group, second line is
                    group_id[i] = group_id[j+i+1]

                    groups[group_id[j+i+1]].append(line1)

                elif group_id[i] != 0 and group_id[j+i+1] != 0:
                    #Both lines are in a group
                    if group_id[i] != group_id[j+i+1]:
                        #Merge groups
                        group1 = groups[group_id[i]]
                        group2 = groups[group_id[j+i+1]]

                        groups[group_id[i]] = group1 + group2

                        for line in group2:
                            group_id[lines.index(line)] = group_id[i]

                        del groups[group_id[j+i+1]]

    for group in groups.values():
        rho = np.mean([line[0] for line in group])
        theta = np.mean([line[1] for line in group])
        grouped_lines.append((rho, theta))

    # Append lines that are not in any group
    for i, line in enumerate(lines):
        if group_id[i] == 0:
            grouped_lines.append(line)

    return grouped_lines

def draw_lines(img, lines, color):
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), color, 2)



img_path = Path("Chessboard_detection", "TestImages", "Temp", "manual_saved.jpg")
img_path_str = str(img_path)
img = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)

vertical_lines, horizontal_lines, edges = detect_lines(img)

# Group lines that are within 10 pixels of each other
grouped_vertical_lines = group_lines(vertical_lines)
grouped_horizontal_lines = group_lines(horizontal_lines)

# Create a copy of the original image to draw lines on
img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw vertical and horizontal lines
draw_lines(img_with_lines, vertical_lines, (0, 255, 0))  # Green for vertical lines
draw_lines(img_with_lines, horizontal_lines, (255, 0, 0))  # Blue for horizontal lines

# Create a copy of the original image to draw lines on
img_with_grouped_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw vertical and horizontal lines
draw_lines(img_with_grouped_lines, grouped_vertical_lines, (0, 255, 0))  # Green for vertical lines
draw_lines(img_with_grouped_lines, grouped_horizontal_lines, (255, 0, 0))  # Blue for horizontal lines

def find_closest_distances(lines):
    distances = []
    for i, (rho1, theta1) in enumerate(lines):
        min_distance = float('inf')
        for j, (rho2, theta2) in enumerate(lines):
            if i != j:
                distance = abs(rho1 - rho2)
                if distance < min_distance:
                    min_distance = distance
        distances.append(min_distance)
    return distances

def discard_outliers(lines, distances, threshold=3):
    median = np.median(distances)
    diff = np.abs(distances - median)
    mad = np.median(diff)
    modified_z_scores = np.abs(0.6745 * diff / mad)
    filtered_lines = [line for line, score in zip(lines, modified_z_scores) if score < threshold]
    return filtered_lines

# Find closest distances for vertical and horizontal lines
vertical_distances = find_closest_distances(grouped_vertical_lines)
horizontal_distances = find_closest_distances(grouped_horizontal_lines)

# Discard outliers
filtered_vertical_lines = discard_outliers(grouped_vertical_lines, vertical_distances)
filtered_horizontal_lines = discard_outliers(grouped_horizontal_lines, horizontal_distances)

# Create a copy of the original image to draw filtered lines on
img_with_filtered_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw filtered vertical and horizontal lines
draw_lines(img_with_filtered_lines, filtered_vertical_lines, (0, 255, 0))  # Green for vertical lines
draw_lines(img_with_filtered_lines, filtered_horizontal_lines, (255, 0, 0))  # Blue for horizontal lines

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(2, 4, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')

plt.subplot(2, 4, 3)
plt.title('Detected Vert and Hor Lines')
plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 4)
plt.title('Grouped Lines')
plt.imshow(cv2.cvtColor(img_with_grouped_lines, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 5)
plt.title('Discard Outliers')
plt.imshow(cv2.cvtColor(img_with_filtered_lines, cv2.COLOR_BGR2RGB))

plt.show()