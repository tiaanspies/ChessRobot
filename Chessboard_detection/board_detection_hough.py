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


def group_lines(lines, image_width, image_height):
    if not lines:
        return []

    grouped_lines = []
    group_id = [0] * len(lines)  # Initialize group IDs for each line
    current_group_id = 0  # Initialize current group ID
    groups = {}  # Dictionary to store groups of lines

    int_width_min = 0 - image_width*0.2
    int_width_max = image_width*1.2
    int_height_min = 0 - image_height*0.2
    int_height_max = image_height*1.2

    def add_to_group(i, j):
        nonlocal current_group_id
        if group_id[i] == 0 and group_id[j] == 0:
            # If both lines are not in any group, create a new group
            current_group_id += 1
            group_id[i] = current_group_id
            group_id[j] = current_group_id
            groups[current_group_id] = [lines[i], lines[j]]
        elif group_id[i] != 0 and group_id[j] == 0:
            # If line i is in a group and line j is not, add line j to the group of line i
            group_id[j] = group_id[i]
            groups[group_id[i]].append(lines[j])
        elif group_id[i] == 0 and group_id[j] != 0:
            # If line j is in a group and line i is not, add line i to the group of line j
            group_id[i] = group_id[j]
            groups[group_id[j]].append(lines[i])
        elif group_id[i] != 0 and group_id[j] != 0 and group_id[i] != group_id[j]:
            # If both lines are in different groups, merge the groups
            merge_groups(group_id[i], group_id[j])

    def merge_groups(id1, id2):
        # Merge two groups into one
        group1 = groups[id1]
        group2 = groups[id2]
        groups[id1] = group1 + group2
        for line in group2:
            group_id[lines.index(line)] = id1
        del groups[id2]

    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            x, y = intersection(lines[i], lines[j])
            if not (x is None or y is None) and in_range(x, y, int_width_min, int_width_max, int_height_min, int_height_max):
                # If the lines intersect within the specified range, add them to a group
                add_to_group(i, j)

    for group in groups.values():
        # Calculate the average rho and theta for each group
        rho = np.mean([line[0] for line in group])
        theta = np.mean([line[1] for line in group])
        grouped_lines.append((rho, theta))

    for i, line in enumerate(lines):
        if group_id[i] == 0:
            # If a line is not in any group, add it as a separate line
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

def sort_lines(lines):
    return sorted(lines, key=lambda x: x[0])

img_path = Path("Chessboard_detection", "TestImages", "Temp", "manual_saved.jpg")
img_path_str = str(img_path)
img = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
image_height, image_width  = img.shape

vertical_lines, horizontal_lines, edges = detect_lines(img)

# Group lines that are within 10 pixels of each other
grouped_vertical_lines = group_lines(vertical_lines, image_width, image_height)
grouped_horizontal_lines = group_lines(horizontal_lines, image_width, image_height)

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

def find_all_intersections(lines_vertical, lines_horizontal):
    """
    Find all intersections between lines
    :param lines_vertical: List of vertical lines in the format (rho, theta)
    :param lines_horizontal: List of horizontal lines in the format (rho, theta)
    :return: List of intersection points in the format (x, y)
    """
    intersection_points = []
    for line1 in lines_horizontal:
        for line2 in lines_vertical:
            x, y = intersection(line1, line2)
            if x is not None and y is not None:
                intersection_points.append((x, y))

    # Reshape intersection points to be 7x7x2
    intersection_points = np.array(intersection_points)

    return intersection_points

def calculate_rho_theta(point1, point2):
    """
    Calculate rho and theta for a line passing through two points
    :param point1: First point (x1, y1)
    :param point2: Second point (x2, y2)
    :return: (rho, theta) for the line passing through the points
    """
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:  # Vertical line
        theta = np.pi / 2
        rho = x1
    else:
        # Calculate the slope and intercept of the line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Calculate theta
        theta = np.arctan(-1 / slope)
        if theta < 0:
            theta += np.pi

        # Calculate rho
        rho = intercept * np.sin(theta)

    return rho, theta

def expand_board_pts(int_points, vertical_lines, horizontal_lines):
    """
    Expand the intersection points to a 9x9 grid
    :param int_points: List of intersection points in the shape (7, 7, 2) format (x, y)
    :param vertical_lines: List of vertical lines in the format (rho, theta)
    :param horizontal_lines: List of horizontal lines in the format (rho, theta)
    :return: List of expanded points in the shape (9, 9, 2) format (x, y)
    """

    assert int_points.shape == (7, 7, 2)

    expanded_points = np.zeros((9, 9, 2), dtype=int)

    # Copy the original 7x7 points to the center of the 9x9 grid
    expanded_points[1:8, 1:8] = int_points
    
    # Plot diffs on a scatter plot and fit a best fit line
    anti_diag_diffs = np.array([int_points[i + 1, 5 - i] - int_points[i, 6 - i] for i in range(6)])

    def fit_line_and_get_offsets(diffs, start_idx, end_idx):
        x = np.arange(len(diffs))
        y_x_diffs = diffs[:, 0]  # x-differences
        m_x, b_x = np.polyfit(x, y_x_diffs, 1)
        x_diff_start = m_x * start_idx + b_x
        x_diff_end = m_x * end_idx + b_x

        y_y_diffs = diffs[:, 1]  # y-differences
        m_y, b_y = np.polyfit(x, y_y_diffs, 1)
        y_diff_start = m_y * start_idx + b_y
        y_diff_end = m_y * end_idx + b_y

        return np.array([x_diff_start, y_diff_start]), np.array([x_diff_end, y_diff_end])

    # Fit lines and get offsets for anti-diagonal
    top_right_offset, bot_left_offset = fit_line_and_get_offsets(anti_diag_diffs, -1, 6)

    # Calculate mean differences for diagonals
    diag_diffs = np.array([int_points[i + 1, i + 1] - int_points[i, i] for i in range(6)])

    # Fit lines and get offsets for diagonal
    top_left_offset, bot_right_offset = fit_line_and_get_offsets(diag_diffs, -1, 6)

    # Expand corners
    expanded_points[0, 0] = expanded_points[1, 1] - top_left_offset
    expanded_points[8, 8] = expanded_points[7, 7] + bot_right_offset
    expanded_points[0, 8] = expanded_points[1, 7] - top_right_offset
    expanded_points[8, 0] = expanded_points[7, 1] + bot_left_offset

    # Calculate rho and theta for the border lines passing through the outer corners
    top_left = expanded_points[0, 0]
    top_right = expanded_points[0, 8]
    bottom_left = expanded_points[8, 0]
    bottom_right = expanded_points[8, 8]

    top_border = calculate_rho_theta(top_left, top_right)
    bottom_border = calculate_rho_theta(bottom_left, bottom_right)
    left_border = calculate_rho_theta(top_left, bottom_left)
    right_border = calculate_rho_theta(top_right, bottom_right)

    left_points = find_all_intersections([left_border], horizontal_lines)
    right_points = find_all_intersections([right_border], horizontal_lines)
    top_points = find_all_intersections([top_border], vertical_lines)
    bottom_points = find_all_intersections([bottom_border], vertical_lines)

    # Expand the border points to the 9x9 grid
    expanded_points[0, 1:8] = top_points
    expanded_points[8, 1:8] = bottom_points
    expanded_points[1:8, 0] = left_points
    expanded_points[1:8, 8] = right_points

    return expanded_points


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

# sort lines
sorted_vertical_lines = sort_lines(filtered_vertical_lines)
sorted_horizontal_lines = sort_lines(filtered_horizontal_lines)

intersection_points = find_all_intersections(sorted_vertical_lines, sorted_horizontal_lines).reshape(7, 7, 2)

# expand points to 9x9 grid
expanded_points = expand_board_pts(intersection_points, sorted_vertical_lines, sorted_horizontal_lines)

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

plt.subplot(2, 4, 6)
plt.title('Intersection points')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Plot intersection points
for (x, y) in intersection_points.reshape(-1, 2):
    plt.plot(x, y, 'r.')  # Red dots for intersection points

plt.subplot(2, 4, 7)
plt.title('Expanded points')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Plot expanded points
for (x, y) in expanded_points.reshape(-1, 2):
    plt.plot(x, y, 'r.')  # Red dots for expanded points

plt.show()
