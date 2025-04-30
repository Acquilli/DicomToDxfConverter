import pydicom as dicom
import ezdxf
import numpy as np
import os
from skimage import measure
from skimage.measure import approximate_polygon
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import re

# Set working directory
working_dir = r"C:\Users\User\Desktop\magisterka\skany_dcm"

# Custom sort function to sort filenames numerically
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    if numbers:
        return int(numbers[-1])
    else:
        return value

# List DICOM files
dicom_files = [os.path.join(working_dir, f) for f in sorted(os.listdir(working_dir), key=numerical_sort) if f.endswith(".dcm")]

print("Lista posortowanych plików DICOM:")
for i, file in enumerate(dicom_files):
    print(f"{i}: {file}")

# Create folder for DXF files (if not exists)
dxf_folder = r"C:\Users\User\Desktop\magisterka\skany_dxf"

if not os.path.exists(dxf_folder):
    os.makedirs(dxf_folder)

print("DICOM files found:", dicom_files)
print("Destination folder for DXF files:", dxf_folder)

# Prompt user to select starting file index
start_index = int(input(f"Enter the starting index (0 to {len(dicom_files) - 1}): ")) -1
if start_index < 0 or start_index >= len(dicom_files):
    print("Invalid index. Exiting.")
    exit()

# Filter DICOM files based on the starting index
dicom_files = dicom_files[start_index:]

# Initial contour intensity threshold
initial_contour_threshold = 0.46
# Tolerance for contour simplification
simplification_tolerance = 0.5

# Sigma value for Gaussian filter
sigma_value = 1.0
# Minimum contour length to keep
min_contour_length = 30

def update_contours(image, threshold):
    smoothed_image = gaussian_filter(image, sigma=sigma_value)
    contours = measure.find_contours(smoothed_image, threshold)
    # Filter out small contours
    contours = [contour for contour in contours if len(contour) >= min_contour_length]
    return contours

def process_dicom_file(dicom_file):
    print(f"Processing {dicom_file}")

    # Read DICOM file
    ds = dicom.read_file(dicom_file)

    # Check if pixel_array is None
    if ds.pixel_array is None:
        print(f"Skipping {dicom_file}: No pixel data found.")
        return [], None

    # Get pixel values
    image_data = ds.pixel_array.astype(np.float32)

    # Invert and rotate the image
    image_data = np.rot90(image_data)
    image_data = np.flipud(image_data)
    image_data = np.fliplr(image_data)

    # Normalize pixel values to range 0-1
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Enhance contrast
    image_data = np.clip(image_data * 1.5, 0, 1)

    # Initial contours
    contours = update_contours(image_data, initial_contour_threshold)

    # Display and adjust contours globally
    selected_contours = adjust_contours_globally(image_data, contours, dicom_file)

    return selected_contours, image_data

def adjust_contours_globally(image, contours, dicom_file):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(image, cmap='gray')

    contour_lines = []
    deleted_contours = []
    initial_contours = [contour.copy() for contour in contours]
    selected_line = None
    selected_point = None
    move_history = []
    selected_contours = []
    is_selecting = False
    is_cutting = False
    is_splitting = False
    is_drawing = False
    cut_points = []
    split_points = []
    drawing_points = []  # List of points for drawing custom lines

    def plot_contours(contours):
        for contour in contours:
            line, = ax.plot(contour[:, 1], contour[:, 0], linewidth=1, picker=True, color='blue')
            line.set_pickradius(5)
            contour_lines.append((line, contour))

    plot_contours(contours)

    def onclick(event):
        nonlocal selected_line, selected_point, cut_points, split_points, drawing_points
        if event.button == 1:  # Left click
            if is_drawing:  # In drawing mode, add points for the custom line
                if event.xdata is not None and event.ydata is not None:
                    drawing_points.append((event.xdata, event.ydata))
                    fig.canvas.draw_idle()
                    if len(drawing_points) == 2:
                        draw_custom_line()
            elif is_cutting:  # Cutting mode
                for line, contour in contour_lines:
                    if line.contains(event)[0]:
                        cut_points.append((line, contour, event.xdata, event.ydata))
                        if len(cut_points) == 2:
                            cut_contour()
                        break
            elif is_selecting:  # Selecting mode
                for line, contour in contour_lines:
                    if line.contains(event)[0]:
                        if (line, contour) in selected_contours:
                            selected_contours.remove((line, contour))
                            line.set_color('blue')
                        else:
                            selected_contours.append((line, contour))
                            line.set_color('red')
                        fig.canvas.draw_idle()
                        break
            elif is_splitting:  # Splitting mode
                for line, contour in contour_lines:
                    if line.contains(event)[0]:
                        split_points.append((line, contour, event.xdata, event.ydata))
                        if len(split_points) == 2:
                            split_contour()
                        break
            else:  # If not in cutting or selecting mode, delete the contour
                for line, contour in contour_lines:
                    if line.contains(event)[0]:
                        line.remove()
                        contour_lines.remove((line, contour))
                        deleted_contours.append((line, contour))
                        move_history.append(('delete', line, contour))
                        fig.canvas.draw_idle()
                        break
        elif event.button == 3:  # Right click to select a point on the contour
            for line, contour in contour_lines:
                if line.contains(event)[0]:
                    selected_line = line
                    # Find the closest point on the contour to the click
                    distances = np.sqrt((contour[:, 1] - event.xdata) ** 2 + (contour[:, 0] - event.ydata) ** 2)
                    selected_point = np.argmin(distances)
                    break

    def ondblclick(event):
        if event.dblclick:
            # Add a new contour as a small circle at the clicked point
            x, y = event.xdata, event.ydata
            radius = 1
            theta = np.linspace(0, 2 * np.pi, 100)
            new_contour = np.column_stack((y + radius * np.sin(theta), x + radius * np.cos(theta)))
            line, = ax.plot(new_contour[:, 1], new_contour[:, 0], 'b-', linewidth=1, picker=True)
            contour_lines.append((line, new_contour))
            fig.canvas.draw_idle()
            move_history.append(('add', line, new_contour))

    def onrelease(event):
        nonlocal selected_line, selected_point
        selected_line = None
        selected_point = None

    def draw_custom_line():
        nonlocal drawing_points
        if len(drawing_points) == 2:
            # Original points where the line was supposed to start and end
            x1, y1 = drawing_points[0]
            x2, y2 = drawing_points[1]

            # Snap the starting and ending points to the nearest endpoints of existing contours
            snapped_start, start_distance = find_nearest_endpoint(x1, y1, threshold=10)
            snapped_end, end_distance = find_nearest_endpoint(x2, y2, threshold=10)

            if start_distance < 10:
                x1, y1 = snapped_start
            if end_distance < 10:
                x2, y2 = snapped_end

            # Create a new line contour with these fixed endpoints and a movable midpoint
            num_points = 20  # Number of points in the middle segment
            x_middle = np.linspace(x1, x2, num_points)[1:-1]
            y_middle = np.linspace(y1, y2, num_points)[1:-1]

            # Create a new line with fixed endpoints and adjustable middle points
            new_line = np.vstack([[y1, x1], np.column_stack([y_middle, x_middle]), [y2, x2]])

            # Plot the new line on the graph
            line, = ax.plot(new_line[:, 1], new_line[:, 0], 'b-', linewidth=1, picker=True)
            contour_lines.append((line, new_line))

            fig.canvas.draw_idle()
            drawing_points = []  # Clear the drawing points after use

    def find_nearest_endpoint(x, y, threshold=10):
        """
        Finds the nearest endpoint on existing contours to the given point (x, y).
        If the distance to the nearest endpoint is within the specified threshold, 
        it snaps the point to that endpoint; otherwise, it returns the original point.
        """
        nearest_endpoint = (x, y)
        min_distance = float('inf')
        for _, contour in contour_lines:
            # Consider only endpoints (first and last points of the contour)
            endpoints = [contour[0], contour[-1]]
            for point in endpoints:
                distance = np.sqrt((point[1] - x) ** 2 + (point[0] - y) ** 2)
                if distance < min_distance and distance <= threshold:
                    min_distance = distance
                    nearest_endpoint = (point[1], point[0])
        return nearest_endpoint, min_distance

    def smooth_contour_segment(contour, start, end, iterations=5):
        # Apply a more aggressive smoothing algorithm to the specified segment of the contour
        segment = contour[start:end]
        for _ in range(iterations):
            smoothed_segment = np.copy(segment)
            smoothed_segment[1:-1] = 0.5 * segment[1:-1] + 0.25 * (segment[:-2] + segment[2:])
            segment = smoothed_segment
        contour[start:end] = segment
        return contour

    def onmotion(event):
        if selected_line is not None and selected_point is not None and event.button == 3:
            # Update the position of the selected point and nearby points
            for line, contour in contour_lines:
                if line == selected_line:
                    original_contour = contour.copy()

                    # Ensure endpoints are fixed
                    if selected_point in [0, len(contour) - 1]:
                        continue  # Skip editing if it's an endpoint

                    dx = event.xdata - contour[selected_point, 1]
                    dy = event.ydata - contour[selected_point, 0]

                    # Influence function to affect only points on the selected line
                    influence_radius = 5  # Radius within which points are affected
                    distances = np.sqrt((contour[:, 1] - contour[selected_point, 1]) ** 2 + (contour[:, 0] - contour[selected_point, 0]) ** 2)
                    influence = np.exp(-distances**2 / (2 * influence_radius**2))

                    # Update all points on the selected line
                    for i in range(len(contour)):  # Allow influence on all points except endpoints
                        if i not in [0, len(contour) - 1]:
                            contour[i, 1] += dx * influence[i]  # Apply influence to x-coordinate
                            contour[i, 0] += dy * influence[i]  # Apply influence to y-coordinate

                    # Smooth the modified segment of the contour to avoid sharp peaks
                    start = max(0, selected_point - 10)
                    end = min(len(contour), selected_point + 10)
                    contour = smooth_contour_segment(contour, start, end)

                    line.set_xdata(contour[:, 1])
                    line.set_ydata(contour[:, 0])
                    fig.canvas.draw_idle()

                    # Save move history for undo functionality
                    move_history.append(('move', line, original_contour, contour.copy()))
                    break

    def undo(event):
        if move_history:
            action = move_history.pop()
            if action[0] == 'move':
                line, original_contour = action[1], action[2]
                line.set_xdata(original_contour[:, 1])
                line.set_ydata(original_contour[:, 0])
            elif action[0] == 'delete':
                line, contour = action[1], action[2]
                line, = ax.plot(contour[:, 1], contour[:, 0], linewidth=1, picker=True, color='blue')
                contour_lines.append((line, contour))
            elif action[0] == 'add' or action[0] == 'draw':
                line = action[1]
                line.remove()
                contour_lines.remove((line, action[2]))
            fig.canvas.draw_idle()

    def restart(event):
        nonlocal contour_lines
        nonlocal contours
        for line, _ in contour_lines:
            line.remove()
        contour_lines.clear()
        contours = [contour.copy() for contour in initial_contours]
        plot_contours(contours)
        global_thresh_slider.set_val(initial_contour_threshold)
        fig.canvas.draw_idle()
        move_history.clear()

    def merge_contours(event):
        nonlocal selected_contours
        if len(selected_contours) < 2:
            print("Select at least two contours to merge.")
            return

        # Extract the contours
        contours_to_merge = [contour for _, contour in selected_contours]

        # Merge the contours using convex hull
        merged_contour = merge_multiple_contours(contours_to_merge)

        # Close the merged contour
        merged_contour = np.vstack([merged_contour, merged_contour[0]])

        # Store the original contours for undo
        original_contours = selected_contours.copy()

        # Remove the selected contours
        for line, contour in selected_contours:
            line.remove()
            contour_lines.remove((line, contour))
        selected_contours.clear()

        # Add the new merged contour
        line, = ax.plot(merged_contour[:, 1], merged_contour[:, 0], 'b-', linewidth=1, picker=True)
        contour_lines.append((line, merged_contour))
        move_history.append(('merge', line, original_contours))
        fig.canvas.draw_idle()

    def merge_multiple_contours(contours):
        points = np.concatenate(contours)
        hull = ConvexHull(points)
        return points[hull.vertices]

    def cut_contour():
        nonlocal cut_points
        (line1, contour1, x1, y1), (line2, contour2, x2, y2) = cut_points
        if line1 != line2:
            print("Please select two points on the same contour.")
            cut_points = []
            return

        contour = contour1
        distances1 = np.sqrt((contour[:, 1] - x1) ** 2 + (contour[:, 0] - y1) ** 2)
        distances2 = np.sqrt((contour[:, 1] - x2) ** 2 + (contour[:, 0] - y2) ** 2)
        idx1, idx2 = np.argmin(distances1), np.argmin(distances2)

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # Define new_contour1 and new_contour2 to start at the intersection points
        new_contour1 = contour[idx1:idx2 + 1].copy()  # From idx1 to idx2, inclusive
        new_contour2 = np.concatenate([contour[idx2:], contour[:idx1 + 1]]).copy()  # From idx2 to end, then start to idx1

        # Plot new contours with different colors for differentiation
        line1_new, = ax.plot(new_contour1[:, 1], new_contour1[:, 0], 'b-', linewidth=1, picker=False, marker=None)
        line2_new, = ax.plot(new_contour2[:, 1], new_contour2[:, 0], 'b-', linewidth=1, picker=False, marker=None)

        # Add new lines and contours to the list
        contour_lines.append((line1_new, new_contour1))
        contour_lines.append((line2_new, new_contour2))

        fig.canvas.draw_idle()

        def onclick_remove(event):
            # Check if the click event is within either line1_new or line2_new
            if line1_new.contains(event)[0]:
                # Check the color of the clicked line
                if line1_new.get_color() == 'b':  # Blue contour
                    line1_new.remove()
                    contour_lines.remove((line1_new, new_contour1))
                    move_history.append(('remove', (line1_new, new_contour1), None))
            elif line2_new.contains(event)[0]:
                # Check the color of the clicked line
                if line2_new.get_color() == 'b':  # Red contour
                    line2_new.remove()
                    contour_lines.remove((line2_new, new_contour2))
                    move_history.append(('remove', (line2_new, new_contour2), None))

            # Disconnect the event after removal to prevent further action
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.draw_idle()

        # Connect the click event to the onclick_remove function
        cid = fig.canvas.mpl_connect('button_press_event', onclick_remove)
        cut_points = []

    def split_contour():
        nonlocal split_points
        (line1, contour1, x1, y1), (line2, contour2, x2, y2) = split_points
        if line1 != line2:
            print("Please select two points on the same contour.")
            split_points = []
            return

        contour = contour1
        distances1 = np.sqrt((contour[:, 1] - x1) ** 2 + (contour[:, 0] - y1) ** 2)
        distances2 = np.sqrt((contour[:, 1] - x2) ** 2 + (contour[:, 0] - y2) ** 2)
        idx1, idx2 = np.argmin(distances1), np.argmin(distances2)

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # Create two new closed contours
        new_contour1 = np.concatenate([contour[:idx1+1], contour[idx2:], [contour[0]]])
        new_contour2 = np.concatenate([contour[idx1:idx2+1], [contour[idx1]]])

        line1.remove()
        contour_lines.remove((line1, contour1))

        # Draw new closed contours
        line1, = ax.plot(new_contour1[:, 1], new_contour1[:, 0], 'b-', linewidth=1, picker=True)
        line2, = ax.plot(new_contour2[:, 1], new_contour2[:, 0], 'b-', linewidth=1, picker=True)

        contour_lines.append((line1, new_contour1))
        contour_lines.append((line2, new_contour2))
        move_history.append(('split', line1, line2, contour1))

        fig.canvas.draw_idle()
        split_points = []

    def on_key_press(event):
        nonlocal is_selecting, is_cutting, is_splitting, is_drawing
        if event.key == 'm':
            is_selecting = True
        elif event.key == 't':
            is_cutting = True
        elif event.key == 'x':
            is_splitting = True
        elif event.key == 'd':
            is_drawing = True  # Activate drawing mode

    def on_key_release(event):
        nonlocal is_selecting, is_cutting, is_splitting, is_drawing
        if event.key == 'm':
            is_selecting = False
        elif event.key == 't':
            is_cutting = False
            if len(cut_points) == 2:
                cut_contour()
        elif event.key == 'x':
            is_splitting = False
            if len(split_points) == 2:
                split_contour()
        elif event.key == 'd':
            is_drawing = False  # Deactivate drawing mode

    # Disable default matplotlib key bindings for 'c'
    plt.rcParams['keymap.save'] = ['ctrl+s']  # Change 'c' to 'ctrl+s' for save
    plt.rcParams['keymap.fullscreen'] = ['ctrl+f']  # Change 'f' to 'ctrl+f' for fullscreen
    plt.rcParams['keymap.pan'] = ['p']  # Change 'p' to 'p' for pan
    plt.rcParams['keymap.zoom'] = ['z']  # Change 'o' to 'z' for zoom

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect('motion_notify_event', onmotion)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    fig.canvas.mpl_connect('button_press_event', ondblclick)

    # Slider for global contour threshold
    ax_thresh = plt.axes([0.2, 0.1, 0.65, 0.03])
    global_thresh_slider = Slider(ax_thresh, 'Global Threshold', 0.0, 1.0, valinit=initial_contour_threshold, valstep=0.001)

    def update_global(val):
        threshold = global_thresh_slider.val
        global_contours = update_contours(image, threshold)
        for line, _ in contour_lines:
            line.remove()
        contour_lines.clear()
        plot_contours(global_contours)
        fig.canvas.draw_idle()

    global_thresh_slider.on_changed(update_global)

    # Button for undoing the last deletion or move
    ax_undo = plt.axes([0.59, 0.025, 0.1, 0.04])
    undo_button = Button(ax_undo, 'Undo', color='lightgoldenrodyellow', hovercolor='0.975')
    undo_button.on_clicked(undo)

    # Button for restarting the entire process
    ax_restart = plt.axes([0.47, 0.025, 0.1, 0.04])
    restart_button = Button(ax_restart, 'Restart', color='lightgoldenrodyellow', hovercolor='0.975')
    restart_button.on_clicked(restart)

    # Button for finalizing selection
    ax_button = plt.axes([0.71, 0.025, 0.1, 0.04])
    button = Button(ax_button, 'Finalize', color='lightgoldenrodyellow', hovercolor='0.975')

    ax_merge = plt.axes([0.25, 0.025, 0.1, 0.04])
    b_merge = Button(ax_merge, 'Merge')
    b_merge.on_clicked(merge_contours)

    def finalize(event):
        # Save the DXF file
        doc = ezdxf.new()
        msp = doc.modelspace()

        for line, contour in contour_lines:
            contour = np.column_stack((line.get_xdata(), line.get_ydata()))
            # Simplify the contour
            simplified_contour = approximate_polygon(contour, tolerance=simplification_tolerance)
            if len(simplified_contour) < 3:
                continue  # Skip too small contours

            # Add lines for the simplified contour, ensuring continuity
            for i in range(len(simplified_contour)):
                x_start, y_start = simplified_contour[i]
                x_end, y_end = simplified_contour[(i + 1) % len(simplified_contour)]
                msp.add_line((x_start, y_start, 0), (x_end, y_end, 0), dxfattribs={'color': 7, 'layer': 'Contours'})

        # Add the frame contour
        height, width = image.shape
        frame_contour = np.array([
            [0, 0],
            [0, height-1],
            [width-1, height-1],
            [width-1, 0],
            [0, 0]
        ])
        for i in range(len(frame_contour) - 1):
            x_start, y_start = frame_contour[i]
            x_end, y_end = frame_contour[i + 1]
            msp.add_line((x_start, y_start, 0), (x_end, y_end, 0), dxfattribs={'color': 1, 'layer': 'Frame'})

        output_dxf_file = os.path.join(dxf_folder, f"{os.path.splitext(os.path.basename(dicom_file))[0]}.dxf")
        doc.saveas(output_dxf_file)
        print(f"DXF file saved to {output_dxf_file}")
        plt.close(fig)

    button.on_clicked(finalize)
    plt.show()

    return [np.column_stack((line.get_xdata(), line.get_ydata())) for line, contour in contour_lines if line.get_visible()]

# Function to convert DICOM to DXF with global segmentation adjustment
def convert_dicom_to_dxf_with_global_adjustment(dicom_files, dxf_folder, simplification_tolerance, sigma_value):
    try:
        for index, dicom_file in enumerate(dicom_files):
            selected_contours, image = process_dicom_file(dicom_file)

            if selected_contours:
                # Create DXF drawing
                doc = ezdxf.new()
                msp = doc.modelspace()

                # Add contours to DXF drawing
                for contour in selected_contours:
                    # Simplify the contour
                    simplified_contour = approximate_polygon(contour, tolerance=simplification_tolerance)
                    if len(simplified_contour) < 3:
                        continue  # Skip too small contours

                    # Add lines for the simplified contour, ensuring continuity
                    for i in range(len(simplified_contour)):
                        x_start, y_start = simplified_contour[i]
                        x_end, y_end = simplified_contour[(i + 1) % len(simplified_contour)]
                        msp.add_line((x_start, y_start), (x_end, y_end), dxfattribs={'color': 7, 'layer': 'Contours'})

                # Add the frame contour
                height, width = image.shape
                frame_contour = np.array([
                    [0, 0],
                    [0, height-1],
                    [width-1, height-1],
                    [width-1, 0],
                    [0, 0]
                ])
                for i in range(len(frame_contour) - 1):
                    x_start, y_start = frame_contour[i]
                    x_end, y_end = frame_contour[i + 1]
                    msp.add_line((x_start, y_start), (x_end, y_end), dxfattribs={'color': 1, 'layer': 'Frame'})

                # Create path to DXF file
                dxf_filename = f"{os.path.splitext(os.path.basename(dicom_file))[0]}_combined_model.dxf"
                dxf_file = os.path.join(dxf_folder, dxf_filename)

                # Save DXF file
                doc.saveas(dxf_file)
                print(f"Combined model saved to {dxf_file}")

    except FileNotFoundError as e:
        print(f"Error: DICOM file not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Convert and combine DICOM files
if dicom_files:
    convert_dicom_to_dxf_with_global_adjustment(dicom_files, dxf_folder, simplification_tolerance, sigma_value)
else:
    print("No DICOM files found in the specified directory.")
 