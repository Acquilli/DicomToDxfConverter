# DicomToDxfConverter

## How It Works 🛠️ 
This Python script provides an interactive interface to convert medical DICOM images into DXF vector drawings. The main workflow includes image preprocessing, contour detection, interactive editing, and exporting vector shapes.

### 1. DICOM File Loading 📂 
The script scans a specified directory for .dcm files (DICOM images).

Files are numerically sorted to preserve scan order.

### 2. Image Preprocessing 🎚️ 
Pixel data from each DICOM file is normalized and enhanced for better contrast.

A Gaussian filter is applied to smooth the image.

Contours are detected using intensity thresholding.

### 3. Interactive Contour Editing 🖱️ 
A user interface with matplotlib allows manual adjustments of the automatically detected contours:

Interaction Modes:

Left Click: select, draw, cut, or delete contours depending on the mode

Right Click + Drag: edit individual points on contours

Double Click: add a new contour

Keyboard Shortcuts:

m: Select multiple contours for merging

t: Cut a contour between two points

x: Split a contour between two points

d: Draw a custom line between two endpoints

UI Controls:

Slider: Adjust global contour threshold

Buttons: Undo, Restart, Finalize, Merge

### 4. Contour Operations 🧠 
Simplification: Each contour is simplified using the Douglas-Peucker algorithm.

Merge: Selected contours can be merged into a convex hull.

Cut/Split: Contours can be split or cut interactively.

Draw: Custom lines can be drawn and snapped to existing endpoints.

### 5. Export to DXF 📤 
Finalized contours are exported to DXF format using ezdxf.

Each contour becomes a closed polyline in the DXF.

A frame representing the image border is also added.

### Notes 💡 
Contours with fewer than 3 points are skipped to ensure valid geometry.

You are prompted to select a starting index to allow partial processing of DICOM sequences.
