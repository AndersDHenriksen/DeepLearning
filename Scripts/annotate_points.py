import sys
from pathlib import Path
import cv2
import numpy as np

# Pyinstaller build command: pyinstaller --onefile --icon=pencil.ico annotate_points.py
directory_exe = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent


def mouse_click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points = np.r_[points, np.array([[x, y]])]
        update_points()
        
    if event == cv2.EVENT_RBUTTONDOWN:
        points = points[:-1]
        update_points()
    
    
def update_points():
    global points
    global im_draw
    global im
    
    im_draw = im.copy()
    for point in points:
        cv2.circle(im_draw, (point[0], point[1]), 10, (255, 0, 0), -1)
     
        
data_path = directory_exe
scale_image = 0.5

global im
global im_draw
global points

print("Instructions:\n--Mouse--\nRight: Add point\n Left: Remove last added point\n--Keyboard--\ns: Save points\nn: Next image\nc: Clear points\np: Print points\nq: Quit")
for im_file in data_path.glob('*.png'):
    im = cv2.imread(str(im_file))
    im = cv2.resize(im, (int(im.shape[1]*scale_image), int(im.shape[0]*scale_image)))  # downscale image for viewing
    im_draw = im.copy()

    annotation_filename = data_path / (im_file.stem + '_annotation.npy')
    if annotation_filename.exists():
        print('Loading annotation')
        points = np.load(annotation_filename)
        points = points*scale_image
        points = points.astype(int)
    else:
        points = np.empty((0, 2), dtype=np.int)
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_click)
    
    update_points()
    
    while True:
        cv2.imshow('image', im_draw)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            sys.exit()
            
        elif key == ord('c'):  # clear points
            points = np.empty((0, 2), dtype=np.int)
            update_points()

        elif key == ord('p'):  # print points
            print(points)

        elif key == ord('s'):  # Save and continue
            points_upscaled = (1/scale_image)*np.array(points)  # Scale points to original image size
            np.save(annotation_filename, points_upscaled)
            break
    
        elif key == ord('n'):  # Next - continue without saving
            break

cv2.destroyAllWindows()
