import cv2
import json
import os
import argparse
from pathlib import Path
import glob

roi_pts = []
line_pts = []
drawing_mode = "polygon"

def mouse_callback(event, x, y, flags, param):
    global roi_pts, line_pts, drawing_mode
    frame_disp = param['frame'].copy()
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_mode == "polygon":
            roi_pts.append([x, y])
        elif drawing_mode == "line":
            if len(line_pts) < 2:
                line_pts.append([x, y])
                
    if len(roi_pts) > 0:
        for i in range(len(roi_pts)-1):
            cv2.line(frame_disp, tuple(roi_pts[i]), tuple(roi_pts[i+1]), (0, 255, 0), 2)
        cv2.line(frame_disp, tuple(roi_pts[-1]), tuple(roi_pts[0]), (0, 255, 0), 2)
        for pt in roi_pts:
            cv2.circle(frame_disp, tuple(pt), 4, (0, 0, 255), -1)
            
    if len(line_pts) > 0:
        for pt in line_pts:
            cv2.circle(frame_disp, tuple(pt), 4, (255, 0, 0), -1)
        if len(line_pts) == 2:
            cv2.line(frame_disp, tuple(line_pts[0]), tuple(line_pts[1]), (255, 0, 0), 2)
            
    cv2.putText(frame_disp, f"Mode: {drawing_mode} ('d'=switch, 'c'=clear, 'Enter'=save)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(param['win_name'], frame_disp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    args = parser.parse_args()
    
    out_json = Path(__file__).parent / "config" / "camera_rois.json"
    
    if out_json.exists():
        with open(out_json, "r") as f:
            db = json.load(f)
    else:
        db = {}
        
    video_files = glob.glob(os.path.join(args.video_dir, "*.mp4")) + glob.glob(os.path.join(args.video_dir, "*.dav"))
    
    global roi_pts, line_pts, drawing_mode
    
    for vpath in video_files:
        basename = os.path.basename(vpath)
        if basename in db:
            print(f"Skipping {basename}, already in JSON.")
            continue
            
        cap = cv2.VideoCapture(vpath)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
            
        roi_pts = []
        line_pts = []
        drawing_mode = "polygon"
        
        win_name = f"Draw for {basename}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        param = {'frame': frame, 'win_name': win_name}
        cv2.setMouseCallback(win_name, mouse_callback, param)
        
        mouse_callback(0, 0, 0, 0, param)
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == 13: # Enter
                if len(roi_pts) >= 3 and len(line_pts) == 2:
                    db[basename] = {
                        "road_polygon": roi_pts,
                        "counting_line": line_pts
                    }
                    print(f"Saved for {basename}")
                    break
                else:
                    print("Need at least 3 points for polygon and exactly 2 for line!")
            elif key == ord('d'):
                drawing_mode = "line" if drawing_mode == "polygon" else "polygon"
                mouse_callback(0, 0, 0, 0, param)
            elif key == ord('c'):
                if drawing_mode == "polygon": roi_pts = []
                else: line_pts = []
                mouse_callback(0, 0, 0, 0, param)
            elif key == 27: # Esc
                print("Exiting...")
                cv2.destroyAllWindows()
                return
                
        cv2.destroyWindow(win_name)
        
        with open(out_json, "w") as f:
            json.dump(db, f, indent=4)

if __name__ == "__main__":
    main()
