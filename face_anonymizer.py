# face_anonymizer.py
import cv2
import numpy as np
import csv
import os
import time

# ---------- simple IoU + track manager ----------
def iou(boxA, boxB):
    # boxes: x, y, w, h
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2]*boxA[3]
    areaB = boxB[2]*boxB[3]
    union = areaA + areaB - inter + 1e-6
    return inter / union

class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.box = box
        self.missed = 0

def assign_tracks(tracks, detections, iou_thresh=0.3):
    assigned = set()
    used = set()
    # greedy match by IoU
    for ti, t in enumerate(tracks):
        best_j, best_iou = -1, 0
        for j, det in enumerate(detections):
            if j in used: 
                continue
            i = iou(t.box, det)
            if i > best_iou:
                best_iou = i
                best_j = j
        if best_j != -1 and best_iou >= iou_thresh:
            # update track
            t.box = detections[best_j]
            t.missed = 0
            used.add(best_j)
            assigned.add(ti)
        else:
            t.missed += 1
    # new tracks for unmatched detections
    new_tracks = []
    for j, det in enumerate(detections):
        if j not in used:
            new_tracks.append(det)
    return assigned, new_tracks

def mosaic_region(img, x, y, w, h, downscale=0.08):
    x, y = max(0,x), max(0,y)
    x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
    roi = img[y:y2, x:x2]
    if roi.size == 0:
        return img
    # pixelate
    small = cv2.resize(roi, (max(1,int(roi.shape[1]*downscale)), max(1,int(roi.shape[0]*downscale))), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    img[y:y2, x:x2] = pix
    return img

def blur_region(img, x, y, w, h, k=35):
    x, y = max(0,x), max(0,y)
    x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
    roi = img[y:y2, x:x2]
    if roi.size == 0:
        return img
    roi = cv2.GaussianBlur(roi, (k|1, k|1), 0)  # ensure odd kernel
    img[y:y2, x:x2] = roi
    return img

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/input.mp4", help="Video file path or webcam index (0/1/...)")
    parser.add_argument("--out", default="outputs/redacted.mp4", help="Output video path")
    parser.add_argument("--log", default="outputs/anonymization_log.csv", help="CSV log path")
    parser.add_argument("--method", choices=["mosaic","blur"], default="mosaic", help="Anonymization method")
    parser.add_argument("--scale", type=float, default=1.1, help="Haar scaleFactor")
    parser.add_argument("--neighbors", type=int, default=5, help="Haar minNeighbors")
    parser.add_argument("--minsize", type=int, default=32, help="Minimum face size (pixels)")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold for ID matching")
    parser.add_argument("--max_missed", type=int, default=10, help="Frames to keep lost tracks")
    parser.add_argument("--display", action="store_true", help="Show live preview")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # Open source (video file or webcam)
    src = args.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")

    # Haar face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Prepare writer
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    # Tracks + logging
    tracks = []
    next_id = 1
    logf = open(args.log, "w", newline="")
    logger = csv.writer(logf)
    logger.writerow(["frame","id","x","y","w","h"])

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=args.scale, minNeighbors=args.neighbors, minSize=(args.minsize, args.minsize)
        )
        detections = [tuple(map(int, f)) for f in faces]

        # match detections to tracks
        assigned_idx, new_dets = assign_tracks(tracks, detections, iou_thresh=args.iou)

        # prune old tracks
        tracks = [t for t in tracks if t.missed <= args.max_missed]

        # create tracks for new detections
        for det in new_dets:
            tracks.append(Track(next_id, det))
            next_id += 1

        # anonymize + draw
        for t in tracks:
            x, y, w, h = t.box
            if args.method == "mosaic":
                frame = mosaic_region(frame, x, y, w, h, downscale=0.08)
            else:
                frame = blur_region(frame, x, y, w, h, k=41)

            # optional thin box + id for audits (comment out if you don’t want overlays)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
            cv2.putText(frame, f"ID {t.id}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # log
            logger.writerow([frame_idx, t.id, x, y, w, h])

        writer.write(frame)

        if args.display:
            cv2.imshow("Face Anonymizer", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC to quit
                break

        frame_idx += 1

    cap.release()
    writer.release()
    logf.close()
    cv2.destroyAllWindows()
    print(f"Done. Video -> {args.out} | Log -> {args.log} | FPS ~ {frame_idx / (time.time()-t0):.1f}")

if __name__ == "__main__":
    main()
