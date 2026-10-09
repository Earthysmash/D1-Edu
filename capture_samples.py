"""Save raw camera frames for manual annotation, one recording session per folder."""
import argparse
import os
import time
from pathlib import Path
from datetime import datetime
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','rtsp_transport;tcp')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='rtsp://127.0.0.1:18554/test')
    p.add_argument('--interval',type=float,default=2)
    p.add_argument('--limit',type=int,default=100)
    args=p.parse_args()
    if args.interval<=0 or args.limit<1:
        p.error('interval and limit must be positive')
    import cv2
    folder=Path('raw_samples')/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    source=int(args.source) if args.source.isdecimal() else args.source
    cap=cv2.VideoCapture(source,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,
                         cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000]) if isinstance(source,str) and source.startswith('rtsp://') else cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            p.exit(1,'Camera unavailable; check PC SSH tunnel\n')
        saved=0; last=-float('inf')
        while saved<args.limit:
            ok,frame=cap.read()
            if not ok:
                p.exit(1,'Camera read failed\n')
            now=time.monotonic()
            if now-last>=args.interval:
                path=folder/f'{saved:05d}.jpg'
                if not cv2.imwrite(str(path),frame):
                    p.exit(1,'Image write failed\n')
                saved+=1; last=now
                print(path,flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__=='__main__':
    main()
