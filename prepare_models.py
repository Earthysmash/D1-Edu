"""Download official models on the PC while Internet is available."""
import argparse
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-pose',action='store_true')
    args=p.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    from ultralytics import YOLO
    models=['yolov8n.pt'] + ([] if args.skip_pose else ['yolov8n-pose.pt'])
    for name in models:
        try:
            model=YOLO(name)
            print('READY:',Path(name).resolve(), 'task:',model.task)
        except Exception as exc:
            p.exit(1,f'Could not prepare {name}: {exc}\nConnect PC to Internet and retry.\n')
    print('Gun and face-covering custom weights are NOT included.')


if __name__=='__main__':
    main()
