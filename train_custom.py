"""Validate a YOLO detection dataset, train, or evaluate local weights."""
import argparse
import hashlib
import math
from pathlib import Path


def check_dataset(yaml_path, require_test=False):
    import yaml
    spec_path=Path(yaml_path).resolve()
    data=yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    names=data.get('names', {})
    if isinstance(names, list):
        names=dict(enumerate(names))
    if not names or set(names)!=set(range(len(names))):
        raise ValueError('names must use consecutive integer IDs starting at 0')
    if len(set(names.values()))!=len(names):
        raise ValueError('Class names must be unique')
    root=Path(data.get('path','.'))
    if not root.is_absolute():
        root=(spec_path.parent/root).resolve()
    seen={}
    summary={}
    splits=['train','val']+(['test'] if require_test or data.get('test') else [])
    for split in splits:
        if not data.get(split):
            raise ValueError(f'Missing {split} split')
        directory=root/data[split]
        images=sorted(p for p in directory.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
        if not images:
            raise ValueError(f'No images in {directory}')
        boxes=0
        negatives=0
        classes=set()
        for image in images:
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f'Duplicate image across splits: {image}')
            seen[digest]=split
            relative=image.relative_to(root)
            parts=list(relative.parts)
            if parts[0]!='images':
                raise ValueError('Use images/train, images/val, images/test layout')
            parts[0]='labels'
            label=(root/Path(*parts)).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'Missing {label}; use an empty txt for a reviewed negative image')
            lines=[line for line in label.read_text().splitlines() if line.strip()]
            if not lines:
                negatives+=1
            for line in lines:
                fields=line.split()
                if len(fields)!=5:
                    raise ValueError(f'{label}: expected class x y w h (detection only)')
                cls=int(fields[0]); x,y,w,h=map(float,fields[1:])
                if cls not in names or not all(math.isfinite(v) for v in (x,y,w,h)):
                    raise ValueError(f'{label}: invalid class or nonfinite coordinates')
                if not (0<w<=1 and 0<h<=1 and x-w/2>=-1e-5 and y-h/2>=-1e-5
                        and x+w/2<=1.00001 and y+h/2<=1.00001):
                    raise ValueError(f'{label}: box outside normalized image bounds')
                boxes+=1; classes.add(cls)
        summary[split]=dict(images=len(images), boxes=boxes, negatives=negatives,
                            missing_classes=[names[i] for i in names if i not in classes])
        if not boxes:
            raise ValueError(f'{split} contains no annotated objects')
    # Write a resolved config for Ultralytics, avoiding ambiguity in relative path handling.
    data['path']=str(root)
    return data,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='dataset/data.yaml')
    p.add_argument('--model',default='yolov8n.pt')
    p.add_argument('--mode',choices=['check','train','val','test'],default='check')
    p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--name',default='bank_objects')
    args=p.parse_args()
    try:
        data,report=check_dataset(args.data,args.mode=='test')
        for split,stats in report.items():
            print(split, stats)
        if args.mode=='check':
            return
        if args.epochs<1 or args.batch<1 or args.imgsz<32:
            raise ValueError('epochs/batch must be positive and imgsz >= 32')
        if not Path(args.model).is_file():
            raise ValueError('Weights missing: run python prepare_models.py while online first')
        import yaml
        resolved=Path(args.data).resolve().with_name('data.resolved.yaml')
        resolved.write_text(yaml.safe_dump(data,allow_unicode=True),encoding='utf-8')
        from ultralytics import YOLO
        model=YOLO(args.model)
        if model.task!='detect':
            raise ValueError('This trainer accepts detection models, not pose models')
        if args.mode=='train':
            model.train(data=str(resolved),epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,
                        device=args.device,workers=0,project='runs',name=args.name,seed=42)
            print('Training output:',model.trainer.save_dir)
        else:
            model.val(data=str(resolved),split='test' if args.mode=='test' else 'val',
                      device=args.device,imgsz=args.imgsz,batch=args.batch,workers=0)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1,f'ERROR: {exc}\n')


if __name__=='__main__':
    main()
