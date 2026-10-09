# D1 Vision: ตรวจคน สิ่งของ มีด/อาวุธ และ human skeleton

โปรเจกต์นี้รันบน PC ทั้งหมด ใช้ Ultralytics + OpenCV + Flask โดยรับกล้องจาก D1 ผ่าน RTSP/SSH
แยกโมเดลตรวจสิ่งของ, โมเดล pose และโมเดลเฉพาะที่เทรนเอง ไม่มีคำสั่งควบคุมหุ่น

## 1. ติดตั้งบน PC ขณะที่มีอินเทอร์เน็ต ไม่ต้องสร้าง environment

เปิด PowerShell ในโฟลเดอร์นี้ (มี detect_web.py):

```powershell
python -m pip install -r requirements.txt
python prepare_models.py
```

ไฟล์น้ำหนักไม่รวมใน repository ให้รัน prepare_models.py เพื่อเตรียม yolov8n.pt และ yolov8n-pose.pt ก่อนใช้งาน
หากต่อ Wi-Fi หุ่นแล้วเข้าอินเทอร์เน็ตไม่ได้ ให้เตรียมโมเดลก่อน แล้วกลับมาต่อหุ่น
prepare_models.py ดาวน์โหลดโมเดลทั่วไปและ skeleton เท่านั้น ไม่ได้ดาวน์โหลดโมเดลปืน

## 2. เปิดภาพพร้อม detection + skeleton + knife events

PowerShell หน้าต่างที่ 1 บน PC (ต้องเห็น PS C:\... ไม่ใช่ firefly@firefly):

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 18554:127.0.0.1:8554 firefly@192.168.234.1
```

ใส่รหัสผ่าน แล้วเปิดค้างไว้
PowerShell หน้าต่างที่ 2 ในโฟลเดอร์โปรเจกต์นี้:

```powershell
python detect_web.py --source rtsp://127.0.0.1:18554/test --pose --knife-alerts --imgsz 640 --device cpu
```

เปิด http://127.0.0.1:8080 แล้ว Ctrl+F5 หากหน้าเดิมค้าง
หยุด server ตัวเดิมก่อน หรือใช้ --port 8081 แล้วเปิด URL พอร์ตนั้น
หากยังไม่ได้โหลด pose ให้ตัด --pose ออกเพื่อใช้ object detection และ knife alerts ก่อน
--preview-only แสดงภาพอย่างเดียว จะปิดทุกโมเดล รวม pose/knife

Dashboard: กรอบคน/สิ่งของ, total tracks แยกคลาส, คนในภาพ, จำนวน skeleton, จุดข้อต่อและเส้นเชื่อม,
เหตุการณ์มีด/อาวุธพร้อมภาพหลักฐาน, ปุ่มยืนยัน/แจ้งเตือนผิด
Skeleton เป็นการประมาณตำแหน่ง 2D 17 จุด ไม่ใช่โครงกระดูกจริงหรือการทำนายพฤติกรรม
คนที่ถูกบังอาจแสดง skeleton ได้ไม่ครบ ใช้ --keypoint-conf 0.5 กรองจุดที่ไม่มั่นใจ
โมเดลสิ่งของและ pose อาจนับคนไม่เท่ากัน เพราะตรวจแยกกัน ไม่มีการจับคู่ ID ของสองโมเดล
CPU ต้องรันหลายโมเดลจึงอาจช้า หากมี CUDA ที่ติดตั้งถูกต้อง ใช้ --device 0

## 3. เก็บภาพจาก D1 สำหรับ dataset

คง SSH tunnel ไว้ แล้วรันบน PC:

```powershell
python capture_samples.py --interval 2 --limit 100
```

บันทึกภาพดิบไป raw_samples/วันที่เวลา/ ทุก 2 วินาที สูงสุด 100 ภาพ หยุดด้วย Ctrl+C
ไม่มี label อัตโนมัติ ใช้ CVAT หรือเครื่องมือ annotation ที่ export YOLO Detection ได้เพื่อตีกรอบ
ภาพตัวอย่างควรครอบคลุมแสง ระยะ มุมกล้อง และวัตถุคล้ายเป้าหมาย เช่น โทรศัพท์ ปากกา ร่ม
ใช้ข้อมูลที่ได้รับอนุญาต และกำหนดนิยามการตีกรอบให้เหมือนกันทั้งชุด

## 4. สร้าง dataset

โครงสร้างที่ตัวตรวจรองรับ (ผู้ใช้ต้องใส่ภาพและ label จริง ไม่มี dataset แถมมา):

```text
dataset/
  data.yaml
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

รูป abc.jpg ต้องมี abc.txt ที่ตรงกัน บรรทัดละ `class_id center_x center_y width height`
พิกัดเป็นสัดส่วน 0..1; ภาพ negative ที่ตรวจ annotation แล้วใช้ txt ว่าง
ค่าเริ่มต้น data.yaml: knife, gun, masked_face, helmet, balaclava ปรับให้ตรงกับข้อมูลที่มีจริง
อย่าใส่คลาสที่ไม่มีข้อมูลแล้วคาดหวังว่าโมเดลจะตรวจได้
แบ่งตามคลิป/รอบถ่าย/สถานที่ เช่น 70/20/10 ไม่สุ่มเฟรมติดกันไปคนละ split
การรวม dataset ต้อง remap class IDs และ label วัตถุของทุกคลาสที่เห็นให้ครบ
หน้ากากอนามัยไม่ใช่ป้ายกำกับอันตราย ห้ามแทน masked_face เป็น covered_face โดยไม่กำหนดนิยามใหม่

แหล่งเริ่มต้น:
- COCO มี knife: https://docs.ultralytics.com/datasets/detect/coco/
- FMLD annotations เป็น Pascal VOC ต้องแปลง: https://github.com/borutb-fri/FMLD
- Medical masks: https://humansintheloop.org/resources/datasets/medical-mask-dataset/
ตรวจ license ภาพต้นทางและสิทธิ์ใช้งานก่อนนำ dataset ไปใช้งานจริง
โค้ดนี้ใช้เฉพาะ object labels ไม่ใช้ ethnicity/gender annotations ที่อาจมีในบาง dataset

## 5. ตรวจข้อมูล แล้ว train / evaluate

ทั้งหมดรันบน PC:

```powershell
python train_custom.py --mode check
python train_custom.py --mode train --epochs 80 --batch 4 --device cpu
```

หากมี NVIDIA GPU และ PyTorch CUDA ใช้งานได้ เปลี่ยน --device cpu เป็น --device 0
รองรับการเทรน object detection ไม่ใช่ custom pose labels; skeleton ใช้ pretrained pose พร้อมใช้
ตัวตรวจจับ label ผิด, พิกัดเกินภาพ, ภาพซ้ำแบบไฟล์เหมือนกันข้าม splits และรายงานคลาสที่ขาด
ยังต้องตรวจภาพเสียและภาพใกล้เคียงด้วยตนเอง ตัวตรวจไม่ได้รับประกัน annotation ถูกต้อง
เมื่อสำเร็จโปรแกรมแสดง training output จริง โดยครั้งแรกปกติเป็น runs/bank_objects
ครั้งต่อไปอาจต่อท้ายเลข ให้ใช้ path ที่โปรแกรมแสดง

```powershell
python train_custom.py --mode test --model runs/bank_objects/weights/best.pt --device cpu
```

ดู precision/recall แยกคลาส และลองคลิป D1 ที่ไม่เคยใช้ train
บันทึก false alarms ต่อชั่วโมงและเหตุการณ์ที่พลาด ไม่ใช้ mAP รวมเพียงอย่างเดียว
ยังไม่มีการ train โมเดลเฉพาะให้ เพราะไม่มี annotated dataset ของคุณ

## 6. ต่อโมเดลที่เทรนเองเข้า dashboard

```powershell
python detect_web.py --source rtsp://127.0.0.1:18554/test --pose --knife-alerts --event-model runs/bank_objects/weights/best.pt --weapon-labels knife,gun --device cpu
```

ถ้าโมเดล custom มี knife จะใช้ผล custom เท่านั้น ไม่สร้าง knife event ซ้ำจากโมเดลทั่วไป
ปืนจะเปิดใช้งานเมื่อโมเดล custom มีคลาสที่ตรงจริง ไม่ใช้ YOLOv8n อ้างว่าตรวจปืนได้
ถ้าตั้งนโยบายตรวจหน้าที่ถูกบัง ต้องมีโมเดล/คลาสที่นิยามตรงและ --policy-zone X1 Y1 X2 Y2
โดย default คลาสชื่อ covered_face; สามารถตั้ง --covered-labels ได้ แต่ต้องทดสอบนิยามก่อน
โซนเป็นพิกัดในภาพ ใช้กล้องอยู่กับที่เท่านั้น ไม่ได้ติดตามพื้นที่ธนาคารเมื่อหุ่นเดิน

## ข้อจำกัดและผลลัพธ์

- เหตุการณ์แดงเป็นวัตถุคล้ายอาวุธ ไม่ยืนยันว่าเป็นของจริง ไม่ระบุผู้ถือจาก proximity
- ยืนยัน ID ต่อเนื่องอย่างน้อย 3 เฟรมและ 1 วินาที ปรับ --event-frames / --event-seconds ได้
- เมื่อภาพขาดเกิน 3 วินาทีหรือไม่มีผลตรวจ กฎยืนยันเริ่มใหม่
- วัตถุหายแล้วกลับมาอาจนับใหม่ ยอดไม่ใช่จำนวนวัตถุจริงที่ไม่ซ้ำตลอดเวลา
- event_data เก็บภาพและ events.jsonl แบบ append-only รวมการ review ไม่มี auto-delete
- ยอด dashboard เริ่มใหม่ทุกการเปิดโปรแกรม ประวัติบน disk ยังคงอยู่
- skeleton ไม่เพิ่มระดับแจ้งเตือนจากท่าทาง และไม่ระบุตัวบุคคล

## ทดสอบโค้ด

```powershell
python -m unittest discover -p "test_*.py" -v
```

API อ้างอิง: https://docs.ultralytics.com/tasks/pose/ และ https://docs.ultralytics.com/modes/train/
ใช้ library ที่ติดตั้งใน PC ไม่มีการสร้าง environment หรือเปลี่ยนไฟล์บนหุ่น
