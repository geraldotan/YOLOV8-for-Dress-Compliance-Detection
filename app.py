from flask import Flask, render_template, request, send_from_directory, redirect, url_for, Response
from ultralytics import YOLO
import os
import cv2
import numpy as np
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

app = Flask(__name__)

model_person = YOLO("yolov8n.pt")
model_cloth = YOLO("best.pt")

for d in ["static", "output", "output/example_frames"]:
    os.makedirs(d, exist_ok=True)

cloth_labels = model_cloth.names
print("Model pakaian berhasil dimuat dengan label:", cloth_labels)

def iou(boxA, boxB):
    """Hitung Intersection over Union (IoU) antara dua bounding box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def is_center_inside(box_inner, box_outer):
    """Cek apakah titik tengah box_inner ada di dalam box_outer."""
    x_center = (box_inner[0] + box_inner[2]) / 2
    y_center = (box_inner[1] + box_inner[3]) / 2
    return (box_outer[0] <= x_center <= box_outer[2]) and (box_outer[1] <= y_center <= box_outer[3])

def detect_violation(frame):
    results_person = model_person(frame, classes=[0])[0]
    results_cloth = model_cloth(frame)[0]

    person_boxes = results_person.boxes.xyxy.cpu().numpy() if results_person.boxes else []
    cloth_boxes = results_cloth.boxes.xyxy.cpu().numpy() if results_cloth.boxes else []
    cloth_cls = results_cloth.boxes.cls.cpu().numpy() if results_cloth.boxes else []
    cloth_conf = results_cloth.boxes.conf.cpu().numpy() if results_cloth.boxes else []

    labels_out = []

    for person_box in person_boxes:
        px1, py1, px2, py2 = person_box

        has_upper = False
        has_pants = False
        has_violation = False
 
        for box, cls, conf in zip(cloth_boxes, cloth_cls, cloth_conf):

            if conf < 0.3:
                continue

            label = cloth_labels[int(cls)].lower()

            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if not (px1 <= cx <= px2 and py1 <= cy <= py2):
                continue

            if label in ["shirt", "tshirt", "sweater", "jacket"]:
                has_upper = True

            elif label == "pants":
                has_pants = True

            elif label in ["short", "skirt"]:
                has_violation = True

        if has_violation:
            status = "Melanggar Aturan"

        elif has_upper and not has_pants:
            status = "Melanggar Aturan"

        else:
            status = "Aman"

        labels_out.append(status)

        color = (0, 0, 255) if status == "Melanggar Aturan" else (0, 255, 0)

        cv2.rectangle(
            frame,
            (int(px1), int(py1)),
            (int(px2), int(py2)),
            color,
            2
        )

        cv2.putText(
            frame,
            status,
            (int(px1), int(py1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return frame, labels_out

def buat_laporan_pdf(filename, summary, video_name, example_frames=None):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    elements += [
        Paragraph("LAPORAN DETEKSI PELANGGARAN PAKAIAN", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"Video: {video_name}", styles['Normal']),
        Paragraph(f"Tanggal proses: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
        Spacer(1, 12)
    ]

    data_summary = [["Kategori", "Jumlah"]] + [[k, v] for k, v in summary.items() if k != "Kesimpulan"]
    table = Table(data_summary, hAlign='LEFT')
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(table)

    if "Kesimpulan" in summary:
        elements += [
            Spacer(1, 12),
            Paragraph("Kesimpulan", styles['Heading2']),
            Paragraph(summary["Kesimpulan"], styles['Normal'])
        ]

    if example_frames:
        for cat, frames in example_frames.items():
            if frames:
                elements += [Spacer(1, 12), Paragraph(f"Contoh frame: {cat}", styles['Heading2'])]
                for f in frames:
                    elements += [Image(f, width=400, height=300), Spacer(1, 6)]

    doc.build(elements)
    return filename

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame, _ = detect_violation(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if not file or file.filename == '':
                return "Tidak ada gambar yang diunggah.", 400

            input_path = 'static/input.jpg'
            output_path = 'output/output.jpg'
            file.save(input_path)

            frame = cv2.imread(input_path)
            if frame is None:
                return "Gagal membaca gambar.", 500

            frame, _ = detect_violation(frame)
            cv2.imwrite(output_path, frame)

            return redirect(url_for('result', filename='output.jpg', type='image'))

        if 'video' in request.files:
            file = request.files['video']
            if not file or file.filename == '':
                return "Tidak ada video yang diunggah.", 400

            input_path = 'static/input_video.mp4'
            output_path = 'output/output_video.mp4'
            file.save(input_path)

            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None
            summary = {"Melanggar Aturan": 0, "Aman": 0}
            example_frames = {"Melanggar Aturan": [], "Aman": []}
            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, labels = detect_violation(frame)
                for l in labels:
                    summary[l] += 1

                for cat in ["Melanggar Aturan", "Aman"]:
                    if cat in labels and len(example_frames[cat]) < 2:
                        path = f'output/example_frames/{cat.lower().replace(" ", "_")}_{frame_id}.jpg'
                        cv2.imwrite(path, frame)
                        example_frames[cat].append(path)

                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
                out.write(frame)
                frame_id += 1

            cap.release()
            if out:
                out.release()

            total = summary["Melanggar Aturan"] + summary["Aman"]
            if summary["Melanggar Aturan"] == 0:
                summary["Kesimpulan"] = "Tidak ditemukan pelanggaran (shorts) pada video."
            else:
                pct = (summary["Melanggar Aturan"] / total) * 100
                summary["Kesimpulan"] = f"Terdapat pelanggaran pada {pct:.1f}% frame video."

            pdf_name = "report_violation.pdf"
            pdf_path = os.path.join("output", pdf_name)
            buat_laporan_pdf(pdf_path, summary, file.filename, example_frames)

            return redirect(url_for('result', filename='output_video.mp4', type='video', report=pdf_name))

    return render_template('index2.html')

@app.route('/result')
def result():
    filename = request.args.get('filename')
    media_type = request.args.get('type', 'image')
    report = request.args.get('report')
    return render_template('result.html',
                           output_file=filename,
                           media_type=media_type,
                           report_file=report)


@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory('output', filename)


@app.route('/report/<filename>')
def report_file(filename):
    return send_from_directory('output', filename)


if __name__ == "__main__":
    app.run(debug=True)
