import cv2
from ultralytics import YOLO

video_path = (r"D:\code\hackathons\train РЖД ПДД\Hackathon-.Russia-is-a-country-of-opportunity\videos"
              r"\akn00006_fqGg6dtL.mov")
output_path = 'processed_video'
CHOSEN_FORMAT = "mp4"

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

formats = {
    "mp4": "mp4v",
    "avi": "XVID"
}
output_path += f'.{CHOSEN_FORMAT}'

fourcc = cv2.VideoWriter_fourcc(*formats[CHOSEN_FORMAT])
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

try:
    for result in model.track(source=video_path, show=True, stream=True):
        frame = result.orig_img

        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            confidence = box.conf[0]

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

except KeyboardInterrupt:
    print(
        "\nОбработка прервана пользователем. Сохранение обработанного видео...")  # НЕ ВСЕГДА РАБОТАЕТ, ЧАЩЕ НЕОБХОДИМО УБИВАТЬ ПРОЦЕСС, ЕСЛИ НУЖНО ЗАВЕРШИТЬ ДОСРОЧНО

finally:
    cap.release()
    out.release()
    print("Видео успешно сохранено в", output_path)
