import csv
import os.path
import subprocess
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from structures import Violation


def bird_eye_view(
        image: np.ndarray, src_points: List[np.ndarray], dst_points: List[np.ndarray]
) -> np.ndarray:
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, M, (w, h))


def yolo_obj_extraction(model: YOLO, frame: np.ndarray, **kwargs: Any) -> List[Results]:
    detections = model.track(source=frame, **kwargs)
    for obj in detections:
        frame = obj.orig_img

        for box in obj.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            confidence = box.conf[0]

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return detections


def lane_detection(frame: np.ndarray) -> None:
    # TODO
    pass


def traffic_light_detection(frame: np.ndarray) -> None:
    # TODO
    pass


def pipeline(frame: np.ndarray, model: YOLO) -> Tuple[np.ndarray, List[Violation]]:
    violations: List[Violation] = []
    h, w = frame.shape[:2]

    original_frame = frame.copy()

    objs = yolo_obj_extraction(model, frame, verbose=False)

    src_points = np.array(
        [[w * 0.4, h * 0.65], [w * 0.6, h * 0.65], [w, h], [0, h]], dtype=np.float32
    )
    dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    bird_eye_view_image = bird_eye_view(frame, src_points, dst_points)

    # TODO
    # traffic_light_detection(bird_eye_view_image)
    # lane_detection(bird_eye_view_image)

    cv2.polylines(
        frame, [np.int32(src_points)], isClosed=True, color=(0, 255, 0), thickness=2
    )

    text = "BEV"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = 10
    text_y = 30 + text_size[1]

    cv2.putText(
        frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    bev_height, bev_width = 200, 200
    bird_eye_view_image_resized = cv2.resize(
        bird_eye_view_image, (bev_width, bev_height)
    )
    frame[0:bev_height, 0:bev_width] = bird_eye_view_image_resized

    return frame, violations


def append_to_submission(vns: List[Violation], csv_path: PathLike) -> None:
    rows = [[vn.text, vn.fine, vn.duration] for vn in vns]
    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)


def main_loop(
        vid_path: str, sub_path: str, offset: int, frames_to_process: int
) -> None:
    if not os.path.exists(vid_path):
        raise FileNotFoundError(vid_path)

    sub_path = Path(sub_path)
    sub_path.touch(exist_ok=True)

    video = cv2.VideoCapture(vid_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {vid_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        f"processed_{offset}.mp4", fourcc, 30.0, (int(video.get(3)), int(video.get(4)))
    )

    performance_postfix = "ns"[1]
    model = YOLO(f"yolo11{performance_postfix}.pt")

    frames_counter = 0
    processed_frames = 0
    frames_to_skip = 15

    while frames_counter < offset:
        ret, _ = video.read()
        if not ret:
            break
        frames_counter += 1

    while processed_frames < frames_to_process:
        ret, frame = video.read()
        if not ret:
            break

        frames_counter += 1
        processed_frames += 1

        if frames_counter % frames_to_skip == 0:
            print(f"{offset}: {processed_frames}")
            frame, violations = pipeline(frame, model)

            if violations:
                append_to_submission(violations, sub_path)

            out.write(frame)

    video.release()
    out.release()


def process_video_chunk(args: Tuple[str, str, int, int]) -> None:
    video_path, submission_path, offset, frames_to_process = args
    main_loop(video_path, submission_path, offset, frames_to_process)


def concat_tmp_files(temp_files):
    try:
        with open("temp_file_list.txt", "w") as f:
            for temp_file in temp_files:
                f.write(f"file '{temp_file}'\n")

        output_file = "OUTPUT.mp4"
        command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "temp_file_list.txt",
            "-c",
            "copy",
            output_file,
        ]
        subprocess.run(command, check=True)
        print(f"Fused video saved at: {output_file}")
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                # os.remove(temp_file)
                pass


def schedule_tasks(video_path: str, submission_path: str, num_cpus: int) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    if total_frames == 0:
        raise ValueError("The video file is empty or could not retrieve frame count.")

    frames_per_task = total_frames // num_cpus
    offsets = [i * frames_per_task for i in range(num_cpus)]

    if offsets:
        remaining_frames = total_frames - (offsets[-1] + frames_per_task)
        if remaining_frames > 0:
            frames_per_task += remaining_frames // len(offsets)

    frames_to_process = frames_per_task if frames_per_task > 0 else total_frames

    temp_files = [f"processed_{offset}.mp4" for offset in offsets]
    args = [
        (video_path, submission_path, offset, frames_to_process) for offset in offsets
    ]

    print(
        f"Starting video processing with the following configuration:\n"
        f"Video path: {video_path}\n"
        f"Submission path: {submission_path}\n"
        f"Number of CPUs: {num_cpus}\n"
        f"Total frames: {total_frames}\n"
        f"Frames per task: {frames_per_task}\n"
        f"Task offsets: {offsets}\n"
        f"Temporary files: {temp_files}\n"
    )

    with Pool(num_cpus) as pool:
        pool.map(process_video_chunk, args)

    concat_tmp_files(temp_files)


if __name__ == "__main__":
    video_path = "videos/akn00006_fqGg6dtL.mov"
    submission_path = "submissions/submission.csv"
    cpus = min((cpu_count() or 1) - 2, 4)
    schedule_tasks(video_path, submission_path, cpus)
