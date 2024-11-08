import os
import threading
import time
from pathlib import Path

import werkzeug
from flask import Flask, request, render_template
from loguru import logger

from baseline import detection_task

app = Flask(__name__)

ROOT_DIR = Path(__file__).parent.parent.resolve()
UPLOAD_FOLDER = ROOT_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Глобальный словарь для хранения результатов обработки
processing_results = {}


def run_and_clean(func, video_path, task_id):
    try:
        results = func(video_path)
        processing_results[task_id] = {"status": "completed", "results": results}
        logger.info(f"Завершена обработка видео: {video_path}")
    except Exception as e:
        processing_results[task_id] = {"status": "error", "results": str(e)}
        logger.error(f"Ошибка при обработке видео: {e}")
    finally:
        try:
            os.remove(video_path)
            logger.info(f"Файл {video_path} удален после обработки.")
        except OSError as e:
            logger.error(f"Ошибка при удалении файла {video_path}: {e}")


@app.route("/")
def upload_form():
    return render_template("index.html")


@app.route("/upload-video", methods=["POST"])
def upload_video():
    logger.info("Запрос на обработку видео")

    if "video" not in request.files:
        return render_template(
            "index.html",
            message={"success": False, "message": "Ошибка: Видео не загружено."},
        )

    file = request.files["video"]
    mimetype = file.content_type
    logger.debug(f"{file}: {mimetype}")

    if mimetype not in ["video/mp4", "video/webm", "video/quicktime"]:
        return render_template(
            "index.html",
            message={"success": False, "message": "Ошибка: Неподдерживаемый тип видео."},
        )

    filename = werkzeug.utils.secure_filename(file.filename)
    video_path = UPLOAD_FOLDER / filename
    file.save(video_path)

    task_id = filename + time.strftime("%Y%m%d%H%M%S")
    processing_results[task_id] = {"status": "processing", "results": None}

    task_thread = threading.Thread(
        target=run_and_clean, args=(detection_task, video_path, task_id)
    )
    task_thread.start()

    return render_template(
        "index.html",
        message={"success": True, "message": "Видео загружено. Обработка начата.", "task_id": task_id},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
