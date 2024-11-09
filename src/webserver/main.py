import base64
import os
import random
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from loguru import logger
import werkzeug

from src.detection.main import detection_task

app = Flask(__name__)

ROOT_DIR = Path(__file__).parent.parent.resolve()
UPLOAD_FOLDER = ROOT_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

processing_results = {}


def np_array_to_base64(np_array):
    img = Image.fromarray(np_array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def run_and_clean(func, video_path, task_id):
    try:
        results = func(video_path)
        text_data, images = results

        # Кодируем изображения в base64
        base64_images = [np_array_to_base64(img) for img in images]

        # Сохраняем текст и изображения
        processing_results[task_id] = {
            "status": "completed",
            "results": {"text": text_data, "images": base64_images},
        }

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


def generate_random_image(width=100, height=100):
    """Generate a random image for testing purposes."""
    random_array = np.random.randint(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    image = Image.fromarray(random_array)
    return image


@app.route("/mock-results/<task_id>")
def mock_results(task_id):
    """Return random images as results for the template."""
    images = [generate_random_image(width=640, height=384) for _ in range(5)]
    base64_images = [np_array_to_base64(np.array(img)) for img in images]

    processing_results[task_id] = {
        "status": "completed",
        "results": {"text": "Hello world!", "images": base64_images},
    }

    return render_template(
        "index.html",
        message={
            "success": True,
            "message": "Видео загружено. Обработка начата.",
            "task_id": task_id,
        },
    )


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

    # Проверка допустимых форматов и размера файла (например, до 100 МБ)
    allowed_types = ["video/mp4", "video/webm", "video/quicktime"]
    if mimetype not in allowed_types:
        return render_template(
            "index.html",
            message={
                "success": False,
                "message": "Ошибка: Неподдерживаемый тип видео.",
            },
        )

    if file.content_length > 100 * 1024 * 1024:  # Ограничение 100 МБ
        return render_template(
            "index.html",
            message={"success": False, "message": "Ошибка: Файл слишком большой."},
        )

    filename = werkzeug.utils.secure_filename(file.filename)
    video_path = UPLOAD_FOLDER / filename

    task_id = filename + time.strftime("%Y%m%d%H%M%S")
    if processing_results.get(task_id):
        return render_template(
            "index.html",
            message={"success": False, "message": "Ошибка: Видео уже обрабатывается."},
        )

    try:
        file.save(video_path)
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        return render_template(
            "index.html",
            message={"success": False, "message": "Ошибка при сохранении видео."},
        )

    processing_results[task_id] = {"status": "processing", "results": None}

    task_thread = threading.Thread(
        target=run_and_clean, args=(detection_task, video_path, task_id)
    )
    task_thread.start()

    return render_template(
        "index.html",
        message={
            "success": True,
            "message": "Видео загружено. Обработка начата.",
            "task_id": task_id,
        },
    )


@app.route("/get-results/<task_id>")
def get_results(task_id):
    result = processing_results.get(task_id, {"status": "not_found", "results": None})
    return jsonify(result)
