import cv2
import numpy as np
from typing import Annotated
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class TrimVideoRequest(BaseModel):
    input_path: str
    output_path: str
    start_sec: float
    end_sec: float


def detect_board_by_color(frame: np.ndarray) -> np.ndarray:
    """Фильтрация шахматной доски по цвету."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([10, 30, 50])
    upper_brown = np.array([30, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    return mask


def find_board_contour(mask: np.ndarray) -> np.ndarray | None:
    """Находит контур шахматной доски."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)
    if len(approx) == 4:
        return approx
    else:
        return None


@app.get("/detect_first_move/")
def detect_first_move(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_frame = None
    start_position = None
    move_detected = False

    while cap.isOpened():
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break

        mask = detect_board_by_color(frame)
        board_contour = find_board_contour(mask)

        if board_contour is not None:
            if start_position is None:
                start_position = frame.copy()
            else:
                diff = cv2.absdiff(cv2.cvtColor(start_position, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if np.sum(diff) > 100000:
                    move_detected = True
                    move_time = frame_idx / fps
                    print(f"Первый ход зафиксирован на {move_time:.2f} секунде")
                    break

    cap.release()
    if not move_detected:
        print("Первый ход не обнаружен.")

@app.post("/trim_video/")
def trim_video(request: TrimVideoRequest) -> dict:
    cap = cv2.VideoCapture(request.input_path)

    if not cap.isOpened():
        return {"error": "Ошибка: не удалось открыть видео"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(request.start_sec * fps)
    end_frame = int(request.end_sec * fps)

    if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
        return {"error": "Ошибка: некорректные значения start_sec и end_sec"}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(request.output_path, fourcc, fps, (width, height))

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return {"message": f"Видео сохранено как {request.output_path}"}


request = TrimVideoRequest(input_path="video_1.MP4", output_path="output.MP4", start_sec=1, end_sec=3)
trim_video(request)
detect_first_move("video_1.MP4")

