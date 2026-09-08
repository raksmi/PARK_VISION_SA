from pathlib import Path
from datetime import datetime
import io
import tempfile
import threading
import time
import base64
import wave
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from parking_logic import parking_insights


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path("best.pt")


st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --pv-border: rgba(255, 255, 255, 0.10);
        --pv-purple: #8b5cf6;
        --pv-cyan: #22d3ee;
        --pv-muted: #a1a1b5;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(139, 92, 246, 0.20), transparent 28%),
            radial-gradient(circle at 90% 8%, rgba(34, 211, 238, 0.14), transparent 25%),
            radial-gradient(circle at 70% 90%, rgba(236, 72, 153, 0.10), transparent 30%),
            linear-gradient(135deg, #05050b 0%, #09091a 48%, #050510 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: linear-gradient(to bottom, black, transparent 85%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(12,12,28,0.88), rgba(6,6,15,0.94));
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid rgba(139,92,246,0.16);
        box-shadow: 12px 0 45px rgba(0,0,0,0.22);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-weight: 800 !important;
        letter-spacing: -2.5px !important;
        background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 42%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 22px rgba(139,92,246,0.18));
    }

    h2 {
        font-weight: 750 !important;
        letter-spacing: -1.2px !important;
    }

    h3 {
        font-weight: 700 !important;
    }

    .stCaption {
        color: #9ca3af !important;
        letter-spacing: 2px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025)),
            rgba(15,15,30,0.58);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--pv-border);
        border-radius: 22px;
        padding: 20px 20px 18px;
        min-height: 122px;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.06),
            0 18px 55px rgba(0,0,0,0.24);
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    }

    div[data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        right: -35px;
        top: -45px;
        border-radius: 50%;
        background: rgba(139,92,246,0.14);
        filter: blur(18px);
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(139,92,246,0.38);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.08),
            0 22px 65px rgba(76,29,149,0.22);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--pv-muted) !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    div[data-testid="stExpander"],
    div[data-testid="stFileUploader"],
    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"] {
        border: 1px solid var(--pv-border);
        border-radius: 20px;
        background: rgba(18,18,34,0.48);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.04),
            0 14px 45px rgba(0,0,0,0.18);
    }

    div[data-testid="stFileUploader"] {
        padding: 6px;
        border-style: dashed;
        border-color: rgba(139,92,246,0.35);
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(34,211,238,0.55);
        box-shadow: 0 0 35px rgba(34,211,238,0.08);
    }

    .image-container {
        border: 1px solid rgba(255,255,255,0.11);
        border-radius: 24px;
        padding: 8px;
        background:
            linear-gradient(135deg, rgba(139,92,246,0.08), rgba(34,211,238,0.04)),
            rgba(10,10,22,0.62);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.06),
            0 25px 80px rgba(0,0,0,0.30);
    }

    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 9px 16px;
        border-radius: 999px;
        background: rgba(16,185,129,0.10);
        border: 1px solid rgba(52,211,153,0.26);
        color: #6ee7b7;
        font-weight: 750;
        font-size: 13px;
        box-shadow: 0 0 28px rgba(16,185,129,0.08);
    }

    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 12px rgba(52,211,153,0.85);
        animation: pvPulse 1.8s infinite;
    }

    @keyframes pvPulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.45); opacity: 0.55; }
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(135deg, rgba(139,92,246,0.16), rgba(34,211,238,0.07));
        color: #f8fafc;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(139,92,246,0.45);
        box-shadow:
            0 10px 28px rgba(76,29,149,0.20),
            inset 0 1px 0 rgba(255,255,255,0.08);
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #0891b2) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        box-shadow: 0 12px 35px rgba(124,58,237,0.28) !important;
    }

    button[kind="primary"]:hover {
        filter: brightness(1.08);
        box-shadow: 0 16px 42px rgba(124,58,237,0.38) !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 14px;
        background: rgba(255,255,255,0.045);
        border-color: rgba(255,255,255,0.09);
    }

    section[data-testid="stSidebar"] label[data-baseweb="radio"] {
        border-radius: 12px;
        padding: 7px 9px;
    }

    section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
        background: rgba(139,92,246,0.08);
    }

    hr {
        border: none !important;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.25), rgba(34,211,238,0.20), transparent);
        margin: 1.4rem 0;
    }

    .js-plotly-plot {
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        overflow: hidden;
        background: rgba(15,15,30,0.34);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }


    .hero-panel {
        position: relative;
        overflow: hidden;
        padding: 28px 30px 26px;
        margin: 4px 0 28px;
        border-radius: 26px;
        border: 1px solid rgba(139,92,246,0.22);
        background:
            radial-gradient(circle at 88% 20%, rgba(34,211,238,0.12), transparent 28%),
            radial-gradient(circle at 55% 100%, rgba(139,92,246,0.14), transparent 34%),
            linear-gradient(135deg, rgba(255,255,255,0.065), rgba(255,255,255,0.018)),
            rgba(12,12,27,0.52);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.07),
            0 24px 70px rgba(0,0,0,0.25);
    }

    .hero-panel::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -110px;
        top: -150px;
        border-radius: 50%;
        background: rgba(139,92,246,0.18);
        filter: blur(35px);
    }

    .hero-eyebrow {
        color: #a78bfa;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2.2px;
        margin-bottom: 7px;
    }

    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4.4rem);
        line-height: 0.98;
        font-weight: 800;
        letter-spacing: -3px;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 46%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1.2px;
        margin-top: 12px;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 19px;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 8px 12px;
        border-radius: 999px;
        color: #dbeafe;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.6px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
    }

    .hero-pill .pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #8b5cf6;
        box-shadow: 0 0 10px rgba(139,92,246,0.8);
    }

    .hero-pill:nth-child(2) .pill-dot {
        background: #22d3ee;
        box-shadow: 0 0 10px rgba(34,211,238,0.8);
    }

    .hero-pill:nth-child(3) .pill-dot {
        background: #34d399;
        box-shadow: 0 0 10px rgba(52,211,153,0.8);
    }

    .dashboard-card {
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 22px;
        padding: 20px;
        background:
            radial-gradient(circle at 100% 0%, rgba(139,92,246,0.10), transparent 32%),
            rgba(15,15,30,0.52);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.05),
            0 18px 55px rgba(0,0,0,0.22);
        margin-bottom: 18px;
    }

    .card-kicker {
        color: #a1a1b5;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .card-title {
        color: #f8fafc;
        font-size: 19px;
        font-weight: 800;
        margin-top: 4px;
    }

    .card-subtitle {
        color: #8f91a5;
        font-size: 12px;
        margin-top: 3px;
    }

    .insight-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0;
        color: #d7d9e4;
        font-size: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .insight-row:last-child {
        border-bottom: none;
    }

    .insight-check {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 21px;
        height: 21px;
        border-radius: 50%;
        color: #34d399;
        border: 1px solid rgba(52,211,153,0.28);
        background: rgba(52,211,153,0.08);
        font-weight: 800;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.8px;
        border: 1px solid rgba(245,158,11,0.30);
        color: #fbbf24;
        background: rgba(245,158,11,0.08);
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 18px;
    }

    .sidebar-logo {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        font-size: 22px;
        font-weight: 900;
        color: white;
        background: linear-gradient(135deg, #7c3aed, #0891b2);
        box-shadow: 0 0 28px rgba(124,58,237,0.35);
        border: 1px solid rgba(255,255,255,0.20);
    }

    .sidebar-name {
        font-size: 17px;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.05;
    }

    .sidebar-tagline {
        color: #85879a;
        font-size: 8px;
        letter-spacing: 1.1px;
        margin-top: 4px;
    }

    .nav-note {
        color: #717387;
        font-size: 10px;
        line-height: 1.5;
        margin: 10px 2px 0;
    }

    .live-card {
        border: 1px solid rgba(34,211,238,0.16);
        background:
            radial-gradient(circle at 100% 0%, rgba(34,211,238,0.11), transparent 30%),
            rgba(10,16,30,0.52);
        border-radius: 22px;
        padding: 18px;
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 18px 50px rgba(0,0,0,0.20);
    }


    .page-heading {
        margin: 8px 0 24px;
    }

    .page-kicker {
        color: #a78bfa;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }

    .page-title {
        color: #f8fafc;
        font-size: 34px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -1.5px;
    }

    .page-subtitle {
        color: #8f91a5;
        font-size: 13px;
        margin-top: 8px;
    }

    .empty-card {
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .history-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 16px 18px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 17px;
        background: rgba(15,15,30,0.48);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .history-name {
        color: #f8fafc;
        font-size: 13px;
        font-weight: 750;
        word-break: break-word;
    }

    .history-meta {
        color: #777a8d;
        font-size: 10px;
        margin-top: 4px;
    }

    .history-stats {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #aeb1c1;
        font-size: 10px;
        white-space: nowrap;
    }

    .history-stats strong {
        color: #c4b5fd;
        font-size: 15px;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        padding-top: 4px;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
        border-radius: 13px;
        margin: 4px 0;
        padding: 9px 11px;
        border: 1px solid transparent;
        transition: all 0.18s ease;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
        background: rgba(139,92,246,0.09);
        border-color: rgba(139,92,246,0.15);
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
        background: linear-gradient(135deg, rgba(124,58,237,0.30), rgba(8,145,178,0.15));
        border-color: rgba(139,92,246,0.34);
        box-shadow: 0 8px 24px rgba(76,29,149,0.16);
    }

    .footer {
        text-align: center;
        color: #64647a;
        font-size: 11px;
        padding-top: 35px;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return YOLO(
        str(MODEL_PATH)
    )


model = load_model()


class LiveStats:

    def __init__(self):

        self.lock = threading.Lock()

        self.data = {
            "occupied": 0,
            "empty": 0,
            "total": 0,
            "occupancy": 0.0,
            "fps": 0.0,
            "updated": "Waiting",
            "active": False,
            "frames": 0,
            "peak_occupancy": 0.0
        }

        self.history = []

    def update(self, **values):

        with self.lock:
            self.data.update(values)

    def read(self):

        with self.lock:
            return dict(
                self.data
            )

    def add_history(self):

        with self.lock:

            self.history.append(
                {
                    "Time":
                        datetime.now().strftime(
                            "%H:%M:%S"
                        ),
                    "Occupancy":
                        round(
                            self.data["occupancy"],
                            1
                        ),
                    "Available":
                        self.data["empty"],
                    "Occupied":
                        self.data["occupied"],
                    "Total":
                        self.data["total"]
                }
            )

            if len(self.history) > 120:
                self.history.pop(0)

    def read_history(self):

        with self.lock:
            return list(
                self.history
            )

    def reset(self):

        with self.lock:

            self.data = {
                "occupied": 0,
                "empty": 0,
                "total": 0,
                "occupancy": 0.0,
                "fps": 0.0,
                "updated": "Waiting",
                "active": False,
                "frames": 0,
                "peak_occupancy": 0.0
            }

            self.history = []


@st.cache_resource
def get_live_stats():
    return LiveStats()


live_stats = get_live_stats()


def make_tone(
    frequency,
    duration=0.25,
    volume=0.35
):

    sample_rate = 44100

    samples = int(
        sample_rate * duration
    )

    audio = bytearray()

    for i in range(samples):

        value = math.sin(
            2
            * math.pi
            * frequency
            * i
            / sample_rate
        )

        value *= volume

        sample = int(
            value * 32767
        )

        audio.extend(
            int(sample).to_bytes(
                2,
                byteorder="little",
                signed=True
            )
        )

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb"
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(
            sample_rate
        )

        wav.writeframes(
            bytes(audio)
        )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


GREEN_SOUND = make_tone(
    880,
    0.22
)

YELLOW_SOUND = make_tone(
    520,
    0.22
)

RED_SOUND = make_tone(
    220,
    0.35
)


def sound_alert(
    level,
    enabled=True
):

    if not enabled:
        return

    if level == "green":
        sound = GREEN_SOUND

    elif level == "yellow":
        sound = YELLOW_SOUND

    else:
        sound = RED_SOUND

    st.markdown(
        f"""
        <audio autoplay>
            <source
                src="data:audio/wav;base64,{sound}"
                type="audio/wav"
            >
        </audio>
        """,
        unsafe_allow_html=True
    )


def parking_level(
    available_ratio
):

    if available_ratio >= 60:
        return "green"

    if available_ratio >= 30:
        return "yellow"

    return "red"


def parking_status(
    level
):

    if level == "green":
        return "🟢 PARKING AVAILABLE"

    if level == "yellow":
        return "🟡 PARKING FILLING UP"

    return "🔴 PARKING NEARLY FULL"


def run_detection(
    image,
    confidence
):

    image = image.convert(
        "RGB"
    )

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as temp:

        image.save(
            temp.name,
            quality=95
        )

        image_path = temp.name

    results = model.predict(
        source=image_path,
        imgsz=512,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    names = result.names

    occupied = 0
    empty = 0

    detections = []

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        for i in range(
            len(result.boxes)
        ):

            class_id = int(
                result.boxes.cls[i].item()
            )

            confidence_score = float(
                result.boxes.conf[i].item()
            )

            label = str(
                names[class_id]
            )

            label_lower = label.lower()

            if "occupied" in label_lower:

                occupied += 1

                status = "Occupied"

            elif (
                "empty" in label_lower
                or "free" in label_lower
            ):

                empty += 1

                status = "Available"

            else:

                status = label

            box = (
                result
                .boxes
                .xyxy[i]
                .tolist()
            )

            detections.append(
                {
                    "Slot":
                        len(detections) + 1,
                    "Status":
                        status,
                    "Confidence (%)":
                        round(
                            confidence_score * 100,
                            1
                        ),
                    "X1":
                        round(box[0], 1),
                    "Y1":
                        round(box[1], 1),
                    "X2":
                        round(box[2], 1),
                    "Y2":
                        round(box[3], 1)
                }
            )

    insights = parking_insights(
        occupied,
        empty
    )

    return (
        annotated,
        insights,
        pd.DataFrame(
            detections
        )
    )


def annotated_bytes(
    annotated
):

    rgb = annotated[
        :,
        :,
        ::-1
    ]

    image = Image.fromarray(
        rgb
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


def add_upload_history(
    insights,
    filename
):

    if (
        "upload_history"
        not in st.session_state
    ):

        st.session_state.upload_history = []

    st.session_state.upload_history.append(
        {
            "Time":
                datetime.now().strftime(
                    "%H:%M:%S"
                ),
            "Image":
                filename,
            "Occupancy":
                round(
                    insights["occupancy"],
                    1
                ),
            "Available":
                insights["available"],
            "Occupied":
                insights["occupied"],
            "Total":
                insights["total"]
        }
    )

    st.session_state.upload_history = (
        st.session_state.upload_history[-60:]
    )


def reset_upload():

    st.session_state.upload_result = None

    st.session_state.upload_detections = None

    st.session_state.upload_filename = None

    st.session_state.upload_history = []

    st.session_state.upload_sound_played = False


def analyse_uploaded_image(
    image,
    filename,
    confidence
):

    with st.spinner(
        "Analysing parking..."
    ):

        (
            annotated,
            insights,
            detections
        ) = run_detection(
            image,
            confidence
        )

    st.session_state.upload_result = (
        annotated,
        insights
    )

    st.session_state.upload_detections = (
        detections
    )

    st.session_state.upload_filename = (
        filename
    )

    add_upload_history(
        insights,
        filename
    )

    st.session_state.upload_sound_played = False


def render_upload_results():

    if (
        st.session_state.upload_result
        is None
    ):
        return

    (
        annotated,
        insights
    ) = (
        st.session_state.upload_result
    )

    detections = (
        st.session_state.upload_detections
    )

    filename = (
        st.session_state.upload_filename
    )

    st.divider()

    st.header(
        "Parking Statistics"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Total",
            insights["total"]
        )

    with col2:

        st.metric(
            "Available",
            insights["available"]
        )

    with col3:

        st.metric(
            "Occupied",
            insights["occupied"]
        )

    with col4:

        st.metric(
            "Occupancy",
            f'{insights["occupancy"]:.1f}%'
        )

    st.divider()

    st.header(
        "Parking Status"
    )

    availability_ratio = 0.0

    if insights["total"] > 0:

        availability_ratio = (
            insights["available"]
            / insights["total"]
            * 100
        )

    level = parking_level(
        availability_ratio
    )

    status1, status2 = (
        st.columns(2)
    )

    with status1:

        if level == "green":

            st.success(
                "🟢 PARKING AVAILABLE"
            )

        elif level == "yellow":

            st.warning(
                "🟡 PARKING FILLING UP"
            )

        else:

            st.error(
                "🔴 PARKING NEARLY FULL"
            )

    with status2:

        st.metric(
            "Availability",
            f"{availability_ratio:.1f}%"
        )

    if insights["total"] == 0:

        st.warning(
            "No parking slots were detected."
        )

    else:

        st.info(
            insights["recommendation"]
        )

    if (
        sound_enabled
        and insights["total"] > 0
        and not st.session_state.upload_sound_played
    ):

        sound_alert(
            level,
            True
        )

        st.session_state.upload_sound_played = True

    st.divider()

    st.header(
        "Parking Map"
    )

    st.markdown(
        '<div class="image-container">',
        unsafe_allow_html=True
    )

    st.image(
        annotated,
        channels="BGR",
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇️ Download Annotated Image",
        data=annotated_bytes(
            annotated
        ),
        file_name="parkvision_annotated.png",
        mime="image/png",
        use_container_width=True
    )

    st.divider()

    st.header(
        "Parking Intelligence"
    )

    occupancy = float(
        insights["occupancy"]
    )

    st.progress(
        min(
            max(
                occupancy / 100,
                0.0
            ),
            1.0
        ),
        text=(
            f"{occupancy:.1f}% occupied"
        )
    )

    congestion = (
        insights["congestion"]
    )

    if congestion == "Low":

        st.success(
            "🟢 LOW CONGESTION"
        )

    elif congestion == "Moderate":

        st.warning(
            "🟡 MODERATE CONGESTION"
        )

    elif congestion == "High":

        st.error(
            "🔴 HIGH CONGESTION"
        )

    else:

        st.info(
            congestion
        )

    st.info(
        insights["recommendation"]
    )

    info1, info2, info3 = (
        st.columns(3)
    )

    with info1:

        st.write(
            "Source"
        )

        st.write(
            filename
        )

    with info2:

        st.write(
            "Analysed"
        )

        st.write(
            datetime.now().strftime(
                "%d %b %Y • %I:%M:%S %p"
            )
        )

    with info3:

        st.write(
            "Confidence"
        )

        st.write(
            f"{confidence:.2f}"
        )

    st.divider()

    st.header(
        "Image Analytics"
    )

    graph1, graph2 = (
        st.columns(2)
    )

    with graph1:

        graph_metric = st.selectbox(
            "Metric",
            [
                "Parking Distribution",
                "Detection Confidence"
            ],
            key="upload_graph_metric"
        )

    with graph2:

        graph_type = st.selectbox(
            "Chart Type",
            [
                "Bar",
                "Pie"
            ],
            key="upload_graph_type"
        )

    fig = go.Figure()

    if (
        graph_metric
        == "Parking Distribution"
    ):

        labels = [
            "Available",
            "Occupied"
        ]

        values = [
            insights["available"],
            insights["occupied"]
        ]

        if graph_type == "Bar":

            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    text=values,
                    textposition="auto"
                )
            )

        else:

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.45
                )
            )

    else:

        confidence_values = []

        if (
            detections is not None
            and not detections.empty
        ):

            confidence_values = (
                detections[
                    "Confidence (%)"
                ].tolist()
            )

        if confidence_values:

            if graph_type == "Bar":

                fig.add_trace(
                    go.Bar(
                        x=[
                            f"Slot {i + 1}"
                            for i in range(
                                len(
                                    confidence_values
                                )
                            )
                        ],
                        y=confidence_values,
                        text=confidence_values,
                        textposition="auto"
                    )
                )

            else:

                fig.add_trace(
                    go.Pie(
                        labels=[
                            f"Slot {i + 1}"
                            for i in range(
                                len(
                                    confidence_values
                                )
                            )
                        ],
                        values=confidence_values,
                        hole=0.45
                    )
                )

        else:

            st.info(
                "No confidence data available."
            )

    fig.update_layout(
        height=390,
        margin={
            "l": 10,
            "r": 10,
            "t": 30,
            "b": 10
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#cbd5e1"
        },
        showlegend=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False
        }
    )

    st.divider()

    st.header(
        "Detection Details"
    )

    if (
        detections is None
        or detections.empty
    ):

        st.info(
            "No parking slots detected."
        )

    else:

        available_count = int(
            (
                detections["Status"]
                == "Available"
            ).sum()
        )

        occupied_count = int(
            (
                detections["Status"]
                == "Occupied"
            ).sum()
        )

        d1, d2 = (
            st.columns(2)
        )

        with d1:

            st.metric(
                "Available Detections",
                available_count
            )

        with d2:

            st.metric(
                "Occupied Detections",
                occupied_count
            )

        with st.expander(
            "View Individual Detections"
        ):

            st.dataframe(
                detections,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Export Detection Data",
                data=(
                    detections
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                ),
                file_name=(
                    "parkvision_detections.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

    st.divider()

    st.header(
        "Session Analytics"
    )

    history = st.session_state.get(
        "upload_history",
        []
    )

    if history:

        history_df = pd.DataFrame(
            history
        )

        peak = (
            history_df[
                "Occupancy"
            ].max()
        )

        average = (
            history_df[
                "Occupancy"
            ].mean()
        )

        session1, session2, session3 = (
            st.columns(3)
        )

        with session1:

            st.metric(
                "Images Analysed",
                len(history_df)
            )

        with session2:

            st.metric(
                "Peak Occupancy",
                f"{peak:.1f}%"
            )

        with session3:

            st.metric(
                "Average Occupancy",
                f"{average:.1f}%"
            )

        if len(history_df) >= 2:

            st.subheader(
                "Occupancy History"
            )

            history_fig = go.Figure()

            history_fig.add_trace(
                go.Scatter(
                    x=history_df["Time"],
                    y=history_df[
                        "Occupancy"
                    ],
                    mode="lines+markers",
                    line={
                        "width": 3
                    },
                    marker={
                        "size": 7
                    },
                    name="Occupancy"
                )
            )

            history_fig.update_layout(
                height=350,
                margin={
                    "l": 10,
                    "r": 10,
                    "t": 30,
                    "b": 10
                },
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={
                    "color": "#cbd5e1"
                },
                xaxis={
                    "showgrid": False
                },
                yaxis={
                    "title": "Occupancy (%)",
                    "gridcolor":
                        "rgba(148,163,184,0.10)"
                },
                showlegend=False
            )

            st.plotly_chart(
                history_fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "displaylogo": False
                }
            )

        st.download_button(
            "⬇️ Download Session Analytics",
            data=(
                history_df
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            ),
            file_name=(
                "parkvision_image_analytics.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


if model is None:

    st.error(
        "Trained model could not be found."
    )

    st.stop()


if (
    "upload_result"
    not in st.session_state
):

    st.session_state.upload_result = None


if (
    "upload_detections"
    not in st.session_state
):

    st.session_state.upload_detections = None


if (
    "upload_filename"
    not in st.session_state
):

    st.session_state.upload_filename = None


if (
    "upload_history"
    not in st.session_state
):

    st.session_state.upload_history = []


if (
    "upload_sound_played"
    not in st.session_state
):

    st.session_state.upload_sound_played = False



if model is None:

    st.error(
        "Trained model could not be found."
    )

    st.stop()


if (
    "upload_result"
    not in st.session_state
):

    st.session_state.upload_result = None


if (
    "upload_detections"
    not in st.session_state
):

    st.session_state.upload_detections = None


if (
    "upload_filename"
    not in st.session_state
):

    st.session_state.upload_filename = None


if (
    "upload_history"
    not in st.session_state
):

    st.session_state.upload_history = []


if (
    "upload_sound_played"
    not in st.session_state
):

    st.session_state.upload_sound_played = False


with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">P</div>
            <div>
                <div class="sidebar-name">ParkVision</div>
                <div class="sidebar-tagline">AI PARKING INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="live-status"><span class="dot"></span>AI SYSTEM READY</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.write("Confidence")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        label_visibility="collapsed"
    )

    st.divider()

    sound_enabled = st.toggle(
        "🔊 Sound alerts",
        value=True,
        key="global_sound_alerts"
    )

    st.divider()

    st.caption("PARKVISION AI")
    st.caption("SMARTER PARKING • GREENER CITIES")



def render_home_page():
    st.markdown(
        """
        <div class="hero-panel home-hero">
            <div class="hero-eyebrow">WELCOME TO PARKVISION</div>
            <div class="hero-title">Smarter Parking.</div>
            <div class="hero-subtitle">AI-POWERED PARKING SPACE DETECTION</div>
            <div class="hero-pills">
                <span class="hero-pill"><span class="pill-dot"></span>REAL-TIME</span>
                <span class="hero-pill"><span class="pill-dot"></span>ACCURATE</span>
                <span class="hero-pill"><span class="pill-dot"></span>SMARTER CITIES</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    history = st.session_state.get("upload_history", [])
    live = live_stats.read()

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric("Total Spaces", live["total"])

    with h2:
        st.metric("Available", live["empty"])

    with h3:
        st.metric("Occupied", live["occupied"])

    with h4:
        st.metric("Occupancy", f'{live["occupancy"]:.1f}%')

    st.divider()

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">GET STARTED</div>
                <div class="card-title">Choose how you want to monitor the parking area</div>
                <div class="card-subtitle">Analyse a saved parking image or switch to the live camera for continuous detection.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        x, y = st.columns(2)

        with x:
            if st.button(
                "▧  Analyse an Image",
                type="primary",
                use_container_width=True
            ):
                st.session_state["nav_target"] = "▧  Upload Image"
                st.rerun()

        with y:
            if st.button(
                "◉  Open Live Monitor",
                use_container_width=True
            ):
                st.session_state["nav_target"] = "◉  Live Monitor"
                st.rerun()

    with right:
        st.markdown(
            f"""
            <div class="live-card">
                <div class="card-kicker">SYSTEM STATUS</div>
                <div class="card-title">{"LIVE DETECTION ACTIVE" if live["active"] else "READY TO DETECT"}</div>
                <div class="card-subtitle">{"Last update: " + str(live["updated"]) if live["active"] else "Start a camera session to begin real-time analysis."}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    if history:
        history_df = pd.DataFrame(history)
        latest = history_df.iloc[-1]

        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">LATEST ANALYSIS</div>
                <div class="card-title">Most recent parking image</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        q1, q2, q3, q4 = st.columns(4)

        with q1:
            st.write("**Image**")
            st.write(str(latest.get("Image", "Unknown")))

        with q2:
            st.metric("Available", int(latest["Available"]))

        with q3:
            st.metric("Occupied", int(latest["Occupied"]))

        with q4:
            st.metric("Occupancy", f'{float(latest["Occupancy"]):.1f}%')
    else:
        st.info("No image analyses yet. Your first result will appear on this dashboard.")



def render_analytics_page():
    st.markdown(
        """
        <div class="page-heading">
            <div class="page-kicker">ANALYTICS</div>
            <div class="page-title">Parking Analytics</div>
            <div class="page-subtitle">Review how occupancy changed across your analysed parking images.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    history = st.session_state.get(
        "upload_history",
        []
    )

    if not history:
        st.markdown(
            """
            <div class="dashboard-card empty-card">
                <div class="card-title">No image history yet</div>
                <div class="card-subtitle">Analyse a parking image first. Your results will appear here automatically.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    history_df = pd.DataFrame(history)

    total_images = len(history_df)
    peak = float(history_df["Occupancy"].max())
    average = float(history_df["Occupancy"].mean())
    latest = float(history_df.iloc[-1]["Occupancy"])

    a, b, c, d = st.columns(4)

    with a:
        st.metric("Images Analysed", total_images)

    with b:
        st.metric("Peak Occupancy", f"{peak:.1f}%")

    with c:
        st.metric("Average Occupancy", f"{average:.1f}%")

    with d:
        st.metric("Latest Occupancy", f"{latest:.1f}%")

    st.divider()

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">OCCUPANCY TREND</div>
                <div class="card-title">Historical parking utilisation</div>
                <div class="card-subtitle">Occupancy recorded after every image analysis.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=history_df["Time"],
                y=history_df["Occupancy"],
                mode="lines+markers",
                fill="tozeroy",
                line={"width": 3},
                marker={"size": 7},
                name="Occupancy"
            )
        )

        fig.update_layout(
            height=410,
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#cbd5e1"},
            xaxis={"showgrid": False},
            yaxis={
                "title": "Occupancy (%)",
                "gridcolor": "rgba(148,163,184,0.10)"
            },
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False}
        )

    with right:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">LATEST IMAGE</div>
                <div class="card-title">Most recent analysis</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        latest_row = history_df.iloc[-1]

        st.metric(
            "Image",
            str(latest_row.get("Image", "Unknown"))
        )

        r1, r2 = st.columns(2)

        with r1:
            st.metric("Available", int(latest_row["Available"]))

        with r2:
            st.metric("Occupied", int(latest_row["Occupied"]))

        st.write(f"Analysed at **{latest_row['Time']}**")

    st.divider()

    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-kicker">ANALYSIS HISTORY</div>
            <div class="card-title">Previous parking results</div>
            <div class="card-subtitle">Every analysed image, its occupancy, and detected space counts.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    display_df = history_df.copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Analysis History",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="parkvision_analysis_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    if len(history_df) >= 2:
        compare = history_df[["Time", "Occupancy", "Available", "Occupied", "Total"]].copy()

        st.divider()

        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">SPACE DISTRIBUTION</div>
                <div class="card-title">Available vs occupied over time</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        dist_fig = go.Figure()

        dist_fig.add_trace(
            go.Scatter(
                x=compare["Time"],
                y=compare["Available"],
                mode="lines+markers",
                name="Available",
                line={"width": 3}
            )
        )

        dist_fig.add_trace(
            go.Scatter(
                x=compare["Time"],
                y=compare["Occupied"],
                mode="lines+markers",
                name="Occupied",
                line={"width": 3}
            )
        )

        dist_fig.update_layout(
            height=360,
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#cbd5e1"},
            xaxis={"showgrid": False},
            yaxis={"gridcolor": "rgba(148,163,184,0.10)"}
        )

        st.plotly_chart(
            dist_fig,
            use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False}
        )



def render_history_page():
    st.markdown(
        """
        <div class="page-heading">
            <div class="page-kicker">DETECTION HISTORY</div>
            <div class="page-title">Previous Analyses</div>
            <div class="page-subtitle">A clean record of every parking image analysed in this session.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    history = st.session_state.get(
        "upload_history",
        []
    )

    if not history:
        st.info("No detections yet. Go to Upload Image and analyse a parking image.")
        return

    history_df = pd.DataFrame(history)

    for i, row in history_df.iloc[::-1].iterrows():
        st.markdown(
            f"""
            <div class="history-row">
                <div>
                    <div class="history-name">{row.get("Image", "Parking image")}</div>
                    <div class="history-meta">{row["Time"]} • {int(row["Total"])} total spaces</div>
                </div>
                <div class="history-stats">
                    <span>{int(row["Available"])} available</span>
                    <span>{int(row["Occupied"])} occupied</span>
                    <strong>{float(row["Occupancy"]):.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.download_button(
        "⬇️ Export Full History",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="parkvision_detection_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button(
        "🗑 Clear Detection History",
        use_container_width=True
    ):
        st.session_state.upload_history = []
        st.rerun()



def render_settings_page():
    st.markdown(
        """
        <div class="page-heading">
            <div class="page-kicker">SETTINGS</div>
            <div class="page-title">System Controls</div>
            <div class="page-subtitle">Tune ParkVision without changing the underlying model.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">DETECTION</div>
                <div class="card-title">Confidence threshold</div>
                <div class="card-subtitle">Higher values show only more confident detections.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"Current threshold: **{confidence:.2f}**")

        st.info("Use the Confidence slider in the sidebar to change this setting.")

    with s2:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">AUDIO</div>
                <div class="card-title">Sound alerts</div>
                <div class="card-subtitle">The sidebar switch controls image and live-camera alerts.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if sound_enabled:
            st.success("Sound alerts are ON.")
        else:
            st.warning("Sound alerts are OFF.")

    st.divider()

    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-kicker">SESSION DATA</div>
            <div class="card-title">Reset local analysis data</div>
            <div class="card-subtitle">This clears image history and current live statistics for this session.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "↻ Reset All Session Data",
        type="primary",
        use_container_width=True
    ):
        reset_upload()
        live_stats.reset()
        st.rerun()



def render_about_page():
    st.markdown(
        """
        <div class="page-heading">
            <div class="page-kicker">ABOUT PARKVISION</div>
            <div class="page-title">Smarter Parking.</div>
            <div class="page-subtitle">AI-powered parking-space detection designed for faster decisions and better utilisation.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">01</div>
                <div class="card-title">Detect</div>
                <div class="card-subtitle">YOLO computer vision identifies occupied and available parking spaces.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">02</div>
                <div class="card-title">Understand</div>
                <div class="card-subtitle">Occupancy, availability and congestion are converted into useful insights.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-kicker">03</div>
                <div class="card-title">Act</div>
                <div class="card-subtitle">Live monitoring and historical analytics help operators respond quickly.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-kicker">PROJECT STACK</div>
            <div class="card-title">ParkVision AI</div>
            <div class="card-subtitle">Streamlit • YOLO • Python • Plotly • WebRTC</div>
        </div>
        """,
        unsafe_allow_html=True
    )



def render_upload_page():
    st.header(
        "Analyse Parking"
    )

    uploaded_file = st.file_uploader(
        "Upload parking image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = (
            Image
            .open(uploaded_file)
            .convert("RGB")
        )

        st.subheader(
            "Input Image"
        )

        st.markdown(
            '<div class="image-container">',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        image_info1, image_info2, image_info3 = (
            st.columns(3)
        )

        with image_info1:

            st.metric(
                "Width",
                f"{image.width}px"
            )

        with image_info2:

            st.metric(
                "Height",
                f"{image.height}px"
            )

        with image_info3:

            st.metric(
                "Confidence",
                f"{confidence:.2f}"
            )

        st.write(
            uploaded_file.name
        )

        action1, action2 = (
            st.columns(2)
        )

        with action1:

            analyse_clicked = st.button(
                "🔍 Analyse Parking",
                type="primary",
                use_container_width=True,
                key="analyse_upload"
            )

        with action2:

            reset_clicked = st.button(
                "↻ Reset Analysis",
                use_container_width=True,
                key="reset_upload"
            )

        if reset_clicked:

            reset_upload()

            st.rerun()

        if analyse_clicked:

            analyse_uploaded_image(
                image,
                uploaded_file.name,
                confidence
            )

            st.rerun()

    else:

        st.info(
            "Upload an image to begin."
        )

    render_upload_results()


def render_live_page():
    try:

        from streamlit_autorefresh import (
            st_autorefresh
        )

        st_autorefresh(
            interval=1000,
            limit=None,
            key="parkvision_live_refresh"
        )

    except ImportError:

        st.error(
            "Install streamlit-autorefresh."
        )

        st.code(
            "python -m pip install streamlit-autorefresh"
        )

    st.header(
        "Live Monitor"
    )

    st.markdown(
        """
        <div class="live-status">
            <span class="dot"></span>
            LIVE DETECTION
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    try:

        import av

        from streamlit_webrtc import (
            VideoProcessorBase,
            RTCConfiguration,
            WebRtcMode,
            webrtc_streamer
        )

        class ParkVisionProcessor(
            VideoProcessorBase
        ):

            def __init__(self):

                self.confidence = 0.35

                self.last_fps_time = (
                    time.time()
                )

                self.frame_count = 0

                self.last_history_time = (
                    time.time()
                )

            def recv(
                self,
                frame
            ):

                image = frame.to_ndarray(
                    format="bgr24"
                )

                results = model.predict(
                    source=image,
                    imgsz=512,
                    conf=self.confidence,
                    verbose=False
                )

                result = results[0]

                annotated = result.plot()

                names = result.names

                occupied = 0

                available = 0

                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):

                    for class_id in (
                        result
                        .boxes
                        .cls
                        .tolist()
                    ):

                        label = str(
                            names[
                                int(
                                    class_id
                                )
                            ]
                        ).lower()

                        if (
                            "occupied"
                            in label
                        ):

                            occupied += 1

                        elif (
                            "empty"
                            in label
                            or "free"
                            in label
                        ):

                            available += 1

                total = (
                    occupied
                    + available
                )

                if total > 0:

                    occupancy = (
                        occupied
                        / total
                        * 100
                    )

                else:

                    occupancy = 0.0

                self.frame_count += 1

                current_time = (
                    time.time()
                )

                elapsed = (
                    current_time
                    - self.last_fps_time
                )

                if elapsed >= 1:

                    fps = (
                        self.frame_count
                        / elapsed
                    )

                    self.frame_count = 0

                    self.last_fps_time = (
                        current_time
                    )

                else:

                    fps = (
                        live_stats
                        .read()
                        ["fps"]
                    )

                current_stats = (
                    live_stats.read()
                )

                peak = max(
                    current_stats[
                        "peak_occupancy"
                    ],
                    occupancy
                )

                frames = (
                    current_stats[
                        "frames"
                    ]
                    + 1
                )

                live_stats.update(
                    occupied=occupied,
                    empty=available,
                    total=total,
                    occupancy=occupancy,
                    fps=fps,
                    updated=(
                        datetime.now()
                        .strftime(
                            "%H:%M:%S"
                        )
                    ),
                    active=True,
                    frames=frames,
                    peak_occupancy=peak
                )

                history_time = (
                    time.time()
                )

                if (
                    history_time
                    - self.last_history_time
                    >= 1
                ):

                    live_stats.add_history()

                    self.last_history_time = (
                        history_time
                    )

                return (
                    av.VideoFrame
                    .from_ndarray(
                        annotated,
                        format="bgr24"
                    )
                )

        st.subheader(
            "Camera"
        )

        context = webrtc_streamer(
            key="parkvision-live-final",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {
                    "iceServers": [
                        {
                            "urls": [
                                "stun:stun.l.google.com:19302"
                            ]
                        }
                    ]
                }
            ),
            video_processor_factory=(
                ParkVisionProcessor
            ),
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        if context.video_processor:

            context.video_processor.confidence = (
                confidence
            )

        st.divider()

        live = live_stats.read()

        st.header(
            "Parking Statistics"
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Total",
                live["total"]
            )

        with col2:

            st.metric(
                "Available",
                live["empty"]
            )

        with col3:

            st.metric(
                "Occupied",
                live["occupied"]
            )

        with col4:

            st.metric(
                "Occupancy",
                f'{live["occupancy"]:.1f}%'
            )

        availability_ratio = 0.0

        if live["total"] > 0:

            availability_ratio = (
                live["empty"]
                / live["total"]
                * 100
            )

        if live["total"] == 0:

            st.warning(
                "Waiting for detection..."
            )

        elif availability_ratio >= 60:

            st.success(
                "🟢 PARKING AVAILABLE"
            )

            sound_alert(
                "green",
                sound_enabled
            )

        elif availability_ratio >= 30:

            st.warning(
                "🟡 PARKING FILLING UP"
            )

            sound_alert(
                "yellow",
                sound_enabled
            )

        else:

            st.error(
                "🔴 PARKING NEARLY FULL"
            )

            sound_alert(
                "red",
                sound_enabled
            )

        st.divider()

        st.header(
            "Live Analytics"
        )

        graph_col1, graph_col2, graph_col3 = (
            st.columns(3)
        )

        with graph_col1:

            graph_metric = st.selectbox(
                "Metric",
                [
                    "Occupancy",
                    "Available",
                    "Occupied",
                    "Total"
                ],
                key="graph_metric"
            )

        with graph_col2:

            graph_range = st.selectbox(
                "Time range",
                [
                    "15 seconds",
                    "30 seconds",
                    "60 seconds",
                    "120 seconds"
                ],
                index=2,
                key="graph_range"
            )

        with graph_col3:

            graph_type = st.selectbox(
                "Chart",
                [
                    "Line",
                    "Bar"
                ],
                key="graph_type"
            )

        history = (
            live_stats
            .read_history()
        )

        if len(history) >= 2:

            history_df = pd.DataFrame(
                history
            )

            range_map = {
                "15 seconds": 15,
                "30 seconds": 30,
                "60 seconds": 60,
                "120 seconds": 120
            }

            points = range_map[
                graph_range
            ]

            history_df = (
                history_df
                .tail(points)
            )

            fig = go.Figure()

            if graph_type == "Line":

                fig.add_trace(
                    go.Scatter(
                        x=history_df["Time"],
                        y=history_df[
                            graph_metric
                        ],
                        mode="lines+markers",
                        line={
                            "width": 3
                        },
                        marker={
                            "size": 6
                        },
                        name=graph_metric
                    )
                )

            else:

                fig.add_trace(
                    go.Bar(
                        x=history_df["Time"],
                        y=history_df[
                            graph_metric
                        ],
                        name=graph_metric
                    )
                )

            fig.update_layout(
                height=390,
                margin={
                    "l": 10,
                    "r": 10,
                    "t": 30,
                    "b": 10
                },
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={
                    "color": "#cbd5e1"
                },
                xaxis={
                    "showgrid": False
                },
                yaxis={
                    "title": graph_metric,
                    "gridcolor":
                        "rgba(148,163,184,0.10)"
                },
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "displaylogo": False
                }
            )

            download_data = (
                history_df
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )

            st.download_button(
                "⬇️ Download Graph Data",
                data=download_data,
                file_name=(
                    "parkvision_analytics.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "Start the camera and wait a few seconds for the live graph."
            )

        st.divider()

        st.header(
            "Session Analytics"
        )

        analytics1, analytics2, analytics3 = (
            st.columns(3)
        )

        with analytics1:

            st.metric(
                "Peak Occupancy",
                f'{live["peak_occupancy"]:.1f}%'
            )

        with analytics2:

            st.metric(
                "Frames Processed",
                f'{live["frames"]:,}'
            )

        with analytics3:

            st.metric(
                "Processing Speed",
                f'{live["fps"]:.1f} FPS'
            )

        st.divider()

        st.header(
            "Quick Controls"
        )

        control1, control2 = (
            st.columns(2)
        )

        with control1:

            if st.button(
                "↻ Reset Session",
                use_container_width=True
            ):

                live_stats.reset()

                st.rerun()

        with control2:

            if st.button(
                "⟳ Refresh Dashboard",
                use_container_width=True
            ):

                st.rerun()

        st.divider()

        st.header(
            "System Status"
        )

        status1, status2 = (
            st.columns(2)
        )

        with status1:

            if live["active"]:

                st.success(
                    "● SYSTEM ACTIVE"
                )

            else:

                st.warning(
                    "● SYSTEM WAITING"
                )

        with status2:

            st.metric(
                "Last Detection",
                live["updated"]
            )

    except ImportError:

        st.error(
            "Live camera packages are missing."
        )

        st.code(
            "python -m pip install streamlit-webrtc av streamlit-autorefresh plotly"
        )


pages = {
    "": [
        st.Page(
            render_home_page,
            title="Home",
            icon="⌂"
        ),
        st.Page(
            render_upload_page,
            title="Upload Image",
            icon="▧"
        ),
        st.Page(
            render_live_page,
            title="Live Monitor",
            icon="◉"
        ),
        st.Page(
            render_analytics_page,
            title="Analytics",
            icon="▥"
        ),
        st.Page(
            render_history_page,
            title="Detection History",
            icon="◷"
        ),
        st.Page(
            render_settings_page,
            title="Settings",
            icon="⚙"
        ),
        st.Page(
            render_about_page,
            title="About",
            icon="ⓘ"
        )
    ]
}


navigation = st.navigation(
    pages,
    position="sidebar",
    expanded=True
)


st.markdown(
    """
    <div class="dashboard-card" style="text-align:center;margin-top:28px;">
        <div style="color:#67e8f9;font-size:12px;font-weight:800;letter-spacing:1px;">
            ✦ SMARTER PARKING. GREENER CITIES. HAPPIER PEOPLE.
        </div>
        <div style="color:#707286;font-size:10px;margin-top:6px;">
            Powered by AI • ParkVision
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

navigation.run()
