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

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "parkvision_yolo26s"
    / "weights"
    / "best.pt"
)

st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(37, 99, 235, 0.16), transparent 32%),
            radial-gradient(circle at 5% 100%, rgba(14, 165, 233, 0.08), transparent 30%),
            #060a12;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a101d 0%, #060a12 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2.3rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-weight: 800 !important;
        letter-spacing: -2px !important;
    }

    h2 {
        font-weight: 750 !important;
        letter-spacing: -1px !important;
    }

    h3 {
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(17, 25, 42, 0.98),
            rgba(9, 15, 27, 0.98)
        );
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 18px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
        transition: all 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.30);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
    }

    .image-container {
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 20px;
        padding: 8px;
        background: #0b1220;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }

    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 9px 16px;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.25);
        color: #34d399;
        font-weight: 750;
        font-size: 13px;
    }

    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.8);
    }

    .sound-card {
        background: linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.96),
            rgba(9, 15, 27, 0.96)
        );
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 18px;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #475569;
        font-size: 11px;
        padding-top: 35px;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 42px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return YOLO(str(MODEL_PATH))


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
            return dict(self.data)

    def add_history(self):
        with self.lock:
            self.history.append(
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Occupancy": round(self.data["occupancy"], 1),
                    "Available": self.data["empty"],
                    "Occupied": self.data["occupied"],
                    "Total": self.data["total"]
                }
            )

            if len(self.history) > 120:
                self.history.pop(0)

    def read_history(self):
        with self.lock:
            return list(self.history)

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


def make_tone(frequency, duration=0.25, volume=0.35):
    sample_rate = 44100
    samples = int(sample_rate * duration)
    audio = bytearray()

    for i in range(samples):
        value = math.sin(
            2 * math.pi * frequency * i / sample_rate
        )

        value *= volume

        sample = int(value * 32767)

        audio.extend(
            int(sample).to_bytes(
                2,
                byteorder="little",
                signed=True
            )
        )

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(audio))

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


GREEN_SOUND = make_tone(880, 0.22)
YELLOW_SOUND = make_tone(520, 0.22)
RED_SOUND = make_tone(220, 0.35)


def sound_alert(level, enabled=True):
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


def get_parking_level(available_ratio):
    if available_ratio >= 60:
        return "green"

    if available_ratio >= 30:
        return "yellow"

    return "red"


def get_level_text(level):
    if level == "green":
        return "🟢 PARKING AVAILABLE"

    if level == "yellow":
        return "🟡 PARKING FILLING UP"

    return "🔴 PARKING NEARLY FULL"


def run_detection(image, confidence):
    image = image.convert("RGB")

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
        for i in range(len(result.boxes)):
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

            box = result.boxes.xyxy[i].tolist()

            detections.append(
                {
                    "Slot": len(detections) + 1,
                    "Status": status,
                    "Confidence (%)": round(
                        confidence_score * 100,
                        1
                    ),
                    "X1": round(box[0], 1),
                    "Y1": round(box[1], 1),
                    "X2": round(box[2], 1),
                    "Y2": round(box[3], 1)
                }
            )

    insights = parking_insights(
        occupied,
        empty
    )

    return (
        annotated,
        insights,
        pd.DataFrame(detections)
    )


def annotated_bytes(annotated):
    rgb = annotated[:, :, ::-1]

    image = Image.fromarray(rgb)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


def update_upload_history(insights):
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []

    st.session_state.upload_history.append(
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Occupancy": round(
                insights["occupancy"],
                1
            ),
            "Available": insights["available"],
            "Occupied": insights["occupied"],
            "Total": insights["total"]
        }
    )

    st.session_state.upload_history = (
        st.session_state.upload_history[-30:]
    )


def reset_upload_session():
    st.session_state.upload_history = []
    st.session_state.upload_result = None
    st.session_state.upload_detections = None
    st.session_state.upload_filename = None


def display_image_analysis(
    image,
    filename,
    confidence
):
    with st.spinner("Analysing parking..."):
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

    st.session_state.upload_detections = detections
    st.session_state.upload_filename = filename

    update_upload_history(insights)

    st.header("Parking Statistics")

    col1, col2, col3, col4 = st.columns(4)

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

    st.header("Parking Status")

    availability_ratio = 0.0

    if insights["total"] > 0:
        availability_ratio = (
            insights["available"]
            / insights["total"]
            * 100
        )

    level = get_parking_level(
        availability_ratio
    )

    status_col1, status_col2 = st.columns(2)

    with status_col1:
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

    with status_col2:
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

    sound_col1, sound_col2 = st.columns(2)

    with sound_col1:
        sound_enabled = st.toggle(
            "🔊 Sound alert",
            value=False,
            key="upload_sound_enabled"
        )

    with sound_col2:
        if st.button(
            "🔊 Test Sound",
            use_container_width=True,
            key="upload_test_sound"
        ):
            sound_alert(
                level,
                True
            )

    if sound_enabled and insights["total"] > 0:
        sound_alert(
            level,
            True
        )

    st.divider()

    st.header("Parking Map")

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

    st.header("Parking Intelligence")

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
        text=f"{occupancy:.1f}% occupied"
    )

    congestion = insights["congestion"]

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

    info1, info2, info3 = st.columns(3)

    with info1:
        st.write("Source")
        st.write(filename)

    with info2:
        st.write("Analysed")
        st.write(
            datetime.now().strftime(
                "%d %b %Y • %I:%M:%S %p"
            )
        )

    with info3:
        st.write("Confidence")
        st.write(
            f"{confidence:.2f}"
        )

    st.divider()

    st.header("Image Analytics")

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        image_graph_metric = st.selectbox(
            "Metric",
            [
                "Parking Distribution",
                "Confidence"
            ],
            key="image_graph_metric"
        )

    with graph_col2:
        image_graph_type = st.selectbox(
            "Chart",
            [
                "Bar",
                "Pie"
            ],
            key="image_graph_type"
        )

    if image_graph_metric == "Parking Distribution":
        labels = [
            "Available",
            "Occupied"
        ]

        values = [
            insights["available"],
            insights["occupied"]
        ]

        fig = go.Figure()

        if image_graph_type == "Bar":
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

        if not detections.empty:
            confidence_values = (
                detections[
                    "Confidence (%)"
                ].tolist()
            )

        fig = go.Figure()

        if confidence_values:
            if image_graph_type == "Bar":
                fig.add_trace(
                    go.Bar(
                        x=list(
                            range(
                                1,
                                len(
                                    confidence_values
                                ) + 1
                            )
                        ),
                        y=confidence_values,
                        text=confidence_values,
                        textposition="auto"
                    )
                )
            else:
                fig.add_trace(
                    go.Pie(
                        values=confidence_values,
                        labels=[
                            f"Slot {i + 1}"
                            for i in range(
                                len(
                                    confidence_values
                                )
                            )
                        ],
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

    st.header("Detection Details")

    if detections.empty:
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

        d1, d2 = st.columns(2)

        with d1:
            st.metric(
                "Available detections",
                available_count
            )

        with d2:
            st.metric(
                "Occupied detections",
                occupied_count
            )

        with st.expander(
            "View individual detections"
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
                file_name="parkvision_detections.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.divider()

    st.header("Session Analytics")

    upload_history = st.session_state.get(
        "upload_history",
        []
    )

    session_col1, session_col2, session_col3 = st.columns(3)

    with session_col1:
        st.metric(
            "Images Analysed",
            len(upload_history)
        )

    with session_col2:
        if upload_history:
            peak = max(
                item["Occupancy"]
                for item in upload_history
            )
        else:
            peak = 0.0

        st.metric(
            "Peak Occupancy",
            f"{peak:.1f}%"
        )

    with session_col3:
        if upload_history:
            average = sum(
                item["Occupancy"]
                for item in upload_history
            ) / len(upload_history)
        else:
            average = 0.0

        st.metric(
            "Average Occupancy",
            f"{average:.1f}%"
        )

    if len(upload_history) >= 2:
        st.subheader(
            "Occupancy History"
        )

        history_df = pd.DataFrame(
            upload_history
        )

        history_fig = go.Figure()

        history_fig.add_trace(
            go.Scatter(
                x=history_df["Time"],
                y=history_df["Occupancy"],
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
                "gridcolor": "rgba(148,163,184,0.10)"
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
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name="parkvision_image_analytics.csv",
            mime="text/csv",
            use_container_width=True
        )


if model is None:
    st.error(
        "Trained model could not be found."
    )

    st.stop()


if "upload_history" not in st.session_state:
    st.session_state.upload_history = []

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

if "upload_detections" not in st.session_state:
    st.session_state.upload_detections = None

if "upload_filename" not in st.session_state:
    st.session_state.upload_filename = None


with st.sidebar:
    st.markdown(
        "## 🅿️ ParkVision"
    )

    st.divider()

    st.success(
        "● AI READY"
    )

    st.write(
        "Confidence"
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        label_visibility="collapsed"
    )

    st.divider()

    mode = st.radio(
        "Mode",
        [
            "Upload image",
            "Live camera"
        ]
    )

    if mode == "Live camera":
        st.divider()

        sound_enabled = st.toggle(
            "🔊 Sound alerts",
            value=True
        )


st.title(
    "ParkVision AI"
)

st.caption(
    "SMART PARKING • COMPUTER VISION"
)


if mode == "Upload image":

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

        image_info1, image_info2, image_info3 = st.columns(3)

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

        action1, action2 = st.columns(2)

        with action1:
            analyse_clicked = st.button(
                "🔍 Analyse Parking",
                type="primary",
                use_container_width=True
            )

        with action2:
            reset_clicked = st.button(
                "↻ Reset Analysis",
                use_container_width=True
            )

        if reset_clicked:
            reset_upload_session()
            st.rerun()

        if analyse_clicked:
            display_image_analysis(
                image,
                uploaded_file.name,
                confidence
            )

    else:

        st.info(
            "Upload an image to begin."
        )

        if st.session_state.upload_history:
            st.divider()

            st.header(
                "Session Analytics"
            )

            upload_history = pd.DataFrame(
                st.session_state.upload_history
            )

            s1, s2, s3 = st.columns(3)

            with s1:
                st.metric(
                    "Images Analysed",
                    len(upload_history)
                )

            with s2:
                st.metric(
                    "Peak Occupancy",
                    f"{upload_history['Occupancy'].max():.1f}%"
                )

            with s3:
                st.metric(
                    "Average Occupancy",
                    f"{upload_history['Occupancy'].mean():.1f}%"
                )

            if len(upload_history) >= 2:
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=upload_history["Time"],
                        y=upload_history["Occupancy"],
                        mode="lines+markers",
                        line={
                            "width": 3
                        },
                        marker={
                            "size": 7
                        }
                    )
                )

                fig.update_layout(
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
                        "gridcolor": "rgba(148,163,184,0.10)"
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

else:

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

            def recv(self, frame):

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
                        result.boxes.cls.tolist()
                    ):

                        label = str(
                            names[
                                int(class_id)
                            ]
                        ).lower()

                        if "occupied" in label:
                            occupied += 1

                        elif (
                            "empty" in label
                            or "free" in label
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

                current_time = time.time()

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
                    updated=datetime.now().strftime(
                        "%H:%M:%S"
                    ),
                    active=True,
                    frames=frames,
                    peak_occupancy=peak
                )

                history_time = time.time()

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
            video_processor_factory=
                ParkVisionProcessor,
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

        col1, col2, col3, col4 = st.columns(4)

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

        graph_col1, graph_col2, graph_col3 = st.columns(3)

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
                    "gridcolor": "rgba(148,163,184,0.10)"
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
                file_name="parkvision_analytics.csv",
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

        analytics1, analytics2, analytics3 = st.columns(3)

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

        control1, control2 = st.columns(2)

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

        status1, status2 = st.columns(2)

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


st.markdown(
    '<div class="footer">ParkVision AI</div>',
    unsafe_allow_html=True
)
