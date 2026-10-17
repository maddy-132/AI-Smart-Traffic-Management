from pathlib import Path
from collections import defaultdict
import tempfile
import time

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="AI Smart Traffic Management", page_icon="🚦", layout="wide")
st.title("🚦 AI-Based Smart Traffic Management System")
st.caption("Vehicle detection • Unique-ID counting • Traffic density • Adaptive signal simulation")

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
DEFAULT_MODEL = "yolov8n.pt"

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def density_level(count):
    if count <= 5:
        return "Low"
    if count <= 15:
        return "Moderate"
    return "High"

def signal_for_counts(counts):
    # Educational simulation only: choose the approach with the highest detected count.
    approaches = ["North", "East", "South", "West"]
    winner = max(approaches, key=lambda a: counts.get(a, 0))
    max_count = counts.get(winner, 0)
    if max_count == 0:
        return "All-red / idle", "No vehicles detected"
    return f"{winner} GREEN", f"{winner} has the highest estimated vehicle count ({max_count})"

def approach_from_center(cx, cy, width, height):
    # Divide the frame into four regions for a simple demonstration.
    # This is not lane calibration and must not control real traffic lights.
    nx, ny = cx / max(width, 1), cy / max(height, 1)
    if ny < 0.5 and nx < 0.5:
        return "North"
    if ny < 0.5 and nx >= 0.5:
        return "East"
    if ny >= 0.5 and nx >= 0.5:
        return "South"
    return "West"

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    sample_every = st.select_slider("Process every Nth frame", options=[1, 2, 3, 5], value=2)
    max_frames = st.number_input("Maximum frames to process (0 = all)", min_value=0, max_value=100000, value=0, step=500)
    model_file = st.text_input("YOLO model file", value=DEFAULT_MODEL, help="Uses yolov8n.pt by default; first run may download model weights.")
    st.caption("For faster processing, choose every 2nd or 3rd frame.")

uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
if not uploaded:
    st.info("Upload an MP4, AVI, MOV, or MKV traffic video to begin.")
    st.stop()

if st.button("▶ Start Detection & Analysis", type="primary", use_container_width=True):
    temp_input = None
    temp_output = None
    try:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getbuffer())
            temp_input = f.name

        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            st.error("Could not open this video. Try converting it to MP4 (H.264) and upload again.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if width <= 0 or height <= 0:
            st.error("Video dimensions could not be read.")
            st.stop()

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output = output_file.name
        output_file.close()
        writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            st.error("Could not create the output video on this system.")
            st.stop()

        model = load_model(model_file.strip() or DEFAULT_MODEL)
        counts_by_class = defaultdict(set)
        frame_count = 0
        processed_count = 0
        last_approach_counts = {a: 0 for a in ["North", "East", "South", "West"]}
        progress = st.progress(0)
        status = st.empty()
        preview = st.empty()
        started = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            if (frame_count - 1) % sample_every != 0:
                # Keep output video continuous by writing the original frame.
                writer.write(frame)
                continue

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                verbose=False
            )
            result = results[0]
            annotated = result.plot()
            approach_counts = {a: 0 for a in ["North", "East", "South", "West"]}

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(cls_ids)

                for coords, class_id, track_id in zip(xyxy, cls_ids, track_ids):
                    if class_id not in VEHICLE_CLASSES:
                        continue
                    name = VEHICLE_CLASSES[class_id]
                    if track_id is not None:
                        counts_by_class[name].add(int(track_id))
                    cx = int((coords[0] + coords[2]) / 2)
                    cy = int((coords[1] + coords[3]) / 2)
                    approach = approach_from_center(cx, cy, width, height)
                    approach_counts[approach] += 1

            last_approach_counts = approach_counts
            total_visible = sum(approach_counts.values())
            level = density_level(total_visible)
            signal, reason = signal_for_counts(approach_counts)

            cv2.putText(annotated, f"Visible vehicles: {total_visible} | Density: {level}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(annotated, f"Simulated signal: {signal}",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(annotated)
            processed_count += 1

            if frame_count % max(1, int(fps)) == 0 or frame_count == 1:
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                if total_frames:
                    progress.progress(min(frame_count / min(total_frames, max_frames or total_frames), 1.0))
                status.write(f"Processed frame {frame_count:,} of {total_frames or 'unknown'}")

        cap.release()
        writer.release()
        progress.progress(1.0)
        status.success(f"Analysis complete in {time.time() - started:.1f} seconds.")

        total_unique = sum(len(ids) for ids in counts_by_class.values())
        st.session_state["traffic_result"] = {
            "counts": {k: len(counts_by_class[k]) for k in VEHICLE_CLASSES.values()},
            "approaches": last_approach_counts,
            "output": temp_output,
            "frames": frame_count,
            "processed": processed_count,
            "unique": total_unique,
        }
    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.info("Check that the model file exists and that the video is a supported format.")
    finally:
        if temp_input and Path(temp_input).exists():
            try:
                Path(temp_input).unlink()
            except OSError:
                pass

result_data = st.session_state.get("traffic_result")
if result_data:
    st.divider()
    st.header("📊 Traffic Analytics Dashboard")
    counts = result_data["counts"]
    total = result_data["unique"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Unique tracked vehicles", total)
    c2.metric("Cars", counts.get("Car", 0))
    c3.metric("Motorcycles", counts.get("Motorcycle", 0))
    c4.metric("Buses", counts.get("Bus", 0))
    c5.metric("Trucks", counts.get("Truck", 0))

    chart_data = pd.DataFrame({"Vehicle": list(counts.keys()), "Count": list(counts.values())})
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Vehicle Count by Type")
        st.bar_chart(chart_data.set_index("Vehicle"))
        st.download_button(
            "⬇ Download counts as CSV",
            data=chart_data.to_csv(index=False).encode("utf-8"),
            file_name="vehicle_counts.csv",
            mime="text/csv"
        )
    with right:
        st.subheader("Traffic Density")
        total_count = sum(counts.values())
        level = density_level(total_count)
        st.metric("Estimated density level", level)
        st.caption("Density thresholds are simple demo settings: Low ≤ 5, Moderate 6–15, High > 15 unique tracked vehicles.")
        approach_df = pd.DataFrame({
            "Approach (image region)": list(result_data["approaches"].keys()),
            "Visible vehicles": list(result_data["approaches"].values())
        })
        st.dataframe(approach_df, use_container_width=True, hide_index=True)
        signal, reason = signal_for_counts(result_data["approaches"])
        st.subheader("🚦 Adaptive Signal Simulation")
        st.success(f"Suggested simulated phase: **{signal}**")
        st.write(reason)
        st.warning("Simulation only. Image quadrants are not calibrated road lanes; do not connect this demo to real traffic signals.")

    st.subheader("🎞️ Processed Video")
    output_path = result_data["output"]
    if output_path and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇ Download processed video", f.read(), file_name="traffic_detection_output.mp4", mime="video/mp4")
    st.caption(f"Frames read: {result_data['frames']:,} • Detection frames processed: {result_data['processed']:,}")
