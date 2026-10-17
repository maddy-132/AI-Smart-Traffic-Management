# AI-Based Smart Traffic Management System

## Features
- YOLOv8 vehicle detection using a pretrained model
- ByteTrack tracking IDs and approximate unique counts for cars, motorcycles, buses, and trucks
- Basic traffic-density level
- Educational adaptive signal simulation based on vehicle counts in four image regions
- Streamlit dashboard, chart, CSV export, annotated video preview and download

## Run in VS Code (Windows)
1. Extract this ZIP and open the extracted folder in VS Code.
2. Open Terminal → New Terminal.
3. (Recommended) Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   python -m streamlit run app.py
   ```
6. Open the local URL shown in the terminal and upload an MP4/AVI/MOV/MKV traffic video.

## Notes
- First run may download `yolov8n.pt` from Ultralytics.
- The count is based on tracking IDs and may be inaccurate if an object is missed or assigned a new ID.
- Density thresholds are illustrative and can be adjusted for your video.
- The four-region signal selection is a simulation, not calibrated lane detection.
- Never connect this educational demo to real traffic lights or road infrastructure.
