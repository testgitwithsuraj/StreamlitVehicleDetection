import streamlit as st
import cv2
import numpy as np
import os
from vehicle_counter import process_video

# Set up the Streamlit app layout
st.set_page_config(page_title="Vehicle Detection and Counting", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .button {
        background-color: darkgrey;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .button:hover {
        background-color: grey;
        color: white;
    }
    .sidebar .sidebar-content {
        padding: 2rem;
    }
    .main {
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Vehicle Detection and Counting")

# Sidebar for video source selection and settings
st.sidebar.header("Video Source")
video_source = st.sidebar.selectbox("Select a video source", ["Upload a video", "Video 1", "Video 2", "Video 3"])

# Path to pre-defined videos
video_paths = {
    "Video 1": "videos/video1.mp4",
    "Video 2": "videos/video2.mp4",
    "Video 3": "videos/video3.mp4"
}

# Initialize video_path
video_path = None

# Handle video source
if video_source == "Upload a video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        video_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(video_path, mode='wb') as f:
            f.write(uploaded_file.read())
else:
    video_path = video_paths.get(video_source)

# Settings
st.sidebar.header("Settings")
resolution = st.sidebar.selectbox("Resolution", ["480p", "720p", "1080p"])
frame_rate = st.sidebar.slider("Frame Rate", 1, 60, 30)
detection_sensitivity = st.sidebar.slider("Detection Sensitivity", 1, 10, 5)

st.sidebar.header("Display Options")
show_bounding_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
show_object_ids = st.sidebar.checkbox("Show Object IDs", True)
show_tracking_paths = st.sidebar.checkbox("Show Tracking Paths", False)

st.sidebar.header("Statistics Options")
show_avg_speed = st.sidebar.checkbox("Show Average Speed", True)
show_vehicle_density = st.sidebar.checkbox("Show Vehicle Density", True)

# Initialize session state for controls
if 'play' not in st.session_state:
    st.session_state.play = False
if 'stop' not in st.session_state:
    st.session_state.stop = True

# Handle button click actions
def set_play_state(play):
    st.session_state.play = play
    st.session_state.stop = not play

if video_path:
    st.sidebar.header("Controls")
    
    # Play button
    if st.sidebar.button("Play"):
        set_play_state(True)

    # Pause button
    if st.sidebar.button("Pause"):
        set_play_state(False)

    # Stop button
    if st.sidebar.button("Stop"):
        st.session_state.play = False
        st.session_state.stop = True

    stframe = st.empty()
    progress_bar = st.progress(0)
    frame_counter = st.sidebar.empty()
    vehicle_counter = st.sidebar.empty()
    average_speed = st.sidebar.empty()
    vehicle_density = st.sidebar.empty()

    # Process video and display frames
    for frame, count, avg_speed, avg_density in process_video(video_path, resolution, frame_rate, detection_sensitivity, show_bounding_boxes, show_object_ids, show_tracking_paths, show_avg_speed, show_vehicle_density):
        if st.session_state.play and not st.session_state.stop:
            stframe.image(frame, channels="BGR")
            progress_bar.progress(min(count / 100, 1.0))  # Adjusted progress calculation
            frame_counter.write(f"Frame: {count}")
            vehicle_counter.write(f"Vehicles Detected: {count}")
            if show_avg_speed:
                average_speed.write(f"Average Speed: {avg_speed:.2f} km/h")
            if show_vehicle_density:
                vehicle_density.write(f"Vehicle Density: {avg_density:.2f} vehicles/km")

    if st.session_state.stop:
        st.session_state.play = False
        st.stop()

    st.success("Processing Complete!")

else:
    st.info("Please select a video source or upload a video file to start.")

# Add a footer
st.sidebar.markdown("""
---
**Developed by [Your Name]**
""")
