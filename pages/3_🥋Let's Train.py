import streamlit as st
import os
import cv2
import mediapipe as mp
import math # for calculating angles
import numpy as np
import tempfile
import google.generativeai as genai
import base64
from PIL import Image
import IPython.display as display
from threading import Thread

st.set_page_config(page_title='Sports Training Assistant App',page_icon='🏌',layout='wide')

def get_dic(landmarks,selected_sport):
    hip = [landmarks[23].x, landmarks[23].y]
    knee = [landmarks[25].x, landmarks[25].y]
    ankle = [landmarks[27].x, landmarks[27].y]
    shoulder = [landmarks[11].x, landmarks[11].y]
    elbow = [landmarks[13].x, landmarks[13].y]  # Left elbow
    wrist = [landmarks[15].x, landmarks[15].y]  # Left wrist
    ground_ref = [landmarks[23].x, 1.0]  # Vertical line reference

    if selected_sport=='High Jump':
        knee_angle = calculate_angle(hip, knee, ankle)
        lean_angle = calculate_angle(ground_ref, hip, shoulder)
        back_arch_angle = calculate_angle(shoulder, hip, knee)
        dic={'knee angle':knee_angle,'lean angle':lean_angle,'back arch angle':back_arch_angle}
    elif selected_sport=='Long Jump':
        lean_angle = calculate_angle(ground_ref, hip, shoulder)
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        flight_angle = calculate_angle(ground_ref, hip, shoulder)
        arm_angle = calculate_angle(elbow, shoulder, wrist)
        dic={'knee angle':knee_angle,'lean angle':lean_angle,'hip angle':hip_angle,
        'flight angle':flight_angle,'arm angle':arm_angle}
    elif selected_sport=='Sprinting':
        ground_ref = [hip[0], 1.0]  # Vertical reference
        lean_angle = calculate_angle(ground_ref, hip, shoulder)
        knee_angle = calculate_angle(hip, knee, ankle)
        thigh_drive_angle = calculate_angle(shoulder, hip, knee)
        alignment_angle = calculate_angle(hip, knee, ankle)
        arm_swing_angle = calculate_angle(shoulder, elbow, wrist)
        ground_ref = [ankle[0], 1.0]  # Vertical line as ground reference
        ground_contact_angle = calculate_angle(ground_ref, ankle, knee)
        dic={'knee angle':knee_angle,'lean angle':lean_angle,'thigh drive angle':thigh_drive_angle,
        'alignment angle':alignment_angle,'arm swing angle':arm_swing_angle,
        'ground contact angle':ground_contact_angle}
    else:
         pass
    return dic

def play_video(video_path,dic):
    video=cv2.VideoCapture(video_path)
    while True:
        suc,img=video.read()
        if not suc:
            break
        img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result=pose.process(img1)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            dic=get_dic(landmarks,selected_sport)
            mpdrawing.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            y_start = 250  # Starting y-coordinate for the first line of text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            line_spacing = 30 
            bg_color = (255, 255, 255)
            half_length=len(dic)//2
            text1=f'{dict(list(dic.items())[:half_length])}'
            text2=f'{dict(list(dic.items())[half_length:])}'
            (text_width, text_height), baseline = cv2.getTextSize(text1, font, font_scale, font_thickness)
            position1 = (20,y_start)  # Position of the text
            rect_x1 = position1[0] - 10  # 10px padding on left
            rect_y1 = position1[1] + baseline   #  padding on top
            rect_x2 = position1[0] + text_width + 10  # 10px padding on right
            rect_y2 = position1[1] - text_height   # 12px padding on bottom

            (text_width, text_height), baseline = cv2.getTextSize(text2, font, font_scale, font_thickness)
            position2 = (20, y_start + line_spacing)  # Position of the text
            rect_x11 = position2[0] - 10  # 10px padding on left
            rect_y11 = position2[1] + baseline  # 10px padding on top
            rect_x21 = position2[0] + text_width + 10  # 10px padding on right
            rect_y21 = position2[1] - text_height  # 10px padding on bottom 

            cv2.rectangle(img, (rect_x1, rect_y2), (rect_x2, rect_y1), bg_color, -1) 
            cv2.rectangle(img, (rect_x11, rect_y21), (rect_x21, rect_y11), bg_color, -1)
            cv2.putText(img,text1,(20,y_start),font,font_scale,(255,0,0),font_thickness)
            cv2.putText(img,text2,(20, y_start + line_spacing),font,font_scale,(255,0,0),font_thickness)
        cv2.imshow('image',img)
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

# Add custom CSS for background image
def set_background(image_file):
    with open(image_file, "rb") as img:
        img_data = img.read()
        # Encode image in base64
        img_base64 = base64.b64encode(img_data).decode()
    # Add background CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
        }}
        h1, h2, h3, .css-1d391kg, .css-1cv9hz2 {{
        color: white; /* Set color for headers and subheaders */
    }}
    .stMarkdown, .stWrite, .css-1d391kg, .css-1cv9hz2 {{
        color: white; /* Set color for content inside st.write() */
    }}
    .css-1d391kg {{
        color: white; /* Set color for text */
    }}
     /* Change label color for selectbox and file uploader */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stFileUploader"] label {{
        color: white; /* Set the label color */
        font-size: 16px; /* Optional: Adjust font size */
        font-weight: bold; /* Optional: Make the font bold */
    }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Set the background for home page
set_background("C:/Users/user/Desktop/Data Science/DL/DL Projects/black-background.avif")

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint (vertex)
    c = np.array(c)  # Third point
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(int(angle))


key='key'
# genai.configure(api_key=key)
model=genai.GenerativeModel('gemini-1.5-flash')
final_model=genai.GenerativeModel('gemini-pro')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mpdrawing=mp.solutions.drawing_utils

st.subheader('Sports Training Assistant App🤸‍♂️')
selected_sport=st.selectbox('Select Sport:',['Long Jump','Sprinting','High Jump'])
frame_sampling_interval = 7
if selected_sport:
    vdo=st.file_uploader('Upload a video file:',type=["mp4", "avi", "mov"])
    if vdo is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(vdo.read())
            video_path = tmp_file.name  #save video temporarily
        st.video(video_path) #display video
        video=cv2.VideoCapture(video_path)
        # Processing loop with error handling, progress bar, and improved structure
        progress = st.progress(0)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        entire_feedback = ""
        while video.isOpened():
            suc,img=video.read()
            if not suc:
                break
            frame_count += 1
            progress.progress(frame_count / total_frames)
            img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if frame_count % frame_sampling_interval == 0:
                result=pose.process(img1)
                cv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_image_rgb)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    dic=get_dic(landmarks,selected_sport)
                    content=f"an image of training for {selected_sport} and monitored angle in this image are {dic}. Detect the phase and provide suggestions for improving the training"
                    response=model.generate_content([content,pil_image])
                    entire_feedback=entire_feedback+' '+response.text
                else:
                    continue
        progress.empty()
        suggestion=final_model.generate_content(f"Summarise this text:'{entire_feedback}' and avoid mentioning limitations, video analysis and coach") 
        st.write(suggestion.text) 
        but=st.button('See angles in video')
        st.write("Press 'q' to stop video playback.")
        if but:
            thread = Thread(target=play_video, args=(video_path,dic))
            thread.start()
            thread.join()
        video.release()
        os.remove(video_path)      
