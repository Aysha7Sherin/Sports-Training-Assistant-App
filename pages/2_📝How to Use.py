import streamlit as st
import base64
st.set_page_config(page_title='Sports Training Assistant App',page_icon='🏌',layout='wide')
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
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background for home page
set_background("C:/Users/user/Desktop/Data Science/DL/DL Projects/gb.jpg")

st.header('Sports Training Assistant App🤸‍♂️')

with st.container():
    st.write('⚪Select Your Sport: Choose the sport you are training for from the dropdown menu. The app supports long jump, sprinting and high jump.')
    st.write('⚪Upload Your Training Video: Click on the file uploader to upload a video of your practice session. Supported formats include MP4, AVI, and MOV.')
    st.write('⚪After calculating relevant body angles for the selected sport, AI model analyses your video and provide suggestions to improve your performance.')
    st.write('⚪Click on the “See angles ” button to view body angles at each moment.')
    st.write('      ▪The app will open a new window using cv2 (OpenCV) to play the video and overlay current angles.')
    st.write('⚪Stopping the Video Playback:')
    st.write('      ▪To stop the video, simply press the ‘q’ key while the cv2 window is active. This will close the window and halt playback.')
    