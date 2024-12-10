import cv2
import streamlit as st

import streamlit as st
st.set_page_config(page_title='Sports Training Assistant App',page_icon='🏌',layout='wide')
import base64

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
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background for home page
set_background("C:/Users/user/Desktop/Data Science/DL/DL Projects/gb.jpg")

with st.container():
    st.subheader('👋Welcome to,')
    st.header('Sports Training Assistant App🤸‍♂️')
with st.container():
    left_column,right_column=st.columns(2)
    with left_column:
        st.write('🤝Our app is designed to help athletes and coaches optimize training by providing real-time feedback on form and technique. Using advanced computer vision, Gen-AI models and pose estimation with Mediapipe, we analyze video footage of your practice sessions to provide insights and suggestions.')
        st.write('💪Train smarter, improve faster, and reach your peak potential with personalized coaching that fits in the palm of your hand.')
    with right_column:
        st.write('')