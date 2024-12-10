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
    st.subheader("Sport-Specific Training Insights")
    st.markdown("""
    - **Long Jump**: Focus on take-off speed, angle, and posture.
    - **Sprinting**: Analyze stride length, stride frequency, and starting block reaction time.
    - **Discus Throw**: Evaluate release angle, spin speed, and footwork precision.
    - **High Jump**: Monitor approach angle, take-off, and bar clearance.
""")