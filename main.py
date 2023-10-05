#!/usr/bin/env python

"""
A streamlit app takes inputs from the user as text and multiple drowpdowns of options and calls leap api to get iamge results.
"""

import streamlit as st
import os
import requests
import time
import replicate
from dotenv import load_dotenv
load_dotenv()

# REPLICATE_API_KEY = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_API_KEY =  st.secrets["REPLICATE_API_KEY"]

# Title
st.title("Image Generator")
# st.write(REPLICATE_API_KEY)

# Model and version
model_input = st.radio("Select model", ["sd", "sdxl",], index=1, captions = ["SD", "SDXL"])

if model_input == "sdxl":
    model = replicate.models.get("stability-ai/sdxl")
    version = model.versions.get("8beff3369e81422112d93b89ca01426147de542cd4684c244b673b105188fe5f")
    scheduler_list = ["DDIM", "DPMSolverMultistep", "HeunDiscrete", "KarrasDPM", "K_EULER_ANCESTRAL", "K_EULER", "PNDM"]
else:
    model = replicate.models.get("stability-ai/stable-diffusion")
    version = model.versions.get("ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4")
    scheduler_list = [ "DDIM", "K_EULER", "DPMSolverMultistep", "K_EULER_ANCESTRAL", "PNDM", "KLMS"]

if REPLICATE_API_KEY:

    prompt = st.text_input("Input prompt", "An astronaut riding a rainbow unicorn")
    negative_prompt = st.text_input("Input Negative Prompt", "")
    # image = st.file_uploader("Input image for img2img or inpaint mode")
    width = st.number_input("Width of output image", value=1024)
    height = st.number_input("Height of output image", value=1024)
    num_outputs = st.number_input("Number of images to output", value=1)
    scheduler = st.selectbox("Scheduler", scheduler_list, index=0)
    prediction_status = None
    if prediction_status == None:

        if st.button("Generate"):
            # Create prediction
            st.write("Generating Image")
            start_time = time.time()
            prediction = replicate.predictions.create(
                version=version,
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_outputs": num_outputs,
                    "scheduler": scheduler,
                }
            )
            prediction_status = prediction.status
            st.write("Prediction ID"" : ", prediction.id)
            # Wait for prediction to complete
            while prediction_status!= "succeeded":
                prediction.reload()
                prediction_status = prediction.status
                st.write("Prediction status: ", prediction_status)
                if prediction_status == "failed":
                    st.write("Prediction failed!")
                    break
                if prediction_status == "succeeded":
                    st.write("Prediction completed!")
                    try:
                        prediction_output = prediction.output
                        # st.write("Prediction output: ", prediction_output)
                        end_time = time.time()
                        st.write("Time taken: ", end_time - start_time)
                        for i in range(num_outputs):
                            image_url = prediction_output[i]
                            response = requests.get(image_url)   
                            st.image(response.content)
                    except Exception as e:
                        st.write("No output!")
                        st.write(e)
                    break
                time.sleep(5)