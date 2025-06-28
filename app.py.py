import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load model (only once, cached)
@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return model

pipe = load_model()

# Streamlit UI
st.title("🖼️ Text-to-Image Generator")
prompt = st.text_input("Enter your text prompt 👇", value="A fantasy landscape with mountains and waterfalls")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)