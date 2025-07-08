import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image, ImageEnhance
import json

# Load animation
def load_lottiefile(path: str):
    with open(path, "r") as f:
        return json.load(f)

lottie_aug = load_lottiefile("lottie/lottie_aug.json")
lottie_success = load_lottiefile("lottie/lottie_suc.json")

# Header and animation
st_lottie(lottie_aug, height=180, key="augment-anim", loop=True)
st.markdown("<div class='title section-anim'>Augmentation Playground</div>", unsafe_allow_html=True)
st.markdown('<div class="subheader">Try live image augmentations and see instant results!</div>', unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📤 Upload Image for Augmentation", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)

    st.subheader("🎛️ Augmentation Controls")

    # Augmentation options
    col1, col2 = st.columns(2)
    with col1:
        rotate_angle = st.slider("🔄 Rotate", -45, 45, 0)
        flip_horizontal = st.checkbox("↔️ Flip Horizontal")
        flip_vertical = st.checkbox("↕️ Flip Vertical")
    with col2:
        zoom = st.slider("🔍 Zoom (%)", 100, 200, 100)
        brightness = st.slider("💡 Brightness", 0.5, 2.0, 1.0)
        contrast = st.slider("🎨 Contrast", 0.5, 2.0, 1.0)

    # Apply augmentations
    if st.button("Apply Augmentation", use_container_width=True):
        with st.spinner("Applying..."):
            st_lottie(lottie_aug, height=60, key="aug-loading", loop=True)
            aug_img = img.rotate(rotate_angle)

            if flip_horizontal:
                aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_vertical:
                aug_img = aug_img.transpose(Image.FLIP_TOP_BOTTOM)

            if zoom > 100:
                w, h = aug_img.size
                zoomed = aug_img.resize((int(w * zoom / 100), int(h * zoom / 100)))
                left = (zoomed.width - w) // 2
                top = (zoomed.height - h) // 2
                aug_img = zoomed.crop((left, top, left + w, top + h))

            aug_img = ImageEnhance.Brightness(aug_img).enhance(brightness)
            aug_img = ImageEnhance.Contrast(aug_img).enhance(contrast)

        # Show result
        st.markdown("<div class='subheader'>🖼️ Augmented Output</div>", unsafe_allow_html=True)
        st.image(aug_img, caption="Augmented Image", use_container_width=True)
        st_lottie(lottie_success, height=60, key="aug-success", loop=True)
