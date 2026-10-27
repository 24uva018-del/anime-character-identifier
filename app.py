import json
import numpy as np
import torch
import joblib
import streamlit as st
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

THRESHOLD = 0.30  # below this, show "not recognized"

st.set_page_config(page_title="Anime Character Identifier")
st.title("Anime Character Identifier")
st.caption("Death Note, Jujutsu Kaisen and Attack on Titan")

@st.cache_resource
def load_all():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clf = joblib.load("clip_classifier.joblib")
    classes = json.load(open("class_names.json"))
    info = json.load(open("info.json", encoding="utf-8"))
    return model, processor, clf, classes, info

model, processor, clf, classes, info = load_all()

def get_feat(img):
    inp = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inp)
    if not torch.is_tensor(out):
        if getattr(out, "image_embeds", None) is not None:
            out = out.image_embeds
        else:
            p = out.pooler_output
            out = p if p.shape[-1] == model.config.projection_dim else model.visual_projection(p)
    out = out / out.norm(dim=-1, keepdim=True)
    return out[0].numpy()

def find_info(name):
    for k, v in info.items():
        if k.lower() == name.lower():
            return v
    return None

file = st.file_uploader("Upload an anime character image", type=["jpg", "jpeg", "png", "webp"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", width=300)

    probs = clf.predict_proba([get_feat(img)])[0]
    order = np.argsort(probs)[::-1]
    best = classes[order[0]]
    conf = float(probs[order[0]])

    if conf < THRESHOLD:
        st.warning(f"Character not recognized (confidence {conf*100:.1f}%). Try another image.")
    else:
        st.success(f"Detected: {best} (confidence {conf*100:.1f}%)")
        st.progress(min(conf, 1.0))
        details = find_info(best)
        if isinstance(details, dict):
            for k, v in details.items():
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                st.markdown(f"**{k}:** {v}")
        elif details:
            st.write(details)
        else:
            st.info("No details found for this character in info.json.")

    with st.expander("Top 3 predictions"):
        for i in order[:3]:
            st.write(f"{classes[i]}: {probs[i]*100:.1f}%")

st.caption("AI predictions can be wrong.")