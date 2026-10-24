import streamlit as st
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import matplotlib.pyplot as plt

MODEL_NAME = "openai/clip-vit-base-patch32"

CATEGORIES = {
    "Cat": [
        "a photograph of a domestic cat",
        "a close-up photograph of a pet cat",
        "a cat sitting or standing"
    ],
    "Elephant": [
        "a photograph of an elephant",
        "a wildlife photograph of an elephant",
        "an elephant with a trunk and large ears"
    ],
    "Lion": [
        "a photograph of a lion",
        "a wildlife photograph of a lion",
        "a lion with a mane"
    ],
    "Panda": [
        "a photograph of a giant panda",
        "a wildlife photograph of a panda bear",
        "a panda eating bamboo"
    ],
    "Tiger": [
        "a photograph of a tiger",
        "a wildlife photograph of a tiger",
        "an orange tiger with black stripes"
    ],
    "Non-animal": [
        "a photograph of a car or automobile",
        "a photograph of a yellow vehicle",
        "a photograph of a mobile phone or computer",
        "a photograph of a chair or furniture",
        "a photograph of a shoe or clothing",
        "a photograph of food or fruit",
        "a photograph of a building or house",
        "a photograph of a road, landscape, or scenery",
        "a photograph of a boat, machine, or tool",
        "a photograph of an everyday object",
        "a photograph of a bicycle or motorcycle",
        "a photograph of a person"
    ]
}

ANIMALS = ["Cat", "Elephant", "Lion", "Panda", "Tiger"]

st.set_page_config(
    page_title="WildGuard AI | Animal Identifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 15% 0%, #18372e 0%, #101a1c 34%, #0c1218 75%);
    color: #f3f6f4;
}
[data-testid="stSidebar"] {
    background: #111b1b;
    border-right: 1px solid #29433a;
}
h1, h2, h3 { color: #f4f7f4; }
.wg-subtitle { color: #b8c9c0; font-size: 1.05rem; }
.wg-card {
    background: rgba(31, 48, 47, 0.88);
    border: 1px solid #35564a;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    margin: 0.4rem 0 1rem 0;
}
.wg-result {
    background: linear-gradient(135deg, rgba(35, 78, 57, .95), rgba(25, 47, 43, .95));
    border: 1px solid #5b9a72;
    border-radius: 18px;
    padding: 1.2rem;
}
div[data-testid="stFileUploader"] {
    background: rgba(27, 42, 42, .7);
    border: 1px dashed #6a9d7e;
    border-radius: 14px;
    padding: 0.7rem;
}
.stButton button {
    border-radius: 10px;
    border: 1px solid #75a98a;
}
small, .stCaption { color: #b6c5be !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return processor, model, device

def extract_tensor(output, preferred_names):
    """Handle differences between Transformers versions."""
    if isinstance(output, torch.Tensor):
        return output
    for name in preferred_names:
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"Could not find a feature tensor in {type(output).__name__}.")

def classify_image(image, processor, model, device):
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_output = model.get_image_features(**image_inputs)
        image_features = extract_tensor(
            image_output, ["image_embeds", "pooler_output"]
        )
        image_features = F.normalize(image_features, dim=-1)

    scores = {}
    for category, prompts in CATEGORIES.items():
        text_inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        with torch.no_grad():
            text_output = model.get_text_features(**text_inputs)
            text_features = extract_tensor(
                text_output, ["text_embeds", "pooler_output"]
            )
            text_features = F.normalize(text_features, dim=-1)
            similarities = (image_features @ text_features.T).squeeze(0)
            scores[category] = float(similarities.mean().item())

    best_animal = max(ANIMALS, key=lambda name: scores[name])
    animal_score = scores[best_animal]
    nonanimal_score = scores["Non-animal"]

    # Initial conservative heuristic; these thresholds need evaluation on a larger test set.
    if nonanimal_score >= animal_score:
        prediction = "Non-animal"
        reason = "The image matched non-animal descriptions best."
    elif animal_score < 0.24:
        prediction = "Uncertain"
        reason = "No supported animal matched strongly enough."
    elif animal_score - nonanimal_score < 0.015:
        prediction = "Uncertain"
        reason = "The animal and non-animal scores were too close."
    else:
        prediction = best_animal
        reason = "A supported animal matched best."

    return prediction, reason, scores

with st.sidebar:
    st.markdown("## 🌿 WildGuard Controls")
    st.markdown("**Choose your forest mood**")
    theme = st.selectbox(
        "Background style",
        ["Wild Forest", "Midnight", "Safari"],
        index=0
    )
    st.markdown("---")
    st.markdown("### 🐾 Supported animals")
    for animal in ANIMALS:
        st.write(f"• {animal}")
    st.caption("Other objects can be flagged as non-animal or uncertain.")
    st.markdown("---")
    st.caption("WildGuard AI is a prototype, not a wildlife monitoring or safety system.")

if theme == "Midnight":
    st.markdown("""<style>.stApp{background:radial-gradient(circle at 15% 0%,#25253e 0%,#11121d 42%,#090b12 85%)}</style>""", unsafe_allow_html=True)
elif theme == "Safari":
    st.markdown("""<style>.stApp{background:radial-gradient(circle at 15% 0%,#51442b 0%,#22271e 40%,#101713 85%)}</style>""", unsafe_allow_html=True)

st.markdown("# 🐾 WildGuard AI")
st.markdown('<p class="wg-subtitle">Discover wildlife with AI. Identify five supported animals or flag images that appear to show something else.</p>', unsafe_allow_html=True)
st.markdown("")

left, right = st.columns([1.02, 0.98], gap="large")

with left:
    st.markdown("### 📷 Upload an image")
    uploaded_file = st.file_uploader(
        "Choose a clear image to identify",
        type=["jpg", "jpeg", "png", "webp"],
        help="For best results, upload one main subject in a clear, well-lit image."
    )
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your uploaded image", use_container_width=True)
        except Exception:
            st.error("We couldn't open this image. Please try a JPG, PNG, or WEBP file.")

    st.info("Tip: try a clear photo of a cat, elephant, lion, panda, tiger, or an everyday object.")

with right:
    st.markdown("### 🔎 Identification results")
    if not uploaded_file:
        st.markdown("""
        <div class="wg-card">
            <h3>Ready to explore?</h3>
            <p>Upload an image to see the model's best-matching category and comparison scores.</p>
            <p>🐈 Cat &nbsp; 🐘 Elephant &nbsp; 🦁 Lion<br>🐼 Panda &nbsp; 🐅 Tiger</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Loading the AI model and analyzing your image… The first run may take a few minutes."):
                processor, model, device = load_model()
                prediction, reason, scores = classify_image(image, processor, model, device)

            if prediction in ANIMALS:
                st.markdown(f"""
                <div class="wg-result">
                    <p>🐾 BEST MATCH</p>
                    <h2>{prediction}</h2>
                    <p>{reason}</p>
                </div>
                """, unsafe_allow_html=True)
                st.success(f"Supported animal match: {prediction}")
            elif prediction == "Non-animal":
                st.warning("🚗 Non-animal image")
                st.write(reason)
            else:
                st.warning("❔ Uncertain result")
                st.write(reason)

            st.caption(f"Model running on: {device.upper()}")
            st.markdown("### 📊 Category similarity")
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            chart_labels = [item[0] for item in sorted_scores]
            chart_values = [item[1] for item in sorted_scores]
            fig, ax = plt.subplots(figsize=(8, 3.8))
            bars = ax.barh(chart_labels[::-1], chart_values[::-1])
            ax.set_xlabel("Cosine similarity (not probability)")
            ax.set_title("Similarity by category")
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            with st.expander("See all similarity scores"):
                for name, score in sorted_scores:
                    st.write(f"**{name}:** {score:.3f}")

            st.markdown("### ℹ️ About this result")
            st.caption(
                "These are cosine similarity scores, not calibrated probabilities. "
                "CLIP can make mistakes, especially with unusual images, multiple subjects, "
                "or animals outside the supported categories. The rejection rule is a prototype "
                "heuristic and has not been validated for real-world wildlife use."
            )
        except Exception as exc:
            st.error("The model couldn't analyze this image.")
            st.exception(exc)

st.markdown("---")
st.markdown("### 🌍 Built for curious wildlife explorers")
st.caption("WildGuard AI • Image classification prototype • Please verify predictions independently.")
