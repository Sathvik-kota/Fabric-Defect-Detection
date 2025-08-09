
import tensorflow as tf
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
!pip install streamlit pyngrok
!python -V
!cp -r /content/drive/MyDrive/saved_model/ /content/
warnings.filterwarnings("ignore")
print(tf.__version__)


img_size = (256,256)
save_model_path = '/content/saved_model/fabric_best_model.keras'
model = tf.keras.models.load_model(save_model_path)
model.summary()

label_to_class_mapping_file = '/content/saved_model/label_to_class_mapping.json'
with open(label_to_class_mapping_file, 'r') as f:
    label_to_class_mapping = json.load(f)
label_to_class_mapping

%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# Set page configuration and styling
st.set_page_config(
    page_title="Fabric Defect Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS to enhance the UI
st.markdown("""

""", unsafe_allow_html=True)

# App title and introduction
st.markdown("🛡️ Fabric Defect Detection System", unsafe_allow_html=True)

st.markdown("""

    Upload fabric images to detect defects and analyze quality using our advanced AI model.

""", unsafe_allow_html=True)
# Constants
img_size = (64, 64)

# Paths (these would need to be updated to your actual paths)
save_model_path = '/content/saved_model/fabric_best_model.keras'
label_to_class_mapping_file = '/content/saved_model/label_to_class_mapping.json'

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(save_model_path)
        with open(label_to_class_mapping_file, 'r') as f:
            label_to_class_mapping = json.load(f)
        return model, label_to_class_mapping
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Create a dummy model and mapping for demonstration
        st.warning("Using a demonstration model since the actual model couldn't be loaded.")
        dummy_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        dummy_mapping = {"0": "good", "1": "color_defect", "2": "hole", "3": "cut"}
        return dummy_model, dummy_mapping
      # Load model at app startup
with st.spinner("Loading model..."):
    model, label_to_class_mapping = load_model()
    class_to_label_mapping = {v: k for k, v in label_to_class_mapping.items()}
    class_names = list(class_to_label_mapping.keys())

def predict_class(img_path):
    img = Image.open(img_path)
    resized_img = img.resize(img_size).convert("RGB")
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    # Normalize the image if your model expects normalized input
    # img = img / 255.0

    probs = model.predict(img)[0]
    # Convert numpy float32 to Python float
    probs = [float(p) for p in probs]
    predicted_class = label_to_class_mapping[str(np.argmax(probs))]
    return probs, predicted_class

def create_probability_plot(probs, class_names):
    # Create a horizontal bar chart using Plotly
    colors = ['#3498db' if p != max(probs) else '#e74c3c' for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=class_names,
        x=[p * 100 for p in probs],  # Convert to percentage
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p * 100:.2f}%" for p in probs],
        textposition='auto'
    ))
  fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis_title="Defect Class",
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(range=[0, 100]),
        yaxis=dict(categoryorder='total ascending')
    )

    return fig

def get_result_styling(predicted_class):
    """Return appropriate styling based on the predicted class."""
    if predicted_class == "good":
        return "🟢 Good", "#10B981"  # Green for good
    else:
        return "🔴 Defect Detected", "#EF4444"  # Red for defects

# Main app layout with columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("Upload Fabric Image", unsafe_allow_html=True)

    # File uploader
    st.markdown("", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("", unsafe_allow_html=True)

    # Display upload instructions
    if not uploaded_file:
        st.info("Please upload a fabric image to analyze for defects.")
              # Sample images section
        st.markdown("About Fabric Defects", unsafe_allow_html=True)
        st.markdown("""
        This system can detect the following fabric defects:
        - **Color Defects**: Inconsistencies in color or dyeing
        - **Holes**: Small to medium-sized holes in the fabric
        - **Cuts**: Linear tears or cuts in the material
        - **Good**: Fabric with no defects
        """)

with col2:
    st.markdown("Analysis Results", unsafe_allow_html=True)

    if uploaded_file:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp_img.jpg")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        st.image(temp_file_path, caption="Uploaded Fabric Image")

        # Analyze with a loading animation
        with st.spinner("Analyzing fabric for defects..."):
            # Add a small delay for effect
            time.sleep(1)
            probs, predicted_class = predict_class(temp_file_path)

        # Display results
        status_text, status_color = get_result_styling(predicted_class)

        st.markdown(f"""
        
            Analysis Result:
            {status_text}
                            Detected Class: {predicted_class.replace('_', ' ').title()}
            
        
        """, unsafe_allow_html=True)

        # Display probability chart
        st.markdown("", unsafe_allow_html=True)
        fig = create_probability_plot(probs, [name.replace('_', ' ').title() for name in class_names])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("", unsafe_allow_html=True)

        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Footer
st.markdown("© 2025 Fabric Defect Detection System | AI-Powered Quality Control", unsafe_allow_html=True)



from pyngrok import ngrok

ngrok_key = "2wwrghyfuHRS5ipbFGHcXbceafq_69z2V5DNkAUEPnptkBE4y"
port = 8501

ngrok.set_auth_token(ngrok_key)
ngrok.connect(port).public_url
