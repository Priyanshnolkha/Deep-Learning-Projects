import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "augmented_yolo26n"
    / "weights"
    / "best.pt"
)


st.set_page_config(
    page_title="Airport Baggage Screening",
    page_icon="🛫",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()

st.title("🛫 Airport Baggage Screening System")

st.markdown(
    """
    **Deep Learning Based X-Ray Baggage Screening**

    Upload an X-ray baggage image and the YOLO model will
    detect suspicious/prohibited objects and display their
    confidence scores.
    """
)

st.divider()

st.sidebar.header("⚙️ Screening Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.write(
    f"Current threshold: **{confidence_threshold:.0%}**"
)

uploaded_file = st.file_uploader(
    "Upload Baggage X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Input X-Ray")

        st.image(
            image,
            use_container_width=True
        )


    # Save temporary image
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as temp_file:

        image.save(temp_file.name)

        temp_path = temp_file.name


    # Run YOLO
    results = model.predict(
        source=temp_path,
        imgsz=640,
        conf=confidence_threshold,
        device="mps",
        verbose=False
    )


    result = results[0]
    
    annotated_image = result.plot()

    with col2:

        st.subheader("🔍 Detection Result")

        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )


    st.divider()


    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            detections.append(
                {
                    "object": class_name,
                    "confidence": confidence
                }
            )



    if detections:

        st.error(
            "⚠️ SUSPICIOUS OBJECT DETECTED"
        )

        st.subheader("Detected Objects")

        for detection in detections:

            object_name = detection["object"]

            confidence = detection["confidence"]

            st.write(
                f"🔴 **{object_name}**"
            )

            st.progress(
                confidence,
                text=f"Confidence: {confidence:.2%}"
            )


        st.warning(
            "Bag should be referred for further human inspection."
        )


    else:

        st.success(
            "✅ NO SUSPICIOUS OBJECT DETECTED"
        )

        st.info(
            "No trained prohibited-object class exceeded "
            "the selected confidence threshold."
        )


else:

    st.info(
        "👆 Upload an X-ray baggage image to begin screening."
    )