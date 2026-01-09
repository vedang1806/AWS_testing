import streamlit as st
import tempfile
import uuid
from main import run_pipeline

st.set_page_config(page_title="Call PII Redaction & Sentiment", layout="wide")

st.title("🎙️ Call Transcription, PII Redaction & Sentiment Analysis")

st.markdown("""
Upload a call recording to:
- Transcribe using **AWS Transcribe**
- Detect & redact **PII**
- Analyze sentiment using **Gemini**
- Generate **redacted audio**
""")

uploaded_file = st.file_uploader("Upload WAV/mp3 audio file", type=["wav", "mp3"])

if uploaded_file:
    call_id = st.text_input(
        "Call ID (optional)",
        value=f"call_{uuid.uuid4().hex[:8]}"
    )

    if st.button("🚀 Run Analysis"):
        with st.spinner("Processing audio... this may take a few minutes ⏳"):

            suffix = ".mp3" if uploaded_file.type == "audio/mpeg" else ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.close()

            audio_path = tmp.name

            try:
                result = run_pipeline(audio_path, call_id)
                st.success("Processing completed successfully 🎉")

                # --- Transcript ---
                st.subheader("📝 Transcript")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original**")
                    st.text_area(
                        "Original Transcript",
                        result["original_text"],
                        height=300,
                        label_visibility="collapsed"
                    )

                with col2:
                    st.markdown("**PII Redacted**")
                    st.text_area(
                        "Redacted Transcript",
                        result["redacted_text"],
                        height=300,
                        label_visibility="collapsed"
                    )

                # --- PII ---
                st.subheader("🔒 Detected PII")
                if result["pii_entities"]:
                    st.json(result["pii_entities"])
                else:
                    st.info("No PII detected")

                # --- Sentiment ---
                st.subheader("🎭 Sentiment Analysis")
                for seg in result["sentiment"]:
                    with st.expander(
                        f"{seg['order']}. {seg['speaker']} ({seg['sentiment']})"
                    ):
                        st.markdown(f"**Text:** {seg['text']}")
                        st.markdown(f"**Time:** {seg['start_time']} → {seg['end_time']}")
                        st.markdown(f"**Tone:** {seg['tone_note']}")
                        st.progress(seg["confidence"])

                # --- Audio ---
                st.subheader("🔊 Redacted Audio")
                audio_url = result.get("redacted_audio_url")
                if audio_url:
                    st.audio(audio_url, format="audio/wav")
                    st.markdown(f"[⬇️ Download from S3]({audio_url})")
                else:
                    st.info("No redacted audio generated")

                st.subheader("🔗 Redacted Audio S3 URL")

                audio_url = result.get("redacted_audio_url")

                if audio_url:
                    st.text_input(
                    "S3 URL",
                    value=audio_url,
                    disabled=True
                )

                    st.markdown(f"[⬇️ Download redacted audio]({audio_url})")
                else:
                    st.info("No S3 URL available")


            except Exception as e:
                st.error(f"Error: {e}")

