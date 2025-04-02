import streamlit as st

from modules.utils import load_profanity_database, process_audio

database_path = "./database/en-us"

profanity_database = load_profanity_database(database_path)


def main():
    st.set_page_config(page_title="ENGI9821 - Final Project", layout="wide")
    st.markdown(
        "<h1 style='text-align: center;'>ENGI9821 - Final Project</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='text-align: center;'>Profanity Detection and Replacement in Audio Using DSP Techniques</h1>",
        unsafe_allow_html=True,
    )
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Process Audio"):
            output_audio = process_audio(
                input_audio_path=uploaded_file, profanity_database=profanity_database
            )

            # Display table
            st.write("### Processed Audio File")
            st.audio(output_audio)
            # st.write("### Processed Data Table")
            # st.dataframe(df)


if __name__ == "__main__":
    main()
