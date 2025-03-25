# Profanity Detection and Replacement in Audio

This project implements a Digital Signal Processing (DSP)-based system to detect and replace profane words in audio recordings using word segmentation, spectral analysis, and dynamic time warping. The system is wrapped in a simple **Streamlit** interface for ease of testing and demonstration.

---

## Features

- Short-time energy and STFT-based speech segmentation
- Boundary refinement using spectral flux
- Profanity detection using MFCC + DTW
- Beep overlay to censor profane segments
- Interactive Streamlit app to upload and process audio

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/profanity-audio-filter.git
cd profanity-audio-filter
```
### 2. Create and Activate Virtual Environment
- For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
- For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

##  Running the Streamlit App
```bash
streamlit run app.py
```

## Project Structure
```bash
.
├── database               # Profanity database
├── modules
    ├── utils              # Detection logic and DSP functions
├── sample                 # Audio samples for testing
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # You're reading it!
```
