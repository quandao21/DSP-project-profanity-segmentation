import os
import wave
from io import BytesIO

import librosa
import librosa.display
import noisereduce as nr
import numpy as np
from scipy.signal import medfilt


def load_profanity_database(database_path, sr=24000):
    """Load profanity database from stored mp3 files."""
    profanity_database = {}
    for file in os.listdir(database_path):
        if file.endswith(".mp3"):
            word = os.path.splitext(file)[0]
            audio, _ = librosa.load(os.path.join(database_path, file), sr=sr)
            profanity_database[word] = audio
    return profanity_database


def generate_beep(frequency=1000, duration=0.5, samplerate=24000, volume=0.3):
    """Generate a beep sound."""
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    beep = (np.sin(2 * np.pi * frequency * t) * 32767 * volume).astype(np.int16)
    return beep


def preprocess_audio(audio, sr):
    """Apply noise reduction and normalization."""
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return librosa.util.normalize(reduced_noise)


def detect_speech_regions(
    audio,
    sr,
    frame_length=2048,
    hop_length=1024,
    energy_threshold_ratio=0.5,
    min_frames=4,
    gap_tolerance=2,
    refine=True,
):
    """
    Detect speech segments in an audio signal and return start and end times (in seconds) of each segment.

    Parameters:
        audio (np.array): Audio time series.
        sr (int): Sampling rate of the audio.
        frame_length (int): Number of samples per frame.
        hop_length (int): Number of samples between successive frames.
        energy_threshold_ratio (float): Ratio to multiply with the mean energy to set the threshold.
        min_frames (int): Minimum consecutive frames to count as a valid speech segment.
        gap_tolerance (int): Maximum allowed gap (in frames) within a speech segment.
        refine (bool): If True, refine boundaries using STFT-based spectral flux.

    Returns:
        Each tuple contains (start_time, end_time) in seconds for a detected speech region.
    """
    # Step 1: Compute short-time energy for each frame
    energy = np.array(
        [
            np.sum(audio[i : i + frame_length] ** 2)
            for i in range(0, len(audio), hop_length)
        ]
    )
    print(len(energy))

    # Step 2: Determine energy threshold for speech detection
    energy_threshold = np.mean(energy) * energy_threshold_ratio

    # Create a binary decision: True if energy exceeds the threshold
    speech_flags = energy > energy_threshold
    # print(speech_flags.astype(float))

    # Step 3: Smooth the decision using a median filter to remove spurious detections
    speech_flags = medfilt(speech_flags.astype(float), kernel_size=5) > 0.5
    # print(speech_flags)
    # Step 4: Group contiguous frames
    segments = []
    start = None
    last = None
    gap_count = 0
    for i, flag in enumerate(speech_flags):
        if flag:
            if start is None:
                start = i
            gap_count = 0
            last = i
        else:
            if start is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    last = i - gap_count
                    # If a valid segment is detected, save the segment boundaries
                    if last - start + 1 >= min_frames:
                        segments.append((start, last))
                    start = None
                    last = None
                    gap_count = 0
    # Append the final segment if the audio ends while in speech
    if start is not None and last - start + 1 >= min_frames:
        segments.append((start, last))

    # Step 5: (Optional) Refine boundaries using STFT-based spectral flux
    if refine:
        # Compute the magnitude spectrogram using STFT
        S = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
        print(len(S))
        # Calculate spectral flux: the Euclidean norm of the difference between consecutive frames
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        # Normalize flux to the [0,1] range for consistency
        flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-6)

        refined_boundaries = []
        window = 5  # Number of frames to search around the boundary for refinement
        for start_frame, end_frame in segments:
            # Refine start boundary: look for the local minimum in spectral flux near the start
            search_start = max(0, start_frame - window)
            search_end = min(len(flux), start_frame + window)
            refined_start_frame = search_start + np.argmin(
                flux[search_start:search_end]
            )

            # Refine end boundary similarly
            search_start_e = max(0, end_frame - window)
            search_end_e = min(len(flux), end_frame + window)
            refined_end_frame = search_start_e + np.argmin(
                flux[search_start_e:search_end_e]
            )

            refined_boundaries.append(
                (
                    refined_start_frame * hop_length / sr,
                    refined_end_frame * hop_length / sr,
                )
            )
        boundaries = refined_boundaries

    return boundaries


def detect_profanity(
    audio, sr, speech_segments, profanity_database, dtw_threshold=3000
):
    """
    Detect profanity words by comparing detected speech segments with a profanity database
    using MFCC features and dynamic time warping (DTW).

    Parameters:
        audio (np.array): Audio time series.
        sr (int): Sampling rate.
        speech_segments (list of tuples): Each tuple contains (start_time, end_time) in seconds.
        profanity_database (dict): Dictionary where keys are profanity words and values are audio templates.
        dtw_threshold (float): DTW distance threshold below which the segment is considered a match.

    Returns:
        Each tuple contains (detected_word, start_time, end_time) for a detected profanity.
    """
    profanity_positions = []

    # Precompute MFCC features for each profanity template once.
    template_mfccs = {}
    for word, template_audio in profanity_database.items():
        # print(word)
        mfcc = librosa.feature.mfcc(y=template_audio, sr=sr, n_mfcc=13)
        template_mfccs[word] = mfcc

    # Process each detected speech segment.
    for start, end in speech_segments:
        # Extract the audio for the segment.
        segment_audio = audio[int(start * sr) : int(end * sr)]
        # Compute MFCC features for the segment.
        segment_mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)

        # Compare the segment's MFCCs to each profanity template using DTW.
        for word, template_mfcc in template_mfccs.items():
            # Compute the DTW distance between the segment and the template.
            D, _ = librosa.sequence.dtw(
                X=segment_mfcc, Y=template_mfcc, metric="euclidean"
            )
            distance = D[-1, -1]  # Total accumulated distance from DTW
            # print(distance)
            # If the distance is below our threshold, mark it as a match.
            if distance < dtw_threshold:
                profanity_positions.append((word, start, end))
                # Stop checking other words once a match is found for this segment.
                # break

    return profanity_positions


def process_audio(input_audio_path, profanity_database):
    """Main function to process audio for profanity detection and replacement."""
    y, sr = librosa.load(input_audio_path, sr=24000)
    y = preprocess_audio(y, sr)

    speech_segments = detect_speech_regions(
        audio=y, sr=sr, frame_length=1024, hop_length=512, energy_threshold_ratio=0.6
    )
    profanity_detections = detect_profanity(
        audio=y,
        sr=sr,
        speech_segments=speech_segments,
        profanity_database=profanity_database,
        dtw_threshold=4000,
    )

    # Convert audio to int16 for wave file
    y_int16 = (y * 32767).astype(np.int16)

    # Replace profanity words with beeps of matching length
    for word, start, end, _ in profanity_detections:
        duration = end - start
        beep = generate_beep(frequency=1000, duration=duration, samplerate=sr)
        start_sample = int(start * sr)
        end_sample = min(start_sample + len(beep), len(y_int16))
        y_int16[start_sample:end_sample] = beep[: end_sample - start_sample]

    output_buffer = BytesIO()
    with wave.open(output_buffer, "wb") as output_wav:
        output_wav.setnchannels(1)
        output_wav.setsampwidth(2)
        output_wav.setframerate(sr)
        output_wav.writeframes(y_int16.tobytes())
    output_buffer.seek(0)

    # # Create results dataframe
    # df = pd.DataFrame(profanity_detections, columns=["Word", "Position (s)"])

    return output_buffer
