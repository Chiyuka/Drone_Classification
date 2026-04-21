import torch
import torchaudio
import pandas as pd
from pydub import AudioSegment
import os
import math
import shutil
import numpy as np


# CONFIGURATION 

AUDIO_FILE_PATH = 'Bruel 4006 - Bal elso_01.wav'
METADATA_PATH = 'drone_dataset_spectrograms/master_drone_labels.csv' 
SPECTROGRAM_OUTPUT_FOLDER = 'drone_dataset_spectrograms'

#Spectrogram Parameters (Standard for CNN Audio Input)
SAMPLE_RATE = 44100  # Target Sample Rate (Hz)
N_FFT = 512         # Window size for the Fourier Transform
N_MELS = 128         # Number of Mel bands (Vertical resolution of the image)

# Initialize the Spectrogram Transform
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
)

# --- LOAD DATA AND SETUP ENVIRONMENT ---
try:
    df = pd.read_csv(METADATA_PATH)
    os.makedirs(SPECTROGRAM_OUTPUT_FOLDER, exist_ok=True)

    # Load the 2GB audio file once using Pydub
    audio_segment = AudioSegment.from_file(AUDIO_FILE_PATH)
    print(f"Loaded {len(df)} segments for feature extraction.")
    
except FileNotFoundError as e:
    print(f"ERROR: Could not find required file. Check paths. {e}")
    exit()
except Exception as e:
    print(f"ERROR loading audio. Ensure FFmpeg is installed correctly. {e}")
    exit()

# PROCESSING LOOP: SLICE -> EXTRACT -> TRANSFORM -> SAVE 
new_metadata = []

for index, row in df.iterrows():
    start_ms = row['start_ms']
    end_ms = row['end_ms']
    segment_id = row['segment_id']
    
    # 1. Efficient Audio Slicing (Pydub handles the large file)
    pydub_segment = audio_segment[start_ms:end_ms]
    
    # 2. Extract raw audio data using pydub's native sample handling
    # This automatically handles the 24-bit to 32-bit conversion and sign extension
    
    # Get array of samples (returns array.array, convert to numpy for PyTorch)
    sample_array = np.array(pydub_segment.get_array_of_samples()) 
    
    # Convert the numpy array to a PyTorch tensor (1D tensor)
    segment_tensor = torch.from_numpy(sample_array).float()
    
    # Normalize the audio to the range [-1.0, 1.0] (Max value for 24-bit signed is 2^23 - 1)
    # The normalization constant remains correct.
    segment_tensor = segment_tensor / (2**23 - 1)
    
    # Normalize the 24-bit audio to the range [-1.0, 1.0]
    # Max value for 24-bit signed int is 2^23 - 1
    segment_tensor = segment_tensor / (2**23 - 1)
    
    # Get original sample rate from Pydub segment
    original_sr = pydub_segment.frame_rate  # Should be 192000 Hz
    
    # Ensure the tensor is 2D: [1, N]
    if segment_tensor.dim() == 1:
        segment_tensor = segment_tensor.unsqueeze(0)  # Shape: [1, N]
    
    # 3. Resample from 192 kHz to 44.1 kHz if needed
    if original_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=SAMPLE_RATE)
        segment_tensor = resampler(segment_tensor)
    
    # 4. Apply Mel-Spectrogram Transformation
    mel_spectrogram = mel_spectrogram_transform(segment_tensor)
    
    # 5. Apply LOG transformation AND STANDARDIZATION
    epsilon = 1e-8
    log_mel_spectrogram = torch.log(mel_spectrogram + epsilon)
    
    # --- LOCAL STANDARDIZATION FIX (Zero Mean, Unit Variance) ---
    # This solves the Dying ReLU problem by ensuring values are centered around zero.
    mean = log_mel_spectrogram.mean()
    std = log_mel_spectrogram.std()
    
    # Avoid dividing by zero if the segment is silent
    if std.item() < epsilon:
        std = torch.tensor(1.0).float()
    
    # Final Spectrogram output (Zero Mean / Unit Variance)
    standardized_spec = (log_mel_spectrogram - mean) / std
    
    # 6. Save the Tensor and Update Metadata
    output_tensor_path = os.path.join(SPECTROGRAM_OUTPUT_FOLDER, f'seg_{segment_id}.pt')
    
    # SAVE THE STANDARDIZED TENSOR
    torch.save(standardized_spec, output_tensor_path)
    
    # --- Metadata Collection Fix ---
    # 1. Convert the Pandas Series (row) into a standard Python dictionary
    row_data = row.to_dict()
    # 2. Add the new feature path to the dictionary
    row_data['feature_path'] = output_tensor_path
    # 3. Append the clean dictionary to the metadata list
    new_metadata.append(row_data)

    
    # Progress indicator
    if (index + 1) % 1000 == 0:
        print(f"Processed {index + 1}/{len(df)} segments...")


# --- FINAL CLEANUP AND OUTPUT ---
# Save the updated CSV with the path to the feature files
df_final = pd.DataFrame(new_metadata)
output_csv_path = os.path.join(SPECTROGRAM_OUTPUT_FOLDER, 'final_labeled_dataset.csv')
df_final.to_csv(output_csv_path, index=False)

print(f"\n{'='*50}")
print("FEATURE GENERATION COMPLETE")
print(f"{'='*50}")
print(f"Total segments processed: {len(df)}")
print(f"Spectrograms saved to: {SPECTROGRAM_OUTPUT_FOLDER}")
print(f"Metadata saved to: {output_csv_path}")

# Print sample statistics
print(f"\nSample spectrogram statistics:")
sample_path = os.path.join(SPECTROGRAM_OUTPUT_FOLDER, 'seg_0.pt')
if os.path.exists(sample_path):
    sample_spec = torch.load(sample_path)
    print(f"  Shape: {sample_spec.shape}")
    print(f"  Range: [{sample_spec.min():.3f}, {sample_spec.max():.3f}]")
    print(f"  Mean: {sample_spec.mean():.3f}, Std: {sample_spec.std():.3f}")
    print(f"  This should be LOG-MEL values (positive, not negative dB)")