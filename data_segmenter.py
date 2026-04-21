from pydub import AudioSegment
import pandas as pd
import os
import math

# --- Configuration ---
AUDIO_FILE_PATH = 'Bruel 4006 - Bal elso_01.wav'
OUTPUT_FOLDER = 'drone_dataset_spectrograms'
SEGMENT_DURATION_MS = 100  # 100 milliseconds
SAMPLE_RATE = 192000 # From the marker file 

# Define Drone Flight Periods (Manual Parsing based on Marker File ) ---
# These periods define when the drone is guaranteed to be present (Label=1).
# Time is defined in milliseconds (ms).

FLIGHT_EVENTS = [
    # Flight 1: Mavic (starts at 3:05, ends at 20:26)
    {'start': 185000, 'end': 1226000, 'drone_type': 'Mavic_1'},
    # Flight 2: Mavic 2 (starts at 24:51, ends at 37:56)
    {'start': 1491000, 'end': 2276000, 'drone_type': 'Mavic_2'},
    # Flight 3: Mavic Mini (starts at 41:48, ends at 53:10)
    {'start': 2508000, 'end': 3190000, 'drone_type': 'Mavic_Mini'},
]

# Load Audio and Initialize Data Structure ---
try:
    print(f"Loading audio file: {AUDIO_FILE_PATH}...")
    audio = AudioSegment.from_file(AUDIO_FILE_PATH)
    total_length_ms = len(audio)
    print(f"Total audio length: {total_length_ms / (1000 * 60):.2f} minutes.")
except Exception as e:
    # This error handles the case where the path is wrong or ffmpeg isn't found
    print(f"ERROR: Could not load audio file. Check the AUDIO_FILE_PATH.")
    print(f"System Error Message: {e}")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# This is the dynamic container for all segment data. It starts blank.
dataset_metadata = []

# 3. Iteratively Segment and Label the Audio (The Chunking Solution) ---
num_segments = math.ceil(total_length_ms / SEGMENT_DURATION_MS)

print(f"Total number of 100ms segments to process: {num_segments}")

for i in range(num_segments):
    start_ms = i * SEGMENT_DURATION_MS
    end_ms = min((i + 1) * SEGMENT_DURATION_MS, total_length_ms) 
    
    # 3.1 Extract the Segment (Memory Efficient Slicing)
    segment = audio[start_ms:end_ms]
    
    # 3.2 Determine the Label (Binary Classification: Drone Present=1 or Not=0)
    is_drone_present = 0
    drone_type = 'None'
    
    # Check if the current 100ms segment falls within any defined flight event
    for event in FLIGHT_EVENTS:
        if event['start'] <= start_ms < event['end']:
            is_drone_present = 1
            drone_type = event['drone_type']
            break # Once found, assign the label and move to the next segment

    # 3.3 Store Metadata
    segment_info = {
        'segment_id': i,
        'start_ms': start_ms,
        'end_ms': end_ms,
        'duration_ms': SEGMENT_DURATION_MS,
        'label_binary': is_drone_present,
        'label_type': drone_type,         # Label for the classification task
    }
    dataset_metadata.append(segment_info)

# --- Final Output ---
df_metadata = pd.DataFrame(dataset_metadata)
output_csv_path = os.path.join(OUTPUT_FOLDER, 'master_drone_labels.csv')
df_metadata.to_csv(output_csv_path, index=False)
print(f"Finished processing. Total segments created: {len(dataset_metadata)}")
print(f"Master label file saved to {output_csv_path}")