import os
import subprocess
import uuid
import sqlite3
import json
from datetime import datetime
#Connect to the DB for meta data
connect = sqlite3.connect('audio_metadata.db')
cursor = connect.cursor()
#Create a table if one does not already exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS audio_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    format TEXT,
    creation_date TEXT,
    duration REAL,
    bitrate INTEGER,
    size_kb INTEGER
)
''')
connect.commit() #Commit to db
dataPath = os.path.join(".", "data") # For crossplatform compatibility, Unix "/" vs Windows "\\"
video_extensions = (".mov", ".mp4", ".mkv", ".avi", ".MOV", ".MP4", ".MKV", ".AVI") #consider all video types and file extensions on all platforms
def handle_metadata(connect, file_path, format, creation_date,duration, bitrate, size_kb):
    sql = '''
    INSERT INTO audio_metadata (
        file_path, format, creation_date, duration, bitrate, size_kb
    ) VALUES (?, ?, ?, ?, ?, ?)
    '''
    cursor = connect.cursor()
    cursor.execute(sql, (
        file_path, format, creation_date, duration, bitrate, size_kb
    ))
    connect.commit()
def get_metadata(file_path):
    try:
        ffprobe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration:format=bit_rate:format=codec_name:format=tags:streams=tags",
            "-of", "json",
            file_path
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        # Debug: Print the entire metadata output
        print(f"Metadata for {file_path}:")
        print(json.dumps(metadata, indent=4))
        format_info = metadata.get('format', {})
        # Extract specific metadata
        creation_time = format_info.get('tags', {}).get('creation_time', 'Unknown')
        # Use os.path.getsize() to get file size in bytes
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes // 1024  # Convert to KB
        return {
            'duration': float(format_info.get('duration', 0)),
            'bitrate': int(format_info.get('bit_rate', 0)) // 1000,  # Bitrate in kbps
            'size_kb': size_kb,
            'creation_time': creation_time,
        }
    except subprocess.CalledProcessError:
        print(f"Failed to retrieve metadata for {file_path}")
        return {
            'duration': 0,
            'bitrate': 0,
            'size_kb': 0,
            'creation_time': 'Unknown'
        }
def video_to_mp3(input_file, output_file, start_time, end_time):
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "192k",
        "-ar", "44100",
        "-ss", str(start_time),  # Start time
        "-to", str(end_time),    # End time
        "-y",
        output_file
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Successfully converted segment ({start_time}-{end_time}) to MP3!")
    except subprocess.CalledProcessError as e: 
        print("Conversion failed!")

def generate_unique_identifier(filename):
    base_name, _ = os.path.splitext(filename)  # Split the filename into base name and extension
    while True:
        identifier = uuid.uuid4().hex
        output_filename = f"{base_name}_audio_{identifier}.mp3"
        output_file = os.path.join(dataPath, output_filename)
        if not os.path.exists(output_file):
            return output_filename

for i, filename in enumerate(os.listdir(dataPath)):
    print(i, filename)
    if filename.endswith(video_extensions):
        input_file = os.path.join(dataPath, filename)
        output_filename =  generate_unique_identifier(filename)
        output_file = os.path.join(dataPath, output_filename)
                # Debug print for paths
        print(f"Input file path: {input_file}")
        print(f"Output file path: {output_file}")
        if os.path.exists(input_file):
            video_to_mp3(input_file, output_file)
            # Extract metadata from the converted MP3 file
            metadata = get_metadata(output_file)
            handle_metadata(
                connect, output_file,
                format='mp3',
                creation_date=datetime.now().isoformat(),
                duration=metadata['duration'],
                bitrate=metadata['bitrate'],
                size_kb=metadata['size_kb']
            )
        else:
            print(f"File does not exist: {input_file}")
#Close database connection
connect.close()