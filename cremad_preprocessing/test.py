import soundfile as sf
import os

audio_folder = '/home/yeskendir/Downloads/crema-d-mirror-main'
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
test_file = os.path.join(audio_folder, audio_files[0])

print(f"Testing file: {test_file}")
print(f"File size: {os.path.getsize(test_file)} bytes")

try:
    data, sr = sf.read(test_file)
    print(f"soundfile direct read works: {len(data)} samples at {sr}Hz")
except Exception as e:
    print(f"soundfile failed: {repr(e)}")

try:
    import librosa
    y, sr = librosa.load(test_file, sr=None)
    print(f"librosa works: {len(y)} samples at {sr}Hz")
except Exception as e:
    print(f"librosa failed: {repr(e)}")