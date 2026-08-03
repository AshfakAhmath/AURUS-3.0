import sys
import os

try:
    import pyaudio
except ImportError:
    print("ERROR: PyAudio is not installed. Install it using: pip install pyaudio")
    sys.exit(1)

def main():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("=========================================================")
    print("                AVAILABLE AUDIO DEVICES                  ")
    print("=========================================================")
    
    input_devices = []
    for i in range(0, numdevices):
        dev_info = p.get_device_info_by_host_api_device_index(0, i)
        if dev_info.get('maxInputChannels') > 0:
            print(f"Device ID {i}: {dev_info.get('name')} (Max Input Channels: {dev_info.get('maxInputChannels')})")
            input_devices.append(i)
            
    print("=========================================================\n")
    
    if not input_devices:
        print("ERROR: No input audio devices (microphones) found!")
        p.terminate()
        sys.exit(1)
        
    # Get MIC_INDEX from environment
    from dotenv import load_dotenv
    load_dotenv()
    mic_idx = os.getenv("MIC_INDEX")
    
    if mic_idx and mic_idx.strip():
        try:
            device_index = int(mic_idx)
            print(f"Current MIC_INDEX in .env: {device_index}")
        except ValueError:
            print(f"Warning: Invalid MIC_INDEX in .env: '{mic_idx}'. Defaulting to auto-selection.")
            device_index = input_devices[0]
    else:
        print("No MIC_INDEX set in .env. PyAudio will auto-select default device.")
        device_index = None # Default device
        
    print("\nStarting 3-second recording test. Speak into the microphone now...")
    
    try:
        # 16000 Hz, 1 channel, 16-bit PCM (same as Vosk settings)
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            input_device_index=device_index
        )
        
        frames = []
        for _ in range(0, int(16000 / 1024 * 3)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            
        stream.close()
        p.terminate()
        
        # Calculate volume
        import numpy as np
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        volume = np.abs(audio_np).mean()
        
        print("\nRecording finished!")
        print(f"Average Sound Amplitude (Volume): {volume:.1f}")
        
        if volume < 50:
            print("\nWARNING: Volume level is extremely low. The microphone might be muted or not capturing any sound.")
            print("Try adjusting your mic volume with 'alsamixer' in your terminal or checking your mic hardware.")
        else:
            print("\nSUCCESS: Microphone is capturing audio successfully!")
            
    except Exception as e:
        print(f"\nERROR opening microphone stream: {e}")
        p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
