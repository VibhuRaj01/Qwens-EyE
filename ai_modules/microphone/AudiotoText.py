from RealtimeSTT import AudioToTextRecorder

def get_transcribed_text():
    """
    Returns the transcribed text from the AudioToTextRecorder.
    """
    print('Say "Jarvis" to start recording.')
    with AudioToTextRecorder(wake_words="jarvis") as recorder:
        text = recorder.text()
    return text

# if __name__ == '__main__':
#     transcribed_text = get_transcribed_text()
#     print(transcribed_text)