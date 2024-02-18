import speech_recognition as sr

recognizer = sr.Recognizer()

# Energy Threshold
recognizer.energy_threshold = 4000

try:
    while True:
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=1)
            print("Listening...")
            
            # Time
            audio = recognizer.listen(mic, timeout=30)

        try:
            if audio:
                text = recognizer.recognize_google(audio)
                text = text.lower()
                print(f"Recognized: {text}")

        # When Speaker is Quiet
        except sr.UnknownValueError:
            print("Could not understand audio.")

except KeyboardInterrupt:
    print("\nProgram terminated by user.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
