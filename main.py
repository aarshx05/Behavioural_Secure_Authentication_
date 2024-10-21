import time
import csv
import numpy as np
from pynput import keyboard
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KeystrokeManager:
    def __init__(self, csv_filename):
        self.keystroke_data = []
        self.csv_filename = csv_filename
        self.passwords = []
        self.max_attempts = 5
        self.baseline_data = []
        self.labels = []
        self.model = None

    def on_press(self, key):
        try:
            key_char = key.char
            press_time = time.time()
            self.keystroke_data.append({
                'key': key_char,
                'press_time': press_time,
                'release_time': None,
                'hold_duration': None
            })
        except AttributeError:
            pass

    def on_release(self, key):
        release_time = time.time()
        if self.keystroke_data:
            last_entry = self.keystroke_data[-1]
            if last_entry['release_time'] is None:
                last_entry['release_time'] = release_time
                last_entry['hold_duration'] = release_time - last_entry['press_time']

        if key == keyboard.Key.enter:
            return False

    def capture_keystrokes(self):
        for attempt in range(self.max_attempts):
            print(f"\nAttempt {attempt + 1}/{self.max_attempts}: Start typing your password... (Press 'Enter' to stop)")
            self.keystroke_data.clear()

            with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                listener.join()

            password = ''.join(entry['key'] for entry in self.keystroke_data if entry['key'])
            self.passwords.append(password)
            print(f"Captured Password: {password}")

            timings = self.extract_timings(self.keystroke_data)
            self.baseline_data.append(timings)
            # Label the first attempt as genuine (1) and subsequent ones as impostor (0)
            self.labels.append(1 if attempt == 0 else 0)

    def extract_timings(self, keystroke_data):
        timings = []
        for i in range(len(keystroke_data) - 1):
            hold_duration = keystroke_data[i]['hold_duration']
            time_between_keys = keystroke_data[i + 1]['press_time'] - keystroke_data[i]['release_time']
            timings.extend([hold_duration, time_between_keys])
        return np.array(timings)

    def train_model(self):
        data = np.array(self.baseline_data)
        labels = np.array(self.labels)

        if len(data) > 1:
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
        else:
            print("Not enough data to train the model.")

    def authenticate_password(self):
        entered_password = input("Please enter your password for verification: ")
        
        if entered_password in self.passwords:
            print("Password verified successfully.")
            return True
        else:
            print("Password verification failed.")
            return False

    def login_verification(self):
        print("Start typing your password for login verification... (Press 'Enter' to stop)")
        self.keystroke_data.clear()

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

        attempt_timings = self.extract_timings(self.keystroke_data)

        if self.model:
            # Reshape attempt_timings to fit the model's expected input shape
            attempt_timings = attempt_timings.reshape(1, -1)
            prediction = self.model.predict(attempt_timings)
            if prediction[0] == 1:
                print("Login successful based on keystroke dynamics.")
            else:
                print("Login failed due to mismatched keystroke pattern.")
        else:
            print("Model not trained yet.")

def main():
    csv_filename = "keystroke_data.csv"
    keystroke_manager = KeystrokeManager(csv_filename)

    keystroke_manager.capture_keystrokes()
    keystroke_manager.train_model()

    # Authenticate the password
    is_verified = keystroke_manager.authenticate_password()
    if is_verified:
        # Perform login verification using keystroke dynamics
        keystroke_manager.login_verification()

if __name__ == "__main__":
    main()
