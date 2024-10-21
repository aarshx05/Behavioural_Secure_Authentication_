import time
import csv
import numpy as np
from pynput import keyboard

class KeystrokeManager:
    def __init__(self, csv_filename):
        self.keystroke_data = []
        self.csv_filename = csv_filename
        self.passwords = []  # List to store captured passwords
        self.max_attempts = 5
        self.baseline_data = []  # Store baseline timing data for model training

    def on_press(self, key):
        try:
            key_char = key.char
            press_time = time.time()
            print(f"Key pressed: {key_char}")  # Debugging statement
            # Log press event
            self.keystroke_data.append({
                'key': key_char,
                'press_time': press_time,
                'release_time': None,
                'hold_duration': None
            })
        except AttributeError:
            # Handle special keys if necessary
            print(f"Special key pressed: {key}")  # Debugging statement

    def on_release(self, key):
        release_time = time.time()
        if self.keystroke_data:
            # Update the last key pressed
            last_entry = self.keystroke_data[-1]
            if last_entry['release_time'] is None:
                last_entry['release_time'] = release_time
                last_entry['hold_duration'] = release_time - last_entry['press_time']
                print(f"Key released: {last_entry['key']}")  # Debugging statement
                print(f"Keystroke data: {last_entry}")  # Debugging statement

        if key == keyboard.Key.enter:
            print("Enter key pressed. Stopping capture for this attempt.")  # Debugging statement
            return False  # Stop the listener

    def capture_keystrokes(self):
        for attempt in range(self.max_attempts):
            print(f"\nAttempt {attempt + 1}/{self.max_attempts}: Start typing your password... (Press 'Enter' to stop)")
            self.keystroke_data.clear()  # Clear previous data

            with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                listener.join()

            # Save the captured keystrokes as a password
            password = ''.join(entry['key'] for entry in self.keystroke_data)
            self.passwords.append(password)
            print(f"Captured Password: {password}")  # Debugging statement

            # Collect baseline data for each password attempt
            self.baseline_data.append(self.extract_timings(self.keystroke_data))

    def extract_timings(self, keystroke_data):
        """Extract timing data from keystrokes."""
        timings = []
        for i in range(len(keystroke_data) - 1):
            hold_duration = keystroke_data[i]['hold_duration']
            time_between_keys = keystroke_data[i + 1]['press_time'] - keystroke_data[i]['release_time']
            timings.append((hold_duration, time_between_keys))
        return np.array(timings)  # Convert to numpy array for further processing

    def save_to_csv(self):
        if not self.keystroke_data:
            print("No keystroke data to save.")  # Debugging statement
            return
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['key', 'press_time', 'release_time', 'hold_duration', 'attempt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only once when creating a new file
            if csvfile.tell() == 0:  # Check if the file is empty
                writer.writeheader()

            for attempt_number, entry in enumerate(self.keystroke_data, start=1):
                entry['attempt'] = attempt_number  # Add attempt number
                writer.writerow(entry)
                print(f"Writing to CSV: {entry}")  # Debugging statement

    def authenticate_password(self):
        # Prompt user to enter their password for verification
        entered_password = input("Please enter your password for verification: ")
        
        # Check if the entered password matches any of the captured passwords
        if entered_password in self.passwords:
            print("Password verified successfully.")
            return True, self.baseline_data[self.passwords.index(entered_password)]
        else:
            print("Password verification failed.")
            return False, None

    def verify_keystroke_pattern(self, attempt_timings, baseline_timings):
        """Compare the timings of the current attempt with the baseline."""
        # You can implement a more sophisticated distance metric here (e.g., Euclidean distance)
        if len(attempt_timings) != len(baseline_timings):
            return False
        
        # Calculate the differences
        differences = np.abs(attempt_timings - baseline_timings)
        threshold = 0.2  # Define a threshold for acceptable variation

        # Check if all differences are below the threshold
        return np.all(differences < threshold)

    def train_model(self):
        # Implement training logic using the captured keystroke data
        print("Training model with the following passwords:")
        for index, password in enumerate(self.passwords):
            print(f"Attempt {index + 1}: {password}")  # Debugging statement

        # Placeholder for ML training logic
        # Here you can include logic to train a model with the captured keystroke patterns

def main():
    csv_filename = "keystroke_data.csv"
    keystroke_manager = KeystrokeManager(csv_filename)

    keystroke_manager.capture_keystrokes()  # Capture keystrokes
    keystroke_manager.save_to_csv()  # Save to CSV after capturing

    # Authenticate the password
    is_verified, baseline_timings = keystroke_manager.authenticate_password()
    if is_verified:
        # Capture the user's login attempt to verify timing patterns
        print("Start typing your password for login verification... (Press 'Enter' to stop)")
        keystroke_manager.keystroke_data.clear()  # Clear previous data for login attempt
        
        with keyboard.Listener(on_press=keystroke_manager.on_press, on_release=keystroke_manager.on_release) as listener:
            listener.join()

        # Extract timings for the login attempt
        attempt_timings = keystroke_manager.extract_timings(keystroke_manager.keystroke_data)
        
        # Verify the login attempt's keystroke pattern against the baseline
        if keystroke_manager.verify_keystroke_pattern(attempt_timings, baseline_timings):
            print("Login successful based on keystroke dynamics.")
        else:
            print("Login failed due to mismatched keystroke pattern.")
        
        if keystroke_manager.verify_keystroke_pattern(attempt_timings, baseline_timings):
            print("Login successful based on keystroke dynamics.")
        else:
            print("Login failed due to mismatched keystroke pattern.")
    
    keystroke_manager.train_model()  # Train the model with captured data

if __name__ == "__main__":
    main()
