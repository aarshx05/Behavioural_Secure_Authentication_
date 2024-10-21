import time
import csv
import re
import numpy as np
from pynput import keyboard
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


key_press_times = {}
key_release_times = {}
keystroke_data = []

csv_filename = "keystroke_data.csv"


def check_password_strength(password):
    if (len(password) >= 8 and
        re.search(r'[A-Z]', password) and
        re.search(r'[a-z]', password) and
        re.search(r'[0-9]', password) and
        re.search(r'[!@#$%^&*(),.?":{}|<>]', password)):
        return True
    return False

def on_press(key):
    try:
        key_char = key.char
        key_press_times[key_char] = time.time()
        print(f"Key pressed: {key_char}")  # Debugging statement
    except AttributeError:
        pass


def on_release(key):
    try:
        key_char = key.char
        key_release_times[key_char] = time.time()
        print(f"Key released: {key_char}")  # Debugging statement
        
        press_time = key_press_times[key_char]
        release_time = key_release_times[key_char]
        hold_duration = release_time - press_time

        if len(keystroke_data) > 0:
            last_key_time = keystroke_data[-1]['release_time']
            flight_time = press_time - last_key_time
        else:
            flight_time = 0
        
        keystroke_data.append({
            'key': key_char,
            'press_time': press_time,
            'release_time': release_time,
            'hold_duration': hold_duration,
            'flight_time': flight_time
        })
        print(f"Keystroke data: {keystroke_data[-1]}")  # Debugging statement

    except AttributeError:
        pass

    if key == keyboard.Key.enter:
        print("Enter key pressed. Stopping capture.")  # Debugging statement
        return False


def capture_keystrokes():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def extract_features(keystroke_data):
    hold_durations = [d['hold_duration'] for d in keystroke_data]
    flight_times = [d['flight_time'] for d in keystroke_data[1:]]  
    return np.array(hold_durations + flight_times)


def train_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    return model


def save_to_single_csv(password_number):
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['password_attempt', 'key', 'press_time', 'release_time', 'hold_duration', 'flight_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only once when creating a new file
        if password_number == 1:  
            writer.writeheader()

        for entry in keystroke_data:
            entry['password_attempt'] = password_number
            writer.writerow(entry)
            print(f"Writing to CSV: {entry}")  # Debugging statement

# Simulate user login with multiple attempts
def user_login(model):
    attempts = 3
    while attempts > 0:
        print(f"\nYou have {attempts} attempts remaining.")
        keystroke_data.clear()
        print("Enter your password:")
        password = input()

        print("Start typing...")
        capture_keystrokes()

        features = extract_features(keystroke_data).reshape(1, -1)
        prediction = model.predict(features)

        if prediction[0] == 1:
            print("Login Success!")
            return True
        else:
            print("Login Failed. Try again.")
            attempts -= 1

    print("Maximum login attempts reached.")
    return False

# Main logic for training and testing
def main():
    collected_data = []
    labels = []
    
    # Collect 10 sets of password data and keystrokes for training
    for i in range(5):
        print(f"Enter password attempt {i+1}:")
        password = input()

        if check_password_strength(password):
            print("Password is strong.")
            print("Now start typing the password. Press 'Enter' when done.")
            
            keystroke_data.clear()
            capture_keystrokes()
            
            features = extract_features(keystroke_data)
            collected_data.append(features)
            labels.append(1)  # All data is from the genuine user, hence label as '1'
            
            # Save to the single CSV file
            save_to_single_csv(i + 1)
        else:
            print("Password is weak. Please try again.")
    
    # Convert data to numpy arrays for training
    collected_data = np.array(collected_data)
    labels = np.array(labels)

    # Train the model
    model = train_model(collected_data, labels)

    # Simulate user login
    user_login(model)

if __name__ == "__main__":
    main()
