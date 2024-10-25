import time
import csv
import os
import numpy as np
from pynput import keyboard
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox


class KeystrokeManager:
    def __init__(self, csv_filename):
        self.keystroke_data = []
        self.csv_filename = csv_filename
        self.user_id = None
        self.max_attempts = 5
        self.baseline_data = []
        self.labels = []
        self.model = None
        self.password = None

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
            return False  # Stop listener on Enter

    def clear_keystroke_data(self):
        """Clear the keystroke data after each attempt."""
        self.keystroke_data.clear()

    def extract_timings(self, keystroke_data):
        timings = []
        for i in range(len(keystroke_data) - 1):
            hold_duration = keystroke_data[i]['hold_duration']
            time_between_keys = keystroke_data[i + 1]['press_time'] - keystroke_data[i]['release_time']
            timings.extend([hold_duration, time_between_keys])
        return np.array(timings)

    def load_user_data(self, user_id):
        data = []
        labels = []
        if os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == user_id:  # Check if the row belongs to the current user
                        data.append([float(x) for x in row[2:-1]])  # Skip user_id and password
                        labels.append(int(row[-1]))  # The last value is the label
        return np.array(data), np.array(labels)

    def train_model(self, user_id):
        data, labels = self.load_user_data(user_id)

        if len(data) > 1:
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            messagebox.showinfo("Model Training", f"Model trained with accuracy: {accuracy * 100:.2f}%")
        else:
            messagebox.showwarning("Training Error", "Not enough data to train the model.")

    def authenticate_password(self, user_id, entered_password):
        with open(self.csv_filename, 'r') as file:
            for row in csv.reader(file):
                if row[0] == user_id and row[1] == entered_password:
                    return True
        return False

    def login_verification(self, feedback_label):
        self.clear_keystroke_data()
        feedback_label.config(text="Start typing your password for keystroke verification.")
        
        # Start keystroke listener
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        listener.join()  # Wait for listener to finish

        # Debugging: Check if keystroke data is captured
        print("Keystroke Data Captured:", self.keystroke_data)
        
        attempt_timings = self.extract_timings(self.keystroke_data)

        # Check if attempt_timings is empty before predicting
        if attempt_timings.size == 0:
            feedback_label.config(text="No keystroke data captured. Please try again.")
            messagebox.showerror("No Data", "No keystroke data captured. Please try again.")
            return

        if self.model:
            print("Attempt Timings Shape:", attempt_timings.shape)
            attempt_timings = attempt_timings.reshape(1, -1)  # Ensure correct input shape
            prediction = self.model.predict(attempt_timings)
            if prediction[0] == 1:
                feedback_label.config(text="Login successful based on keystroke dynamics.")
                messagebox.showinfo("Login Success", "Login successful based on keystroke dynamics.")
            else:
                feedback_label.config(text="Login failed due to mismatched keystroke pattern.")
                messagebox.showerror("Login Failed", "Login failed due to mismatched keystroke pattern.")
        else:
            messagebox.showwarning("Model Not Trained", "Model not trained yet.")


# Registration Window
def register_window():
    root = tk.Tk()
    root.title("Keystroke Authentication - Register")

    manager = KeystrokeManager("keystroke_data_gui.csv")

    def on_register():
        user_id = user_id_entry.get()
        password = password_entry.get()
        if user_id and password:
            # Clear existing data
            manager.baseline_data.clear()
            manager.labels.clear()

            feedback_label.config(text="Start typing your password... (Press 'Enter' to stop)")
            root.update()

            for attempt in range(manager.max_attempts):
                manager.clear_keystroke_data()
                feedback_label.config(text=f"Attempt {attempt + 1}/{manager.max_attempts}: Start typing your password... (Press 'Enter' to stop)")
                root.update()

                with keyboard.Listener(on_press=manager.on_press, on_release=manager.on_release) as listener:
                    listener.join()

                # Collect the password timings
                captured_password = ''.join(entry['key'] for entry in manager.keystroke_data if entry['key'])
                if captured_password != password:
                    feedback_label.config(text="Incorrect password, please try again.")
                    root.update()
                    continue  # Skip to the next attempt if password is incorrect

                # Extract timings and save
                timings = manager.extract_timings(manager.keystroke_data)
                manager.baseline_data.append(timings)
                # Label the first attempt as genuine (1) and subsequent ones as impostor (0)
                manager.labels.append(1 if attempt == 0 else 0)

            # Save data to CSV
            with open(manager.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for i in range(manager.max_attempts):
                    writer.writerow([user_id, password] + manager.baseline_data[i].tolist() + [manager.labels[i]])

            messagebox.showinfo("Registration Success", "Registration successful!")
            root.destroy()
        else:
            messagebox.showwarning("Input Error", "Please enter both User ID and Password")

    # UI Setup
    tk.Label(root, text="User ID:").grid(row=0, column=0)
    user_id_entry = tk.Entry(root)
    user_id_entry.grid(row=0, column=1)

    tk.Label(root, text="Password:").grid(row=1, column=0)
    password_entry = tk.Entry(root, show="*")
    password_entry.grid(row=1, column=1)

    tk.Button(root, text="Register", command=on_register).grid(row=2, columnspan=2)
    feedback_label = tk.Label(root, text="")
    feedback_label.grid(row=3, columnspan=2)

    root.mainloop()


# Login Window
def login_window():
    root = tk.Tk()
    root.title("Keystroke Authentication - Login")

    manager = KeystrokeManager("keystroke_data_gui.csv")

    def on_login():
        user_id = user_id_entry.get()
        password = password_entry.get()

        if manager.authenticate_password(user_id, password):
            manager.user_id = user_id
            feedback_label.config(text="Password verified successfully. Start typing for keystroke verification...")
            root.update()
            
            # Train model only if there is enough data
            manager.train_model(user_id)  # Train model with user data
            
            # Now perform keystroke verification
            manager.login_verification(feedback_label)
        else:
            feedback_label.config(text="Password verification failed.")
            messagebox.showerror("Login Failed", "Password verification failed.")

    # UI Setup
    tk.Label(root, text="User ID:").grid(row=0, column=0)
    user_id_entry = tk.Entry(root)
    user_id_entry.grid(row=0, column=1)

    tk.Label(root, text="Password:").grid(row=1, column=0)
    password_entry = tk.Entry(root, show="*")
    password_entry.grid(row=1, column=1)

    tk.Button(root, text="Login", command=on_login).grid(row=2, columnspan=2)
    feedback_label = tk.Label(root, text="")
    feedback_label.grid(row=3, columnspan=2)

    root.mainloop()


def main_gui():
    root = tk.Tk()
    root.title("Keystroke Authentication")

    tk.Button(root, text="Register", command=register_window).pack(pady=10)
    tk.Button(root, text="Login", command=login_window).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main_gui()
