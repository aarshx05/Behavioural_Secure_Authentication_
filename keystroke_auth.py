import time
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from pynput import keyboard
import sys

# Constants
USER_DATA_PATH = "user_data"
THRESHOLD = 0.4

# Ensure the directory for user data exists
if not os.path.exists(USER_DATA_PATH):
    os.makedirs(USER_DATA_PATH)

# Global variables for keystroke handling
key_times, hold_times, flight_times, password_typed = [], [], [], []
# Store match probabilities for dynamic threshold
user_match_probabilities = {}


# Utility Functions
def reset_keystroke_data():
    """Resets keystroke data for a fresh capture."""
    global key_times, hold_times, flight_times
    key_times, hold_times, flight_times = [], [], []

# Keystroke Event Handling
def on_press(key):
    """Captures the time a key is pressed."""
    try:
        if key == keyboard.Key.enter:
            return False  # Stops listening once Enter is pressed
        elif key.char == password_typed[len(key_times)]:
            key_times.append(time.time())
    except AttributeError:
        pass

def on_release(key):
    """Captures the time a key is released and calculates key hold and flight times."""
    try:
        if key.char == password_typed[len(hold_times)]:
            hold_times.append(time.time() - key_times[len(hold_times)])
            if len(key_times) > 1:
                flight_times.append(key_times[-1] - key_times[-2])
            if len(hold_times) == len(password_typed):
                return False  # Stops listening once password is fully typed
    except AttributeError:
        pass

# Keystroke Data Collection
def collect_keystroke_data(password):
    global password_typed
    password_typed = list(password)
    print("\nPress Enter to start typing your password.")
    input("")  # Wait for user to press Enter to start

    print("\nType your password and press Enter to stop.")
    reset_keystroke_data()
    user_typed_password = ""
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        user_typed_password += input("")  # Wait for user to press Enter to stop
        listener.join()
   
    # Check if the password typed is valid
    if len(hold_times) != len(password_typed):
        print("\nPassword length mismatch. This attempt will be skipped.")
        return None,None

    # Calculate total typing time and return the sample
    total_time = sum(hold_times) + sum(flight_times)
    return np.array([total_time] + hold_times + flight_times), user_typed_password

# Data Collection for Enrollment
def create_user_data(user_id, password, num_samples=10):
    """Collects a specific number of authentic keystroke samples."""
    data = []
    attempts = 0
    while len(data) < num_samples:
        print(f"\nCollecting sample {len(data) + 1}/{num_samples}...")
        sample, pwd = collect_keystroke_data(password)
        if (sample is not None and pwd==password):
            data.append(sample)
        else:
            print("\nIncorrect password attempt. Please try again.")
        attempts += 1

        if attempts >= num_samples * 2:  # Allow twice as many attempts as required samples
            print("\nToo many incorrect attempts. Enrollment failed.")
            break

    return np.array(data)

# Model Training
def train_models(authentic_data, non_authentic_data,choice_train=1):
    """Trains the model using authentic and non-authentic data."""
    X = np.vstack([authentic_data, non_authentic_data])
    y = np.hstack([np.ones(len(authentic_data)), np.zeros(len(non_authentic_data))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if(choice_train==1):
        # SVM Model
        svm_model = SVC(kernel='linear', probability=True) 

        # KNN Model
        knn_model = KNeighborsClassifier(n_neighbors=3)  

        # RF Model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    else:
        # SVM Model: RBF kernel with moderate C and default gamma
        svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)  # Reasonable C for moderate regularization

        # KNN Model: 5 neighbors with distance weighting
        knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')  # Use distance weighting for sensitivity

        # Random Forest Model: Moderate depth with 100 estimators
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    
  
    voting_model = VotingClassifier(estimators=[ 
        ('svm', svm_model),
        ('knn', knn_model),
        ('rf', rf_model)
    ], voting='soft')

    voting_model.fit(X_scaled, y)
    return voting_model, scaler


def adjust_threshold(user_id):
    """Adjust the threshold dynamically based on the last 10 match probabilities."""
    if user_id not in user_match_probabilities:
        return THRESHOLD  # Use static threshold if there are no past attempts

    probabilities = user_match_probabilities[user_id]
    if len(probabilities) < 10:
        return THRESHOLD  # Use static threshold if there are fewer than 10 attempts

    mean_prob = np.mean(probabilities)
    std_prob = np.std(probabilities)
    dynamic_threshold = mean_prob + std_prob  # Dynamic threshold adjustment
    return max(dynamic_threshold, 0.3)  # Ensure the threshold is not too low


# File Operations
def save_user_data(user_id, model, scaler, password, authentic_data=None, synthetic_data=None):
    """Saves the model, scaler, password, and data for a user."""
    user_path = os.path.join(USER_DATA_PATH, user_id)
    os.makedirs(user_path, exist_ok=True)

    with open(os.path.join(user_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(user_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(user_path, 'metadata.pkl'), 'wb') as f:
        pickle.dump({'password': password}, f)

    if authentic_data is not None:
        np.save(os.path.join(user_path, 'authentic_data.npy'), authentic_data)
    if synthetic_data is not None:
        np.save(os.path.join(user_path, 'synthetic_data.npy'), synthetic_data)

    # Save match probabilities for dynamic threshold
    with open(os.path.join(user_path, 'match_probabilities.pkl'), 'wb') as f:
        pickle.dump(user_match_probabilities.get(user_id, []), f)


def load_user_data(user_id):
    """Loads the model, scaler, metadata, and data for a user."""
    user_path = os.path.join(USER_DATA_PATH, user_id)
    if not os.path.exists(user_path):
        print(f"\nUser '{user_id}' not found.")
        return None, None, None, None, None

    with open(os.path.join(user_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(user_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(user_path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    authentic_data = np.load(os.path.join(user_path, 'authentic_data.npy')) if os.path.exists(os.path.join(user_path, 'authentic_data.npy')) else None
    synthetic_data = np.load(os.path.join(user_path, 'synthetic_data.npy')) if os.path.exists(os.path.join(user_path, 'synthetic_data.npy')) else None

    # Load match probabilities
    with open(os.path.join(user_path, 'match_probabilities.pkl'), 'rb') as f:
        user_match_probabilities[user_id] = pickle.load(f)

    return model, scaler, metadata, authentic_data, synthetic_data


# Generate Non-Authentic Data (Synthetic Data)
def generate_non_authentic_data(authentic_data, num_samples=100):
    """Generates synthetic non-authentic data based on authentic data."""
    synthetic_data = []
    for sample in authentic_data:
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.1, size=sample.shape)  # Add Gaussian noise
            synthetic_sample = sample + noise  # Add noise to the original sample
            synthetic_data.append(synthetic_sample)
    return np.array(synthetic_data)

def register_user(user_id, password,choice_train=1):
    # Check if the user already exists
    model, scaler, metadata, _, _ = load_user_data(user_id)
    if model is not None:
        print(f"\nUser '{user_id}' already exists. Use the retrain option to update the model.")
        return

    # Collect new authentic data samples for registration
    new_authentic_data = create_user_data(user_id, password, num_samples=10)
    if len(new_authentic_data) == 0:
        print("\nRegistration failed due to insufficient data.")
        return

    # Generate synthetic data and train the model
    synthetic_data = generate_non_authentic_data(new_authentic_data, num_samples=100)
    model, scaler = train_models(new_authentic_data, synthetic_data,choice_train)

    # Save the model, scaler, password, and data
    save_user_data(user_id, model, scaler, password, new_authentic_data, synthetic_data)
    print(f"\nUser '{user_id}' registered successfully.")

def retrain_user_model(user_id, password, choice_train=1):
    # Load existing user data
    model, scaler, metadata, old_authentic_data, old_synthetic_data = load_user_data(user_id)
    if model is None or metadata is None:
        print(f"\nUser '{user_id}' does not exist. Please register first.")
        return
    if password != metadata['password']:
        print("\nPassword mismatch! Cannot retrain model.")
        return

    # Collect new authentic data samples for retraining
    new_authentic_data = create_user_data(user_id, password, num_samples=10)
    if len(new_authentic_data) == 0:
        print("\nRetraining failed due to insufficient data.")
        return

    # Combine old and new data, then generate additional synthetic data
    combined_authentic_data = np.vstack([old_authentic_data, new_authentic_data]) if old_authentic_data is not None else new_authentic_data
    new_synthetic_data = generate_non_authentic_data(combined_authentic_data, num_samples=50)
    combined_synthetic_data = np.vstack([old_synthetic_data, new_synthetic_data]) if old_synthetic_data is not None else new_synthetic_data

    # Retrain the model with combined data
    model, scaler = train_models(combined_authentic_data, combined_synthetic_data, choice_train)

    # Save the updated model, scaler, and data
    save_user_data(user_id, model, scaler, password, combined_authentic_data, combined_synthetic_data)
    print(f"\nUser '{user_id}' model retrained successfully.")

def check_password(p1,user_id):
    model, scaler, metadata, old_authentic_data, old_synthetic_data = load_user_data(user_id)
    if model is None or metadata is None:
        print(f"\nUser '{user_id}' does not exist. Please register first.")
        return
    if p1!= metadata['password']:
        return 0


def verify_user(model, scaler, password, user_id):
    chk= check_password(password,user_id)
    if(chk==0):
        print("Wrong Initial Password")
        return False
    print("\nPress Enter to start typing your password for verification.")
    new_sample, user_typed_password= collect_keystroke_data(password)
    if new_sample is None:
        print("\nPassword mismatch during verification.")
        return False
    
    chk= check_password(user_typed_password,user_id)
    if(chk==0):
        print("Wrong Initial Password")
        return False
    # Scale the new sample and make predictions
    new_sample_scaled = scaler.transform([new_sample])
    prediction_prob = model.predict_proba(new_sample_scaled)[0][1]  # Get the probability of being authentic

    # Adjust the threshold based on previous attempts
    dynamic_threshold = adjust_threshold(user_id)
    
    print(f"Dynamic Threshold: {dynamic_threshold}, Prediction Probability: {prediction_prob}")
    
    if prediction_prob > dynamic_threshold:
        # Store the match probability for the user
        if user_id not in user_match_probabilities:
            user_match_probabilities[user_id] = []
        
        user_match_probabilities[user_id].append(prediction_prob)
        
        # Keep only the last 10 probabilities
        if len(user_match_probabilities[user_id]) > 10:
            user_match_probabilities[user_id].pop(0)
        
        return True  # User is verified
    else:
        return False  # Verification failed


# Example Command-line Interface for the Keystroke Authentication System
def cli():
    print("Welcome to the Keystroke Authentication System!")
    while True:
        print("\nSelect an option:")
        print("1. Register a new user")
        print("2. Retrain an existing user model")
        print("3. Verify an existing user")
        print("4. Exit")
        choice = input("Enter choice (1-4): ")

        if choice == '1':
            user_id = input("Enter user ID: ")
            password = input("Enter password: ")
            choice_train = input("Enter Choice Of Model (1 - Harsh / 2 - Easy):")
            register_user(user_id, password, choice_train)
        elif choice == '2':
            user_id = input("Enter user ID: ")
            password = input("Enter password: ")
            choice_train = input("Enter Choice Of Model (1 - Harsh / 2 - Easy):")
            retrain_user_model(user_id, password)
        elif choice == '3':
            user_id = input("Enter user ID: ")
            password = input("Enter password: ")

            model, scaler, metadata, _, _ = load_user_data(user_id)
            if model is None:
                print(f"\nUser '{user_id}' does not exist. Please register first.")
            else:
                if verify_user(model, scaler, password,user_id):
                    print("\nUser verified successfully!")
                else:
                    print("\nUser verification failed.")
        elif choice == '4':
            print("\nExiting Keystroke Authentication System.")
            sys.exit()
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    cli()
