# Behavioural_Secure_Authentication

This project is a keystroke dynamics-based authentication system designed to enhance security by analyzing the typing patterns of users. **Note:** The project is currently in development, and future versions will include advanced features such as enhanced dataset handling, increased reference points, and more sophisticated machine learning (ML) algorithms.

## Features
- **Password Strength Checker**: Ensures users create strong passwords based on length, uppercase, lowercase, digits, and special characters.
- **Keystroke Data Capture**: Records and analyzes keystroke timing, including hold and flight times, to create a unique typing pattern for each user.
- **Machine Learning Model**: Trains a Random Forest model using the captured keystroke data to authenticate users based on their typing behavior.
- **Login System**: A simulated login system using the trained ML model to verify users.

## Future Development
- **Enhanced Dataset Collection**: Expanding the number of reference points to include more variations of keystrokes and users, creating a more robust dataset for training.
- **Advanced Feature Engineering**: Incorporating technical terms like **Hesitation Analysis**, **Key Pair Latency**, and **Sequence Pattern Recognition** to refine keystroke dynamics.
- **Model Improvement**: Implementation of deep learning techniques such as **LSTM (Long Short-Term Memory)** networks for better pattern recognition and prediction accuracy.

## Requirements
- Python 3.x
- **Packages**: 
  - `pynput`
  - `numpy`
  - `scikit-learn`
  
Install required packages using:
```bash
pip install pynput numpy scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/keystroke-auth-system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd keystroke-auth-system
   ```
3. Run the main Python file:
   ```bash
   python keystroke_auth.py
   ```

## How It Works
1. **Password Entry**: Users are prompted to enter and confirm a password. Passwords are checked for strength before proceeding.
2. **Keystroke Capture**: As the user types their password, the system captures:
   - **Hold Duration**: Time each key is pressed.
   - **Flight Time**: Time between the release of one key and the press of the next key.
3. **Feature Extraction**: Keystroke features are extracted and used to train a Random Forest classifier.
4. **Authentication**: The user is prompted to log in, and their typing pattern is compared with the trained model.

## File Structure
- **keystroke_auth.py**: Main program file that handles user input, data capture, and ML model training.
- **keystroke_data.csv**: CSV file where keystroke data is logged for analysis and model training.

## Known Issues
- Limited dataset: The model is currently trained with limited data, and accuracy might improve with more diverse data.
- Requires console-based password entry, which might be improved in future versions with a GUI.

## Contributions
Contributions are welcome! Please open an issue for discussion before submitting a pull request.

## License
This project is licensed under the MIT License.

