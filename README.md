

# 🛡️ Keystroke Dynamics-Based Authentication System

## Project Overview
This project aims to develop a keystroke dynamics-based authentication system that enhances user security by analyzing individual typing patterns. By capturing how users type their passwords, this system provides an additional layer of authentication, making it harder for unauthorized users to gain access. As online security threats continue to rise, innovative solutions like keystroke dynamics offer a promising way to improve user safety and experience.

## Objectives
- **Objective 1**: Develop a reliable system for capturing and analyzing keystroke dynamics.
- **Objective 2**: Implement a machine learning model to authenticate users based on their unique typing behavior.
- **Objective 3**: Ensure a user-friendly registration and login process through a graphical user interface (GUI).
- **Objective 4**: Improve the accuracy and reliability of the model by expanding the dataset over time.

## Problem Statement
Traditional authentication methods, such as passwords and PINs, are increasingly vulnerable to breaches. Users often create weak passwords, reuse them across multiple sites, or fall prey to phishing attacks. The challenge lies in developing a more robust and user-friendly authentication mechanism that minimizes these risks without sacrificing convenience. 

## Proposed Solution
The proposed solution involves creating a keystroke dynamics-based authentication system that:
- Captures keystroke timing data, including hold duration and flight time between key presses.
- Utilizes machine learning techniques, specifically Random Forest classifiers, to analyze the captured data and predict user authentication.
- Offers a graphical user interface (GUI) for easy interaction during registration and login processes.

### Tools and Libraries
The following tools and libraries are used in this project:
- **Python 3.x**: The programming language used for development.
- **pynput**: A library for capturing keystroke data.
- **numpy**: Used for numerical operations and data handling.
- **scikit-learn**: A library for implementing machine learning algorithms, particularly for the Random Forest model.

## Features and Functionalities
- **Password Strength Checker**: Ensures that users create strong passwords by analyzing criteria such as length, character variety (uppercase, lowercase, digits, special characters), and overall complexity.
- **Keystroke Data Capture**: Records various keystroke timing metrics, including:
  - **Hold Duration**: The time each key is pressed down.
  - **Flight Time**: The time between the release of one key and the press of the next key.
- **Machine Learning Model**: 
  - Trains a Random Forest classifier using the captured keystroke data to identify unique typing patterns for each user.
  - Utilizes the extracted features from the keystroke data for effective user authentication.
- **User Registration**: 
  - Allows users to register by entering a user ID and password.
  - Captures and saves multiple typing attempts to create a baseline for future authentication.
- **Login System**: 
  - Prompts users to enter their credentials and verifies their passwords.
  - Captures typing patterns during the login process to authenticate users against the trained model.
- **Feedback Mechanism**: 
  - Provides real-time feedback to users during registration and login, indicating the success or failure of authentication attempts.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/keystroke-auth-system.git
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd keystroke-auth-system
   ```
3. **Install Required Packages**:
   Make sure you have Python 3.x installed. Then, install the necessary packages:
   ```bash
   pip install pynput numpy scikit-learn
   ```
4. **Run the Main Python File**:
   ```bash
   python keystroke_auth.py
   ```
5. **Registration**:
   - Open the application and select the "Register" option.
   - Enter a user ID and a password.
   - Follow the prompts to capture your typing pattern.
6. **Login**:
   - Open the application and select the "Login" option.
   - Enter your user ID and password.
   - Type your password when prompted to verify your keystroke dynamics.

## Known Issues
- **Login Model Issue**: The login functionality currently faces challenges in capturing keystroke data correctly, leading to frequent authentication failures. Further debugging and refinement of the keystroke capture and model training processes are required to enhance reliability.
- **Limited Dataset**: The model is currently trained with limited data, and accuracy might improve with more diverse data.

## Risks and Challenges
- **Limited Dataset**: The initial model may not perform well due to insufficient data. This can be mitigated by gathering more user data over time and continuously updating the model.
- **Model Accuracy**: The current version may show low accuracy, particularly with small datasets. Regular updates and retraining of the model will help improve its performance.

## Resources Needed
- **Software**: Python 3.x and the necessary libraries (`pynput`, `numpy`, `scikit-learn`).
- **Data Sets**: Keystroke timing data from user interactions for model training.
- **Hardware**: A computer with adequate specifications for running the machine learning model and capturing keystroke data effectively.

## Conclusion
This project is designed to provide a secure and user-friendly authentication mechanism by leveraging the unique typing patterns of users. The expected outcome is a robust system that enhances security and user experience while addressing the growing need for improved authentication methods in today's digital landscape.

## References
- Relevant literature on keystroke dynamics and authentication methods.
- Documentation for libraries used (`pynput`, `numpy`, `scikit-learn`).

