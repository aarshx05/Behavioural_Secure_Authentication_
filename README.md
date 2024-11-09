# Keystroke Authentication System

Welcome to the **Keystroke Authentication System**, a cutting-edge security solution that authenticates users based on their unique typing patterns. By measuring keystroke dynamics, such as the time taken between pressing and releasing keys, this system provides an additional layer of security for user authentication.

This system is designed to work with stronger, more complex passwords, as users who are "cyber-aware" tend to use better, more secure passwords. These stronger passwords often come with associated muscle memory from frequent use, which in turn creates distinct keystroke patterns. The logic behind this model is that it performs more effectively with real, well-structured passwords, leveraging the unique typing rhythm that users develop over time when entering such passwords.

This project uses various machine learning models—SVM, KNN, and Random Forest—combined into an ensemble voting classifier. These models are trained to distinguish between legitimate users and impostors based on their typing behavior.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Registering a New User](#registering-a-new-user)
   - [Retraining the User Model](#retraining-the-user-model)
   - [Verifying a User](#verifying-a-user)
   - [Testing with Existing User Profile](#testing-with-existing-user-profile)
5. [System Architecture](#system-architecture)
6. [Machine Learning Models](#machine-learning-models)
7. [Dynamic Threshold Adjustment](#dynamic-threshold-adjustment)
8. [Data Storage and Management](#data-storage-and-management)
9. [Future Enhancements](#future-enhancements)
10. [Demo](#demo-placeholders)

---

## Introduction

Keystroke dynamics is a biometric technique that identifies users based on how they type. This project captures the user’s typing patterns, processes the data, and classifies the user as authentic or non-authentic based on the keystroke dynamics.

The system includes:
- Keystroke data collection
- User registration
- Retraining the model with new data
- User verification with dynamic threshold adjustment based on previous login attempts

## Features

- **Multi-Model Authentication**: Uses SVM, KNN, and Random Forest in a voting classifier to enhance performance.
- **Dynamic Thresholding**: Automatically adjusts the threshold for user verification based on previous login attempts, making the system more adaptable to user variability.
- **Keystroke Data Collection**: Collects data related to the user's key hold times and flight times between keystrokes.
- **User Model Retraining**: Users can retrain their profile to improve accuracy over time by adding more authentic samples.

## Installation

To get started with the Keystroke Authentication System, follow these steps:

### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `pickle`, `sklearn`, `pynput`

Install the required dependencies by running:
```bash
pip install numpy scikit-learn pynput
```

### Clone the Repository
```bash
git clone https://github.com/aarshx05/Behavioural_Secure_Authentication_.git
cd Behavioural_Secure_Authentication
```

### Running the Program
You can launch the program using Python from your terminal or command prompt:
```bash
python keystroke_auth.py
```

## Usage

The system provides a command-line interface (CLI) for interacting with the authentication system. Upon running the program, you will see the following options:

```plaintext
1. Register a new user
2. Retrain an existing user model
3. Verify an existing user
4. Exit
```

### Registering a New User
To register a new user, follow these steps:
1. Enter the user ID (unique identifier for the user).
2. Enter the user's password.
3. The system will prompt you to type the password multiple times to gather sufficient data.
4. After collecting the keystroke data, the system will generate synthetic non-authentic data and train a machine learning model for that user.

### Retraining the User Model
If a user feels their profile is not performing well, they can retrain their model:
1. Enter the user ID and password.
2. The system will collect new samples and combine them with previously collected data.
3. The model will be retrained with the updated dataset.

### Verifying a User
To verify a user:
1. Enter the user ID and password.
2. The system will compare the entered password's keystroke dynamics with the stored model for that user.
3. Based on a dynamic threshold, the system will classify the user as authentic or non-authentic.

### Testing with Existing User Profile

(**Note:** This system is designed to work with stronger, more complex passwords, as users who are "cyber-aware" tend to use better, more secure passwords but this simple password will be good for you to understand how it works)

For testing purposes, you can log in using a pre-existing user profile with the following credentials:
- **User ID**: `1`
- **Password**: `a123`

Try logging in with this profile to see how the system works in real-time.

---

## System Architecture

The following flowchart demonstrates the overall system architecture:

```plaintext
            +--------------------------------------+
            |   Start: Keystroke Authentication    |
            +--------------------------------------+
                            |
                            v
            +-------------------------------+
            |   User Selects an Option       |
            +-------------------------------+
             |         |          |        |
   Register a new    Retrain    Verify    Exit System
      User          Model     User  
             v         v          v
  +-------------------+    +------------------+
  | Collect Keystroke  |    | Load User Model  |
  | Data               |    | from Disk        |
  +-------------------+    +------------------+
             v         v          v
  +----------------------------------------+
  | Save or Update User Model on Disk      |
  +----------------------------------------+
```

---

## Machine Learning Models

The system uses an ensemble of three machine learning models:

1. **Support Vector Machine (SVM)**: A robust classifier that works well with smaller datasets. The linear kernel is used for initial models, with an option to switch to an RBF kernel for greater flexibility.

2. **K-Nearest Neighbors (KNN)**: This model classifies data points based on their proximity to other points. The number of neighbors and weighting system can be adjusted based on the user’s choice of model harshness.

3. **Random Forest (RF)**: A flexible, high-performing model that works well with larger datasets. It creates multiple decision trees and outputs the class that is the mode of the individual trees.

**Voting Classifier**: These three models are combined using a "soft" voting mechanism, meaning that the prediction probabilities of each model are averaged, and the final decision is made based on the highest probability.

---

## Dynamic Threshold Adjustment

The system introduces a dynamic threshold mechanism for user verification, which adjusts based on the user's previous login attempts. This ensures that users who might type slightly differently on different occasions are not falsely rejected.

- **Initial Threshold**: A static threshold (default: 0.3) is set for new users.
- **Dynamic Adjustment**: Over time, the system records the user’s match probabilities and adjusts the threshold by averaging the last 10 attempts. This makes the system more flexible for genuine users while maintaining security.

---

## Data Storage and Management

All user-related data (models, scalers, keystroke data) is stored locally in the `user_data` folder:
- **model.pkl**: Stores the trained machine learning model for a user.
- **scaler.pkl**: Stores the data scaler for normalizing keystroke data.
- **metadata.pkl**: Stores user metadata, including the password.
- **authentic_data.npy**: Stores authentic keystroke samples collected from the user.
- **synthetic_data.npy**: Stores synthetic (non-authentic) data used for training.

---

## Future Enhancements

The system is functional, but there are several areas for improvement:

1. **Improving Dynamic Threshold**: The current threshold adjustment is based on simple statistical measures (mean and standard deviation). Future iterations could incorporate more sophisticated techniques such as anomaly detection or personalized thresholds.
   
2. **Advanced User Feedback**: Implement real-time feedback on typing patterns and suggestions to help users adjust their typing for better recognition.

3. **Graphical User Interface (GUI)**: Currently, the system uses a CLI. Adding a GUI will make the system more user-friendly. A flowchart of the proposed interface is shown below:

---

## Demo Placeholders

### Screenshot Example

Here's a placeholder for a screenshot of the CLI in action:

![CLI Example](![image](https://github.com/user-attachments/assets/bdfad6e9-abe3-4607-bf9a-39d782dad469))

### Video Demo Placeholder

A video demonstration of the system in action.


https://github.com/user-attachments/assets/b014b3cb-87a7-4c23-8adc-e27134c701ee


---

### Author
**[Aarsh Chaurasia - aarsh.chaurasia.201007@gmail.com]**

If you have any questions or would like to contribute, feel free to reach out.
