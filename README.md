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
   - [Profile Status and Drift](#profile-status-and-drift)
   - [Verifying a User](#verifying-a-user)
   - [Testing with Existing User Profile](#testing-with-existing-user-profile)
5. [System Architecture](#system-architecture)
6. [Captured Attributes](#captured-attributes)
7. [Machine Learning Models](#machine-learning-models)
8. [Dynamic Threshold Adjustment](#dynamic-threshold-adjustment)
9. [Adaptive Retraining](#adaptive-retraining)
10. [Data Storage and Management](#data-storage-and-management)
11. [Privacy Note](#privacy-note)
12. [Future Enhancements](#future-enhancements)
13. [Demo](#demo)
14. [Capturing the Demo Assets](#capturing-the-demo-assets)

---

## Introduction

Keystroke dynamics is a biometric technique that identifies users based on how they type. This project captures the user’s typing patterns, processes the data, and classifies the user as authentic or non-authentic based on the keystroke dynamics.

The system includes:
- Keystroke data collection, including press *and* release timings
- Contextual capture of the network, device and clock each sample came from
- User registration
- Adaptive retraining that follows the user's typing as it changes over time
- User verification combining a biometric score with a contextual risk score

## Features

- **Multi-Model Authentication**: Uses SVM, KNN, and Random Forest in a voting classifier to enhance performance.
- **Rich Keystroke Capture**: Records dwell time, down-down, up-down (true flight) and up-up latencies for every key, plus statistical aggregates describing rhythm consistency, typing speed and key overlap.
- **Contextual Capture**: Every sample is stamped with the network, device and clock context it was typed in (IP, subnet, hostname, MAC, OS, timezone, hour of day, keyboard layout).
- **Risk-Based Verification**: Contextual attributes form a second decision layer alongside the biometric score, so an unfamiliar device or network raises the bar rather than passing unnoticed.
- **Dynamic Thresholding**: Adjusts the acceptance threshold from the user's own score history, sitting at the lower edge of their genuine distribution.
- **Adaptive Retraining**: The profile follows the user's typing as it changes — a sliding window, recency weighting, automatic adoption of confident logins, and drift detection.

## Installation

To get started with the Keystroke Authentication System, follow these steps:

### Prerequisites
- Python 3.8 or newer
- `numpy`, `scikit-learn`, `pynput` (`pickle` and `os` are part of the standard library and need no installation)

### Clone the Repository
```bash
git clone https://github.com/aarshx05/Behavioural_Secure_Authentication_.git
cd Behavioural_Secure_Authentication_
```

> Note the trailing underscore in the directory name — it is part of the repository name.

### Install dependencies

A virtual environment is recommended so the packages do not land in your system Python:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Running the Program
```bash
python keystroke_auth.py
```

You should see the menu:

```plaintext
Welcome to the Keystroke Authentication System!

Select an option:
1. Register a new user
2. Retrain an existing user model
3. Verify an existing user
4. Profile status and drift report
5. Exit
```

### Platform notes

`pynput` reads keystrokes globally, which some operating systems restrict:

| Platform | Requirement |
|---|---|
| Windows | Works out of the box. If launched from an elevated terminal, keystrokes from non-elevated windows may not register |
| macOS | Grant your terminal **Input Monitoring** and **Accessibility** permission under System Settings → Privacy & Security, then restart the terminal |
| Linux | Requires an X11 session. Under Wayland, `pynput` cannot capture global key events — run in an Xorg session, or use `XDG_SESSION_TYPE=x11` |
| SSH / headless | Not supported — there is no keyboard to read |

Type the password into the **same terminal window** that is running the program. The password is echoed as you type, so avoid capturing your real password in screen recordings — use a throwaway password for any demo.

## Usage

The system provides a command-line interface (CLI) for interacting with the authentication system. Upon running the program, you will see the following options:

```plaintext
1. Register a new user
2. Retrain an existing user model
3. Verify an existing user
4. Profile status and drift report
5. Exit
```

### Registering a New User
To register a new user, follow these steps:
1. Enter the user ID (unique identifier for the user).
2. Enter the user's password.
3. The system will prompt you to type the password multiple times to gather sufficient data.
4. After collecting the keystroke data, the system will generate synthetic non-authentic data and train a machine learning model for that user.

### Retraining the User Model
Typing changes. A password typed for the first time this week is slow and deliberate; the same password six months later is muscle memory. A profile frozen at enrollment drifts away from its owner, and the false rejection rate climbs until the user gives up.

Retraining keeps the profile tracking the user:
1. Enter the user ID and password.
2. The system reports how far your typing has drifted since the profile was built.
3. New samples are collected and merged into a **sliding window** of your most recent typing — samples beyond the window are dropped rather than kept forever.
4. Within that window, recent samples are **weighted more heavily** than older ones (a sample one half-life old counts half as much).
5. The model is refit and the profile is re-anchored.

Most of this happens without being asked. See [Adaptive Retraining](#adaptive-retraining).

### Profile Status and Drift
Option 4 reports the state of a profile: window size and age span, where samples came from, the current threshold, how far the template has moved from its anchor, devices seen, and whether your typing has drifted far enough to justify a retrain.

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

> This profile predates the extended feature set, so it loads as **schema v1** and verifies against the original feature layout. It will not show contextual risk, adaptive learning or drift reporting — those need a v2 profile. Register a new user (or retrain this one, which offers to upgrade it) to exercise the newer features, and for any demo captures.

---

## System Architecture

Verification runs as two independent layers. The biometric layer decides **how**
the password was typed; the contextual layer decides whether **where and when**
looks normal. Neither can override the other — context raises or lowers the bar
the biometric score has to clear.

```plaintext
                        Type password
                              |
              +---------------+---------------+
              v                               v
    +--------------------+          +---------------------+
    |  Keystroke timings |          | Context snapshot    |
    |  hold / DD / UD/UU |          | IP, device, clock   |
    +--------------------+          +---------------------+
              |                               |
              v                               v
    +--------------------+          +---------------------+
    | Voting classifier  |          | Risk assessment     |
    | -> match score     |          | -> risk score       |
    +--------------------+          +---------------------+
              |                               |
              +---------------+---------------+
                              v
                  +-------------------------+
                  | required = threshold    |
                  |          + risk penalty |
                  +-------------------------+
                              v
                     score >= required ?
                       /            \
                    yes              no
                     |                \
                     v                 v
        +-------------------+       Reject
        | Authenticate      |
        +-------------------+
                     |
         confident AND low risk AND
         template near its anchor ?
                     |
                    yes
                     v
        +-------------------------------+
        | Adopt sample -> refit profile |
        +-------------------------------+
```

---

## Captured Attributes

**Timing attributes** (these form the machine learning feature vector, length `4n + 12` for an `n`-character password):

| Attribute | Meaning |
|---|---|
| `hold` | Dwell time — how long each key stays down |
| `dd` | Down-down latency between consecutive keys |
| `ud` | True flight time: previous key released → next pressed. Negative values mean the keys overlapped, which is the rollover typing fast touch-typists produce and hunt-and-peck typists never do |
| `uu` | Up-up latency between consecutive releases |
| aggregates | mean/std/min/max dwell, mean/std of each latency, rhythm consistency (coefficient of variation), typing speed, key overlap ratio |

Only `hold` and `dd` existed previously; release timestamps were captured and discarded.

**Contextual attributes** (recorded per sample, scored separately — *not* in the feature vector):

| Attribute | Attribute |
|---|---|
| Local IP and `/24` subnet | Hostname |
| Public IP *(opt-in)* | MAC address |
| OS name, release, version | Machine and processor |
| Timezone and UTC offset | Hour of day and weekday |
| OS username | Keyboard layout |

### Why context is not in the feature vector

It is tempting to append the IP to the feature array. That fails for three reasons:

- Context is **constant during enrollment**, so the classifier would learn "local IP is 192.168.1.7 ⇒ authentic". That is trivially spoofable, and it collapses the moment the user switches to Wi-Fi, a VPN, or a new DHCP lease.
- Context is **categorical**. Standard-scaling an encoded IP is meaningless, and the synthetic negative generator perturbs *timings* — adding Gaussian noise to an encoded hostname produces nothing an impostor would ever look like.
- A user's network legitimately changes **far more often** than their typing rhythm, so context belongs on a slower, separate axis.

So context is scored on its own axis and used to adjust the bar the biometric score must clear. An unfamiliar device does not prove an impostor, so by default it demands stronger biometric evidence rather than hard-blocking; set `RISK_BLOCK_ENABLED` to refuse high-risk contexts outright.

---

## Machine Learning Models

The system uses an ensemble of three machine learning models:

1. **Support Vector Machine (SVM)**: An RBF-kernel SVM, wrapped in `CalibratedClassifierCV` to produce probabilities.

2. **K-Nearest Neighbors (KNN)**: Classifies data points based on their proximity to other points, with distance weighting. The number of neighbors varies with the chosen preset.

3. **Random Forest (RF)**: A flexible, high-performing model that works well with larger datasets. It creates multiple decision trees and outputs the class that is the mode of the individual trees.

**Voting Classifier**: These three models are combined using a "soft" voting mechanism, meaning that the prediction probabilities of each model are averaged, and the final decision is made based on the highest probability.

### Why both presets use an RBF kernel

The two presets ("Harsh" and "Easy") differ in strictness — regularisation, neighbour count, tree depth — not in kernel. Earlier versions gave "Harsh" a **linear** kernel, which cannot work here: the synthetic negatives surround the authentic cluster in every direction, and no hyperplane separates a blob from a shell enclosing it. Measured on simulated typists, the linear kernel scored genuine samples at ~0.53 against RBF's ~0.73, dragging the ensemble down from ~0.82 to ~0.76.

### Synthetic negatives

Negatives are built in raw timing space and the feature vector re-derived from them, so aggregate features always stay consistent with the per-key values they summarise. Six impostor archetypes are generated:

| Archetype | Models |
|---|---|
| `jitter` | someone typing the password almost right — the hard negative |
| `slow` / `fast` | typists at a different tempo |
| `flat` | uniform intervals: scripted or replayed input |
| `shuffle` | the user's own intervals in the wrong order |
| `random` | broad draws across a plausible human range |

`shuffle` matters most: it has the same overall speed as the genuine user but the wrong rhythm, which forces the model to learn rhythm rather than tempo. Removing it lets a same-speed impostor's score jump from 0.002 to 0.352.

Negatives are regenerated from scratch on every retrain at a fixed ratio to the positive count, rather than accumulated — the earlier approach stacked new negatives onto stored ones each time, so 10 samples became 2,500 negatives after two retrains.

---

## Dynamic Threshold Adjustment

The system adjusts the acceptance threshold from the user's own score history, so users who type slightly differently on different occasions are not falsely rejected.

- **Initial Threshold**: A static threshold (default: 0.4) is used until 5 scores have been recorded.
- **Dynamic Adjustment**: The threshold is placed at the **lower edge** of the user's genuine score distribution — `mean − k × std` of recent successful attempts — and clamped to a sane range. `k` is widened while the history is short, so a handful of unusually consistent logins cannot set a bar the user then struggles to clear.

> **Note on an earlier bug.** Previous versions computed `mean + std`. Because only scores *above* the threshold are ever recorded, that mean is high by construction, and adding a standard deviation pushed the bar above almost every future genuine attempt. The threshold ratcheted upward on each success until the real user was locked out. The bar belongs below the genuine mean, not above it.

Recorded scores are cleared on a **manual** retrain, where the user has deliberately supplied new typing. They are *not* cleared on an automatic refit: that fits nearly the same window plus a few samples, so previous scores still describe the model closely — and clearing on every fifth login would stop the history ever reaching the length the dynamic threshold needs, pinning the bar at the static value permanently.

---

## Adaptive Retraining

Retraining is not only a manual action. Three mechanisms keep the profile tracking the user:

1. **Sliding window with recency weighting.** Only the newest samples are kept, and recent ones dominate the fit. Recency is expressed by *replicating* recent rows rather than via `sample_weight`, because `VotingClassifier` only forwards sample weights when every estimator accepts them — and `KNeighborsClassifier` does not.

2. **Auto-adoption.** A verification that is confidently genuine and comes from an ordinary context is folded back into the profile, so everyday logins become the training data. After a few adoptions the model refits automatically.

3. **Drift detection.** The oldest and newest samples in the window are compared, measured in standard deviations of the profile's own spread — so a naturally variable typist is not flagged for variation that is normal for them.

4. **Rejected-attempt analysis.** Drift measured over stored samples can only see logins that were *accepted*. Once a user's typing moves far enough to start being rejected, nothing new is adopted and that measure goes blind — it will keep reporting "stable" while the user is locked out. Rejected attempts that supplied the correct password are therefore analysed too.

### Drift or attack?

Repeated rejections with a correct password have two very different causes, and the system must not confuse them:

| Evidence | Diagnosis | Told to the user |
|---|---|---|
| Attempts cluster tightly around one new rhythm | The user's own typing has changed | "Your typing appears to have changed — retrain to catch up" |
| Attempts are scattered and inconsistent | Several different people with the password | "That looks like different people — consider changing the password" |

Cohesion — how tightly the rejected attempts agree with each other, relative to the profile's own spread — is what separates the two. Rejected attempts are kept **for reporting only and are never fed to the model**; retraining still requires the password and freshly captured typing.

### Guarding against template poisoning

Auto-adoption is a write path into the model, so it has three independent guards:

- a sample must clear an **absolute confidence floor** *and* beat the bar it was judged against by a margin — merely passing verification is not enough to become training data;
- the **context must look ordinary**, so a login from an unrecognised device never teaches the model anything;
- the window centroid must stay within a bounded distance of the **anchor** recorded at the last password-verified enroll/retrain.

The third guard exists because the per-sample checks alone cannot stop a *walk attack*: an attacker who can pass verification nudging the template toward their own typing a little at a time. Past the bound, auto-adoption stops and an explicit retrain is required to re-anchor. Authentication still works while locked out — only learning pauses.

A share-of-profile cap cannot do this job: the window slides, so enrollment samples age out and the profile legitimately becomes mostly auto-adopted, which would disable adaptation permanently.

---

## Data Storage and Management

All user-related data is stored locally in the `user_data/<user_id>/` folder:

| File | Contents |
|---|---|
| `metadata.pkl` | Schema version, password, feature spec, counters, template anchor |
| `model.pkl` | Trained voting classifier |
| `scaler.pkl` | Fitted scaler for normalizing keystroke data |
| `authentic_data.npy` | Authentic keystroke samples, oldest row first |
| `synthetic_data.npy` | Most recently generated negatives |
| `sample_meta.pkl` | Per-sample timestamp, source (`enroll`/`retrain`/`auto`) and context |
| `context_history.pkl` | Network/device/clock contexts seen for this user |
| `match_probabilities.pkl` | Recent genuine match scores |
| `recent_failures.pkl` | Rejected attempts that had the correct password (reporting only) |
| `history.pkl` | Enrollment / retrain / drift event log |

### Backwards compatibility

Profiles written by earlier versions carry only `{'password': ...}` in `metadata.pkl`. They load as **schema v1** and keep using the original `2n`-length feature layout, so they verify against the model they were actually fit on. The extended feature vector is a strict superset — its first `2n` entries are byte-identical to v1 — so nothing about the old layout had to change. Retraining a v1 profile offers to upgrade it; because v1 never recorded key-release times, the upgrade rebuilds the window from freshly captured typing.

---

## Privacy Note

Contextual capture is **local by default**. Hostname, LAN IP, MAC, OS, timezone and keyboard layout are read from the machine; determining the LAN IP opens a UDP socket against an RFC 5737 documentation address, which performs a routing-table lookup and transmits nothing.

Public IP lookup is the one exception: it contacts a third-party service, so it is **disabled by default**. Enable it with `ENABLE_PUBLIC_IP_LOOKUP` in `bauth/config.py`.

---

## Future Enhancements

The system is functional, but there are several areas for improvement:

1. **Hash the stored password**: `metadata.pkl` still holds the password in plaintext, which is a significant weakness in an authentication project. It should be salted and hashed.

2. **Graphical User Interface (GUI)**: Currently, the system uses a CLI. Adding a GUI will make the system more user-friendly.

3. **Advanced User Feedback**: Implement real-time feedback on typing patterns and suggestions to help users adjust their typing for better recognition.

4. **Real impostor data**: Negatives are currently synthetic. Evaluating against genuine impostor attempts — other people typing the same password — would give trustworthy FAR/FRR figures.

5. **Feature selection**: A long password produces a high-dimensional vector from few samples. Dimensionality reduction or per-feature stability weighting may help.

6. **Free-text keystroke dynamics**: Continuous authentication during a session, rather than only at the login prompt.

---

## Demo

> **Status:** the captures below are placeholders. See [Capturing the demo assets](#capturing-the-demo-assets) for exactly what to record and when.

### 1. Registration

Enrolling a new user. Ten samples are collected, then the model and the first context snapshot are stored.

<!-- REPLACE: drag 01-registration.png onto the PR/issue comment box and paste the generated URL here -->
![Registration](https://placehold.co/900x420?text=01+registration.png)

### 2. Successful verification

The genuine user from their own machine. Note the three reported values: biometric score, required score, and contextual risk.

<!-- REPLACE: 02-verify-success.png -->
![Successful verification](https://placehold.co/900x300?text=02+verify-success.png)

### 3. Impostor rejected

Someone else typing the **correct password**. The password check passes; the biometric layer is what stops them.

<!-- REPLACE: 03-impostor-rejected.png -->
![Impostor rejected](https://placehold.co/900x300?text=03+impostor-rejected.png)

### 4. Unrecognised device or network

Correct password *and* correct typing rhythm, but from a network the profile has never seen. The required score rises above the base threshold.

<!-- REPLACE: 04-risk-elevated.png -->
![Elevated contextual risk](https://placehold.co/900x300?text=04+risk-elevated.png)

### 5. Adaptive learning

A confident, low-risk login is folded back into the profile; after a few of these the model refits itself.

<!-- REPLACE: 05-adaptive-learning.png -->
![Adaptive learning](https://placehold.co/900x320?text=05+adaptive-learning.png)

### 6. Profile status and drift report

Option 4: window size and age span, sample sources, current threshold, template drift, devices seen, and the drift verdict.

<!-- REPLACE: 06-status-report.png -->
![Profile status](https://placehold.co/900x480?text=06+status-report.png)

### 7. Drift detected from rejected attempts

After the user's typing has changed, repeated rejections are diagnosed as drift rather than as an attack.

<!-- REPLACE: 07-drift-detected.png -->
![Drift detected](https://placehold.co/900x340?text=07+drift-detected.png)

### 8. Recovery after retraining

Option 2 rebuilds the profile around current typing, and the previously-rejected user verifies cleanly again.

<!-- REPLACE: 08-retrain-recovery.png -->
![Retrain recovery](https://placehold.co/900x380?text=08+retrain-recovery.png)

### Full walkthrough

<!-- REPLACE: drag demo-walkthrough.mp4 onto the comment box and paste the generated URL on its own line -->
_Video placeholder — a single recording covering register → verify → impostor → status → drift → retrain._

---

## Capturing the demo assets

### Before you start

```bash
# Use a throwaway password - it is echoed on screen and will be visible in captures
# Suggested: Demo!Pass#2026

# Start from a clean slate so the walkthrough is reproducible
# (this deletes local profiles - skip if you have data you want to keep)
rm -rf user_data/demo        # macOS / Linux
Remove-Item -Recurse -Force user_data\demo   # Windows PowerShell
```

Set your terminal to roughly **100×30** characters with a readable font size before recording. Keep the whole prompt-and-output block visible in one frame.

### What to capture, and exactly when

| # | Asset | Menu option | Capture at the moment... |
|---|---|---|---|
| 1 | `01-registration.png` | 1 | The `registered successfully` block appears, showing features per sample, sample count, negatives, and the context line |
| 2 | `02-verify-success.png` | 3 | `User verified successfully!` appears — include the three score lines above it |
| 3 | `03-impostor-rejected.png` | 3 | `User verification failed` appears after **someone else** typed the password |
| 4 | `04-risk-elevated.png` | 3 | `Context risk : elevated/high` appears with a `Required score` above the base |
| 5 | `05-adaptive-learning.png` | 3 | `This sample was added to your profile` and ideally `Profile automatically retrained` appear |
| 6 | `06-status-report.png` | 4 | The full profile block is on screen, including `template drift` and `drift` lines |
| 7 | `07-drift-detected.png` | 3 | `Your typing appears to have changed:` appears after ~3 rejected attempts |
| 8 | `08-retrain-recovery.png` | 2 then 3 | The retrain summary, then a successful verification immediately after |

### Step-by-step

**Registration — asset 1**

1. `python keystroke_auth.py` → `1`
2. User ID `demo`, password `Demo!Pass#2026`, model choice `1`
3. Type the password 10 times, pressing Enter each time. **Type naturally** — a rhythm you can reproduce later
4. Screenshot the success block

**Verification — asset 2**

5. Menu → `3`, user `demo`, same password, type it once
6. Screenshot the score block plus `User verified successfully!`

**Impostor — asset 3**

7. Ask someone else to sit at the keyboard. Menu → `3`, enter user `demo` and the password yourself, then let **them** type the password at the capture prompt
8. Screenshot the failure

> No second person available? Type it deliberately differently — one finger, hunting for each key, with long pauses. That is a genuinely different rhythm and will score low.

**Elevated risk — asset 4**

9. Change your network so the subnet differs — switching Wi-Fi to a **phone hotspot** is the easiest way. Alternatively run the project on a second machine, copying the `user_data/demo` folder across
10. Menu → `3` and verify normally
11. Screenshot the `Context risk : ... new network (...)` line with the raised `Required score`

> The risk layer compares against contexts already seen. Registering and verifying on the same machine and network always scores `low` — you must actually change something for this asset.

**Adaptive learning — asset 5**

12. Return to your normal network. Run option `3` about **five times**, typing consistently
13. Screenshot a run showing `This sample was added to your profile`, ideally one that also prints `Profile automatically retrained on your recent typing`

**Status report — asset 6**

14. Menu → `4`, enter `demo`
15. Screenshot the whole block

**Drift — asset 7**

Natural drift takes months. To stage it honestly, change your typing the way months of practice would:

16. Run option `3` and type the password **noticeably faster** than you enrolled it — the speed muscle memory eventually gives you. Repeat until it fails
17. After roughly the third consecutive failure, screenshot the `Your typing appears to have changed:` message with its measured `sd` shift and percentage

> Typing *inconsistently* instead — different rhythm each attempt — produces the `Warning: ... looks like different people` message. That is asset 7's counterpart and worth capturing too if you want to show both branches.

**Recovery — asset 8**

18. Menu → `2`, user `demo`, password, model choice `1`, then type the password 5 times **at the new faster speed**
19. Screenshot the retrain summary
20. Menu → `3` and verify at the faster speed — it should now pass. Screenshot

**Video walkthrough**

Record steps 1–20 as one continuous take, roughly 3–5 minutes. Windows: `Win + Alt + R` (Xbox Game Bar) or OBS. macOS: `Cmd + Shift + 5`. Linux: OBS or `SimpleScreenRecorder`. Pause a beat on each result block so viewers can read the scores.

### Adding them to the README

GitHub does not serve images from a repo path in the way most people expect, and committing binaries bloats the history. The simplest reliable route:

1. Open any issue or PR comment box on the repository
2. Drag each file in — GitHub uploads it and inserts a `https://github.com/user-attachments/assets/...` URL
3. Copy that URL into the matching `REPLACE` slot above, then **close the comment without submitting it**
4. For the video, paste the URL on its own line — GitHub renders it as an inline player

Keep screenshots under ~1 MB and the video under GitHub's 10 MB attachment limit; trim or lower the resolution if needed.

---

### Author
**[Aarsh Chaurasia - aarsh.chaurasia.201007@gmail.com]**

If you have any questions or would like to contribute, feel free to reach out.
