# Tic-Tac-Toe AI Game

### Created and Developed by Khaled Ziadi

---

## Overview
This project is a web-based **Tic-Tac-Toe** game where the player competes against an AI powered by a machine learning model. The game uses a combination of:
- **Streamlit** for the web interface
- **Keras/TensorFlow** for AI model training
- **Scikit-learn** for preprocessing
- **Minimax algorithm** for optimal gameplay

The application enables players to:
- Play against an intelligent AI opponent
- Save game data for future model improvement
- Retrain the AI model with newly gathered data
- Predict game outcomes based on user moves

---

## Components

### 1. AI Model Training Script (`train_model.py`)

**Functionality:**
- Reads game data from a CSV file (`Tic tac initial results.csv`)
- Cleans and processes the data
- Trains a Keras Sequential model to classify the outcome (win/draw/loss)
- Saves the trained model and label encoder

**Model Architecture:**
- Input Layer: 7 neurons (for `MOVE1` to `MOVE7`)
- Hidden Layers: Dense(128) -> Dense(64) -> Dense(32)
- Output Layer: Dense(3, softmax) - for 3 possible outcomes

**Label Encoding:**
- win = 2
- loss = 1
- draw = 0

**Output:**
- `tic_tac_toe_model.h5`
- `label_encoder.pkl`

---

### 2. Web App (`app.py`)

**Main Features:**
- A 3x3 Tic-Tac-Toe board implemented using Streamlit UI elements
- CSS styling for a visually clean and responsive interface
- Player plays as "X" and the AI as "O"

**Game Mechanics:**
- Game logic uses the Minimax algorithm with alpha-beta pruning to calculate the best AI move
- Player and AI alternate turns until a win/draw occurs

**User Actions:**
- 🔍 **Predict:** Predict game result based on current player moves
- 💾 **Save:** Save finished game and its result
- 📈 **Retrain:** Retrain model with accumulated game data
- 🔄 **Reset:** Reset the game board

**Data File:** `game_data.csv`
- Stores new games in format: `MOVE1` to `MOVE7` and `CLASS`

---

## Folder Structure
```
project/
├── app.py                      # Main Streamlit app
├── train_model.py              # Initial model training script
├── tic_tac_toe_model.h5        # Trained Keras model
├── label_encoder.pkl           # LabelEncoder for class labels
├── game_data.csv               # Collected game data
├── Tic tac initial results.csv # Original training dataset
```

---

## Author
**Khaled Ziadi**
- Creator and Developer of the project
- Designed both the AI model and the Streamlit web application

Feel free to reach out for improvements, feedback, or contributions!

---

## Future Improvements
- Add player difficulty levels
- Allow online multiplayer
- Visualize prediction probabilities
- Export game history

---

## License
This project is open-source. Feel free to use and modify with credit to the author.

