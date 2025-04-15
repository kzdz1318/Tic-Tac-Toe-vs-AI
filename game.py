import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import joblib
import os
import random

# Load the model and LabelEncoder
model = load_model("tic_tac_toe_model.h5")
le = joblib.load("label_encoder.pkl")

# File to store new game data
DATA_FILE = "game_data.csv"

# ---------- Functions ----------

def predict_game(moves):
    moves = moves + [-1] * (7 - len(moves))
    moves = np.array([moves])
    prediction = model.predict(moves, verbose=0)
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class

def retrain_model():
    global model
    data = pd.read_csv(DATA_FILE)
    # Keep only MOVE1 to MOVE7 and CLASS
    valid_columns = [f"MOVE{i+1}" for i in range(7)] + ["CLASS"]
    data = data[[col for col in data.columns if col in valid_columns]]
    data = data.fillna(-1).replace('?', -1)
    for col in valid_columns[:-1]:
        data[col] = data[col].astype(int)
    data['CLASS'] = le.transform(data['CLASS'])
    X = data[valid_columns[:-1]].values
    y = data['CLASS'].values
    model = Sequential([
        Dense(128, activation='relu', input_shape=(7,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    model.save("tic_tac_toe_model.h5")
    st.success("Model updated successfully!")

def save_game_data(moves, result):
    moves = moves[:7] + [-1] * (7 - len(moves))
    new_data = pd.DataFrame([moves + [result]], columns=[f"MOVE{i+1}" for i in range(7)] + ["CLASS"])
    if os.path.exists(DATA_FILE):
        new_data.to_csv(DATA_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(DATA_FILE, index=False)

def check_winner(board):
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for a, b, c in win_conditions:
        if board[a] == board[b] == board[c] and board[a] != " ":
            return board[a]
    return "draw" if " " not in board else None

def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)
    if winner == "O": return 1
    if winner == "X": return -1
    if winner == "draw": return 0

    best_score = -float("inf") if is_maximizing else float("inf")
    for i in range(9):
        if board[i] == " ":
            board[i] = "O" if is_maximizing else "X"
            score = minimax(board, depth + 1, not is_maximizing, alpha, beta)
            board[i] = " "
            if is_maximizing:
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(score, best_score)
                beta = min(beta, best_score)
            if beta <= alpha:
                break
    return best_score

def best_move(board):
    best_score = -float("inf")
    move = -1
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(board, 0, False, -float("inf"), float("inf"))
            board[i] = " "
            if score > best_score:
                best_score = score
                move = i
    return move

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Tic-Tac-Toe AI", layout="centered")

# CSS styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%; height: 60px; font-size: 20px; font-weight: bold;
        border: 2px solid #fff; border-radius: 10px; background-color: #4682b4;
        color: white; box-shadow: 3px 3px 5px rgba(0,0,0,0.3);
        transition: background-color 0.3s, transform 0.1s;
        margin: 5px auto;
    }
    .stButton>button:hover { background-color: #5a9bd4; transform: scale(1.05); }
    .stButton>button:disabled { background-color: #87ceeb; color: white; box-shadow: none; }
    .x { color: #ff4d4d; font-weight: bold; }
    .o { color: #ffffff; font-weight: bold; }
    .center-text { text-align: center; color: white; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='center-text'>🎮 Tic-Tac-Toe vs AI</h1>", unsafe_allow_html=True)

# ---------- Game State ----------

if "board" not in st.session_state:
    st.session_state.board = [" "] * 9
if "moves" not in st.session_state:
    st.session_state.moves = []
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None
if "result" not in st.session_state:
    st.session_state.result = None

# ---------- Display Board ----------

st.markdown("""
<div class='center-text'>You <span class='x'>X</span> - Model <span class='o'>O</span></div>
""", unsafe_allow_html=True)

for i in range(3):
    cols = st.columns(3)
    for j in range(3):
        idx = i * 3 + j
        with cols[j]:
            label = st.session_state.board[idx]
            if label == "X": label = f"<span class='x'>{label}</span>"
            elif label == "O": label = f"<span class='o'>{label}</span>"
            if st.session_state.board[idx] == " " and not st.session_state.game_over:
                if st.button(" ", key=f"cell_{idx}"):
                    st.session_state.board[idx] = "X"
                    st.session_state.moves.append(idx)
                    winner = check_winner(st.session_state.board)
                    if winner:
                        st.session_state.game_over = True
                        st.session_state.winner = winner
                        st.session_state.result = "loss" if winner == "X" else "win" if winner == "O" else "draw"
                    else:
                        move = best_move(st.session_state.board)
                        if move != -1:
                            st.session_state.board[move] = "O"
                            st.session_state.moves.append(move)
                            winner = check_winner(st.session_state.board)
                            if winner:
                                st.session_state.game_over = True
                                st.session_state.winner = winner
                                st.session_state.result = "loss" if winner == "X" else "win" if winner == "O" else "draw"
                    st.rerun()
            else:
                st.markdown(f"<div style='text-align:center;font-size:30px'>{label}</div>", unsafe_allow_html=True)

# ---------- Display Result ----------

if st.session_state.game_over:
    msg = {
        "draw": ("#ffd700", "Draw 🤝"),
        "X": ("#00ff00", "You Win 🎉"),
        "O": ("#ff4d4d", "Model Wins 💻")
    }
    color, text = msg.get(st.session_state.winner, ("white", ""))
    st.markdown(f"<div style='text-align: center; color: {color}; font-size: 24px;'>{text}</div>", unsafe_allow_html=True)

# ---------- Control Buttons ----------

st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)
control_cols = st.columns([1, 1, 1, 1])

with control_cols[0]:
    if st.button("🔍 Predict"):
        if not st.session_state.moves:
            st.error("Start playing first!")
        else:
            result = predict_game(st.session_state.moves)
            st.success(f"Predicted result: {result}")

with control_cols[1]:
    if st.button("💾 Save"):
        if not st.session_state.moves:
            st.error("Start the game first!")
        elif not st.session_state.game_over:
            st.error("You must finish the game to save!")
        else:
            save_game_data(st.session_state.moves, st.session_state.result)
            st.success("Game data saved ✅")

with control_cols[2]:
    if st.button("📈 Retrain"):
        if os.path.exists(DATA_FILE):
            retrain_model()
        else:
            st.error("No data available to retrain the model!")

with control_cols[3]:
    if st.button("🔄 Reset"):
        st.session_state.board = [" "] * 9
        st.session_state.moves = []
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.result = None
        st.rerun()
