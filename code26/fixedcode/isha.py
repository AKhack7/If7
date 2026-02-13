import pyttsx3
import speech_recognition as sr
import datetime
import os
import webbrowser
import pyautogui
import time
import psutil
import subprocess
import re
import threading
import socket
import logging
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import ctypes
import requests
import cv2
import numpy as np
import json
import pickle
from collections import defaultdict, Counter
import math

# === Optional Offline STT (Vosk) ===
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False

# === MediaPipe for Hand Gestures ===
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

# === Personal LLM ===
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PERSONAL_LLM_AVAILABLE = True
except Exception:
    PERSONAL_LLM_AVAILABLE = False

# Initialize logging
logging.basicConfig(filename="isha_assistant.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================
# ADVANCED HAND GESTURE CONTROLLER
# ============================================

class AdvancedGestureController:
    """Ultra-fast hand gesture recognition with 15+ gestures."""
    
    GESTURE_ACTIONS = {
        "PALM": "WAITING",
        "FIST": "CLOSE_FILE",
        "POINT": "OPEN_FILE",
        "PEACE": "SCREENSHOT",
        "THUMBS_UP": "VOLUME_UP",
        "THUMBS_DOWN": "VOLUME_DOWN",
        "OK": "SELFIE",
        "THREE": "OPEN_BROWSER",
        "FOUR": "OPEN_CALCULATOR",
        "FIVE": "OPEN_NOTEPAD",
        "PINCH": "SCROLL",
        "CALL": "PLAY_MUSIC",
        "ROCK": "WEATHER",
        "SPIDERMAN": "JOKE",
        "CROSS": "CLOSE_APP"
    }
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        self.frame_skip = 2
        self.frame_count = 0
        self.gesture_history = []
        self.history_length = 5
        self.last_action_time = 0
        self.action_cooldown = 0.3
        self.scroll_mode = False
        self.last_scroll_y = 0
        self.scroll_speed = 50
        self.mouse_control = False
        self.screen_w, self.screen_h = pyautogui.size()
        self.show_preview = False
        self.active = False
        
    def toggle(self):
        self.active = not self.active
        return self.active
    
    def toggle_preview(self):
        self.show_preview = not self.show_preview
        return self.show_preview
    
    def recognize_gesture(self, hand_landmarks, hand_label):
        """Ultra-fast gesture recognition."""
        lm = hand_landmarks.landmark
        
        tips_ids = [4, 8, 12, 16, 20]
        pip_ids = [2, 6, 10, 14, 18]
        
        fingers = []
        
        if hand_label == "Right":
            fingers.append(1 if lm[4].x > lm[3].x else 0)
        else:
            fingers.append(1 if lm[4].x < lm[3].x else 0)
        
        for tip, pip in zip(tips_ids[1:], pip_ids[1:]):
            fingers.append(1 if lm[tip].y < lm[pip].y else 0)
        
        finger_count = sum(fingers)
        
        pinch_dist = math.sqrt(
            (lm[4].x - lm[8].x)**2 + 
            (lm[4].y - lm[8].y)**2
        )
        
        palm_center_x = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
        palm_center_y = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4
        
        tip_distances = []
        for tip_id in tips_ids:
            dist = math.sqrt(
                (lm[tip_id].x - palm_center_x)**2 + 
                (lm[tip_id].y - palm_center_y)**2
            )
            tip_distances.append(dist)
        
        avg_tip_dist = sum(tip_distances) / len(tip_distances)
        
        # FIST
        if finger_count == 0 and avg_tip_dist < 0.15:
            return "FIST"
        
        # PALM
        if finger_count == 5 and avg_tip_dist > 0.25:
            return "PALM"
        
        # POINT
        if finger_count == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
            return "POINT"
        
        # PEACE
        if finger_count == 2 and fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
            if abs(lm[8].x - lm[12].x) > 0.05:
                return "PEACE"
        
        # THUMBS UP
        if fingers[0] == 1 and sum(fingers[1:]) == 0:
            if lm[4].y < lm[3].y < lm[2].y:
                return "THUMBS_UP"
        
        # THUMBS DOWN
        if fingers[0] == 1 and sum(fingers[1:]) == 0:
            if lm[4].y > lm[3].y > lm[2].y:
                return "THUMBS_DOWN"
        
        # OK
        if pinch_dist < 0.05 and sum(fingers[2:]) == 0:
            return "OK"
        
        # THREE
        if finger_count == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
            return "THREE"
        
        # FOUR
        if finger_count == 4 and sum(fingers[1:5]) == 4:
            return "FOUR"
        
        # FIVE
        if finger_count == 5:
            return "FIVE"
        
        # PINCH
        if pinch_dist < 0.03:
            return "PINCH"
        
        # CALL
        if fingers[0] == 1 and fingers[4] == 1 and sum(fingers[1:4]) == 0:
            return "CALL"
        
        # ROCK
        if fingers[1] == 1 and fingers[4] == 1 and sum(fingers[2:4]) == 0 and fingers[0] == 0:
            return "ROCK"
        
        # SPIDERMAN
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 1 and sum(fingers[3:5]) == 0:
            return "SPIDERMAN"
        
        # CROSS
        if fingers[1] == 1 and fingers[2] == 1:
            if abs(lm[8].x - lm[12].x) < 0.02:
                return "CROSS"
        
        return "UNKNOWN"
    
    def execute_action(self, gesture, frame=None, hand_landmarks=None):
        """Execute the action associated with the gesture."""
        
        current_time = time.time()
        
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        action = self.GESTURE_ACTIONS.get(gesture)
        
        if action == "SCREENSHOT":
            pyautogui.hotkey('win', 'prtsc') if os.name == 'nt' else pyautogui.screenshot()
            self.last_action_time = current_time
            return "Screenshot taken"
        
        elif action == "SELFIE" and frame is not None:
            folder = os.path.join(os.getcwd(), "isha_captures")
            os.makedirs(folder, exist_ok=True)
            filename = f"selfie_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(folder, filename)
            cv2.imwrite(path, frame)
            self.last_action_time = current_time
            return "Selfie captured"
        
        elif action == "VOLUME_UP":
            pyautogui.press('volumeup', presses=2)
            self.last_action_time = current_time
            return "Volume up"
        
        elif action == "VOLUME_DOWN":
            pyautogui.press('volumedown', presses=2)
            self.last_action_time = current_time
            return "Volume down"
        
        elif action == "OPEN_BROWSER":
            webbrowser.open('https://www.google.com')
            self.last_action_time = current_time
            return "Opening browser"
        
        elif action == "OPEN_CALCULATOR":
            os.system('calc' if os.name == 'nt' else 'gnome-calculator')
            self.last_action_time = current_time
            return "Opening calculator"
        
        elif action == "OPEN_NOTEPAD":
            os.system('notepad' if os.name == 'nt' else 'gedit')
            self.last_action_time = current_time
            return "Opening notepad"
        
        elif action == "PLAY_MUSIC":
            os.system('start wmplayer' if os.name == 'nt' else 'rhythmbox')
            self.last_action_time = current_time
            return "Playing music"
        
        elif action == "WEATHER":
            webbrowser.open('https://weather.com')
            self.last_action_time = current_time
            return "Checking weather"
        
        elif action == "JOKE":
            self.last_action_time = current_time
            return "Why don't scientists trust atoms? Because they make up everything!"
        
        elif action == "CLOSE_APP":
            pyautogui.hotkey('alt', 'f4')
            self.last_action_time = current_time
            return "Closing app"
        
        elif action == "OPEN_FILE" and hand_landmarks:
            x = int(hand_landmarks.landmark[8].x * self.screen_w)
            y = int(hand_landmarks.landmark[8].y * self.screen_h)
            pyautogui.moveTo(x, y)
            pyautogui.doubleClick()
            self.last_action_time = current_time
            return "Opening file"
        
        elif action == "SCROLL" and hand_landmarks:
            y = hand_landmarks.landmark[8].y
            if self.last_scroll_y != 0:
                delta = y - self.last_scroll_y
                if abs(delta) > 0.02:
                    pyautogui.scroll(-int(delta * self.scroll_speed))
            self.last_scroll_y = y
            return "Scrolling"
        
        return None
    
    def process_frame(self, frame):
        """Process a single frame - OPTIMIZED FOR SPEED."""
        if not self.active:
            return frame, "Gestures Off", None
        
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0:
            return frame, "Processing...", None
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        gesture_text = "No Hand"
        action_result = None
        
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                gesture = self.recognize_gesture(hand_landmarks, hand_label)
                
                self.gesture_history.append(gesture)
                if len(self.gesture_history) > self.history_length:
                    self.gesture_history.pop(0)
                
                if self.gesture_history:
                    gesture = max(set(self.gesture_history), key=self.gesture_history.count)
                
                gesture_text = f"{hand_label}: {gesture}"
                
                if self.show_preview:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
                    )
                
                action_result = self.execute_action(gesture, frame, hand_landmarks)
                break
        
        return frame, gesture_text, action_result


# ============================================
# PERSONAL LLM (Your custom trained model)
# ============================================

class PersonalVocabulary:
    """Build vocabulary from your personal conversations."""
    
    def __init__(self, min_freq=2):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.min_freq = min_freq
        self.word_counts = Counter()
        
    def build_vocab(self, texts):
        for text in texts:
            words = text.lower().split()
            self.word_counts.update(words)
        
        idx = 4
        for word, count in self.word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"📚 Vocabulary size: {len(self.word2idx)} words")
    
    def encode(self, text, max_len=20):
        words = text.lower().split()[:max_len-2]
        indices = [self.word2idx["<SOS>"]] + \
                  [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words] + \
                  [self.word2idx["<EOS>"]]
        indices += [self.word2idx["<PAD>"]] * (max_len - len(indices))
        return indices
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx["<EOS>"]:
                break
            if idx not in [self.word2idx["<PAD>"], self.word2idx["<SOS>"]]:
                words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)
    
    def save(self, path="personal_vocab.pkl"):
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts
            }, f)
    
    def load(self, path="personal_vocab.pkl"):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.word2idx = data['word2idx']
                self.idx2word = data['idx2word']
                self.word_counts = data['word_counts']
            return True
        return False


class PersonalLLM(nn.Module):
    """Your personal LLM - trained on YOUR conversations."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super(PersonalLLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        out = self.fc(lstm_out)
        return out, hidden
    
    def generate(self, vocab, prompt, max_length=20, temperature=0.8):
        self.eval()
        
        with torch.no_grad():
            input_ids = torch.tensor([vocab.encode(prompt, max_length=10)])
            
            hidden = None
            generated = []
            
            for _ in range(max_length):
                output, hidden = self(input_ids, hidden)
                
                probs = torch.softmax(output[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == vocab.word2idx["<EOS>"]:
                    break
                    
                generated.append(next_token)
                input_ids = torch.tensor([[next_token]])
            
            return vocab.decode(generated)


class PersonalAITrainer:
    """Train your personal LLM on your conversations."""
    
    def __init__(self):
        self.vocab = PersonalVocabulary()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Your personal conversation style
        self.conversations = self.load_or_create_conversations()
    
    def load_or_create_conversations(self):
        conv_file = "my_personal_conversations.txt"
        
        if os.path.exists(conv_file):
            with open(conv_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        
        # YOUR PERSONAL CONVERSATION STYLE
        return [
            "User: hello Assistant: Hello! Your personal ISHA assistant is ready.",
            "User: hi Assistant: Hi there! How can I help you today?",
            "User: how are you Assistant: I'm doing great, ready to assist you!",
            "User: what is your name Assistant: I'm ISHA - your Intelligent System for Human Assistance.",
            "User: thank you Assistant: You're welcome! Always happy to help.",
            "User: thanks Assistant: My pleasure!",
            "User: who made you Assistant: You created me! I'm your personal AI.",
            "User: what can you do Assistant: I can open apps, take screenshots, control hand gestures, check time, and learn from our conversations.",
            "User: activate hand gestures Assistant: Hand gestures activated! Show your palm to the camera.",
            "User: deactivate hand gestures Assistant: Hand gestures deactivated.",
            "User: show camera Assistant: Camera preview turned on.",
            "User: hide camera Assistant: Camera preview turned off.",
            "User: take screenshot Assistant: Taking screenshot now.",
            "User: take selfie Assistant: Opening camera for selfie.",
            "User: open google Assistant: Opening Google in your browser.",
            "User: what time is it Assistant: Let me check the current time for you.",
            "User: what's the date Assistant: Here's today's date.",
        ]
    
    def add_conversation(self, user_msg, assistant_msg):
        conv = f"User: {user_msg} Assistant: {assistant_msg}"
        self.conversations.append(conv)
        
        with open("my_personal_conversations.txt", 'a', encoding='utf-8') as f:
            f.write(conv + '\n')
    
    def train(self, epochs=30):
        if not PERSONAL_LLM_AVAILABLE:
            print("❌ PyTorch not installed")
            return False
        
        print("🧠 Training your personal AI...")
        
        all_texts = []
        for conv in self.conversations:
            all_texts.extend(conv.split())
        self.vocab.build_vocab(all_texts)
        
        input_seqs = []
        target_seqs = []
        
        for conv in self.conversations:
            if "Assistant:" in conv:
                parts = conv.split("Assistant:")
                user_part = parts[0].replace("User:", "").strip()
                assistant_part = parts[1].strip()
                
                if user_part and assistant_part:
                    input_seq = self.vocab.encode(user_part, 15)
                    target_seq = self.vocab.encode(assistant_part, 15)
                    
                    input_seqs.append(input_seq)
                    target_seqs.append(target_seq)
        
        inputs = torch.tensor(input_seqs)
        targets = torch.tensor(target_seqs)
        
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        self.model = PersonalLLM(len(self.vocab.word2idx))
        self.model = self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                output, _ = self.model(batch_inputs)
                
                loss = criterion(
                    output.view(-1, len(self.vocab.word2idx)),
                    batch_targets.view(-1)
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
        }, "my_personal_ai.pt")
        self.vocab.save()
        
        print("✅ Personal AI training complete!")
        return True
    
    def load_model(self):
        if os.path.exists("my_personal_ai.pt"):
            try:
                checkpoint = torch.load("my_personal_ai.pt", map_location=self.device)
                self.vocab.load()
                self.model = PersonalLLM(len(self.vocab.word2idx))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                return True
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
        return False
    
    def generate_response(self, prompt):
        if self.model is None:
            return None
        
        try:
            response = self.model.generate(self.vocab, prompt, max_length=15)
            return response if response else None
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return None


# ============================================
# MAIN ISHA ASSISTANT - WITH BEAUTIFUL ANIMATED GUI
# ============================================

class IshaAssistant:
    """Your personal AI assistant with advanced hand gestures and beautiful GUI."""
    
    def __init__(self):
        # Initialize TTS
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        self.set_female_voice()
        
        # === YOUR PERSONAL AI ===
        print("🧠 Initializing Your Personal AI...")
        self.personal_trainer = PersonalAITrainer()
        
        if not self.personal_trainer.load_model():
            print("🎓 Training your personal AI for the first time...")
            self.personal_trainer.train(epochs=30)
        
        self.personal_ai_enabled = self.personal_trainer.model is not None
        
        # === ADVANCED HAND GESTURES ===
        print("🖐️ Initializing Advanced Hand Gestures...")
        self.gesture_controller = AdvancedGestureController()
        self.gesture_active = False
        self.gesture_thread = None
        self.camera_active = False
        
        # === Speech Recognition ===
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = None
        self.is_listening = False
        
        # Vosk for offline STT
        self.vosk_model = None
        self.vosk_recognizer = None
        self.vosk_sample_rate = 16000
        self.VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "").strip()
        self._init_vosk_if_possible()
        
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            logging.error(f"Microphone initialization failed: {str(e)}")
            self.select_microphone()
        
        # === ORIGINAL COMMAND MAPPINGS ===
        self.setup_original_command_mappings()
        
        # === QUEUE AND PENDING ===
        self.input_queue = queue.Queue()
        self.pending = None
        
        # === WELCOME ===
        self.wish_me()
        
        # === START SERVER WITH BEAUTIFUL ANIMATED GUI ===
        self.start_server()
    
    def _init_vosk_if_possible(self):
        """Original Vosk initialization."""
        try:
            if not VOSK_AVAILABLE:
                logging.info("Vosk not available.")
                return
            if not self.VOSK_MODEL_PATH or not os.path.isdir(self.VOSK_MODEL_PATH):
                logging.info("VOSK_MODEL_PATH not set or invalid. Offline STT disabled.")
                return
            self.vosk_model = VoskModel(self.VOSK_MODEL_PATH)
            self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.vosk_sample_rate)
            logging.info("Vosk offline STT initialized.")
        except Exception as e:
            logging.error(f"Failed to init Vosk: {e}")
            self.vosk_model = None
            self.vosk_recognizer = None
    
    def setup_original_command_mappings(self):
        """YOUR ORIGINAL COMMAND MAPPINGS - COMPLETE AND UNCHANGED."""
        
        # Settings Map
        self.SETTING_MAP = {
            "display setting": ("ms-settings:display", "01"),
            "sound setting": ("ms-settings:sound", "02"),
            "notification & action setting": ("ms-settings:notifications", "03"),
            "focus assist setting": ("ms-settings:quiethours", "04"),
            "power & sleep setting": ("ms-settings:powersleep", "05"),
            "storage setting": ("ms-settings:storagesense", "06"),
            "tablet setting": ("ms-settings:tablet", "07"),
            "multitasking setting": ("ms-settings:multitasking", "08"),
            "projecting to this pc setting": ("ms-settings:project", "09"),
            "shared experiences setting": ("ms-settings:crossdevice", "010"),
            "system components setting": ("ms-settings:appsfeatures-app", "001"),
            "clipboard setting": ("ms-settings:clipboard", "002"),
            "remote desktop setting": ("ms-settings:remotedesktop", "003"),
            "optional features setting": ("ms-settings:optionalfeatures", "004"),
            "about setting": ("ms-settings:about", "005"),
            "system setting": ("ms-settings:system", "006"),
            "devices setting": ("ms-settings:devices", "007"),
            "mobile devices setting": ("ms-settings:mobile-devices", "008"),
            "network & internet setting": ("ms-settings:network", "009"),
            "personalization setting": ("ms-settings:personalization", "000"),
            "apps setting": ("ms-settings:appsfeatures", "10"),
            "account setting": ("ms-settings:yourinfo", "20"),
            "time & language setting": ("ms-settings:dateandtime", "30"),
            "gaming setting": ("ms-settings:gaming", "40"),
            "ease of access setting": ("ms-settings:easeofaccess", "50"),
            "privacy setting": ("ms-settings:privacy", "60"),
            "updated & security": ("ms-settings:windowsupdate", "70")
        }
        
        self.SETTING_MAP4s = {
            "01": "ms-settings:display",
            "02": "ms-settings:sound",
            "03": "ms-settings:notifications",
            "04": "ms-settings:quiethours",
            "05": "ms-settings:powersleep",
            "06": "ms-settings:storagesense",
            "07": "ms-settings:tablet",
            "08": "ms-settings:multitasking",
            "09": "ms-settings:project",
            "010": "ms-settings:crossdevice",
            "001": "ms-settings:appsfeatures-app",
            "002": "ms-settings:clipboard",
            "003": "ms-settings:remotedesktop",
            "004": "ms-settings:optionalfeatures",
            "005": "ms-settings:about",
            "006": "ms-settings:system",
            "007": "ms-settings:devices",
            "008": "ms-settings:mobile-devices",
            "009": "ms-settings:network",
            "000": "ms-settings:personalization",
            "10": "ms-settings:appsfeatures",
            "20": "ms-settings:yourinfo",
            "30": "ms-settings:dateandtime",
            "40": "ms-settings:gaming",
            "50": "ms-settings:easeofaccess",
            "60": "ms-settings:privacy",
            "70": "ms-settings:windowsupdate"
        }
        
        self.apps_commands = {
            "alarms & clock": ("ms-clock:", "a1"),
            "calculator": ("calc", "c1"),
            "calendar": ("outlookcal:", "c2"),
            "camera": ("microsoft.windows.camera:", "c3"),
            "copilot": ("ms-copilot:", "c4"),
            "cortana": ("ms-cortana:", "c5"),
            "game bar": ("ms-gamebar:", "gb1"),
            "groove music": ("mswindowsmusic:", "gm1"),
            "mail": ("outlookmail:", "m1"),
            "maps": ("bingmaps:", "map1"),
            "microsoft edge": ("msedge", "me1"),
            "microsoft solitaire collection": ("ms-solitaire:", "mc1"),
            "microsoft store": ("ms-windows-store:", "mst1"),
            "mixed reality portal": ("ms-mixedreality:", "mp1"),
            "movies & tv": ("mswindowsvideo:", "mt1"),
            "office": ("ms-office:", "o1"),
            "onedrive": ("ms-onedrive:", "oe"),
            "onenote": ("ms-onenote:", "one"),
            "outlook": ("outlookmail:", "ouk"),
            "outlook (classic)": ("ms-outlook:", "oc1"),
            "paint": ("mspaint", "p1"),
            "paint 3d": ("ms-paint:", "p3d"),
            "phone link": ("ms-phonelink:", "pk"),
            "power point": ("ms-powerpoint:", "pt"),
            "settings": ("ms-settings:", "ss"),
            "skype": ("skype:", "sk1"),
            "snip & sketch": ("ms-snip:", "s0h"),
            "sticky note": ("ms-stickynotes:", "s1e"),
            "tips": ("ms-tips:", "ts0"),
            "voice recorder": ("ms-soundrecorder:", "vr0"),
            "weather": ("msnweather:", "w1"),
            "windows backup": ("ms-settings:backup", "wb1"),
            "windows security": ("ms-settings:windowsdefender", "ws1"),
            "word": ("ms-word:", "wrd"),
            "xbox": ("ms-xbox:", "xb"),
            "about your pc": ("ms-settings:about", "apc")
        }
        
        self.apps_commands4q = {
            "a1": "ms-clock:",
            "c1": "calc",
            "c2": "outlookcal:",
            "c3": "microsoft.windows.camera:",
            "c4": "ms-copilot:",
            "c5": "ms-cortana:",
            "gb1": "ms-gamebar:",
            "gm1": "mswindowsmusic:",
            "m1": "outlookmail:",
            "map1": "bingmaps:",
            "me1": "msedge",
            "mc1": "ms-solitaire:",
            "mst1": "ms-windows-store:",
            "mp1": "ms-mixedreality:",
            "mt1": "mswindowsvideo:",
            "o1": "ms-office:",
            "oe": "ms-onedrive:",
            "one": "ms-onenote:",
            "ouk": "outlookmail:",
            "oc1": "ms-outlook:",
            "p1": "mspaint",
            "p3d": "ms-paint:",
            "pk": "ms-phonelink:",
            "pt": "ms-powerpoint:",
            "ss": "ms-settings:",
            "sk1": "skype:",
            "s0h": "ms-snip:",
            "s1e": "ms-stickynotes:",
            "ts0": "ms-tips:",
            "vr0": "ms-soundrecorder:",
            "w1": "msnweather:",
            "wb1": "ms-settings:backup",
            "ws1": "ms-settings:windowsdefender",
            "wrd": "ms-word:",
            "xb": "ms-xbox:",
            "apc": "ms-settings:about"
        }
        
        self.software_dict = {
            "notepad": "notepad",
            "ms word": "winword",
            "command prompt": "cmd",
            "excel": "excel",
            "vscode": "code",
            "word16": "winword",
            "file explorer": "explorer",
            "edge": "msedge",
            "microsoft 365 copilot": "ms-copilot:",
            "outlook": "outlook",
            "microsoft store": "ms-windows-store:",
            "photos": "microsoft.photos:",
            "xbox": "xbox:",
            "solitaire": "microsoft.microsoftsolitairecollection:",
            "clipchamp": "clipchamp",
            "to do": "microsoft.todos:",
            "linkedin": "https://www.linkedin.com",
            "calculator": "calc",
            "news": "bingnews:",
            "one drive": "onedrive",
            "onenote 2016": "onenote",
            "google": "https://www.google.com"
        }
        
        # Merge all command dictionaries
        self.commands_dict = {**self.SETTING_MAP, **self.SETTING_MAP4s,
                              **self.software_dict, **self.apps_commands,
                              **self.apps_commands4q}
        self.commands_dict = {k: v if isinstance(v, str) else v[0]
                              for k, v in self.commands_dict.items()}
        
        self.settings_display_to_cmd = {
            f"{name} ({code})": cmd for name, (cmd, code) in self.SETTING_MAP.items()}
        self.apps_display_to_cmd = {name: cmd for name, (cmd, code) in self.apps_commands.items()}
    
    # ============ BEAUTIFUL ANIMATED GUI - UPDATED WITH YOUR REQUIREMENTS ============
    
    def start_server(self):
        """Start web UI server with BEAUTIFUL ANIMATED GUI."""
        class CustomHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_path = urlparse(self.path)
                query_params = parse_qs(parsed_path.query)
                path = parsed_path.path

                if path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(self.server.assistant.get_html().encode())
                elif path == '/command':
                    cmd = query_params.get('cmd', [None])[0]
                    if cmd is None:
                        self.send_response(400)
                        self.end_headers()
                        return
                    if self.server.assistant.pending:
                        self.server.assistant.input_queue.put(cmd)
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(b'Input received')
                    else:
                        response = self.server.assistant.process_command(cmd)
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(response.encode())
                elif path == '/voice':
                    self.server.assistant.toggle_voice()
                    message = "Microphone toggled"
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(message.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(('localhost', 8000), CustomHandler)
        server.assistant = self
        threading.Thread(target=server.serve_forever, daemon=True).start()
        webbrowser.open('http://localhost:8000/')
    
    def get_html(self):
        """YOUR BEAUTIFUL ANIMATED GUI - WITH NO GESTURE BUTTONS, ANIMATED CIRCLE, AND 'activet 156' PRESERVED."""
        apps_html = ''.join(
            f'<div class="app-item" data-command="open {name}" style="margin:6px 0; cursor:pointer;">• {name}</div>'
            for name in sorted(self.apps_display_to_cmd.keys()))
        settings_html = ''.join(
            f'<div class="setting-item" data-command="open {name}" style="margin:6px 0; cursor:pointer;">• {name}</div>'
            for name in sorted(self.SETTING_MAP.keys()))
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ISHA Assistant</title>
    <style>
        :root {{
            --bg1: #050519;
            --bg2: #0f1636;
            --neon1: #00e0ff;
            --neon2: #7b4bff;
            --glass: rgba(255, 255, 255, 0.04);
        }}

        * {{
            box-sizing: border-box;
            -webkit-font-smoothing: antialiased;
            font-family: "Segoe UI", Inter, system-ui, sans-serif;
            margin: 0;
            padding: 0;
        }}

        html, body {{
            height: 100%;
            margin: 0;
            background: linear-gradient(180deg, var(--bg1), var(--bg2));
            color: #e8f6ff;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}

        .container {{
            width: 540px;
            max-width: calc(100% - 40px);
            height: 680px;
            border-radius: 24px;
            position: relative;
            padding: 30px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.01));
            border: 1px solid rgba(255, 255, 255, 0.04);
            box-shadow: 0 20px 60px rgba(5, 10, 40, 0.6);
            overflow: hidden;
            backdrop-filter: blur(8px);
        }}

        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}

        .title {{
            font-weight: 700;
            font-size: 18px;
            color: #cfeeff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .title .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--neon1), var(--neon2));
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
                box-shadow: 0 0 5px var(--neon1);
            }}
            50% {{
                opacity: 0.8;
                box-shadow: 0 0 15px var(--neon1), 0 0 25px var(--neon2);
            }}
        }}

        .stage {{
            width: 100%;
            height: 420px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .core {{
            width: 220px;
            height: 220px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at 40% 30%, rgba(255, 255, 255, 0.14), rgba(0, 224, 255, 0.07));
            border: 1px solid rgba(0, 224, 255, 0.15);
            position: relative;
            animation: float 4s ease-in-out infinite, glow 3s ease-in-out infinite;
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.1);
        }}

        @keyframes float {{
            0%, 100% {{
                transform: translateY(0px);
            }}
            50% {{
                transform: translateY(-10px);
            }}
        }}

        @keyframes glow {{
            0%, 100% {{
                box-shadow: 0 0 20px rgba(0, 224, 255, 0.1), 0 0 40px rgba(123, 75, 255, 0.1);
            }}
            50% {{
                box-shadow: 0 0 40px rgba(0, 224, 255, 0.3), 0 0 80px rgba(123, 75, 255, 0.2);
            }}
        }}

        .core::before {{
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            border: 2px solid transparent;
            background: linear-gradient(135deg, var(--neon1), var(--neon2)) border-box;
            -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: destination-out;
            mask-composite: exclude;
            opacity: 0.5;
            animation: rotate 6s linear infinite;
        }}

        @keyframes rotate {{
            from {{
                transform: rotate(0deg);
            }}
            to {{
                transform: rotate(360deg);
            }}
        }}

        .label {{
            font-weight: 800;
            font-size: 32px;
            letter-spacing: 3px;
            color: #e9fbff;
            text-shadow: 0 0 10px rgba(0, 224, 255, 0.5), 0 0 20px rgba(123, 75, 255, 0.3);
            position: relative;
            z-index: 2;
            animation: textGlow 2s ease-in-out infinite;
        }}

        @keyframes textGlow {{
            0%, 100% {{
                text-shadow: 0 0 10px rgba(0, 224, 255, 0.5), 0 0 20px rgba(123, 75, 255, 0.3);
            }}
            50% {{
                text-shadow: 0 0 20px rgba(0, 224, 255, 0.8), 0 0 40px rgba(123, 75, 255, 0.5);
            }}
        }}

        .datetime {{
            margin-top: 18px;
            text-align: center;
            color: var(--neon1);
            font-weight: 700;
            font-size: 16px;
        }}

        .datetime .time {{
            font-size: 28px;
            color: #dffbff;
            text-shadow: 0 0 5px rgba(0, 224, 255, 0.3);
        }}

        .controls {{
            margin-top: 24px;
            width: 100%;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .input {{
            flex: 1;
            height: 48px;
            border-radius: 14px;
            padding: 10px 16px;
            background: var(--glass);
            border: 1px solid rgba(255, 255, 255, 0.04);
            color: #e8f6ff;
            outline: none;
            font-size: 15px;
            transition: all 0.3s ease;
        }}

        .input::placeholder {{
            color: rgba(255, 255, 255, 0.3);
        }}

        .input:focus {{
            border-color: var(--neon1);
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.2);
            transform: scale(1.02);
        }}

        .icon-btn {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            border: none;
            background: linear-gradient(180deg, #0c1228, #05051a);
            color: var(--neon1);
            cursor: pointer;
            font-size: 18px;
            border: 1px solid rgba(0, 224, 255, 0.06);
            transition: all 0.2s ease;
        }}

        .icon-btn:hover {{
            border-color: var(--neon1);
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.3);
            transform: scale(1.05);
            color: white;
        }}

        .popup {{
            position: absolute;
            width: 320px;
            min-height: 120px;
            max-height: 500px;
            border-radius: 12px;
            padding: 14px;
            background: rgba(10, 12, 22, 0.98);
            border: 1px solid rgba(0, 224, 255, 0.06);
            display: none;
            z-index: 20;
            cursor: move;
            transition: all 0.4s cubic-bezier(0.2, 0.9, 0.3, 1);
            backdrop-filter: blur(10px);
            overflow-y: auto;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}

        #appPopup {{
            left: -340px;
            top: 120px;
        }}

        #settingsPopup {{
            right: -340px;
            top: 140px;
        }}

        #appPopup.active {{
            left: 40px;
        }}

        #settingsPopup.active {{
            right: 40px;
        }}

        .popup .head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            color: var(--neon1);
            font-weight: 700;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0, 224, 255, 0.2);
        }}

        .popup .close {{
            width: 34px;
            height: 34px;
            border-radius: 8px;
            display: inline-grid;
            place-items: center;
            background: rgba(255, 255, 255, 0.02);
            cursor: pointer;
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.03);
            transition: all 0.2s ease;
        }}

        .popup .close:hover {{
            background: rgba(255, 0, 0, 0.2);
            border-color: #ff4444;
            transform: rotate(90deg);
        }}

        .app-item, .setting-item {{
            margin: 8px 0;
            padding: 6px 10px;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s cubic-bezier(0.2, 0.9, 0.3, 1);
            color: #cfeeff;
            font-size: 14px;
        }}

        .app-item:hover, .setting-item:hover {{
            background: rgba(0, 224, 255, 0.1);
            padding-left: 16px;
            color: white;
            transform: translateX(5px);
        }}

        .right {{
            display: flex;
            gap: 8px;
        }}

        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
        }}

        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
        }}

        ::-webkit-scrollbar-thumb {{
            background: rgba(0, 224, 255, 0.3);
            border-radius: 10px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(0, 224, 255, 0.5);
        }}

        /* Particle effect for background */
        .particle {{
            position: fixed;
            width: 2px;
            height: 2px;
            background: var(--neon1);
            opacity: 0.3;
            border-radius: 50%;
            pointer-events: none;
            animation: particleFloat 8s linear infinite;
        }}

        @keyframes particleFloat {{
            0% {{
                transform: translateY(100vh) scale(0);
                opacity: 0;
            }}
            50% {{
                opacity: 0.5;
            }}
            100% {{
                transform: translateY(-100vh) scale(1);
                opacity: 0;
            }}
        }}
    </style>
</head>
<body>
    <!-- Particle Background -->
    <div id="particles"></div>

    <div class="container" id="container">
        <!-- Top Bar -->
        <div class="topbar">
            <div class="title">
                <span class="dot"></span> ISHA Assistant
            </div>
            <div style="opacity: 0.8; font-size: 13px; color: #bfefff; text-shadow: 0 0 5px rgba(0,224,255,0.3);">
                activet 156
            </div>
        </div>

        <!-- Center Stage with Animated Circle -->
        <div class="stage">
            <div class="core">
                <div class="label">ISHA</div>
            </div>
        </div>

        <!-- Date & Time -->
        <div class="datetime" id="datetime">
            <div class="time" id="time">--:--:--</div>
            <div class="date" id="date">Loading date...</div>
        </div>

        <!-- Controls - NO GESTURE BUTTONS, only A, S, V -->
        <div class="controls">
            <input 
                class="input" 
                id="cmd" 
                placeholder="Type command (e.g., activet 156)..." 
                autofocus
            />
            <div class="right">
                <button class="icon-btn" id="appBtn" title="Applications">A</button>
                <button class="icon-btn" id="settingsBtn" title="Settings">S</button>
                <button class="icon-btn" id="voiceBtn" title="Voice">V</button>
            </div>
        </div>
    </div>

    <!-- Applications Popup -->
    <div class="popup" id="appPopup">
        <div class="head">
            <div>📱 Applications</div>
            <div class="close" data-close="appPopup">✕</div>
        </div>
        <div style="font-size: 14px; color: #cfeeff; max-height: 350px; overflow-y: auto;">
            {apps_html}
        </div>
    </div>

    <!-- Settings Popup -->
    <div class="popup" id="settingsPopup">
        <div class="head">
            <div>⚙️ Settings</div>
            <div class="close" data-close="settingsPopup">✕</div>
        </div>
        <div style="font-size: 14px; color: #cfeeff; max-height: 350px; overflow-y: auto;">
            {settings_html}
        </div>
    </div>

    <script>
        // ============================================
        // LIVE DATE & TIME UPDATER
        // ============================================
        function updateDateTime() {{
            const now = new Date();
            const timeEl = document.getElementById('time');
            const dateEl = document.getElementById('date');
            
            timeEl.textContent = now.toLocaleTimeString([], {{ 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit', 
                hour12: false 
            }});
            
            dateEl.textContent = now.toLocaleDateString([], {{ 
                weekday: 'short', 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric' 
            }});
        }}
        
        updateDateTime();
        setInterval(updateDateTime, 500);

        // ============================================
        // PARTICLE BACKGROUND
        // ============================================
        function createParticles() {{
            const particles = document.getElementById('particles');
            for (let i = 0; i < 50; i++) {{
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 8 + 's';
                particle.style.width = Math.random() * 3 + 'px';
                particle.style.height = particle.style.width;
                particles.appendChild(particle);
            }}
        }}
        createParticles();

        // ============================================
        // DRAGGABLE POPUPS
        // ============================================
        function makeDraggable(el) {{
            let dragging = false;
            let offsetX = 0;
            let offsetY = 0;

            el.addEventListener('mousedown', (e) => {{
                dragging = true;
                offsetX = e.clientX - el.offsetLeft;
                offsetY = e.clientY - el.offsetTop;
                el.style.transition = 'none';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            }});

            window.addEventListener('mousemove', (e) => {{
                if (!dragging) return;
                
                let newLeft = e.clientX - offsetX;
                let newTop = e.clientY - offsetY;
                
                newLeft = Math.max(10, Math.min(newLeft, window.innerWidth - el.offsetWidth - 10));
                newTop = Math.max(10, Math.min(newTop, window.innerHeight - el.offsetHeight - 10));
                
                el.style.left = newLeft + 'px';
                el.style.top = newTop + 'px';
            }});

            window.addEventListener('mouseup', () => {{
                if (dragging) {{
                    dragging = false;
                    el.style.transition = '';
                    document.body.style.userSelect = '';
                }}
            }});
        }}

        // ============================================
        // POPUP TOGGLE
        // ============================================
        const appBtn = document.getElementById('appBtn');
        const settingsBtn = document.getElementById('settingsBtn');
        const voiceBtn = document.getElementById('voiceBtn');
        const appPopup = document.getElementById('appPopup');
        const settingsPopup = document.getElementById('settingsPopup');

        appBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            const isActive = appPopup.classList.toggle('active');
            
            if (isActive) {{
                appPopup.style.display = 'block';
                settingsPopup.classList.remove('active');
                settingsPopup.style.display = 'none';
            }} else {{
                setTimeout(() => {{
                    if (!appPopup.classList.contains('active')) {{
                        appPopup.style.display = 'none';
                    }}
                }}, 200);
            }}
        }});

        settingsBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            const isActive = settingsPopup.classList.toggle('active');
            
            if (isActive) {{
                settingsPopup.style.display = 'block';
                appPopup.classList.remove('active');
                appPopup.style.display = 'none';
            }} else {{
                setTimeout(() => {{
                    if (!settingsPopup.classList.contains('active')) {{
                        settingsPopup.style.display = 'none';
                    }}
                }}, 200);
            }}
        }});

        voiceBtn.addEventListener('click', () => {{
            fetch('/voice').then(res => res.text()).then(_ => {{}});
        }});

        document.querySelectorAll('.close').forEach(btn => {{
            btn.addEventListener('click', (e) => {{
                e.stopPropagation();
                const id = btn.dataset.close;
                const popup = document.getElementById(id);
                popup.classList.remove('active');
                setTimeout(() => {{
                    popup.style.display = 'none';
                }}, 200);
            }});
        }});

        makeDraggable(appPopup);
        makeDraggable(settingsPopup);

        document.querySelectorAll('.app-item').forEach(item => {{
            item.addEventListener('click', () => {{
                const cmd = item.getAttribute('data-command');
                sendCommand(cmd);
                appPopup.classList.remove('active');
                setTimeout(() => appPopup.style.display = 'none', 200);
            }});
        }});
        
        document.querySelectorAll('.setting-item').forEach(item => {{
            item.addEventListener('click', () => {{
                const cmd = item.getAttribute('data-command');
                sendCommand(cmd);
                settingsPopup.classList.remove('active');
                setTimeout(() => settingsPopup.style.display = 'none', 200);
            }});
        }});

        function sendCommand(cmd) {{
            fetch(`/command?cmd=${{encodeURIComponent(cmd)}}`)
            .then(res => res.text())
            .then(text => console.log('Response:', text))
            .catch(err => console.error('Error:', err));
        }}

        document.getElementById('cmd').addEventListener('keydown', (e)=>{{
            if(e.key === 'Enter'){{
                const v = e.target.value.trim();
                if(!v) return;
                sendCommand(v);
                e.target.value = '';
            }}
        }});

        // ============================================
        // CLOSE POPUPS WHEN CLICKING OUTSIDE
        // ============================================
        document.addEventListener('click', (e) => {{
            if (!appPopup.contains(e.target) && !appBtn.contains(e.target)) {{
                if (appPopup.classList.contains('active')) {{
                    appPopup.classList.remove('active');
                    setTimeout(() => {{
                        appPopup.style.display = 'none';
                    }}, 200);
                }}
            }}
            
            if (!settingsPopup.contains(e.target) && !settingsBtn.contains(e.target)) {{
                if (settingsPopup.classList.contains('active')) {{
                    settingsPopup.classList.remove('active');
                    setTimeout(() => {{
                        settingsPopup.style.display = 'none';
                    }}, 200);
                }}
            }}
        }});

        appPopup.addEventListener('mousedown', (e) => {{
            e.stopPropagation();
        }});
        
        settingsPopup.addEventListener('mousedown', (e) => {{
            e.stopPropagation();
        }});

        appPopup.style.display = 'none';
        settingsPopup.style.display = 'none';
    </script>
</body>
</html>
        """
        return html
    
    # ============ ORIGINAL MICROPHONE METHODS ============
    
    def select_microphone(self):
        """Original microphone selection."""
        try:
            mic_names = sr.Microphone.list_microphone_names()
            if not mic_names:
                message = "No microphones detected."
                self.speak(message)
                print(message)
                return

            mic_list = "\n".join([f"{i}: {name}" for i, name in enumerate(mic_names)])
            selected = self.user_input(
                f"Available microphones:\n{mic_list}\nEnter index:")

            if selected is not None:
                try:
                    index = int(selected)
                    if 0 <= index < len(mic_names):
                        self.microphone = sr.Microphone(device_index=index)
                        with self.microphone as source:
                            self.recognizer.adjust_for_ambient_noise(source, duration=2)
                        message = f"Selected: {mic_names[index]}"
                        self.speak(message)
                        print(message)
                    else:
                        message = "Invalid index."
                        self.speak(message)
                        print(message)
                except ValueError:
                    message = "Invalid input."
                    self.speak(message)
                    print(message)
                except (OSError, sr.RequestError) as e:
                    message = f"Failed: {str(e)}"
                    self.speak(message)
                    print(message)
        except Exception as e:
            message = f"Error: {str(e)}"
            self.speak(message)
            print(message)

    def user_input(self, prompt):
        console = ctypes.windll.kernel32.GetConsoleWindow()
        if console:
            ctypes.windll.user32.ShowWindow(console, 5)
        res = input(prompt)
        if console:
            ctypes.windll.user32.ShowWindow(console, 0)
        return res
    
    def set_female_voice(self):
        """Original female voice setup."""
        try:
            voices = self.engine.getProperty('voices')
            selected_voice = None

            logging.info("Available voices: %s", [voice.name for voice in voices])

            for voice in voices:
                if "zira" in voice.name.lower() or "female" in voice.name.lower():
                    selected_voice = voice
                    break

            if not selected_voice:
                selected_voice = voices[0] if voices else None
                logging.warning("No female voice found, using default voice")
                message = "No female voice available, using default voice."
                self.speak(message)
            else:
                logging.info("Selected female voice: %s", selected_voice.name)

            if selected_voice:
                self.engine.setProperty('voice', selected_voice.id)
                self.engine.say("Initializing voice")
                self.engine.runAndWait()
            else:
                logging.error("No voices available")
                message = "No voices available."
                self.speak(message)
        except Exception as e:
            logging.error(f"Failed to set female voice: {str(e)}")
            message = "Failed to initialize TTS."
            self.speak(message)
    
    def check_internet(self):
        """Original internet check."""
        current_time = time.time()
        if current_time - self.last_internet_check < self.internet_check_interval:
            return self.internet_status

        self.last_internet_check = current_time
        for host in [("8.8.8.8", 80), ("1.1.1.1", 80)]:
            try:
                socket.create_connection(host, timeout=2)
                self.internet_status = True
                return True
            except (socket.gaierror, socket.timeout):
                continue
        self.internet_status = False
        return False
    
    # ============ GESTURE CONTROL METHODS ============
    
    def toggle_hand_gestures(self, activate=None):
        """Turn hand gestures on/off by voice or text command."""
        if activate is not None:
            self.gesture_active = activate
        else:
            self.gesture_active = not self.gesture_active
        
        self.gesture_controller.active = self.gesture_active
        
        if self.gesture_active:
            if not self.gesture_thread or not self.gesture_thread.is_alive():
                self.gesture_thread = threading.Thread(target=self._gesture_loop, daemon=True)
                self.gesture_thread.start()
            message = "Hand gestures activated"
        else:
            message = "Hand gestures deactivated"
        
        self.speak(message)
        return message
    
    def toggle_camera_preview(self, show=None):
        """Turn camera preview on/off."""
        if show is not None:
            self.gesture_controller.show_preview = show
        else:
            self.gesture_controller.show_preview = not self.gesture_controller.show_preview
        
        status = "on" if self.gesture_controller.show_preview else "off"
        message = f"Camera preview turned {status}"
        self.speak(message)
        return message
    
    def _gesture_loop(self):
        """Main gesture recognition loop - runs in background."""
        if not MEDIAPIPE_AVAILABLE:
            self.speak("MediaPipe not installed. Cannot use hand gestures.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.speak("Camera not available.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        last_action_message = ""
        action_message_time = 0
        
        while self.gesture_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame, gesture_text, action_result = self.gesture_controller.process_frame(frame)
            
            if action_result:
                last_action_message = action_result
                action_message_time = time.time()
                print(f"🖐️ Gesture: {gesture_text} - {action_result}")
            
            if self.gesture_controller.show_preview:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
                cv2.putText(frame, f"ISHA GESTURES: {'ON' if self.gesture_active else 'OFF'}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Gesture: {gesture_text}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if time.time() - action_message_time < 2:
                    cv2.putText(frame, f"Action: {last_action_message}", 
                               (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(frame, "Press 'Q' to close", 
                           (10, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("ISHA Hand Gestures", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.gesture_active = False
                    break
            else:
                time.sleep(0.01)
        
        cap.release()
        cv2.destroyAllWindows()
    
    # ============ PERSONAL AI RESPONSES ============
    
    def get_personal_response(self, user_input):
        """Get response from YOUR personal AI."""
        if not self.personal_ai_enabled:
            return None
        
        response = self.personal_trainer.generate_response(user_input)
        
        if response:
            self.personal_trainer.add_conversation(user_input, response)
        
        return response
    
    # ============ COMMAND PROCESSING ============
    
    def process_command(self, command):
        """Process user command with personal AI and gesture control."""
        logging.info(f"Processing command: {command}")
        command = command.lower().strip()
        
        if self.pending:
            self.input_queue.put(command)
            self.pending = None
            return "Input received"
        
        # === HAND GESTURE COMMANDS ===
        if any(word in command for word in ["activate hand", "turn on hand", "start hand", "enable hand"]):
            return self.toggle_hand_gestures(True)
        
        elif any(word in command for word in ["deactivate hand", "turn off hand", "stop hand", "disable hand"]):
            return self.toggle_hand_gestures(False)
        
        elif "gesture" in command and ("status" in command or "check" in command):
            status = "ON" if self.gesture_active else "OFF"
            return f"Hand gestures are {status}"
        
        # === CAMERA PREVIEW COMMANDS ===
        elif any(word in command for word in ["show camera", "show preview", "camera on"]):
            return self.toggle_camera_preview(True)
        
        elif any(word in command for word in ["hide camera", "hide preview", "camera off"]):
            return self.toggle_camera_preview(False)
        
        # === TRAINING COMMANDS ===
        elif command == "train my ai":
            self.speak("Training your personal AI. This will take a moment.")
            self.personal_trainer.train(epochs=20)
            self.speak("Training complete!")
            return "Personal AI retrained"
        
        # === GESTURE ACTIVATION CODE ===
        if command == "activet 156":
            return self.toggle_hand_gestures(True)
        
        # === TRY PERSONAL AI FIRST ===
        if self.personal_ai_enabled:
            personal_response = self.get_personal_response(command)
            if personal_response:
                self.speak(personal_response)
                return personal_response
        
        # === ORIGINAL RULE-BASED COMMANDS ===
        
        # Open commands
        if command.startswith("open "):
            app_or_setting = command[5:].strip()
            result = self.handle_settings_apps_commands(app_or_setting)
            if result:
                return result
        
        # Time
        if command in ["what is the time", "tell me the time", "current time", "time now", "what time is it", "what's the time", "time", "what time"]:
            message = datetime.datetime.now().strftime("%H:%M:%S")
            self.speak(message)
            return message
        
        # Date
        elif command in ["what is the date", "tell me the date", "current date", "date now", "what date is it", "what's the date", "date", "what date"]:
            message = datetime.datetime.now().strftime("%A, %B %d, %Y")
            self.speak(message)
            return message
        
        # Google
        elif command in ["open google", "launch google", "go to google"]:
            if self.check_internet():
                return self.open_google()
            else:
                message = "No internet connection. Google cannot be opened."
                self.speak(message)
                return message
        
        # Weather
        elif command in ["weather", "check weather", "what's the weather"]:
            return self.get_weather_offline_first()
        
        # Greetings
        elif command in ["hi", "hello", "hey"]:
            message = "Hello! How can I assist you today?"
            self.speak(message)
            return message
        
        # Screenshot
        elif command in ["screenshot", "take screenshot", "capture screen"]:
            return self.take_screenshot()
        
        # Selfie
        elif command in ["selfie", "take selfie", "capture selfie"]:
            return self.take_selfie()
        
        # Default
        message = f"Command not recognized: {command}"
        self.speak(message)
        return message
    
    def handle_settings_apps_commands(self, command):
        """Original command handler."""
        cmd = None
        if command in self.commands_dict:
            cmd = self.commands_dict[command]
        elif command in self.settings_display_to_cmd:
            cmd = self.settings_display_to_cmd[command]
        elif command in self.apps_display_to_cmd:
            cmd = self.apps_display_to_cmd[command]

        if cmd:
            try:
                if cmd.startswith("http"):
                    webbrowser.open(cmd)
                else:
                    subprocess.run(["start", "", cmd], shell=True)
                message = f"Opening {command}"
                self.speak(message)
                return message
            except Exception as e:
                message = f"Failed to open {command}: {str(e)}"
                self.speak(message)
                return message
        return None
    
    def open_google(self):
        """Original Google opener."""
        try:
            webbrowser.open("https://www.google.com")
            message = "Opening Google"
            self.speak(message)
            return message
        except Exception as e:
            message = f"Failed to open Google: {str(e)}"
            self.speak(message)
            return message
    
    def get_weather_offline_first(self):
        """Original weather with offline cache."""
        message = "Which city's weather do you want to check?"
        self.speak(message)
        self.pending = 'weather_city'
        try:
            city = self.input_queue.get(timeout=30)
        except queue.Empty:
            self.pending = None
            message = "No city entered. Weather check cancelled."
            self.speak(message)
            return message
        self.pending = None

        if not city or city.lower() in ["none", "cancel", "no"]:
            message = "No city provided. Please try again."
            self.speak(message)
            return message

        if self.check_internet():
            try:
                response = requests.get(f"https://wttr.in/{city}?format=%C+%t", timeout=5)
                response.raise_for_status()
                weather_info = response.text.strip()
                with open("weather_cache.txt", "w", encoding="utf-8") as f:
                    f.write(f"{city}:{weather_info}:{int(time.time())}")
                message = f"Weather in {city}: {weather_info}"
                self.speak(message)
                return message
            except Exception as e:
                logging.error(f"Weather fetch failed: {e}")

        try:
            with open("weather_cache.txt", "r", encoding="utf-8") as f:
                cache_data = f.read().strip()
            if not cache_data:
                message = "No internet and no cached weather available."
                self.speak(message)
                return message
            c_city, weather_info, timestamp = cache_data.split(":", 2)
            age = int(time.time()) - int(timestamp)
            if age < 3600:
                message = f"No internet. Showing cached weather for {c_city}: {weather_info}"
            else:
                message = "No internet and cached weather is too old."
            self.speak(message)
            return message
        except Exception as e:
            message = f"No internet and no valid cached weather available: {str(e)}."
            self.speak(message)
            return message
    
    def take_screenshot(self):
        """Original screenshot method."""
        try:
            folder = os.path.join(os.getcwd(), "isha_captures")
            self._ensure_folder(folder)
            fname = self._unique_filename(prefix="screenshot", ext="png")
            path = os.path.join(folder, fname)
            shot = pyautogui.screenshot()
            shot.save(path)
            message = f"Screenshot saved"
            self.speak("Screenshot captured")
            logging.info(f"Screenshot saved: {path}")
            return message
        except Exception as e:
            message = f"Failed to take screenshot: {str(e)}"
            self.speak(message)
            return message
    
    def take_selfie(self):
        """Original selfie method."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Camera not available")
            ret, frame = cap.read()
            cap.release()
            if ret:
                folder = os.path.join(os.getcwd(), "isha_captures")
                self._ensure_folder(folder)
                fname = self._unique_filename(prefix="selfie", ext="jpg")
                path = os.path.join(folder, fname)
                cv2.imwrite(path, frame)
                message = f"Selfie captured"
                self.speak("Selfie captured")
                logging.info(f"Selfie saved: {path}")
                return message
            else:
                raise Exception("Failed to capture image")
        except Exception as e:
            message = f"Failed to take selfie: {str(e)}"
            self.speak(message)
            return message
    
    def _ensure_folder(self, folder):
        os.makedirs(folder, exist_ok=True)
    
    def _unique_filename(self, prefix="capture", ext="png"):
        now = datetime.datetime.now()
        return f"{prefix}__{now.strftime('%d-%m-%Y__%H-%M-%S')}__{int(now.microsecond/1000):03d}.{ext}"
    
    def wish_me(self):
        """Original welcome message."""
        current_hour = datetime.datetime.now().hour
        greeting = (
            "Good morning" if 5 <= current_hour < 12 else
            "Good afternoon" if 12 <= current_hour < 17 else
            "Good evening" if 17 <= current_hour < 21 else
            "Good night"
        )
        self.speak(greeting)
        time.sleep(1)
        message = "I am Isha, Intelligent System for Human Assistance. Welcome! Say 'activate hand gestures' for gesture control."
        self.speak(message)
        time.sleep(2)
    
    # ============ VOICE METHODS ============
    
    def toggle_voice(self):
        """Original voice toggle."""
        if self.microphone is None:
            self.select_microphone()
            if self.microphone is None:
                message = "Voice recognition is disabled. Use text input instead."
                self.speak(message)
                print(message)
                return
        self.is_listening = not self.is_listening
        if self.is_listening:
            message = "Microphone is now on"
            self.speak(message)
            print(message)
            threading.Thread(target=self.listen_voice, daemon=True).start()
        else:
            message = "Microphone is now off"
            self.speak(message)
            print(message)
    
    def listen_voice(self):
        """Original voice listener."""
        while self.is_listening:
            command = self.listen()
            if command:
                self.process_command(command)
            time.sleep(0.5)
    
    def listen(self):
        """Original listen method."""
        if self.vosk_recognizer:
            text = self._listen_vosk(duration_sec=6)
            if text:
                return text

        if self.microphone is None:
            self.select_microphone()
            if self.microphone is None:
                query = self.user_input("Voice not available. Enter your command: ")
                return query.lower() if query else None

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                for _ in range(3):
                    try:
                        audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                        return self.recognizer.recognize_google(audio).lower()
                    except sr.WaitTimeoutError:
                        self.speak("No speech detected, retrying...")
                        continue
                    except sr.UnknownValueError:
                        self.speak("Could not understand, retrying...")
                        continue
                    except sr.RequestError as e:
                        logging.error(f"Google STT error: {e}")
                        query = self.user_input("Voice STT failed. Enter your command: ")
                        return query.lower() if query else None
            query = self.user_input("Voice input failed. Enter your command: ")
            return query.lower() if query else None
        except Exception as e:
            logging.error(f"Voice input failed: {e}")
            query = self.user_input("Voice not available. Enter your command: ")
            return query.lower() if query else None
    
    def _listen_vosk(self, duration_sec=6):
        """Original Vosk listener."""
        if not self.vosk_recognizer or not self.microphone:
            return None
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1,
                             rate=self.vosk_sample_rate, input=True,
                             frames_per_buffer=4000)
            stream.start_stream()
            start = time.time()
            text = ""
            while time.time() - start < duration_sec:
                data = stream.read(4000, exception_on_overflow=False)
                if self.vosk_recognizer.AcceptWaveform(data):
                    res = json.loads(self.vosk_recognizer.Result())
                    text = res.get("text", "") or text
            res = json.loads(self.vosk_recognizer.FinalResult())
            text = res.get("text", "") or text
            stream.stop_stream()
            stream.close()
            pa.terminate()
            return text.strip().lower() if text else None
        except Exception as e:
            logging.error(f"Vosk listen failed: {e}")
            return None
    
    # ============ SPEAK ============
    
    def speak(self, text):
        """Original speak method."""
        def run_speak():
            try:
                self.engine.stop()
                time.sleep(0.2)
                if getattr(self.engine, "_inLoop", False):
                    try:
                        self.engine.endLoop()
                    except Exception:
                        pass
                    time.sleep(0.3)
                    self.engine = pyttsx3.init()
                    self.set_female_voice()
                self.engine.say(text)
                self.engine.runAndWait()
                logging.info(f"Spoke: {text}")
            except Exception as e:
                logging.error(f"Speech error: {str(e)} - Text: {text}")

        threading.Thread(target=run_speak, daemon=True).start()
        time.sleep(0.2)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    try:
        # Hide console on Windows
        console = ctypes.windll.kernel32.GetConsoleWindow()
        if console:
            ctypes.windll.user32.ShowWindow(console, 0)
        
        print("=" * 60)
        print("🤖 ISHA - Your Personal AI Assistant")
        print("🖐️ Advanced Hand Gestures Ready (OFF by default)")
        print("📷 Camera Preview: OFF (say 'show camera' to enable)")
        print("🧠 Personal LLM: Trained on YOUR conversations")
        print("✨ Beautiful Animated GUI with Floating Circle")
        print("=" * 60)
        print("Commands:")
        print("  • 'activate hand gestures' - Turn gestures ON")
        print("  • 'deactivate hand gestures' - Turn gestures OFF")
        print("  • 'show camera' - See yourself")
        print("  • 'hide camera' - Hide preview (gestures still work)")
        print("  • 'screenshot' - Capture screen")
        print("  • 'selfie' - Take photo")
        print("  • 'open [app]' - Open any app")
        print("  • 'train my ai' - Improve your personal AI")
        print("  • 'activet 156' - Quick gesture activation")
        print("=" * 60)
        
        app = IshaAssistant()
        
        # Keep alive
        while True:
            time.sleep(1)
            
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        print(f"Error: Application failed to start: {str(e)}")
        
        
