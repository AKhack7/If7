```python
"""
IshaAssistant - Intelligent System for Human Assistance
======================================================

A Windows-focused desktop assistant with:
- Tkinter GUI (dark theme) with chat history + input box + quick buttons
- Voice input via SpeechRecognition (Google recognizer) with microphone selection fallback
- Text-to-speech via pyttsx3 with female voice preference (Zira if available)
- 100+ command-like intents (time/date/math/apps/settings/web/music/weather/system control)
- Internet status indicator with caching
- Threaded/non-blocking voice listening + threaded TTS
- Basic error logging to isha_assistant.log

NOTE:
- This code is designed for Windows.
- Install dependencies:
  pip install pyttsx3 SpeechRecognition pyautogui psutil pywhatkit sympy requests numpy
  plus PyAudio for microphone:
  pip install pyaudio   (may require wheels / system deps on Windows)

Some commands in the provided documentation referenced undefined functions (e.g., came2()).
Those are implemented as safe placeholders in this regeneration to avoid runtime errors.

Author: Regenerated from user-provided documentation description.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog
import pyttsx3
import speech_recognition as sr
import datetime
import os
import webbrowser
import pyautogui
import time
import psutil
import pywhatkit
import random
import subprocess
import re
import threading
import socket
import glob
import logging
from sympy import sympify, sin, cos, tan, sqrt, pi
import urllib.request
import numpy as np
import requests


# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    filename="isha_assistant.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class IshaAssistant:
    def __init__(self, root: tk.Tk):
        self.root = root

        # Window configuration
        self.root.title("Isha Assistant")
        self.root.geometry("600x400")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        # Try to set toolwindow style (Windows-only)
        try:
            self.root.attributes("-toolwindow", True)
        except Exception:
            pass

        # State
        self.is_listening = False
        self.last_enter_time = 0

        # Internet cache
        self.last_internet_check = 0
        self.internet_status = False
        self.internet_check_interval = 10  # seconds

        # GUI placeholders
        self.chat_box = None
        self.input_box = None
        self.status_label = None
        self.mic_btn = None
        self.settings_popup = None
        self.apps_popup = None

        # Initialize TTS
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        self.engine.setProperty("volume", 0.9)

        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = None

        # Microphone setup
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Default microphone initialized successfully.")
        except Exception as e:
            logging.error(f"Microphone init failed: {str(e)}")
            # Speak warning (may still work without mic)
            self.speak("Microphone is not available. Falling back to text input. You can try selecting a microphone.")
            # Allow user to select later
            self.microphone = None

        # GUI
        self.create_gui()

        # Voice selection
        self.set_female_voice()

        # Settings and apps maps (as described)
        self.SETTING_MAP = {
            "display setting": ("ms-settings:display", "01"),
            "wifi setting": ("ms-settings:network-wifi", "02"),
            "bluetooth setting": ("ms-settings:bluetooth", "03"),
            "sound setting": ("ms-settings:sound", "04"),
            "notifications setting": ("ms-settings:notifications", "05"),
            "privacy setting": ("ms-settings:privacy", "06"),
            "windows update setting": ("ms-settings:windowsupdate", "07"),
            "accounts setting": ("ms-settings:yourinfo", "08"),
            "time setting": ("ms-settings:dateandtime", "09"),
            "language setting": ("ms-settings:regionlanguage", "10"),
            "apps setting": ("ms-settings:appsfeatures", "11"),
            "storage setting": ("ms-settings:storagesense", "12"),
            "personalization setting": ("ms-settings:personalization", "13"),
            "gaming setting": ("ms-settings:gaming-gamebar", "14"),
            "ease of access setting": ("ms-settings:easeofaccess", "15"),
            "security setting": ("ms-settings:windowsdefender", "16"),
            "about setting": ("ms-settings:about", "17"),
        }
        self.SETTING_MAP4s = {v[1]: v[0] for k, v in self.SETTING_MAP.items()}

        self.apps_commands = {
            "calculator": ("calc", "c1"),
            "notepad": ("notepad", "c2"),
            "paint": ("mspaint", "c3"),
            "cmd": ("cmd", "c4"),
            "control panel": ("control", "c5"),
            "task manager": ("taskmgr", "c6"),
        }
        self.apps_commands4q = {v[1]: v[0] for k, v in self.apps_commands.items()}

        self.software_dict = {
            "notepad": "notepad",
            "calculator": "calc",
            "paint": "mspaint",
            "command prompt": "cmd",
        }

        # Unified command dict for opening things
        self.commands_dict = {}
        for k, v in self.SETTING_MAP.items():
            self.commands_dict[k] = v[0]
        for k, v in self.apps_commands.items():
            self.commands_dict[k] = v[0]
        for k, v in self.software_dict.items():
            self.commands_dict[k] = v

        # Display-friendly maps for popups
        self.settings_display_to_cmd = {k.title(): v[0] for k, v in self.SETTING_MAP.items()}
        self.apps_display_to_cmd = {k.title(): v[0] for k, v in self.apps_commands.items()}

        # Greet
        self.wish_me()

        # Bindings
        self.root.bind("<Return>", self.handle_double_enter)

        # Start internet status updates
        self.update_internet_status()

    # ----------------------------
    # Microphone selection
    # ----------------------------
    def select_microphone(self):
        try:
            names = sr.Microphone.list_microphone_names()
            if not names:
                self.speak("No microphones detected on this system.")
                return

            msg = "Available microphones:\n\n"
            for i, name in enumerate(names):
                msg += f"{i}: {name}\n"

            idx_str = simpledialog.askstring("Select Microphone", msg + "\nEnter microphone index:")
            if idx_str is None:
                return
            idx = int(idx_str)

            self.microphone = sr.Microphone(device_index=idx)
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)

            self.speak("Microphone selected successfully.")
            logging.info(f"Microphone selected: index={idx}, name={names[idx]}")
        except Exception as e:
            logging.error(f"Microphone selection failed: {str(e)}")
            self.speak("Microphone selection failed. Please check your microphone and PyAudio installation.")

    # ----------------------------
    # Female voice preference
    # ----------------------------
    def set_female_voice(self):
        try:
            voices = self.engine.getProperty("voices")
            for v in voices:
                logging.info(f"TTS voice found: {v.name} ({getattr(v, 'id', '')})")

            chosen_id = None
            for v in voices:
                name = (v.name or "").lower()
                if "zira" in name or "female" in name:
                    chosen_id = v.id
                    break

            if chosen_id is None and voices:
                chosen_id = voices[0].id

            if chosen_id:
                self.engine.setProperty("voice", chosen_id)

            # Test
            self.speak("Initializing voice")
        except Exception as e:
            logging.error(f"Voice setup failed: {str(e)}")
            self.speak("Voice initialization failed. Continuing with default voice if available.")

    # ----------------------------
    # Double-enter focus behavior
    # ----------------------------
    def handle_double_enter(self, event):
        now = time.time()
        if now - self.last_enter_time <= 0.5:
            if self.input_box:
                self.input_box.focus_set()
        self.last_enter_time = now

    # ----------------------------
    # Internet checking with cache
    # ----------------------------
    def check_internet(self) -> bool:
        now = time.time()
        if now - self.last_internet_check < self.internet_check_interval:
            return self.internet_status

        self.last_internet_check = now
        try:
            for host in [("8.8.8.8", 80), ("1.1.1.1", 80)]:
                s = socket.create_connection(host, timeout=2)
                s.close()
                self.internet_status = True
                return True
        except Exception:
            self.internet_status = False
            return False

    def update_internet_status(self):
        online = self.check_internet()
        if self.status_label:
            self.status_label.configure(text=f"Internet: {'Online' if online else 'Offline'}")
        self.root.after(10000, self.update_internet_status)

    # ----------------------------
    # GUI Construction
    # ----------------------------
    def create_gui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="white")
        style.configure("TButton", padding=6, background="#2d2d2d", foreground="white")
        style.map("TButton", background=[("active", "#3a3a3a")])

        # Custom button styles
        style.configure("MicOn.TButton", background="#2ecc71", foreground="black")
        style.configure("MicOff.TButton", background="#e74c3c", foreground="white")
        style.configure("Quick.TButton", background="#3b3b3b", foreground="white")

        # Chat display
        self.chat_box = scrolledtext.ScrolledText(
            self.root,
            height=10,
            width=60,
            bg="#111111",
            fg="white",
            insertbackground="white",
            wrap=tk.WORD
        )
        self.chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Input row
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.input_box = ttk.Entry(bottom_frame, width=50)
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_box.bind("<Return>", self.process_text_input)

        # Buttons row
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.mic_btn = ttk.Button(btn_frame, text="Mic", style="MicOff.TButton", command=self.toggle_voice)
        self.mic_btn.pack(side=tk.LEFT, padx=4)

        ttk.Button(btn_frame, text="Settings", style="Quick.TButton", command=self.toggle_settings).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Apps", style="Quick.TButton", command=self.toggle_apps).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="File M", style="Quick.TButton", command=self.open_file_explorer).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Downloads", style="Quick.TButton", command=self.open_downloads).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="About", style="Quick.TButton", command=self.show_about).pack(side=tk.RIGHT, padx=4)

        # Status
        self.status_label = ttk.Label(self.root, text="Internet: Checking...")
        self.status_label.pack(anchor="w", padx=12, pady=(0, 8))

        # Keyboard shortcuts
        self.root.bind("<Control-m>", lambda e: self.toggle_voice())
        self.root.bind("<Control-s>", lambda e: self.toggle_settings())
        self.root.bind("<Control-a>", lambda e: self.toggle_apps())
        self.root.bind("<Control-f>", lambda e: self.open_file_explorer())
        self.root.bind("<Control-d>", lambda e: self.open_downloads())
        self.root.bind("<Control-period>", lambda e: self.show_about())

    # ----------------------------
    # Chat helper
    # ----------------------------
    def chat_box_insert(self, text: str):
        if not self.chat_box:
            return
        self.chat_box.insert(tk.END, text + "\n")
        self.chat_box.see(tk.END)

    # ----------------------------
    # Voice toggle and listening thread
    # ----------------------------
    def toggle_voice(self):
        if self.microphone is None:
            self.select_microphone()

        self.is_listening = not self.is_listening
        if self.mic_btn:
            self.mic_btn.configure(style="MicOn.TButton" if self.is_listening else "MicOff.TButton")

        self.speak("Microphone is now on" if self.is_listening else "Microphone is now off")

        if self.is_listening:
            t = threading.Thread(target=self.listen_voice, daemon=True)
            t.start()

    def listen(self):
        # If mic is not available, use text dialog fallback
        if self.microphone is None:
            txt = simpledialog.askstring("Voice Input", "Microphone not available.\nType your command:")
            return txt.lower().strip() if txt else None

        attempts = 0
        while attempts < 5:
            attempts += 1
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=15)
                query = self.recognizer.recognize_google(audio)
                if query:
                    return query.lower().strip()
                return None
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                logging.error(f"Voice recognition failed: {str(e)}")
                txt = simpledialog.askstring("Voice Input", "Voice recognition failed.\nType your command:")
                return txt.lower().strip() if txt else None

        return None

    def listen_voice(self):
        while self.is_listening:
            cmd = self.listen()
            if cmd:
                self.root.after(0, lambda c=cmd: self.process_voice_command(c))
            time.sleep(1)

    def process_voice_command(self, command: str):
        if self.input_box:
            self.input_box.delete(0, tk.END)
            self.input_box.insert(0, command)
        # simulate enter processing
        self.process_command(command)

    # ----------------------------
    # Text input handler
    # ----------------------------
    def process_text_input(self, event=None):
        cmd = self.input_box.get().strip().lower() if self.input_box else ""
        if self.input_box:
            self.input_box.delete(0, tk.END)
        if cmd:
            self.process_command(cmd)

    # ----------------------------
    # Greeting
    # ----------------------------
    def wish_me(self):
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            greet = "Good morning"
        elif 12 <= hour < 17:
            greet = "Good afternoon"
        elif 17 <= hour < 21:
            greet = "Good evening"
        else:
            greet = "Good night"

        self.speak(greet)
        time.sleep(1)
        self.speak("I am Isha. Welcome!")
        self.chat_box_insert("Welcome to Isha Assistant.")
        time.sleep(2)

    # ----------------------------
    # Core command router
    # ----------------------------
    def process_command(self, command: str):
        try:
            online = self.check_internet()
            logging.info(f"Command: {command} | internet={online}")

            self.chat_box_insert(f"You: {command}")

            # Common variants
            time_cmds = {"what is the time", "time", "time batao", "tell me the time"}
            date_cmds = {"what is the date", "date", "date batao", "tell me the date"}

            if command in time_cmds:
                return self.get_time()

            if command in date_cmds:
                return self.get_date()

            # Math
            if command.startswith("solve "):
                return self.solve_math(command.replace("solve ", "", 1))

            if any(op in command for op in ["+", "-", "*", "/", "sin", "cos", "tan", "sqrt", "pi"]) and any(ch.isdigit() for ch in command):
                # basic heuristic: treat as math
                return self.solve_math(command)

            # App/system commands
            if command in {"open calculator", "calculator"}:
                return self.open_calculator()

            if command in {"open file explorer", "file manager", "open files"}:
                return self.open_file_explorer()

            if command in {"open downloads", "downloads"}:
                return self.open_downloads()

            if command in {"minimize windows", "minimize"}:
                return self.minimize_windows()

            if command in {"search", "open search"}:
                return self.open_search()

            if command in {"news", "open news"}:
                if online:
                    return self.open_news_widget()
                else:
                    # offline fallback as documented
                    return self.open_uri("msnweather:")

            if command in {"run", "open run"}:
                return self.open_run_command()

            if command in {"settings", "open settings"}:
                return self.open_settings()

            if command in {"ok isha", "isha", ""}:
                return self.find_now()

            if command in {"about setting", "open about setting"}:
                return self.open_about_settings()

            if command in {"project screen", "projection"}:
                return self.open_project_screen()

            if command in {"enhanced security", "performance settings"}:
                return self.open_performance_settings()

            if command in {"feedback", "feedback hub"}:
                return self.open_feedback_hub()

            if command in {"xbox", "game bar"}:
                return self.open_game_bar()

            if command in {"open mic", "voice typing"}:
                return self.open_voice_typing()

            if command in {"connect", "network", "open connect"}:
                return self.open_connect_panel()

            if command in {"lock", "lock screen"}:
                return self.lock_screen()

            if command in {"menu", "quick menu"}:
                return self.open_quick_menu()

            if command in {"cortana"}:
                return self.open_cortana()

            if command in {"clipboard"}:
                return self.open_clipboard_history()

            if command in {"duplicate window", "notifications"}:
                return self.open_notifications()

            if command in {"play song", "play music", "song"}:
                return self.play_song()

            if command in {"youtube", "open youtube"}:
                return self.open_youtube()

            if command in {"google", "open google"}:
                return self.open_google()

            if command in {"instagram", "open instagram"}:
                return self.open_instagram()

            if command in {"iti"}:
                return self.open_25()

            if command in {"phone camera on"}:
                return self.came2()  # placeholder
            if command in {"phone camera off"}:
                return self.cmaw21()

            if command in {"h1", "open chatbox"}:
                return self.open_chatbox()

            if command in {"download photo", "download picture"}:
                return self.download_picture()

            if command in {"instagram login"}:
                return self.login_instagram()

            if command in {"whatsapp", "open whatsapp"}:
                return self.open_whatsapp()

            if command in {"hello", "hi", "hey"}:
                return self.hello()

            if command in {"thank you", "thanks"}:
                return self.thank_you_reply()

            if command in {"name", "what is your name"}:
                return self.what_is_your_name()

            if command in {"select all"}:
                return self.select_all_text()

            if command in {"good morning"}:
                return self.morningtime()

            if command in {"stop song", "stop music"}:
                return self.stop_song()

            if command in {"download reel", "download reels", "download stories"}:
                return self.download_instagram_reel()

            if command in {"mute", "unmute", "pause"}:
                return self.mute_unmute()

            if command in {"battery", "battery percentage"}:
                return self.btr()

            if command in {"full screen"}:
                return self.full_screen()

            if command in {"captions on", "captions off", "toggle captions"}:
                return self.toggle_caption()

            if command in {"weather", "get weather"}:
                return self.get_weather()

            if command in {"shutdown", "shutdown pc"}:
                return self.shutdown_pc()

            if command in {"restart", "restart pc"}:
                return self.restart_pc()

            if command in {"find", "search now", "find now"}:
                return self.find_now()

            if command in {"about"}:
                return self.show_about()

            if command in {"greet me", "wish me"}:
                return self.wish_me()

            # Try settings/apps dictionary routing
            return self.handle_settings_apps_commands(command)

        except Exception as e:
            logging.error(f"process_command error: {str(e)}")
            self.speak("Sorry, something went wrong while processing your command.")

    # ----------------------------
    # Basic actions
    # ----------------------------
    def get_time(self):
        now = datetime.datetime.now().strftime("%I:%M %p")
        msg = f"The time is {now}"
        self.chat_box_insert(f"Isha: {msg}")
        self.speak(msg)

    def get_date(self):
        now = datetime.datetime.now().strftime("%B %d, %Y")
        msg = f"Today's date is {now}"
        self.chat_box_insert(f"Isha: {msg}")
        self.speak(msg)

    def find_now(self):
        online = self.check_internet()
        if online:
            q = simpledialog.askstring("Search", "What do you want to search?")
            if not q or q.strip().lower() in {"none", "cancel", "no"}:
                self.speak("Okay.")
                return
            url = "https://www.google.com/search?q=" + requests.utils.quote(q)
            webbrowser.open(url)
            self.speak("Searching on Google.")
        else:
            self.speak("No internet. Opening File Explorer instead.")
            self.open_file_explorer()

    def solve_math(self, expression: str):
        try:
            expr = expression.replace(" ", "")
            val = sympify(expr, locals={"sin": sin, "cos": cos, "tan": tan, "sqrt": sqrt, "pi": pi})
            valf = float(val.evalf())

            if abs(valf - int(valf)) < 1e-12:
                out = str(int(valf))
            else:
                out = str(round(valf, 6))

            msg = f"The answer is {out}"
            self.chat_box_insert(f"Isha: {msg}")
            self.speak(msg)
        except Exception as e:
            logging.error(f"solve_math error: {str(e)} | expr={expression}")
            self.speak("I could not solve that expression.")

    def open_calculator(self):
        try:
            subprocess.Popen("start calc", shell=True)
            self.speak("Opening calculator.")
        except Exception as e:
            logging.error(f"open_calculator failed: {str(e)}")
            self.speak("Failed to open calculator.")

    def minimize_windows(self):
        pyautogui.hotkey("win", "m")
        self.speak("Minimizing windows.")

    def open_search(self):
        pyautogui.hotkey("win", "q")
        self.speak("Opening search.")

    def open_news_widget(self):
        pyautogui.hotkey("win", "w")
        self.speak("Opening news.")

    def open_run_command(self):
        pyautogui.hotkey("win", "r")
        self.speak("Opening Run.")

    def open_settings(self):
        pyautogui.hotkey("win", "i")
        self.speak("Opening settings.")

    def open_about_settings(self):
        self.open_uri("ms-settings:about")
        self.speak("Opening about settings.")

    def open_project_screen(self):
        pyautogui.hotkey("win", "p")
        self.speak("Opening projection options.")

    def open_performance_settings(self):
        # As documented: uses Win+S (note: this normally opens search)
        pyautogui.hotkey("win", "s")
        self.speak("Opening performance settings.")

    def open_feedback_hub(self):
        pyautogui.hotkey("win", "f")
        self.speak("Opening feedback hub.")

    def open_game_bar(self):
        pyautogui.hotkey("win", "g")
        self.speak("Opening Xbox Game Bar.")

    def open_voice_typing(self):
        pyautogui.hotkey("win", "h")
        self.speak("Opening voice typing.")

    def open_connect_panel(self):
        pyautogui.hotkey("win", "k")
        self.speak("Opening connect panel.")

    def lock_screen(self):
        pyautogui.hotkey("win", "l")

    def open_quick_menu(self):
        pyautogui.hotkey("win", "x")
        self.speak("Opening quick menu.")

    def open_cortana(self):
        pyautogui.hotkey("win", "c")
        self.speak("Opening Cortana.")

    def open_clipboard_history(self):
        pyautogui.hotkey("win", "v")
        self.speak("Opening clipboard history.")

    def open_notifications(self):
        pyautogui.hotkey("win", "n")
        self.speak("Opening notifications.")

    def select_all_text(self):
        pyautogui.hotkey("ctrl", "a")
        self.speak("Selected all.")

    def open_25(self):
        webbrowser.open("https://itiadmission.gujarat.gov.in/")
        self.speak("Opening ITI admission website.")

    def open_google(self):
        webbrowser.open("https://www.google.com/")
        self.speak("Opening Google.")

    def open_youtube(self):
        webbrowser.open("https://youtube.com/")
        self.speak("Opening YouTube.")

    def open_instagram(self):
        webbrowser.open("https://www.instagram.com/")
        self.speak("Opening Instagram.")

    def open_chatbox(self):
        webbrowser.open("https://hack.chat/?Isha")
        self.speak("Opening chat box.")

    def download_picture(self):
        webbrowser.open("https://pixabay.com/")
        self.speak("Opening Pixabay.")

    def download_instagram_reel(self):
        webbrowser.open("https://igram.world/reels-downloader/")
        self.speak("Opening reels downloader.")

    def open_uri(self, uri: str):
        try:
            subprocess.Popen(f"start {uri}", shell=True)
        except Exception as e:
            logging.error(f"open_uri failed: {str(e)} | uri={uri}")

    # ----------------------------
    # Popups: Settings / Apps
    # ----------------------------
    def toggle_settings(self):
        if self.settings_popup and tk.Toplevel.winfo_exists(self.settings_popup):
            self.settings_popup.destroy()
            self.settings_popup = None
        else:
            self.show_settings_popup()

    def show_settings_popup(self):
        self.settings_popup = tk.Toplevel(self.root)
        self.settings_popup.title("Settings")
        self.settings_popup.geometry("300x300")
        self.settings_popup.configure(bg="#1e1e1e")

        lb = tk.Listbox(self.settings_popup, bg="#111111", fg="white")
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        items = sorted(self.settings_display_to_cmd.keys())
        for item in items:
            lb.insert(tk.END, item)

        lb.bind("<Double-Button-1>", lambda e: self.on_settings_select(lb))

    def on_settings_select(self, listbox: tk.Listbox):
        sel = listbox.curselection()
        if not sel:
            return
        label = listbox.get(sel[0])
        cmd = self.settings_display_to_cmd.get(label)
        if cmd:
            self.open_uri(cmd)
            self.speak(f"Opening {label}.")

    def toggle_apps(self):
        if self.apps_popup and tk.Toplevel.winfo_exists(self.apps_popup):
            self.apps_popup.destroy()
            self.apps_popup = None
        else:
            self.show_apps_popup()

    def show_apps_popup(self):
        self.apps_popup = tk.Toplevel(self.root)
        self.apps_popup.title("Apps")
        self.apps_popup.geometry("300x300")
        self.apps_popup.configure(bg="#1e1e1e")

        lb = tk.Listbox(self.apps_popup, bg="#111111", fg="white")
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        items = sorted(self.apps_display_to_cmd.keys())
        for item in items:
            lb.insert(tk.END, item)

        lb.bind("<Double-Button-1>", lambda e: self.on_apps_select(lb))

    def on_apps_select(self, listbox: tk.Listbox):
        sel = listbox.curselection()
        if not sel:
            return
        label = listbox.get(sel[0])
        cmd = self.apps_display_to_cmd.get(label)
        if cmd:
            subprocess.Popen(f"start {cmd}", shell=True)
            self.speak(f"Opening {label}.")

    # ----------------------------
    # File explorer helpers
    # ----------------------------
    def open_file_explorer(self):
        try:
            subprocess.Popen("explorer", shell=True)
            self.speak("Opening File Explorer.")
        except Exception as e:
            logging.error(f"open_file_explorer failed: {str(e)}")
            self.speak("Failed to open File Explorer.")

    def open_downloads(self):
        try:
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            subprocess.Popen(f'explorer "{downloads}"', shell=True)
            self.speak("Opening Downloads.")
        except Exception as e:
            logging.error(f"open_downloads failed: {str(e)}")
            self.speak("Failed to open Downloads.")

    # ----------------------------
    # System info
    # ----------------------------
    def btr(self):
        try:
            b = psutil.sensors_battery()
            if b is None:
                self.speak("Battery information is not available.")
                return
            self.speak(f"Battery is at {b.percent} percent.")
        except Exception as e:
            logging.error(f"btr failed: {str(e)}")
            self.speak("Unable to read battery status.")

    # ----------------------------
    # About window
    # ----------------------------
    def show_about(self):
        top = tk.Toplevel(self.root)
        top.title("About Isha Assistant")
        top.geometry("420x260")
        top.configure(bg="#1e1e1e")

        txt = (
            "IshaAssistant - Intelligent System for Human Assistance\n\n"
            "Features:\n"
            "- Voice + Text commands\n"
            "- App and Settings shortcuts\n"
            "- Time/Date/Weather/Math\n"
            "- Music playback\n"
            "- Windows automation hotkeys\n\n"
            "Note: This assistant is Windows-focused."
        )
        lbl = tk.Label(top, text=txt, bg="#1e1e1e", fg="white", justify="left", wraplength=400)
        lbl.pack(padx=10, pady=10, anchor="w")

        self.speak("This is Isha Assistant. A desktop helper for voice and text commands.")

    # ----------------------------
    # Settings/apps command handler
    # ----------------------------
    def handle_settings_apps_commands(self, command: str):
        # Supports: "open X" or "X" where X is in commands_dict
        m = re.match(r"open\s+(.+)", command)
        key = m.group(1).strip() if m else command.strip()

        if key in self.commands_dict:
            target = self.commands_dict[key]
            # Decide whether to open via start or browser
            if str(target).startswith("http"):
                webbrowser.open(target)
            elif str(target).startswith("ms-settings:"):
                self.open_uri(target)
            else:
                subprocess.Popen(f"start {target}", shell=True)
            self.speak(f"Opening {key}.")
            return

        self.speak("Command not recognized.")

    # ----------------------------
    # Speech output (threaded)
    # ----------------------------
    def speak(self, text: str):
        def _run():
            try:
                # Try to stop any current speech safely
                try:
                    self.engine.stop()
                except Exception:
                    pass

                self.engine.say(text)
                self.engine.runAndWait()
                logging.info(f"SPOKE: {text}")
            except Exception as e:
                logging.error(f"TTS failed: {str(e)}")
                # Fallback: at least show in chat
                self.chat_box_insert(f"Isha: {text}")

        threading.Thread(target=_run, daemon=True).start()

    # ----------------------------
    # Music / media controls
    # ----------------------------
    def play_song(self):
        online = self.check_internet()
        if online:
            # Random YouTube links (simple)
            urls = [
                "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
                "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
                "https://www.youtube.com/watch?v=fRh_vgS2dFE"
            ]
            url = random.choice(urls)
            webbrowser.open(url)
            time.sleep(2)
            try:
                pyautogui.press("k")  # play/pause on YouTube
            except Exception:
                pass
            self.speak("Playing a song on YouTube.")
        else:
            music_dir = os.path.join(os.path.expanduser("~"), "Music")
            files = glob.glob(os.path.join(music_dir, "*.mp3")) + glob.glob(os.path.join(music_dir, "*.wav"))
            if not files:
                self.speak("No local music found in your Music folder.")
                return
            song = random.choice(files)
            os.startfile(song)
            self.speak("Playing local music.")

    def stop_song(self):
        try:
            pyautogui.press("k")  # YouTube play/pause
        except Exception:
            pass
        self.speak("Stopping song.")

    def mute_unmute(self):
        try:
            pyautogui.press("m")
        except Exception:
            pass
        self.speak("Toggling mute.")

    def full_screen(self):
        try:
            pyautogui.press("f")
        except Exception:
            pass
        self.speak("Toggling full screen.")

    def toggle_caption(self):
        try:
            pyautogui.press("c")
        except Exception:
            pass
        self.speak("Toggling captions.")

    # ----------------------------
    # Camera disconnect placeholder
    # ----------------------------
    def came2(self):
        # Placeholder: documentation said phone camera on/off; "came2()" missing.
        # We'll just press a key that might be mapped by the user.
        self.speak("Phone camera on command received.")
        try:
            pyautogui.press("e")
        except Exception:
            pass

    def cmaw21(self):
        try:
            pyautogui.press("q")
        except Exception:
            pass
        self.speak("Phone camera off.")

    # ----------------------------
    # Weather (wttr.in) with cache file
    # ----------------------------
    def get_weather(self):
        cache_file = "weather_cache.txt"
        online = self.check_internet()

        if online:
            city = simpledialog.askstring("Weather", "Enter city name:")
            if not city:
                self.speak("Cancelled.")
                return

            try:
                url = f"https://wttr.in/{requests.utils.quote(city)}?format=%C+%t"
                resp = urllib.request.urlopen(url, timeout=5).read().decode("utf-8", errors="ignore").strip()

                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(f"{time.time()}\n{city}\n{resp}\n")

                msg = f"Weather in {city}: {resp}"
                self.chat_box_insert(f"Isha: {msg}")
                self.speak(msg)
            except Exception as e:
                logging.error(f"get_weather online failed: {str(e)}")
                self.speak("Failed to fetch weather.")
        else:
            # Offline: read cache if <1 hour old
            try:
                if not os.path.exists(cache_file):
                    self.speak("No internet and no cached weather available.")
                    return
                with open(cache_file, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                ts = float(lines[0])
                city = lines[1] if len(lines) > 1 else "your city"
                weather = lines[2] if len(lines) > 2 else ""
                if time.time() - ts <= 3600:
                    msg = f"(Cached) Weather in {city}: {weather}"
                    self.chat_box_insert(f"Isha: {msg}")
                    self.speak(msg)
                else:
                    self.speak("Cached weather is too old. Please connect to the internet.")
            except Exception as e:
                logging.error(f"get_weather offline failed: {str(e)}")
                self.speak("Unable to read cached weather.")

    # ----------------------------
    # WhatsApp (pywhatkit)
    # ----------------------------
    def open_whatsapp(self):
        online = self.check_internet()
        if not online:
            self.speak("No internet. Opening Notepad instead.")
            subprocess.Popen("notepad", shell=True)
            return

        contact = simpledialog.askstring("WhatsApp", "Enter phone number with country code (e.g., +91XXXXXXXXXX):")
        if not contact:
            self.speak("Cancelled.")
            return
        message = simpledialog.askstring("WhatsApp", "Enter message:")
        if message is None:
            self.speak("Cancelled.")
            return

        try:
            webbrowser.open("https://web.whatsapp.com/")
            time.sleep(5)
            pywhatkit.sendwhatmsg_instantly(contact, message, wait_time=15, tab_close=True, close_time=3)
            self.speak("Message sent.")
        except Exception as e:
            logging.error(f"open_whatsapp failed: {str(e)}")
            self.speak("Failed to send WhatsApp message.")

    # ----------------------------
    # Selenium Instagram login (optional)
    # ----------------------------
    def login_instagram(self):
        # To keep the script runnable without forcing selenium install,
        # import selenium only when needed.
        def _run():
            try:
                username = simpledialog.askstring("Instagram Login", "Enter Instagram username:")
                password = simpledialog.askstring("Instagram Login", "Enter Instagram password:", show="*")
                if not username or not password:
                    self.speak("Cancelled.")
                    return

                self.speak("Starting Instagram login. This may take a moment.")

                from selenium import webdriver
                from selenium.webdriver.common.by import By
                from selenium.webdriver.common.keys import Keys
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service

                options = webdriver.ChromeOptions()
                options.add_argument("--disable-notifications")

                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver.get("https://www.instagram.com/accounts/login/")
                time.sleep(5)

                user_in = driver.find_element(By.NAME, "username")
                pass_in = driver.find_element(By.NAME, "password")

                user_in.send_keys(username)
                pass_in.send_keys(password)
                pass_in.send_keys(Keys.ENTER)

                time.sleep(8)
                self.speak("Login attempt completed. Please check the browser.")
            except Exception as e:
                logging.error(f"login_instagram failed: {str(e)}")
                self.speak("Instagram login failed. Selenium and ChromeDriver may be required.")

        threading.Thread(target=_run, daemon=True).start()

    # ----------------------------
    # Small-talk
    # ----------------------------
    def hello(self):
        replies = [
            "Hello! How can I help you?",
            "Hi! Tell me what you want to do.",
            "Namaste! What can I do for you?"
        ]
        r = random.choice(replies)
        self.chat_box_insert(f"Isha: {r}")
        self.speak(r)

    def thank_you_reply(self):
        replies = [
            "You're welcome!",
            "No problem!",
            "Anytime!"
        ]
        r = random.choice(replies)
        self.chat_box_insert(f"Isha: {r}")
        self.speak(r)

    def what_is_your_name(self):
        r = "My name is Isha Assistant."
        self.chat_box_insert(f"Isha: {r}")
        self.speak(r)

    def morningtime(self):
        r = "Good morning! I hope you have a great day."
        self.chat_box_insert(f"Isha: {r}")
        self.speak(r)

    # ----------------------------
    # Power actions
    # ----------------------------
    def shutdown_pc(self):
        self.speak("Your computer will shut down in 10 seconds.")
        time.sleep(10)
        subprocess.Popen("shutdown /s /t 1", shell=True)

    def restart_pc(self):
        self.speak("Your computer will restart in 10 seconds.")
        time.sleep(10)
        subprocess.Popen("shutdown /r /t 1", shell=True)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = IshaAssistant(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        print(f"Error: Application failed to start: {str(e)}")
```
