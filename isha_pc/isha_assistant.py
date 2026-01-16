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
import requests

from isha_llm import IshaLLM


logging.basicConfig(
    filename="isha_assistant.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class IshaAssistant:
    def __init__(self, root: tk.Tk):
        self.root = root

        # Window setup
        self.root.title("Isha Assistant")
        self.root.geometry("600x400")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)
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
        self.internet_check_interval = 10

        # GUI refs
        self.chat_box = None
        self.input_box = None
        self.status_label = None
        self.mic_btn = None
        self.settings_popup = None
        self.apps_popup = None

        # LLM simulation
        self.llm = IshaLLM(memory_file="isha_memory.json")

        # TTS
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        self.engine.setProperty("volume", 0.9)

        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = None

        # Microphone init
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Default microphone initialized successfully.")
        except Exception as e:
            logging.error(f"Microphone init failed: {str(e)}")
            self.microphone = None

        # Build GUI
        self.create_gui()

        # TTS voice
        self.set_female_voice()

        # Maps
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

        self.commands_dict = {}
        for k, v in self.SETTING_MAP.items():
            self.commands_dict[k] = v[0]
        for k, v in self.apps_commands.items():
            self.commands_dict[k] = v[0]
        for k, v in self.software_dict.items():
            self.commands_dict[k] = v

        self.settings_display_to_cmd = {k.title(): v[0] for k, v in self.SETTING_MAP.items()}
        self.apps_display_to_cmd = {k.title(): v[0] for k, v in self.apps_commands.items()}

        # Greet
        self.wish_me()

        # Keybinds
        self.root.bind("<Return>", self.handle_double_enter)
        self.root.bind("<Control-m>", lambda e: self.toggle_voice())
        self.root.bind("<Control-s>", lambda e: self.toggle_settings())
        self.root.bind("<Control-a>", lambda e: self.toggle_apps())
        self.root.bind("<Control-f>", lambda e: self.open_file_explorer())
        self.root.bind("<Control-d>", lambda e: self.open_downloads())

        # Internet status loop
        self.update_internet_status()

    # ----------------------------
    # GUI
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

        style.configure("MicOn.TButton", background="#2ecc71", foreground="black")
        style.configure("MicOff.TButton", background="#e74c3c", foreground="white")
        style.configure("Quick.TButton", background="#3b3b3b", foreground="white")

        self.chat_box = scrolledtext.ScrolledText(
            self.root, height=10, width=60,
            bg="#111111", fg="white", insertbackground="white", wrap=tk.WORD
        )
        self.chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.input_box = ttk.Entry(bottom_frame, width=50)
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_box.bind("<Return>", self.process_text_input)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.mic_btn = ttk.Button(btn_frame, text="Mic", style="MicOff.TButton", command=self.toggle_voice)
        self.mic_btn.pack(side=tk.LEFT, padx=4)

        ttk.Button(btn_frame, text="Settings", style="Quick.TButton", command=self.toggle_settings).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Apps", style="Quick.TButton", command=self.toggle_apps).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="File M", style="Quick.TButton", command=self.open_file_explorer).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Downloads", style="Quick.TButton", command=self.open_downloads).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="About", style="Quick.TButton", command=self.show_about).pack(side=tk.RIGHT, padx=4)

        self.status_label = ttk.Label(self.root, text="Internet: Checking...")
        self.status_label.pack(anchor="w", padx=12, pady=(0, 8))

    # ----------------------------
    # Helpers
    # ----------------------------
    def chat_box_insert(self, text: str):
        self.chat_box.insert(tk.END, text + "\n")
        self.chat_box.see(tk.END)

    def handle_double_enter(self, event):
        now = time.time()
        if now - self.last_enter_time <= 0.5:
            self.input_box.focus_set()
        self.last_enter_time = now

    def set_female_voice(self):
        try:
            voices = self.engine.getProperty("voices")
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
            self.speak("Initializing voice")
        except Exception as e:
            logging.error(f"Voice setup failed: {str(e)}")

    # ----------------------------
    # Internet
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
        self.status_label.configure(text=f"Internet: {'Online' if online else 'Offline'}")
        self.root.after(10000, self.update_internet_status)

    # ----------------------------
    # Voice
    # ----------------------------
    def select_microphone(self):
        try:
            names = sr.Microphone.list_microphone_names()
            if not names:
                self.speak("No microphones detected.")
                return
            msg = "Available microphones:\n\n" + "\n".join([f"{i}: {n}" for i, n in enumerate(names)])
            idx_str = simpledialog.askstring("Select Microphone", msg + "\n\nEnter mic index:")
            if idx_str is None:
                return
            idx = int(idx_str)
            self.microphone = sr.Microphone(device_index=idx)
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            self.speak("Microphone selected successfully.")
        except Exception as e:
            logging.error(f"Microphone selection failed: {str(e)}")
            self.speak("Microphone selection failed.")

    def toggle_voice(self):
        if self.microphone is None:
            self.select_microphone()

        self.is_listening = not self.is_listening
        self.mic_btn.configure(style="MicOn.TButton" if self.is_listening else "MicOff.TButton")
        self.speak("Microphone is now on" if self.is_listening else "Microphone is now off")

        if self.is_listening:
            threading.Thread(target=self.listen_voice, daemon=True).start()

    def listen(self):
        if self.microphone is None:
            txt = simpledialog.askstring("Voice Input", "Mic not available.\nType command:")
            return txt.lower().strip() if txt else None

        attempts = 0
        while attempts < 5:
            attempts += 1
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=15)
                query = self.recognizer.recognize_google(audio)
                return query.lower().strip() if query else None
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as e:
                logging.error(f"Voice recognition failed: {str(e)}")
                txt = simpledialog.askstring("Voice Input", "Voice failed.\nType command:")
                return txt.lower().strip() if txt else None
        return None

    def listen_voice(self):
        while self.is_listening:
            cmd = self.listen()
            if cmd:
                self.root.after(0, lambda c=cmd: self.process_voice_command(c))
            time.sleep(1)

    def process_voice_command(self, command: str):
        self.input_box.delete(0, tk.END)
        self.input_box.insert(0, command)
        self.process_command(command)

    # ----------------------------
    # Speak (threaded)
    # ----------------------------
    def speak(self, text: str):
        def _run():
            try:
                try:
                    self.engine.stop()
                except Exception:
                    pass
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS failed: {str(e)}")
                self.chat_box_insert(f"Isha: {text}")

        threading.Thread(target=_run, daemon=True).start()

    # ----------------------------
    # Welcome
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
        time.sleep(1)

    # ----------------------------
    # Text input
    # ----------------------------
    def process_text_input(self, event=None):
        cmd = self.input_box.get().strip().lower()
        self.input_box.delete(0, tk.END)
        if cmd:
            self.process_command(cmd)

    # ----------------------------
    # Commands
    # ----------------------------
    def process_command(self, command: str):
        try:
            online = self.check_internet()
            logging.info(f"Command: {command} | internet={online}")
            self.chat_box_insert(f"You: {command}")

            if command in {"time", "what is the time", "time batao"}:
                return self.get_time()

            if command in {"date", "what is the date", "date batao"}:
                return self.get_date()

            if command.startswith("solve "):
                return self.solve_math(command.replace("solve ", "", 1))

            if any(op in command for op in ["+", "-", "*", "/", "sin", "cos", "tan", "sqrt", "pi"]) and any(ch.isdigit() for ch in command):
                return self.solve_math(command)

            if command in {"open calculator", "calculator"}:
                return self.open_calculator()

            if command in {"open downloads", "downloads"}:
                return self.open_downloads()

            if command in {"open file explorer", "file manager", "open files"}:
                return self.open_file_explorer()

            if command in {"minimize windows", "minimize"}:
                pyautogui.hotkey("win", "m")
                self.speak("Minimizing windows.")
                return

            if command in {"search", "open search"}:
                pyautogui.hotkey("win", "q")
                self.speak("Opening search.")
                return

            if command in {"settings", "open settings"}:
                pyautogui.hotkey("win", "i")
                self.speak("Opening settings.")
                return

            if command in {"weather"}:
                return self.get_weather()

            if command in {"play music", "play song"}:
                return self.play_song()

            if command in {"stop music", "stop song"}:
                pyautogui.press("k")
                self.speak("Stopping song.")
                return

            if command in {"battery"}:
                return self.btr()

            if command in {"google", "open google"}:
                webbrowser.open("https://www.google.com/")
                self.speak("Opening Google.")
                return

            if command in {"youtube", "open youtube"}:
                webbrowser.open("https://youtube.com/")
                self.speak("Opening YouTube.")
                return

            if command in {"instagram", "open instagram"}:
                webbrowser.open("https://www.instagram.com/")
                self.speak("Opening Instagram.")
                return

            if command in {"whatsapp", "open whatsapp"}:
                return self.open_whatsapp()

            if command in {"shutdown"}:
                return self.shutdown_pc()

            if command in {"restart"}:
                return self.restart_pc()

            if command in {"about"}:
                return self.show_about()

            # Try opening via settings/apps dictionary
            if self.handle_settings_apps_commands(command):
                return

            # ✅ FINAL FALLBACK: Simulated LLM
            reply, intent = self.llm.generate_response(command)
            self.chat_box_insert(f"Isha: {reply}")
            self.speak(reply)

        except Exception as e:
            logging.error(f"process_command error: {str(e)}")
            self.speak("Sorry, something went wrong.")

    def handle_settings_apps_commands(self, command: str) -> bool:
        m = re.match(r"open\s+(.+)", command)
        key = m.group(1).strip() if m else command.strip()

        if key in self.commands_dict:
            target = self.commands_dict[key]
            if str(target).startswith("http"):
                webbrowser.open(target)
            elif str(target).startswith("ms-settings:"):
                subprocess.Popen(f"start {target}", shell=True)
            else:
                subprocess.Popen(f"start {target}", shell=True)
            self.speak(f"Opening {key}.")
            return True

        return False

    # ----------------------------
    # Core features
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

    def solve_math(self, expression: str):
        try:
            expr = expression.replace(" ", "")
            val = sympify(expr, locals={"sin": sin, "cos": cos, "tan": tan, "sqrt": sqrt, "pi": pi})
            valf = float(val.evalf())
            out = str(int(valf)) if abs(valf - int(valf)) < 1e-12 else str(round(valf, 6))
            msg = f"The answer is {out}"
            self.chat_box_insert(f"Isha: {msg}")
            self.speak(msg)
        except Exception as e:
            logging.error(f"solve_math error: {str(e)} | expr={expression}")
            self.speak("I could not solve that expression.")

    def open_calculator(self):
        subprocess.Popen("start calc", shell=True)
        self.speak("Opening calculator.")

    def open_file_explorer(self):
        subprocess.Popen("explorer", shell=True)
        self.speak("Opening File Explorer.")

    def open_downloads(self):
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        subprocess.Popen(f'explorer "{downloads}"', shell=True)
        self.speak("Opening Downloads.")

    def btr(self):
        b = psutil.sensors_battery()
        if b is None:
            self.speak("Battery info not available.")
            return
        self.speak(f"Battery is at {b.percent} percent.")

    def play_song(self):
        if self.check_internet():
            urls = [
                "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
                "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
                "https://www.youtube.com/watch?v=fRh_vgS2dFE"
            ]
            webbrowser.open(random.choice(urls))
            time.sleep(2)
            try:
                pyautogui.press("k")
            except Exception:
                pass
            self.speak("Playing a song on YouTube.")
        else:
            music_dir = os.path.join(os.path.expanduser("~"), "Music")
            files = glob.glob(os.path.join(music_dir, "*.mp3")) + glob.glob(os.path.join(music_dir, "*.wav"))
            if not files:
                self.speak("No local music found in Music folder.")
                return
            os.startfile(random.choice(files))
            self.speak("Playing local music.")

    def get_weather(self):
        cache_file = "weather_cache.txt"
        if self.check_internet():
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
                logging.error(f"weather error: {str(e)}")
                self.speak("Failed to fetch weather.")
        else:
            try:
                if not os.path.exists(cache_file):
                    self.speak("No internet and no cached weather.")
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
                    self.speak("Cached weather is too old. Please connect internet.")
            except Exception as e:
                logging.error(f"weather cache error: {str(e)}")
                self.speak("Unable to read cached weather.")

    def open_whatsapp(self):
        if not self.check_internet():
            self.speak("No internet. Opening Notepad.")
            subprocess.Popen("notepad", shell=True)
            return

        contact = simpledialog.askstring("WhatsApp", "Enter phone with country code (e.g., +91XXXXXXXXXX):")
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
            logging.error(f"whatsapp error: {str(e)}")
            self.speak("Failed to send WhatsApp message.")

    def show_about(self):
        top = tk.Toplevel(self.root)
        top.title("About Isha Assistant")
        top.geometry("420x260")
        top.configure(bg="#1e1e1e")

        txt = (
            "IshaAssistant - Desktop Assistant\n\n"
            "Features:\n"
            "- Voice + Text commands\n"
            "- Time/Date/Math/Weather\n"
            "- Open Apps/Settings\n"
            "- Music + WhatsApp\n"
            "- Simulated LLM + Memory (remember/recall)\n"
        )
        tk.Label(top, text=txt, bg="#1e1e1e", fg="white", justify="left", wraplength=400).pack(
            padx=10, pady=10, anchor="w"
        )
        self.speak("This is Isha Assistant.")

    def shutdown_pc(self):
        self.speak("Your computer will shut down in 10 seconds.")
        time.sleep(10)
        subprocess.Popen("shutdown /s /t 1", shell=True)

    def restart_pc(self):
        self.speak("Your computer will restart in 10 seconds.")
        time.sleep(10)
        subprocess.Popen("shutdown /r /t 1", shell=True)


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = IshaAssistant(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        print(f"Error: Application failed to start: {str(e)}")
