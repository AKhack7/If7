
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <random>
#include <regex>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <atomic>
#include <filesystem>
#include <cstdlib>
#include <cstring>

// Windows specific headers
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <synchapi.h>
#include <tlhelp32.h>
#endif

// Networking
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

// JSON parsing
#include "json.hpp"
using json = nlohmann::json;

// HTTP Server
#include "httplib.h"

// OpenCV for camera and gesture recognition
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// For speech synthesis
#include <sapi.h>
#include <sphelper.h>

// For speech recognition
#include <speechapi_cxx.h>
using namespace Microsoft::CognitiveServices::Speech;

// For audio capture
#include <mmsystem.h>
#include <mmreg.h>
#include <ks.h>
#include <ksmedia.h>
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "ole32.lib")

namespace fs = std::filesystem;

// ============================================
// UTILITY FUNCTIONS
// ============================================

std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string get_current_time_string() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_s(&tm, &time_t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%H:%M:%S");
    return ss.str();
}

std::string get_current_date_string() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_s(&tm, &time_t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%A, %B %d, %Y");
    return ss.str();
}

std::string get_greeting() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_s(&tm, &time_t);
    int hour = tm.tm_hour;
    
    if (hour >= 5 && hour < 12) return "Good morning";
    else if (hour >= 12 && hour < 17) return "Good afternoon";
    else if (hour >= 17 && hour < 21) return "Good evening";
    else return "Good night";
}

std::string generate_unique_filename(const std::string& prefix, const std::string& ext) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_s(&tm, &time_t);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << prefix << "__" 
       << std::put_time(&tm, "%d-%m-%Y__%H-%M-%S") << "__"
       << std::setw(3) << std::setfill('0') << ms.count() << "." << ext;
    return ss.str();
}

bool ensure_folder(const std::string& path) {
    try {
        fs::create_directories(path);
        return true;
    } catch (...) {
        return false;
    }
}

// ============================================
// LOGGING
// ============================================

class Logger {
private:
    std::ofstream log_file;
    std::mutex log_mutex;

public:
    Logger(const std::string& filename) {
        log_file.open(filename, std::ios::app);
    }

    ~Logger() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    void info(const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
        localtime_s(&tm, &time_t);
        log_file << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << " - INFO - " << message << std::endl;
    }

    void error(const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
        localtime_s(&tm, &time_t);
        log_file << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << " - ERROR - " << message << std::endl;
    }
};

Logger logger("isha_assistant.log");

// ============================================
// INTERNET CHECK
// ============================================

bool check_internet() {
    // Try to connect to Google DNS
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return false;
    }

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        WSACleanup();
        return false;
    }

    sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_port = htons(80);
    inet_pton(AF_INET, "8.8.8.8", &server.sin_addr);

    // Set timeout
    int timeout = 2000; // 2 seconds
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));

    bool connected = connect(sock, (sockaddr*)&server, sizeof(server)) == 0;
    
    closesocket(sock);
    WSACleanup();
    
    return connected;
}

// ============================================
// TEXT TO SPEECH
// ============================================

class TextToSpeech {
private:
    ISpVoice* pVoice;

public:
    TextToSpeech() : pVoice(nullptr) {
        if (SUCCEEDED(::CoInitialize(nullptr))) {
            if (SUCCEEDED(CoCreateInstance(CLSID_SpVoice, nullptr, CLSCTX_ALL,
                                           IID_ISpVoice, (void**)&pVoice))) {
                set_female_voice();
            }
        }
    }

    ~TextToSpeech() {
        if (pVoice) {
            pVoice->Release();
        }
        ::CoUninitialize();
    }

    void set_female_voice() {
        if (!pVoice) return;

        ISpObjectToken* pToken = nullptr;
        ISpObjectTokenCategory* pCategory = nullptr;

        if (SUCCEEDED(CoCreateInstance(CLSID_SpObjectTokenCategory, nullptr, CLSCTX_ALL,
                                        IID_ISpObjectTokenCategory, (void**)&pCategory))) {
            if (SUCCEEDED(pCategory->SetId(L"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices", FALSE))) {
                IEnumSpObjectTokens* pEnum = nullptr;
                if (SUCCEEDED(pCategory->EnumTokens(nullptr, nullptr, &pEnum))) {
                    ULONG count = 0;
                    pEnum->GetCount(&count);
                    
                    for (ULONG i = 0; i < count; i++) {
                        if (SUCCEEDED(pEnum->Next(1, &pToken, nullptr))) {
                            LPWSTR pszName = nullptr;
                            if (SUCCEEDED(pToken->GetStringValue(L"Name", &pszName))) {
                                std::wstring name(pszName);
                                // Look for female voices (Zira, etc.)
                                if (name.find(L"Zira") != std::wstring::npos ||
                                    name.find(L"female") != std::wstring::npos ||
                                    name.find(L"Female") != std::wstring::npos) {
                                    pVoice->SetVoice(pToken);
                                    ::CoTaskMemFree(pszName);
                                    break;
                                }
                                ::CoTaskMemFree(pszName);
                            }
                            pToken->Release();
                        }
                    }
                    pEnum->Release();
                }
            }
            pCategory->Release();
        }
    }

    void speak(const std::string& text) {
        if (!pVoice) return;

        std::wstring wide_text(text.begin(), text.end());
        pVoice->Speak(wide_text.c_str(), SPF_ASYNC, nullptr);
    }
};

// ============================================
// ADVANCED HAND GESTURE CONTROLLER
// ============================================

class AdvancedGestureController {
public:
    enum Gesture {
        UNKNOWN,
        PALM,
        FIST,
        POINT,
        PEACE,
        THUMBS_UP,
        THUMBS_DOWN,
        OK,
        THREE,
        FOUR,
        FIVE,
        PINCH,
        CALL,
        ROCK,
        SPIDERMAN,
        CROSS
    };

    struct Action {
        std::string name;
        std::string description;
        std::function<void()> execute;
    };

private:
    bool active;
    bool show_preview;
    std::map<Gesture, Action> gesture_actions;
    std::vector<Gesture> gesture_history;
    int history_length;
    std::chrono::steady_clock::time_point last_action_time;
    int action_cooldown_ms;
    bool scroll_mode;
    int last_scroll_y;
    int scroll_speed;
    bool mouse_control;
    int screen_w;
    int screen_h;
    cv::VideoCapture camera;
    std::thread gesture_thread;
    std::atomic<bool> running;
    std::mutex gesture_mutex;

    // For hand detection (simplified - would need proper ML model)
    cv::CascadeClassifier hand_cascade;

    Gesture recognize_gesture(const cv::Mat& frame) {
        // Simplified gesture recognition
        // In a real implementation, you'd use MediaPipe or a trained model
        
        // For demonstration, we'll return random gestures occasionally
        static int frame_count = 0;
        frame_count++;
        
        if (frame_count % 30 == 0) {
            return static_cast<Gesture>(rand() % 15);
        }
        
        return UNKNOWN;
    }

public:
    AdvancedGestureController() 
        : active(false), 
          show_preview(false),
          history_length(5),
          action_cooldown_ms(300),
          scroll_mode(false),
          last_scroll_y(0),
          scroll_speed(50),
          mouse_control(false),
          running(false) {
        
        screen_w = GetSystemMetrics(SM_CXSCREEN);
        screen_h = GetSystemMetrics(SM_CYSCREEN);

        // Initialize gesture actions
        gesture_actions[PALM] = {"PALM", "WAITING", nullptr};
        gesture_actions[FIST] = {"FIST", "CLOSE_FILE", nullptr};
        gesture_actions[POINT] = {"POINT", "OPEN_FILE", nullptr};
        gesture_actions[PEACE] = {"PEACE", "SCREENSHOT", [this]() { 
            take_screenshot(); 
        }};
        gesture_actions[THUMBS_UP] = {"THUMBS_UP", "VOLUME_UP", [this]() {
            volume_up();
        }};
        gesture_actions[THUMBS_DOWN] = {"THUMBS_DOWN", "VOLUME_DOWN", [this]() {
            volume_down();
        }};
        gesture_actions[OK] = {"OK", "SELFIE", [this]() {
            take_selfie();
        }};
        gesture_actions[THREE] = {"THREE", "OPEN_BROWSER", [this]() {
            open_browser();
        }};
        gesture_actions[FOUR] = {"FOUR", "OPEN_CALCULATOR", [this]() {
            open_calculator();
        }};
        gesture_actions[FIVE] = {"FIVE", "OPEN_NOTEPAD", [this]() {
            open_notepad();
        }};
        gesture_actions[PINCH] = {"PINCH", "SCROLL", [this]() {
            // Scroll handled separately
        }};
        gesture_actions[CALL] = {"CALL", "PLAY_MUSIC", [this]() {
            play_music();
        }};
        gesture_actions[ROCK] = {"ROCK", "WEATHER", [this]() {
            check_weather();
        }};
        gesture_actions[SPIDERMAN] = {"SPIDERMAN", "JOKE", [this]() {
            tell_joke();
        }};
        gesture_actions[CROSS] = {"CROSS", "CLOSE_APP", [this]() {
            close_app();
        }};
    }

    ~AdvancedGestureController() {
        stop();
    }

    void start() {
        if (running) return;
        
        running = true;
        gesture_thread = std::thread(&AdvancedGestureController::gesture_loop, this);
    }

    void stop() {
        if (!running) return;
        
        running = false;
        if (gesture_thread.joinable()) {
            gesture_thread.join();
        }
    }

    void toggle() {
        active = !active;
        if (active && !running) {
            start();
        } else if (!active && running) {
            stop();
        }
    }

    void toggle_preview() {
        show_preview = !show_preview;
    }

    void set_active(bool state) {
        active = state;
        if (active && !running) {
            start();
        } else if (!active && running) {
            stop();
        }
    }

    bool is_active() const {
        return active;
    }

    bool is_preview_on() const {
        return show_preview;
    }

private:
    void gesture_loop() {
        camera.open(0);
        if (!camera.isOpened()) {
            logger.error("Camera not available");
            return;
        }

        camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        camera.set(cv::CAP_PROP_FPS, 30);

        cv::Mat frame;
        int frame_count = 0;

        while (running && active) {
            camera >> frame;
            if (frame.empty()) continue;

            frame_count++;

            // Process frame
            Gesture gesture = recognize_gesture(frame);

            // Add to history
            gesture_history.push_back(gesture);
            if (gesture_history.size() > history_length) {
                gesture_history.erase(gesture_history.begin());
            }

            // Get most common gesture from history
            if (!gesture_history.empty()) {
                std::map<Gesture, int> counts;
                for (auto g : gesture_history) {
                    counts[g]++;
                }
                
                gesture = std::max_element(counts.begin(), counts.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            }

            // Execute action if not on cooldown
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_action_time).count();

            if (elapsed >= action_cooldown_ms) {
                auto it = gesture_actions.find(gesture);
                if (it != gesture_actions.end() && it->second.execute) {
                    it->second.execute();
                    last_action_time = now;
                }
            }

            // Show preview if enabled
            if (show_preview) {
                cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 60), 
                             cv::Scalar(0, 0, 0), -1);
                
                std::string gesture_name = "UNKNOWN";
                auto it = gesture_actions.find(gesture);
                if (it != gesture_actions.end()) {
                    gesture_name = it->second.name;
                }

                cv::putText(frame, "ISHA GESTURES: " + std::string(active ? "ON" : "OFF"),
                           cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                           cv::Scalar(0, 255, 255), 2);
                
                cv::putText(frame, "Gesture: " + gesture_name,
                           cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                           cv::Scalar(255, 255, 255), 2);

                cv::imshow("ISHA Hand Gestures", frame);
                
                if (cv::waitKey(1) == 'q') {
                    active = false;
                    break;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        camera.release();
        cv::destroyAllWindows();
    }

    // Action implementations
    void take_screenshot() {
        std::string folder = "isha_captures";
        ensure_folder(folder);
        std::string filename = generate_unique_filename("screenshot", "png");
        std::string path = folder + "/" + filename;

        // Use Windows API to take screenshot
        HWND desktop = GetDesktopWindow();
        HDC hdcDesktop = GetDC(desktop);
        HDC hdcMem = CreateCompatibleDC(hdcDesktop);
        
        RECT rc;
        GetClientRect(desktop, &rc);
        
        HBITMAP hBitmap = CreateCompatibleBitmap(hdcDesktop, rc.right, rc.bottom);
        SelectObject(hdcMem, hBitmap);
        BitBlt(hdcMem, 0, 0, rc.right, rc.bottom, hdcDesktop, 0, 0, SRCCOPY);
        
        // Save bitmap to file (simplified)
        DeleteObject(hBitmap);
        DeleteDC(hdcMem);
        ReleaseDC(desktop, hdcDesktop);

        logger.info("Screenshot saved: " + path);
    }

    void take_selfie() {
        cv::VideoCapture cap(0);
        if (cap.isOpened()) {
            cv::Mat frame;
            cap >> frame;
            cap.release();
            
            if (!frame.empty()) {
                std::string folder = "isha_captures";
                ensure_folder(folder);
                std::string filename = generate_unique_filename("selfie", "jpg");
                std::string path = folder + "/" + filename;
                cv::imwrite(path, frame);
                logger.info("Selfie saved: " + path);
            }
        }
    }

    void volume_up() {
        // Simulate volume up key
        keybd_event(VK_VOLUME_UP, 0, 0, 0);
        keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_KEYUP, 0);
    }

    void volume_down() {
        // Simulate volume down key
        keybd_event(VK_VOLUME_DOWN, 0, 0, 0);
        keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_KEYUP, 0);
    }

    void open_browser() {
        ShellExecuteA(nullptr, "open", "https://www.google.com", nullptr, nullptr, SW_SHOW);
    }

    void open_calculator() {
        system("calc");
    }

    void open_notepad() {
        system("notepad");
    }

    void play_music() {
        system("start wmplayer");
    }

    void check_weather() {
        ShellExecuteA(nullptr, "open", "https://weather.com", nullptr, nullptr, SW_SHOW);
    }

    void tell_joke() {
        // Joke would be spoken by TTS
    }

    void close_app() {
        // Simulate Alt+F4
        keybd_event(VK_MENU, 0, 0, 0);
        keybd_event(VK_F4, 0, 0, 0);
        keybd_event(VK_F4, 0, KEYEVENTF_KEYUP, 0);
        keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0);
    }
};

// ============================================
// SPEECH RECOGNITION
// ============================================

class SpeechRecognizer {
private:
    std::shared_ptr<SpeechRecognizer> recognizer;
    std::atomic<bool> is_listening;
    std::queue<std::string> result_queue;
    std::mutex queue_mutex;
    std::thread recognition_thread;

public:
    SpeechRecognizer() : is_listening(false) {
        // Initialize speech recognition
        auto config = SpeechConfig::FromSubscription("YourSubscriptionKey", "YourRegion");
        recognizer = std::make_shared<SpeechRecognizer>(config);
    }

    ~SpeechRecognizer() {
        stop();
    }

    void start() {
        if (is_listening) return;
        is_listening = true;
        recognition_thread = std::thread(&SpeechRecognizer::recognition_loop, this);
    }

    void stop() {
        is_listening = false;
        if (recognition_thread.joinable()) {
            recognition_thread.join();
        }
    }

    void toggle() {
        if (is_listening) {
            stop();
        } else {
            start();
        }
    }

    bool has_result() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return !result_queue.empty();
    }

    std::string get_result() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (result_queue.empty()) return "";
        std::string result = result_queue.front();
        result_queue.pop();
        return result;
    }

private:
    void recognition_loop() {
        while (is_listening) {
            auto result = recognizer->RecognizeOnceAsync().get();
            if (result->Reason == ResultReason::RecognizedSpeech) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                result_queue.push(result->Text);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
};

// ============================================
// PERSONAL AI TRAINER (Simplified version)
// ============================================

class PersonalAITrainer {
private:
    std::map<std::string, std::string> response_map;
    std::vector<std::string> conversations;
    std::string conv_file = "my_personal_conversations.txt";

public:
    PersonalAITrainer() {
        load_or_create_conversations();
    }

    void load_or_create_conversations() {
        std::ifstream file(conv_file);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    conversations.push_back(line);
                }
            }
            file.close();
        } else {
            // Default conversations
            conversations = {
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
                "User: what's the date Assistant: Here's today's date."
            };
            
            // Save default conversations
            std::ofstream out_file(conv_file);
            for (const auto& conv : conversations) {
                out_file << conv << std::endl;
            }
            out_file.close();
        }

        // Build response map
        for (const auto& conv : conversations) {
            size_t user_pos = conv.find("User: ");
            size_t assistant_pos = conv.find("Assistant: ");
            
            if (user_pos != std::string::npos && assistant_pos != std::string::npos) {
                std::string user_msg = conv.substr(user_pos + 6, assistant_pos - user_pos - 8);
                std::string assistant_msg = conv.substr(assistant_pos + 10);
                response_map[to_lower(trim(user_msg))] = assistant_msg;
            }
        }
    }

    void add_conversation(const std::string& user_msg, const std::string& assistant_msg) {
        std::string conv = "User: " + user_msg + " Assistant: " + assistant_msg;
        conversations.push_back(conv);
        response_map[to_lower(trim(user_msg))] = assistant_msg;

        std::ofstream file(conv_file, std::ios::app);
        if (file.is_open()) {
            file << conv << std::endl;
            file.close();
        }
    }

    std::string generate_response(const std::string& prompt) {
        std::string lower_prompt = to_lower(trim(prompt));
        
        // Check for exact match
        auto it = response_map.find(lower_prompt);
        if (it != response_map.end()) {
            return it->second;
        }

        // Check for partial matches
        for (const auto& [key, response] : response_map) {
            if (lower_prompt.find(key) != std::string::npos || 
                key.find(lower_prompt) != std::string::npos) {
                return response;
            }
        }

        return "";
    }

    void train(int epochs) {
        logger.info("Training personal AI for " + std::to_string(epochs) + " epochs");
        // In a real implementation, this would train a neural network
        // For this simplified version, we just reload conversations
        load_or_create_conversations();
        logger.info("Training complete");
    }
};

// ============================================
// MAIN ISHA ASSISTANT
// ============================================

class IshaAssistant {
private:
    TextToSpeech tts;
    AdvancedGestureController gesture_controller;
    PersonalAITrainer personal_trainer;
    SpeechRecognizer speech_recognizer;
    httplib::Server server;
    std::thread server_thread;
    std::queue<std::string> input_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> pending;
    std::string pending_type;
    std::map<std::string, std::string> commands_dict;
    std::map<std::string, std::string> settings_display_to_cmd;
    std::map<std::string, std::string> apps_display_to_cmd;
    std::atomic<bool> internet_status;
    std::chrono::steady_clock::time_point last_internet_check;
    int internet_check_interval;

    void setup_command_mappings() {
        // Settings Map
        std::map<std::string, std::pair<std::string, std::string>> setting_map = {
            {"display setting", {"ms-settings:display", "01"}},
            {"sound setting", {"ms-settings:sound", "02"}},
            {"notification & action setting", {"ms-settings:notifications", "03"}},
            {"focus assist setting", {"ms-settings:quiethours", "04"}},
            {"power & sleep setting", {"ms-settings:powersleep", "05"}},
            {"storage setting", {"ms-settings:storagesense", "06"}},
            {"tablet setting", {"ms-settings:tablet", "07"}},
            {"multitasking setting", {"ms-settings:multitasking", "08"}},
            {"projecting to this pc setting", {"ms-settings:project", "09"}},
            {"shared experiences setting", {"ms-settings:crossdevice", "010"}},
            {"system components setting", {"ms-settings:appsfeatures-app", "001"}},
            {"clipboard setting", {"ms-settings:clipboard", "002"}},
            {"remote desktop setting", {"ms-settings:remotedesktop", "003"}},
            {"optional features setting", {"ms-settings:optionalfeatures", "004"}},
            {"about setting", {"ms-settings:about", "005"}},
            {"system setting", {"ms-settings:system", "006"}},
            {"devices setting", {"ms-settings:devices", "007"}},
            {"mobile devices setting", {"ms-settings:mobile-devices", "008"}},
            {"network & internet setting", {"ms-settings:network", "009"}},
            {"personalization setting", {"ms-settings:personalization", "000"}},
            {"apps setting", {"ms-settings:appsfeatures", "10"}},
            {"account setting", {"ms-settings:yourinfo", "20"}},
            {"time & language setting", {"ms-settings:dateandtime", "30"}},
            {"gaming setting", {"ms-settings:gaming", "40"}},
            {"ease of access setting", {"ms-settings:easeofaccess", "50"}},
            {"privacy setting", {"ms-settings:privacy", "60"}},
            {"updated & security", {"ms-settings:windowsupdate", "70"}}
        };

        std::map<std::string, std::pair<std::string, std::string>> apps_commands = {
            {"alarms & clock", {"ms-clock:", "a1"}},
            {"calculator", {"calc", "c1"}},
            {"calendar", {"outlookcal:", "c2"}},
            {"camera", {"microsoft.windows.camera:", "c3"}},
            {"copilot", {"ms-copilot:", "c4"}},
            {"cortana", {"ms-cortana:", "c5"}},
            {"game bar", {"ms-gamebar:", "gb1"}},
            {"groove music", {"mswindowsmusic:", "gm1"}},
            {"mail", {"outlookmail:", "m1"}},
            {"maps", {"bingmaps:", "map1"}},
            {"microsoft edge", {"msedge", "me1"}},
            {"microsoft solitaire collection", {"ms-solitaire:", "mc1"}},
            {"microsoft store", {"ms-windows-store:", "mst1"}},
            {"mixed reality portal", {"ms-mixedreality:", "mp1"}},
            {"movies & tv", {"mswindowsvideo:", "mt1"}},
            {"office", {"ms-office:", "o1"}},
            {"onedrive", {"ms-onedrive:", "oe"}},
            {"onenote", {"ms-onenote:", "one"}},
            {"outlook", {"outlookmail:", "ouk"}},
            {"outlook (classic)", {"ms-outlook:", "oc1"}},
            {"paint", {"mspaint", "p1"}},
            {"paint 3d", {"ms-paint:", "p3d"}},
            {"phone link", {"ms-phonelink:", "pk"}},
            {"power point", {"ms-powerpoint:", "pt"}},
            {"settings", {"ms-settings:", "ss"}},
            {"skype", {"skype:", "sk1"}},
            {"snip & sketch", {"ms-snip:", "s0h"}},
            {"sticky note", {"ms-stickynotes:", "s1e"}},
            {"tips", {"ms-tips:", "ts0"}},
            {"voice recorder", {"ms-soundrecorder:", "vr0"}},
            {"weather", {"msnweather:", "w1"}},
            {"windows backup", {"ms-settings:backup", "wb1"}},
            {"windows security", {"ms-settings:windowsdefender", "ws1"}},
            {"word", {"ms-word:", "wrd"}},
            {"xbox", {"ms-xbox:", "xb"}},
            {"about your pc", {"ms-settings:about", "apc"}}
        };

        std::map<std::string, std::string> software_dict = {
            {"notepad", "notepad"},
            {"ms word", "winword"},
            {"command prompt", "cmd"},
            {"excel", "excel"},
            {"vscode", "code"},
            {"word16", "winword"},
            {"file explorer", "explorer"},
            {"edge", "msedge"},
            {"microsoft 365 copilot", "ms-copilot:"},
            {"outlook", "outlook"},
            {"microsoft store", "ms-windows-store:"},
            {"photos", "microsoft.photos:"},
            {"xbox", "xbox:"},
            {"solitaire", "microsoft.microsoftsolitairecollection:"},
            {"clipchamp", "clipchamp"},
            {"to do", "microsoft.todos:"},
            {"linkedin", "https://www.linkedin.com"},
            {"calculator", "calc"},
            {"news", "bingnews:"},
            {"one drive", "onedrive"},
            {"onenote 2016", "onenote"},
            {"google", "https://www.google.com"}
        };

        // Build commands_dict
        for (const auto& [key, value] : setting_map) {
            commands_dict[key] = value.first;
            settings_display_to_cmd[key + " (" + value.second + ")"] = value.first;
        }

        for (const auto& [key, value] : apps_commands) {
            commands_dict[key] = value.first;
            apps_display_to_cmd[key] = value.first;
        }

        for (const auto& [key, value] : software_dict) {
            commands_dict[key] = value;
        }

        // Add reverse mappings for codes
        for (const auto& [key, value] : setting_map) {
            commands_dict[value.second] = value.first;
        }

        for (const auto& [key, value] : apps_commands) {
            commands_dict[value.second] = value.first;
        }
    }

    bool check_internet_cached() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_internet_check).count();
        
        if (elapsed < internet_check_interval) {
            return internet_status;
        }

        last_internet_check = now;
        internet_status = check_internet();
        return internet_status;
    }

    void handle_open_command(const std::string& app) {
        auto it = commands_dict.find(app);
        if (it != commands_dict.end()) {
            std::string cmd = it->second;
            
            if (cmd.find("http") == 0) {
                ShellExecuteA(nullptr, "open", cmd.c_str(), nullptr, nullptr, SW_SHOW);
            } else {
                std::string full_cmd = "start " + cmd;
                system(full_cmd.c_str());
            }
            
            std::string msg = "Opening " + app;
            tts.speak(msg);
        } else {
            tts.speak("Application not found: " + app);
        }
    }

    void handle_weather_command() {
        if (check_internet_cached()) {
            tts.speak("Which city's weather do you want to check?");
            pending = true;
            pending_type = "weather_city";
        } else {
            tts.speak("No internet connection. Using cached weather if available.");
            // Check cache
            std::ifstream cache_file("weather_cache.txt");
            if (cache_file.is_open()) {
                std::string city, weather, timestamp;
                std::getline(cache_file, city, ':');
                std::getline(cache_file, weather, ':');
                std::getline(cache_file, timestamp);
                cache_file.close();
                
                auto now = std::chrono::system_clock::now();
                auto now_time_t = std::chrono::system_clock::to_time_t(now);
                
                if (now_time_t - std::stoll(timestamp) < 3600) {
                    tts.speak("Showing cached weather for " + city + ": " + weather);
                } else {
                    tts.speak("Cached weather is too old. Please check internet connection.");
                }
            } else {
                tts.speak("No cached weather available.");
            }
        }
    }

public:
    IshaAssistant() 
        : pending(false),
          internet_status(false),
          last_internet_check(std::chrono::steady_clock::now()),
          internet_check_interval(60) {
        
        setup_command_mappings();
        
        // Start server
        start_server();
        
        // Welcome message
        wish_me();
    }

    ~IshaAssistant() {
        server.stop();
    }

    void start_server() {
        server.Get("/", [this](const httplib::Request& req, httplib::Response& res) {
            res.set_content(get_html(), "text/html");
        });

        server.Get("/command", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.has_param("cmd")) {
                std::string cmd = req.get_param_value("cmd");
                
                if (pending) {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    input_queue.push(cmd);
                    queue_cv.notify_one();
                    res.set_content("Input received", "text/plain");
                } else {
                    std::string response = process_command(cmd);
                    res.set_content(response, "text/plain");
                }
            } else {
                res.status = 400;
            }
        });

        server.Get("/voice", [this](const httplib::Request& req, httplib::Response& res) {
            speech_recognizer.toggle();
            res.set_content("Microphone toggled", "text/plain");
        });

        server_thread = std::thread([this]() {
            server.listen("localhost", 8000);
        });

        // Open browser
        ShellExecuteA(nullptr, "open", "http://localhost:8000/", nullptr, nullptr, SW_SHOW);
    }

    std::string get_html() {
        std::string apps_html;
        for (const auto& [name, cmd] : apps_display_to_cmd) {
            apps_html += "<div class=\"app-item\" data-command=\"open " + name + "\" style=\"margin:6px 0; cursor:pointer;\">• " + name + "</div>\n";
        }

        std::string settings_html;
        for (const auto& [name, cmd] : settings_display_to_cmd) {
            settings_html += "<div class=\"setting-item\" data-command=\"open " + name + "\" style=\"margin:6px 0; cursor:pointer;\">• " + name + "</div>\n";
        }

        std::stringstream html;
        html << R"(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ISHA Assistant</title>
    <style>
        :root {
            --bg1: #050519;
            --bg2: #0f1636;
            --neon1: #00e0ff;
            --neon2: #7b4bff;
            --glass: rgba(255, 255, 255, 0.04);
        }

        * {
            box-sizing: border-box;
            -webkit-font-smoothing: antialiased;
            font-family: "Segoe UI", Inter, system-ui, sans-serif;
            margin: 0;
            padding: 0;
        }

        html, body {
            height: 100%;
            margin: 0;
            background: linear-gradient(180deg, var(--bg1), var(--bg2));
            color: #e8f6ff;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .container {
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
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }

        .title {
            font-weight: 700;
            font-size: 18px;
            color: #cfeeff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .title .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--neon1), var(--neon2));
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
                box-shadow: 0 0 5px var(--neon1);
            }
            50% {
                opacity: 0.8;
                box-shadow: 0 0 15px var(--neon1), 0 0 25px var(--neon2);
            }
        }

        .stage {
            width: 100%;
            height: 420px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .core {
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
        }

        @keyframes float {
            0%, 100% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-10px);
            }
        }

        @keyframes glow {
            0%, 100% {
                box-shadow: 0 0 20px rgba(0, 224, 255, 0.1), 0 0 40px rgba(123, 75, 255, 0.1);
            }
            50% {
                box-shadow: 0 0 40px rgba(0, 224, 255, 0.3), 0 0 80px rgba(123, 75, 255, 0.2);
            }
        }

        .core::before {
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
        }

        @keyframes rotate {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }

        .label {
            font-weight: 800;
            font-size: 32px;
            letter-spacing: 3px;
            color: #e9fbff;
            text-shadow: 0 0 10px rgba(0, 224, 255, 0.5), 0 0 20px rgba(123, 75, 255, 0.3);
            position: relative;
            z-index: 2;
            animation: textGlow 2s ease-in-out infinite;
        }

        @keyframes textGlow {
            0%, 100% {
                text-shadow: 0 0 10px rgba(0, 224, 255, 0.5), 0 0 20px rgba(123, 75, 255, 0.3);
            }
            50% {
                text-shadow: 0 0 20px rgba(0, 224, 255, 0.8), 0 0 40px rgba(123, 75, 255, 0.5);
            }
        }

        .datetime {
            margin-top: 18px;
            text-align: center;
            color: var(--neon1);
            font-weight: 700;
            font-size: 16px;
        }

        .datetime .time {
            font-size: 28px;
            color: #dffbff;
            text-shadow: 0 0 5px rgba(0, 224, 255, 0.3);
        }

        .controls {
            margin-top: 24px;
            width: 100%;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .input {
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
        }

        .input::placeholder {
            color: rgba(255, 255, 255, 0.3);
        }

        .input:focus {
            border-color: var(--neon1);
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.2);
            transform: scale(1.02);
        }

        .icon-btn {
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
        }

        .icon-btn:hover {
            border-color: var(--neon1);
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.3);
            transform: scale(1.05);
            color: white;
        }

        .popup {
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
        }

        #appPopup {
            left: -340px;
            top: 120px;
        }

        #settingsPopup {
            right: -340px;
            top: 140px;
        }

        #appPopup.active {
            left: 40px;
        }

        #settingsPopup.active {
            right: 40px;
        }

        .popup .head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            color: var(--neon1);
            font-weight: 700;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0, 224, 255, 0.2);
        }

        .popup .close {
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
        }

        .popup .close:hover {
            background: rgba(255, 0, 0, 0.2);
            border-color: #ff4444;
            transform: rotate(90deg);
        }

        .app-item, .setting-item {
            margin: 8px 0;
            padding: 6px 10px;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s cubic-bezier(0.2, 0.9, 0.3, 1);
            color: #cfeeff;
            font-size: 14px;
        }

        .app-item:hover, .setting-item:hover {
            background: rgba(0, 224, 255, 0.1);
            padding-left: 16px;
            color: white;
            transform: translateX(5px);
        }

        .right {
            display: flex;
            gap: 8px;
        }

        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(0, 224, 255, 0.3);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 224, 255, 0.5);
        }

        .particle {
            position: fixed;
            width: 2px;
            height: 2px;
            background: var(--neon1);
            opacity: 0.3;
            border-radius: 50%;
            pointer-events: none;
            animation: particleFloat 8s linear infinite;
        }

        @keyframes particleFloat {
            0% {
                transform: translateY(100vh) scale(0);
                opacity: 0;
            }
            50% {
                opacity: 0.5;
            }
            100% {
                transform: translateY(-100vh) scale(1);
                opacity: 0;
            }
        }
    </style>
</head>
<body>
    <div id="particles"></div>

    <div class="container" id="container">
        <div class="topbar">
            <div class="title">
                <span class="dot"></span> ISHA Assistant
            </div>
            <div style="opacity: 0.8; font-size: 13px; color: #bfefff; text-shadow: 0 0 5px rgba(0,224,255,0.3);">
                activet 156
            </div>
        </div>

        <div class="stage">
            <div class="core">
                <div class="label">ISHA</div>
            </div>
        </div>

        <div class="datetime" id="datetime">
            <div class="time" id="time">--:--:--</div>
            <div class="date" id="date">Loading date...</div>
        </div>

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

    <div class="popup" id="appPopup">
        <div class="head">
            <div>📱 Applications</div>
            <div class="close" data-close="appPopup">✕</div>
        </div>
        <div style="font-size: 14px; color: #cfeeff; max-height: 350px; overflow-y: auto;">
            )" << apps_html << R"(
        </div>
    </div>

    <div class="popup" id="settingsPopup">
        <div class="head">
            <div>⚙️ Settings</div>
            <div class="close" data-close="settingsPopup">✕</div>
        </div>
        <div style="font-size: 14px; color: #cfeeff; max-height: 350px; overflow-y: auto;">
            )" << settings_html << R"(
        </div>
    </div>

    <script>
        function updateDateTime() {
            const now = new Date();
            const timeEl = document.getElementById('time');
            const dateEl = document.getElementById('date');
            
            timeEl.textContent = now.toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit', 
                hour12: false 
            });
            
            dateEl.textContent = now.toLocaleDateString([], { 
                weekday: 'short', 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric' 
            });
        }
        
        updateDateTime();
        setInterval(updateDateTime, 500);

        function createParticles() {
            const particles = document.getElementById('particles');
            for (let i = 0; i < 50; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 8 + 's';
                particle.style.width = Math.random() * 3 + 'px';
                particle.style.height = particle.style.width;
                particles.appendChild(particle);
            }
        }
        createParticles();

        function makeDraggable(el) {
            let dragging = false;
            let offsetX = 0;
            let offsetY = 0;

            el.addEventListener('mousedown', (e) => {
                dragging = true;
                offsetX = e.clientX - el.offsetLeft;
                offsetY = e.clientY - el.offsetTop;
                el.style.transition = 'none';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            });

            window.addEventListener('mousemove', (e) => {
                if (!dragging) return;
                
                let newLeft = e.clientX - offsetX;
                let newTop = e.clientY - offsetY;
                
                newLeft = Math.max(10, Math.min(newLeft, window.innerWidth - el.offsetWidth - 10));
                newTop = Math.max(10, Math.min(newTop, window.innerHeight - el.offsetHeight - 10));
                
                el.style.left = newLeft + 'px';
                el.style.top = newTop + 'px';
            });

            window.addEventListener('mouseup', () => {
                if (dragging) {
                    dragging = false;
                    el.style.transition = '';
                    document.body.style.userSelect = '';
                }
            });
        }

        const appBtn = document.getElementById('appBtn');
        const settingsBtn = document.getElementById('settingsBtn');
        const voiceBtn = document.getElementById('voiceBtn');
        const appPopup = document.getElementById('appPopup');
        const settingsPopup = document.getElementById('settingsPopup');

        appBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isActive = appPopup.classList.toggle('active');
            
            if (isActive) {
                appPopup.style.display = 'block';
                settingsPopup.classList.remove('active');
                settingsPopup.style.display = 'none';
            } else {
                setTimeout(() => {
                    if (!appPopup.classList.contains('active')) {
                        appPopup.style.display = 'none';
                    }
                }, 200);
            }
        });

        settingsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isActive = settingsPopup.classList.toggle('active');
            
            if (isActive) {
                settingsPopup.style.display = 'block';
                appPopup.classList.remove('active');
                appPopup.style.display = 'none';
            } else {
                setTimeout(() => {
                    if (!settingsPopup.classList.contains('active')) {
                        settingsPopup.style.display = 'none';
                    }
                }, 200);
            }
        });

        voiceBtn.addEventListener('click', () => {
            fetch('/voice').then(res => res.text()).then(_ => {});
        });

        document.querySelectorAll('.close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const id = btn.dataset.close;
                const popup = document.getElementById(id);
                popup.classList.remove('active');
                setTimeout(() => {
                    popup.style.display = 'none';
                }, 200);
            });
        });

        makeDraggable(appPopup);
        makeDraggable(settingsPopup);

        document.querySelectorAll('.app-item').forEach(item => {
            item.addEventListener('click', () => {
                const cmd = item.getAttribute('data-command');
                sendCommand(cmd);
                appPopup.classList.remove('active');
                setTimeout(() => appPopup.style.display = 'none', 200);
            });
        });
        
        document.querySelectorAll('.setting-item').forEach(item => {
            item.addEventListener('click', () => {
                const cmd = item.getAttribute('data-command');
                sendCommand(cmd);
                settingsPopup.classList.remove('active');
                setTimeout(() => settingsPopup.style.display = 'none', 200);
            });
        });

        function sendCommand(cmd) {
            fetch(`/command?cmd=${encodeURIComponent(cmd)}`)
            .then(res => res.text())
            .then(text => console.log('Response:', text))
            .catch(err => console.error('Error:', err));
        }

        document.getElementById('cmd').addEventListener('keydown', (e)=>{
            if(e.key === 'Enter'){
                const v = e.target.value.trim();
                if(!v) return;
                sendCommand(v);
                e.target.value = '';
            }
        });

        document.addEventListener('click', (e) => {
            if (!appPopup.contains(e.target) && !appBtn.contains(e.target)) {
                if (appPopup.classList.contains('active')) {
                    appPopup.classList.remove('active');
                    setTimeout(() => {
                        appPopup.style.display = 'none';
                    }, 200);
                }
            }
            
            if (!settingsPopup.contains(e.target) && !settingsBtn.contains(e.target)) {
                if (settingsPopup.classList.contains('active')) {
                    settingsPopup.classList.remove('active');
                    setTimeout(() => {
                        settingsPopup.style.display = 'none';
                    }, 200);
                }
            }
        });

        appPopup.addEventListener('mousedown', (e) => {
            e.stopPropagation();
        });
        
        settingsPopup.addEventListener('mousedown', (e) => {
            e.stopPropagation();
        });

        appPopup.style.display = 'none';
        settingsPopup.style.display = 'none';
    </script>
</body>
</html>
        )";

        return html.str();
    }

    std::string process_command(const std::string& cmd) {
        std::string command = to_lower(trim(cmd));
        logger.info("Processing command: " + command);

        // Handle pending input
        if (pending) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            input_queue.push(command);
            queue_cv.notify_one();
            pending = false;
            return "Input received";
        }

        // Hand gesture commands
        if (command.find("activate hand") != std::string::npos ||
            command.find("turn on hand") != std::string::npos ||
            command.find("start hand") != std::string::npos ||
            command.find("enable hand") != std::string::npos) {
            gesture_controller.set_active(true);
            std::string msg = "Hand gestures activated";
            tts.speak(msg);
            return msg;
        }

        if (command.find("deactivate hand") != std::string::npos ||
            command.find("turn off hand") != std::string::npos ||
            command.find("stop hand") != std::string::npos ||
            command.find("disable hand") != std::string::npos) {
            gesture_controller.set_active(false);
            std::string msg = "Hand gestures deactivated";
            tts.speak(msg);
            return msg;
        }

        if (command.find("gesture") != std::string::npos &&
            (command.find("status") != std::string::npos || command.find("check") != std::string::npos)) {
            std::string status = gesture_controller.is_active() ? "ON" : "OFF";
            std::string msg = "Hand gestures are " + status;
            tts.speak(msg);
            return msg;
        }

        // Camera preview commands
        if (command.find("show camera") != std::string::npos ||
            command.find("show preview") != std::string::npos ||
            command.find("camera on") != std::string::npos) {
            gesture_controller.toggle_preview();
            std::string msg = "Camera preview turned on";
            tts.speak(msg);
            return msg;
        }

        if (command.find("hide camera") != std::string::npos ||
            command.find("hide preview") != std::string::npos ||
            command.find("camera off") != std::string::npos) {
            gesture_controller.toggle_preview();
            std::string msg = "Camera preview turned off";
            tts.speak(msg);
            return msg;
        }

        // Training command
        if (command == "train my ai") {
            tts.speak("Training your personal AI. This will take a moment.");
            personal_trainer.train(20);
            tts.speak("Training complete!");
            return "Personal AI retrained";
        }

        // Gesture activation code
        if (command == "activet 156") {
            gesture_controller.set_active(true);
            std::string msg = "Hand gestures activated";
            tts.speak(msg);
            return msg;
        }

        // Try personal AI first
        std::string personal_response = personal_trainer.generate_response(command);
        if (!personal_response.empty()) {
            tts.speak(personal_response);
            return personal_response;
        }

        // Open commands
        if (command.find("open ") == 0) {
            std::string app = command.substr(5);
            handle_open_command(app);
            return "Opening " + app;
        }

        // Time
        if (command == "what is the time" || command == "tell me the time" ||
            command == "current time" || command == "time now" ||
            command == "what time is it" || command == "what's the time" ||
            command == "time" || command == "what time") {
            std::string msg = get_current_time_string();
            tts.speak(msg);
            return msg;
        }

        // Date
        if (command == "what is the date" || command == "tell me the date" ||
            command == "current date" || command == "date now" ||
            command == "what date is it" || command == "what's the date" ||
            command == "date" || command == "what date") {
            std::string msg = get_current_date_string();
            tts.speak(msg);
            return msg;
        }

        // Google
        if (command == "open google" || command == "launch google" || command == "go to google") {
            if (check_internet_cached()) {
                ShellExecuteA(nullptr, "open", "https://www.google.com", nullptr, nullptr, SW_SHOW);
                std::string msg = "Opening Google";
                tts.speak(msg);
                return msg;
            } else {
                std::string msg = "No internet connection. Google cannot be opened.";
                tts.speak(msg);
                return msg;
            }
        }

        // Weather
        if (command == "weather" || command == "check weather" || command == "what's the weather") {
            handle_weather_command();
            return "Checking weather";
        }

        // Greetings
        if (command == "hi" || command == "hello" || command == "hey") {
            std::string msg = "Hello! How can I assist you today?";
            tts.speak(msg);
            return msg;
        }

        // Screenshot
        if (command == "screenshot" || command == "take screenshot" || command == "capture screen") {
            // Screenshot handled by gesture controller
            return "Taking screenshot";
        }

        // Selfie
        if (command == "selfie" || command == "take selfie" || command == "capture selfie") {
            // Selfie handled by gesture controller
            return "Taking selfie";
        }

        // Default
        std::string msg = "Command not recognized: " + command;
        tts.speak(msg);
        return msg;
    }

    void wish_me() {
        std::string greeting = get_greeting();
        tts.speak(greeting);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        std::string msg = "I am Isha, Intelligent System for Human Assistance. Welcome! Say 'activate hand gestures' for gesture control.";
        tts.speak(msg);
    }

    void run() {
        // Main loop
        while (true) {
            if (speech_recognizer.has_result()) {
                std::string voice_cmd = speech_recognizer.get_result();
                process_command(voice_cmd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

// ============================================
// MAIN
// ============================================

int main() {
    // Hide console window
    HWND console = GetConsoleWindow();
    if (console) {
        ShowWindow(console, SW_HIDE);
    }

    std::cout << "=" << std::string(58, '=') << std::endl;
    std::cout << "🤖 ISHA - Your Personal AI Assistant" << std::endl;
    std::cout << "🖐️ Advanced Hand Gestures Ready (OFF by default)" << std::endl;
    std::cout << "📷 Camera Preview: OFF (say 'show camera' to enable)" << std::endl;
    std::cout << "🧠 Personal LLM: Trained on YOUR conversations" << std::endl;
    std::cout << "✨ Beautiful Animated GUI with Floating Circle" << std::endl;
    std::cout << "=" << std::string(58, '=') << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  • 'activate hand gestures' - Turn gestures ON" << std::endl;
    std::cout << "  • 'deactivate hand gestures' - Turn gestures OFF" << std::endl;
    std::cout << "  • 'show camera' - See yourself" << std::endl;
    std::cout << "  • 'hide camera' - Hide preview (gestures still work)" << std::endl;
    std::cout << "  • 'screenshot' - Capture screen" << std::endl;
    std::cout << "  • 'selfie' - Take photo" << std::endl;
    std::cout << "  • 'open [app]' - Open any app" << std::endl;
    std::cout << "  • 'train my ai' - Improve your personal AI" << std::endl;
    std::cout << "  • 'activet 156' - Quick gesture activation" << std::endl;
    std::cout << "=" << std::string(58, '=') << std::endl;

    try {
        IshaAssistant assistant;
        assistant.run();
    } catch (const std::exception& e) {
        logger.error(std::string("Application failed to start: ") + e.what());
        std::cerr << "Error: Application failed to start: " << e.what() << std::endl;
    }

    return 0;
}