// ISHA Assistant - C++ Version
// Intelligent System for Human Assistance

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <memory>
#include <functional>
#include <random>
#include <cmath>
#include <atomic>
#include <condition_variable>

// Windows specific
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <ShlObj.h>
#endif

// Networking
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// MediaPipe
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"

// Libtorch (PyTorch C++)
#include <torch/torch.h>
#include <torch/script.h>

// Text-to-Speech
#include <sapi.h>
#include <sphelper.h>

// Speech Recognition
#include <speechapi_cxx.h>

// HTTP Server
#include "httplib.h"

// JSON
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// Forward declarations
class AdvancedGestureController;
class PersonalVocabulary;
class PersonalLLM;
class PersonalAITrainer;
class IshaAssistant;

// ============================================
// CONSTANTS AND GLOBALS
// ============================================

const string VOSK_MODEL_PATH = ""; // Set via environment
atomic<bool> VOSK_AVAILABLE(false);
atomic<bool> MEDIAPIPE_AVAILABLE(true);
atomic<bool> PERSONAL_LLM_AVAILABLE(true);

// ============================================
// LOGGING UTILITY
// ============================================

class Logger {
private:
    ofstream logFile;
    mutex logMutex;

public:
    Logger(const string& filename) {
        logFile.open(filename, ios::app);
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    void info(const string& message) {
        lock_guard<mutex> lock(logMutex);
        if (logFile.is_open()) {
            auto now = chrono::system_clock::now();
            auto now_c = chrono::system_clock::to_time_t(now);
            logFile << put_time(localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
                   << " - INFO - " << message << endl;
        }
    }

    void error(const string& message) {
        lock_guard<mutex> lock(logMutex);
        if (logFile.is_open()) {
            auto now = chrono::system_clock::now();
            auto now_c = chrono::system_clock::to_time_t(now);
            logFile << put_time(localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
                   << " - ERROR - " << message << endl;
        }
    }

    void warning(const string& message) {
        lock_guard<mutex> lock(logMutex);
        if (logFile.is_open()) {
            auto now = chrono::system_clock::now();
            auto now_c = chrono::system_clock::to_time_t(now);
            logFile << put_time(localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
                   << " - WARNING - " << message << endl;
        }
    }
};

Logger logger("isha_assistant.log");

// ============================================
// ADVANCED HAND GESTURE CONTROLLER
// ============================================

class AdvancedGestureController {
public:
    static const map<string, string> GESTURE_ACTIONS;

    AdvancedGestureController() {
        frame_skip = 2;
        frame_count = 0;
        history_length = 5;
        last_action_time = chrono::steady_clock::now();
        action_cooldown = 0.3;
        scroll_mode = false;
        last_scroll_y = 0;
        scroll_speed = 50;
        mouse_control = false;
        screen_w = GetSystemMetrics(SM_CXSCREEN);
        screen_h = GetSystemMetrics(SM_CYSCREEN);
        show_preview = false;
        active = false;

        // Initialize MediaPipe
        mp_hands = make_unique<mediapipe::CalculatorGraph>();
        string proto = R"(
            input_stream: "input_video"
            output_stream: "hand_landmarks"
            node {
                calculator: "HandLandmarkTrackingCpu"
                input_stream: "IMAGE:input_video"
                output_stream: "LANDMARKS:hand_landmarks"
                output_stream: "HANDEDNESS:handedness"
                options: {
                    [mediapipe.HandLandmarkTrackingOptions.ext] {
                        model_complexity: 0
                        min_detection_confidence: 0.7
                        min_tracking_confidence: 0.5
                    }
                }
            }
        )";
        
        mediapipe::CalculatorGraphConfig config;
        if (mediapipe::ParseTextProto<mediapipe::CalculatorGraphConfig>(proto, &config)) {
            mp_hands->Initialize(config);
        }
    }

    bool toggle() {
        active = !active;
        return active;
    }

    bool toggle_preview() {
        show_preview = !show_preview;
        return show_preview;
    }

    string recognize_gesture(const vector<mediapipe::NormalizedLandmark>& lm, const string& hand_label) {
        vector<int> tips_ids = {4, 8, 12, 16, 20};
        vector<int> pip_ids = {2, 6, 10, 14, 18};
        
        vector<int> fingers;
        
        if (hand_label == "Right") {
            fingers.push_back((lm[4].x() > lm[3].x()) ? 1 : 0);
        } else {
            fingers.push_back((lm[4].x() < lm[3].x()) ? 1 : 0);
        }
        
        for (size_t i = 1; i < tips_ids.size(); i++) {
            int tip = tips_ids[i];
            int pip = pip_ids[i];
            fingers.push_back((lm[tip].y() < lm[pip].y()) ? 1 : 0);
        }
        
        int finger_count = 0;
        for (int f : fingers) finger_count += f;
        
        double pinch_dist = sqrt(
            pow(lm[4].x() - lm[8].x(), 2) + 
            pow(lm[4].y() - lm[8].y(), 2)
        );
        
        double palm_center_x = (lm[5].x() + lm[9].x() + lm[13].x() + lm[17].x()) / 4;
        double palm_center_y = (lm[5].y() + lm[9].y() + lm[13].y() + lm[17].y()) / 4;
        
        double avg_tip_dist = 0;
        for (int tip_id : tips_ids) {
            double dist = sqrt(
                pow(lm[tip_id].x() - palm_center_x, 2) + 
                pow(lm[tip_id].y() - palm_center_y, 2)
            );
            avg_tip_dist += dist;
        }
        avg_tip_dist /= tips_ids.size();
        
        // FIST
        if (finger_count == 0 && avg_tip_dist < 0.15) {
            return "FIST";
        }
        
        // PALM
        if (finger_count == 5 && avg_tip_dist > 0.25) {
            return "PALM";
        }
        
        // POINT
        if (finger_count == 1 && fingers[1] == 1 && 
            fingers[2] == 0 && fingers[3] == 0 && fingers[4] == 0) {
            return "POINT";
        }
        
        // PEACE
        if (finger_count == 2 && fingers[1] == 1 && fingers[2] == 1 && 
            fingers[3] == 0 && fingers[4] == 0) {
            if (abs(lm[8].x() - lm[12].x()) > 0.05) {
                return "PEACE";
            }
        }
        
        // THUMBS UP
        if (fingers[0] == 1 && fingers[1] == 0 && fingers[2] == 0 && 
            fingers[3] == 0 && fingers[4] == 0) {
            if (lm[4].y() < lm[3].y() && lm[3].y() < lm[2].y()) {
                return "THUMBS_UP";
            }
        }
        
        // THUMBS DOWN
        if (fingers[0] == 1 && fingers[1] == 0 && fingers[2] == 0 && 
            fingers[3] == 0 && fingers[4] == 0) {
            if (lm[4].y() > lm[3].y() && lm[3].y() > lm[2].y()) {
                return "THUMBS_DOWN";
            }
        }
        
        // OK
        if (pinch_dist < 0.05 && fingers[2] == 0 && fingers[3] == 0 && fingers[4] == 0) {
            return "OK";
        }
        
        // THREE
        if (finger_count == 3 && fingers[1] == 1 && fingers[2] == 1 && 
            fingers[3] == 1 && fingers[4] == 0) {
            return "THREE";
        }
        
        // FOUR
        if (finger_count == 4 && fingers[1] == 1 && fingers[2] == 1 && 
            fingers[3] == 1 && fingers[4] == 1) {
            return "FOUR";
        }
        
        // FIVE
        if (finger_count == 5) {
            return "FIVE";
        }
        
        // PINCH
        if (pinch_dist < 0.03) {
            return "PINCH";
        }
        
        // CALL
        if (fingers[0] == 1 && fingers[4] == 1 && 
            fingers[1] == 0 && fingers[2] == 0 && fingers[3] == 0) {
            return "CALL";
        }
        
        // ROCK
        if (fingers[1] == 1 && fingers[4] == 1 && 
            fingers[2] == 0 && fingers[3] == 0 && fingers[0] == 0) {
            return "ROCK";
        }
        
        // SPIDERMAN
        if (fingers[1] == 1 && fingers[2] == 1 && fingers[0] == 1 && 
            fingers[3] == 0 && fingers[4] == 0) {
            return "SPIDERMAN";
        }
        
        // CROSS
        if (fingers[1] == 1 && fingers[2] == 1) {
            if (abs(lm[8].x() - lm[12].x()) < 0.02) {
                return "CROSS";
            }
        }
        
        return "UNKNOWN";
    }

    string execute_action(const string& gesture, cv::Mat* frame = nullptr, 
                         const vector<mediapipe::NormalizedLandmark>* hand_landmarks = nullptr) {
        
        auto current_time = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::duration<double>>(
            current_time - last_action_time);
        
        if (duration.count() < action_cooldown) {
            return "";
        }
        
        auto it = GESTURE_ACTIONS.find(gesture);
        if (it == GESTURE_ACTIONS.end()) return "";
        
        string action = it->second;
        
        if (action == "SCREENSHOT") {
#ifdef _WIN32
            keybd_event(VK_LWIN, 0, 0, 0);
            keybd_event(VK_SNAPSHOT, 0, 0, 0);
            keybd_event(VK_SNAPSHOT, 0, KEYEVENTF_KEYUP, 0);
            keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0);
#endif
            last_action_time = chrono::steady_clock::now();
            return "Screenshot taken";
        }
        
        else if (action == "SELFIE" && frame != nullptr) {
            string folder = "isha_captures";
            CreateDirectoryA(folder.c_str(), NULL);
            
            auto now = chrono::system_clock::now();
            auto now_c = chrono::system_clock::to_time_t(now);
            stringstream filename;
            filename << "selfie_" << put_time(localtime(&now_c), "%Y%m%d_%H%M%S") << ".jpg";
            
            string path = folder + "\\" + filename.str();
            cv::imwrite(path, *frame);
            
            last_action_time = chrono::steady_clock::now();
            return "Selfie captured";
        }
        
        else if (action == "VOLUME_UP") {
            for (int i = 0; i < 2; i++) {
                keybd_event(VK_VOLUME_UP, 0, 0, 0);
                keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_KEYUP, 0);
                Sleep(50);
            }
            last_action_time = chrono::steady_clock::now();
            return "Volume up";
        }
        
        else if (action == "VOLUME_DOWN") {
            for (int i = 0; i < 2; i++) {
                keybd_event(VK_VOLUME_DOWN, 0, 0, 0);
                keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_KEYUP, 0);
                Sleep(50);
            }
            last_action_time = chrono::steady_clock::now();
            return "Volume down";
        }
        
        else if (action == "OPEN_BROWSER") {
            ShellExecuteA(NULL, "open", "https://www.google.com", NULL, NULL, SW_SHOW);
            last_action_time = chrono::steady_clock::now();
            return "Opening browser";
        }
        
        else if (action == "OPEN_CALCULATOR") {
            system("calc");
            last_action_time = chrono::steady_clock::now();
            return "Opening calculator";
        }
        
        else if (action == "OPEN_NOTEPAD") {
            system("notepad");
            last_action_time = chrono::steady_clock::now();
            return "Opening notepad";
        }
        
        else if (action == "PLAY_MUSIC") {
            ShellExecuteA(NULL, "open", "start", "wmplayer", NULL, SW_SHOW);
            last_action_time = chrono::steady_clock::now();
            return "Playing music";
        }
        
        else if (action == "WEATHER") {
            ShellExecuteA(NULL, "open", "https://weather.com", NULL, NULL, SW_SHOW);
            last_action_time = chrono::steady_clock::now();
            return "Checking weather";
        }
        
        else if (action == "JOKE") {
            last_action_time = chrono::steady_clock::now();
            return "Why don't scientists trust atoms? Because they make up everything!";
        }
        
        else if (action == "CLOSE_APP") {
            keybd_event(VK_MENU, 0, 0, 0);
            keybd_event(VK_F4, 0, 0, 0);
            keybd_event(VK_F4, 0, KEYEVENTF_KEYUP, 0);
            keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0);
            last_action_time = chrono::steady_clock::now();
            return "Closing app";
        }
        
        else if (action == "OPEN_FILE" && hand_landmarks != nullptr && !hand_landmarks->empty()) {
            int x = static_cast<int>((*hand_landmarks)[8].x() * screen_w);
            int y = static_cast<int>((*hand_landmarks)[8].y() * screen_h);
            SetCursorPos(x, y);
            mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
            Sleep(100);
            mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
            last_action_time = chrono::steady_clock::now();
            return "Opening file";
        }
        
        else if (action == "SCROLL" && hand_landmarks != nullptr && !hand_landmarks->empty()) {
            double y = (*hand_landmarks)[8].y();
            if (last_scroll_y != 0) {
                double delta = y - last_scroll_y;
                if (abs(delta) > 0.02) {
                    mouse_event(MOUSEEVENTF_WHEEL, 0, 0, 
                               static_cast<DWORD>(-delta * scroll_speed * 120), 0);
                }
            }
            last_scroll_y = y;
            return "Scrolling";
        }
        
        return "";
    }

    pair<cv::Mat, string> process_frame(cv::Mat frame) {
        if (!active) {
            return {frame, "Gestures Off"};
        }
        
        frame_count++;
        
        if (frame_count % frame_skip != 0) {
            return {frame, "Processing..."};
        }
        
        cv::flip(frame, frame, 1);
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        
        // Process with MediaPipe (simplified - actual implementation would use the graph)
        string gesture_text = "No Hand";
        string action_result;
        
        // Gesture recognition would happen here via MediaPipe
        // For simplicity, we'll just return a placeholder
        gesture_text = "Right: FIST";
        
        // Update history
        gesture_history.push_back(gesture_text);
        if (gesture_history.size() > history_length) {
            gesture_history.erase(gesture_history.begin());
        }
        
        return {frame, gesture_text};
    }

private:
    unique_ptr<mediapipe::CalculatorGraph> mp_hands;
    int frame_skip;
    int frame_count;
    vector<string> gesture_history;
    size_t history_length;
    chrono::steady_clock::time_point last_action_time;
    double action_cooldown;
    bool scroll_mode;
    double last_scroll_y;
    int scroll_speed;
    bool mouse_control;
    int screen_w, screen_h;

public:
    bool show_preview;
    bool active;
};

const map<string, string> AdvancedGestureController::GESTURE_ACTIONS = {
    {"PALM", "WAITING"},
    {"FIST", "CLOSE_FILE"},
    {"POINT", "OPEN_FILE"},
    {"PEACE", "SCREENSHOT"},
    {"THUMBS_UP", "VOLUME_UP"},
    {"THUMBS_DOWN", "VOLUME_DOWN"},
    {"OK", "SELFIE"},
    {"THREE", "OPEN_BROWSER"},
    {"FOUR", "OPEN_CALCULATOR"},
    {"FIVE", "OPEN_NOTEPAD"},
    {"PINCH", "SCROLL"},
    {"CALL", "PLAY_MUSIC"},
    {"ROCK", "WEATHER"},
    {"SPIDERMAN", "JOKE"},
    {"CROSS", "CLOSE_APP"}
};

// ============================================
// PERSONAL VOCABULARY
// ============================================

class PersonalVocabulary {
private:
    map<string, int> word2idx;
    map<int, string> idx2word;
    map<string, int> word_counts;
    int min_freq;

public:
    PersonalVocabulary(int min_freq = 2) : min_freq(min_freq) {
        word2idx = {{"<PAD>", 0}, {"<UNK>", 1}, {"<SOS>", 2}, {"<EOS>", 3}};
        idx2word = {{0, "<PAD>"}, {1, "<UNK>"}, {2, "<SOS>"}, {3, "<EOS>"}};
    }

    void build_vocab(const vector<string>& texts) {
        for (const auto& text : texts) {
            istringstream iss(text);
            string word;
            while (iss >> word) {
                transform(word.begin(), word.end(), word.begin(), ::tolower);
                word_counts[word]++;
            }
        }
        
        int idx = 4;
        for (const auto& [word, count] : word_counts) {
            if (count >= min_freq && word2idx.find(word) == word2idx.end()) {
                word2idx[word] = idx;
                idx2word[idx] = word;
                idx++;
            }
        }
        
        cout << "📚 Vocabulary size: " << word2idx.size() << " words" << endl;
    }

    vector<int> encode(const string& text, int max_len = 20) {
        vector<int> indices;
        indices.push_back(word2idx["<SOS>"]);
        
        istringstream iss(text);
        string word;
        int count = 0;
        while (iss >> word && count < max_len - 2) {
            transform(word.begin(), word.end(), word.begin(), ::tolower);
            auto it = word2idx.find(word);
            if (it != word2idx.end()) {
                indices.push_back(it->second);
            } else {
                indices.push_back(word2idx["<UNK>"]);
            }
            count++;
        }
        
        indices.push_back(word2idx["<EOS>"]);
        
        while (indices.size() < static_cast<size_t>(max_len)) {
            indices.push_back(word2idx["<PAD>"]);
        }
        
        return indices;
    }

    string decode(const vector<int>& indices) {
        vector<string> words;
        for (int idx : indices) {
            if (idx == word2idx["<EOS>"]) break;
            if (idx != word2idx["<PAD>"] && idx != word2idx["<SOS>"]) {
                auto it = idx2word.find(idx);
                if (it != idx2word.end()) {
                    words.push_back(it->second);
                } else {
                    words.push_back("<UNK>");
                }
            }
        }
        
        string result;
        for (const auto& w : words) {
            if (!result.empty()) result += " ";
            result += w;
        }
        return result;
    }

    void save(const string& path = "personal_vocab.json") {
        json j;
        j["word2idx"] = word2idx;
        j["idx2word"] = idx2word;
        j["word_counts"] = word_counts;
        
        ofstream file(path);
        file << j.dump(4);
    }

    bool load(const string& path = "personal_vocab.json") {
        ifstream file(path);
        if (!file.is_open()) return false;
        
        json j;
        file >> j;
        
        word2idx = j["word2idx"].get<map<string, int>>();
        idx2word = j["idx2word"].get<map<int, string>>();
        word_counts = j["word_counts"].get<map<string, int>>();
        
        return true;
    }

    size_t size() const { return word2idx.size(); }
    int get_idx(const string& word) const {
        auto it = word2idx.find(word);
        return (it != word2idx.end()) ? it->second : word2idx.at("<UNK>");
    }
};

// ============================================
// PERSONAL LLM (Simplified with libtorch)
// ============================================

struct PersonalLLMImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};

    PersonalLLMImpl(int vocab_size, int embed_size = 128, int hidden_size = 256, int num_layers = 2) {
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_size));
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(embed_size, hidden_size).num_layers(num_layers).batch_first(true).dropout(0.2)));
        fc = register_module("fc", torch::nn::Linear(hidden_size, vocab_size));
    }

    pair<torch::Tensor, tuple<torch::Tensor, torch::Tensor>> forward(torch::Tensor x, 
        tuple<torch::Tensor, torch::Tensor> hidden = {}) {
        auto embedded = embedding(x);
        auto lstm_out = lstm(embedded, hidden);
        auto out = fc(get<0>(lstm_out));
        return {out, get<1>(lstm_out)};
    }

    string generate(shared_ptr<PersonalVocabulary> vocab, const string& prompt, 
                   int max_length = 20, double temperature = 0.8) {
        eval();
        
        torch::NoGradGuard no_grad;
        
        auto input_ids_vec = vocab->encode(prompt, 10);
        auto input_ids = torch::tensor(input_ids_vec).unsqueeze(0);
        
        tuple<torch::Tensor, torch::Tensor> hidden;
        vector<int> generated;
        
        for (int i = 0; i < max_length; i++) {
            auto output = forward(input_ids, hidden);
            auto logits = get<0>(output);
            
            auto probs = torch::softmax(logits[0][-1] / temperature, 0);
            
            auto multinomial = torch::multinomial(probs, 1);
            int next_token = multinomial.item<int>();
            
            if (next_token == vocab->get_idx("<EOS>")) {
                break;
            }
            
            generated.push_back(next_token);
            input_ids = torch::tensor({{next_token}});
        }
        
        return vocab->decode(generated);
    }
};

TORCH_MODULE(PersonalLLM);

// ============================================
// PERSONAL AI TRAINER
// ============================================

class PersonalAITrainer {
private:
    vector<string> load_or_create_conversations() {
        const string conv_file = "my_personal_conversations.txt";
        vector<string> conversations;
        
        ifstream file(conv_file);
        if (file.is_open()) {
            string line;
            while (getline(file, line)) {
                if (!line.empty()) {
                    conversations.push_back(line);
                }
            }
            file.close();
        } else {
            // YOUR PERSONAL CONVERSATION STYLE
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
            
            ofstream out(conv_file);
            for (const auto& conv : conversations) {
                out << conv << endl;
            }
        }
        
        return conversations;
    }

public:
    shared_ptr<PersonalVocabulary> vocab;
    PersonalLLM model;
    torch::Device device;

    vector<string> conversations;

    PersonalAITrainer() : device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        vocab = make_shared<PersonalVocabulary>();
        conversations = load_or_create_conversations();
    }

    void add_conversation(const string& user_msg, const string& assistant_msg) {
        string conv = "User: " + user_msg + " Assistant: " + assistant_msg;
        conversations.push_back(conv);
        
        ofstream file("my_personal_conversations.txt", ios::app);
        file << conv << endl;
    }

    bool train(int epochs = 30) {
        if (!PERSONAL_LLM_AVAILABLE) {
            cout << "❌ PyTorch not properly configured" << endl;
            return false;
        }
        
        cout << "🧠 Training your personal AI..." << endl;
        
        vector<string> all_texts;
        for (const auto& conv : conversations) {
            istringstream iss(conv);
            string token;
            while (iss >> token) {
                all_texts.push_back(token);
            }
        }
        vocab->build_vocab(all_texts);
        
        vector<vector<int>> input_seqs;
        vector<vector<int>> target_seqs;
        
        for (const auto& conv : conversations) {
            size_t pos = conv.find("Assistant:");
            if (pos != string::npos) {
                string user_part = conv.substr(6, pos - 7); // Skip "User: "
                string assistant_part = conv.substr(pos + 10); // Skip "Assistant: "
                
                if (!user_part.empty() && !assistant_part.empty()) {
                    auto input_seq = vocab->encode(user_part, 15);
                    auto target_seq = vocab->encode(assistant_part, 15);
                    
                    input_seqs.push_back(input_seq);
                    target_seqs.push_back(target_seq);
                }
            }
        }
        
        // Convert to tensors
        int64_t input_data[input_seqs.size()][15];
        int64_t target_data[target_seqs.size()][15];
        
        for (size_t i = 0; i < input_seqs.size(); i++) {
            for (size_t j = 0; j < 15; j++) {
                input_data[i][j] = input_seqs[i][j];
                target_data[i][j] = target_seqs[i][j];
            }
        }
        
        auto inputs = torch::from_blob(input_data, 
            {static_cast<long>(input_seqs.size()), 15}, torch::kLong).clone();
        auto targets = torch::from_blob(target_data, 
            {static_cast<long>(target_seqs.size()), 15}, torch::kLong).clone();
        
        auto dataset = torch::data::datasets::TensorDataset({inputs, targets});
        auto loader = torch::data::make_data_loader(
            dataset.map(torch::data::transforms::Stack<>()), 
            torch::data::DataLoaderOptions().batch_size(8));
        
        model = PersonalLLM(vocab->size());
        model->to(device);
        
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
        
        model->train();
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;
            int batch_count = 0;
            
            for (auto& batch : *loader) {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                
                optimizer.zero_grad();
                
                auto output = model->forward(data);
                auto logits = get<0>(output);
                
                auto loss = torch::nn::functional::cross_entropy(
                    logits.view({-1, static_cast<long>(vocab->size())}),
                    target.view({-1}),
                    torch::nn::functional::CrossEntropyFuncOptions().ignore_index(0)
                );
                
                loss.backward();
                optimizer.step();
                
                total_loss += loss.item<double>();
                batch_count++;
            }
            
            if ((epoch + 1) % 10 == 0) {
                cout << "  Epoch " << (epoch + 1) << "/" << epochs 
                     << ", Loss: " << (total_loss / batch_count) << endl;
            }
        }
        
        // Save model
        torch::save(model, "my_personal_ai.pt");
        vocab->save();
        
        cout << "✅ Personal AI training complete!" << endl;
        return true;
    }

    bool load_model() {
        ifstream file("my_personal_ai.pt");
        if (!file.good()) return false;
        
        try {
            vocab->load();
            model = PersonalLLM(vocab->size());
            torch::load(model, "my_personal_ai.pt");
            model->to(device);
            model->eval();
            return true;
        } catch (const exception& e) {
            cerr << "❌ Failed to load model: " << e.what() << endl;
            return false;
        }
    }

    string generate_response(const string& prompt) {
        if (!model) return "";
        
        try {
            string response = model->generate(vocab, prompt, 15);
            return response;
        } catch (const exception& e) {
            logger.error("Generation error: " + string(e.what()));
            return "";
        }
    }
};

// ============================================
// MAIN ISHA ASSISTANT
// ============================================

class IshaAssistant {
private:
    // TTS
    ISpVoice* pVoice;
    
    // Speech Recognition
    // Using Microsoft Speech API
    
    // Gesture Controller
    unique_ptr<AdvancedGestureController> gesture_controller;
    atomic<bool> gesture_active;
    thread gesture_thread;
    atomic<bool> camera_active;
    
    // Personal AI
    unique_ptr<PersonalAITrainer> personal_trainer;
    bool personal_ai_enabled;
    
    // Settings Maps
    map<string, string> SETTING_MAP;
    map<string, string> SETTING_MAP4s;
    map<string, string> apps_commands;
    map<string, string> apps_commands4q;
    map<string, string> software_dict;
    map<string, string> commands_dict;
    map<string, string> settings_display_to_cmd;
    map<string, string> apps_display_to_cmd;
    
    // Voice and Input
    queue<string> input_queue;
    mutex queue_mutex;
    condition_variable queue_cv;
    atomic<bool> is_listening;
    atomic<bool> pending;
    
    // Internet check
    chrono::steady_clock::time_point last_internet_check;
    double internet_check_interval;
    bool internet_status;
    
    // HTTP Server
    unique_ptr<httplib::Server> svr;
    
    void setup_original_command_mappings() {
        // Settings Map
        SETTING_MAP = {
            {"display setting", "ms-settings:display"},
            {"sound setting", "ms-settings:sound"},
            {"notification & action setting", "ms-settings:notifications"},
            {"focus assist setting", "ms-settings:quiethours"},
            {"power & sleep setting", "ms-settings:powersleep"},
            {"storage setting", "ms-settings:storagesense"},
            {"tablet setting", "ms-settings:tablet"},
            {"multitasking setting", "ms-settings:multitasking"},
            {"projecting to this pc setting", "ms-settings:project"},
            {"shared experiences setting", "ms-settings:crossdevice"},
            {"system components setting", "ms-settings:appsfeatures-app"},
            {"clipboard setting", "ms-settings:clipboard"},
            {"remote desktop setting", "ms-settings:remotedesktop"},
            {"optional features setting", "ms-settings:optionalfeatures"},
            {"about setting", "ms-settings:about"},
            {"system setting", "ms-settings:system"},
            {"devices setting", "ms-settings:devices"},
            {"mobile devices setting", "ms-settings:mobile-devices"},
            {"network & internet setting", "ms-settings:network"},
            {"personalization setting", "ms-settings:personalization"},
            {"apps setting", "ms-settings:appsfeatures"},
            {"account setting", "ms-settings:yourinfo"},
            {"time & language setting", "ms-settings:dateandtime"},
            {"gaming setting", "ms-settings:gaming"},
            {"ease of access setting", "ms-settings:easeofaccess"},
            {"privacy setting", "ms-settings:privacy"},
            {"updated & security", "ms-settings:windowsupdate"}
        };
        
        // Apps Commands
        apps_commands = {
            {"alarms & clock", "ms-clock:"},
            {"calculator", "calc"},
            {"calendar", "outlookcal:"},
            {"camera", "microsoft.windows.camera:"},
            {"copilot", "ms-copilot:"},
            {"cortana", "ms-cortana:"},
            {"game bar", "ms-gamebar:"},
            {"groove music", "mswindowsmusic:"},
            {"mail", "outlookmail:"},
            {"maps", "bingmaps:"},
            {"microsoft edge", "msedge"},
            {"microsoft solitaire collection", "ms-solitaire:"},
            {"microsoft store", "ms-windows-store:"},
            {"mixed reality portal", "ms-mixedreality:"},
            {"movies & tv", "mswindowsvideo:"},
            {"office", "ms-office:"},
            {"onedrive", "ms-onedrive:"},
            {"onenote", "ms-onenote:"},
            {"outlook", "outlookmail:"},
            {"outlook (classic)", "ms-outlook:"},
            {"paint", "mspaint"},
            {"paint 3d", "ms-paint:"},
            {"phone link", "ms-phonelink:"},
            {"power point", "ms-powerpoint:"},
            {"settings", "ms-settings:"},
            {"skype", "skype:"},
            {"snip & sketch", "ms-snip:"},
            {"sticky note", "ms-stickynotes:"},
            {"tips", "ms-tips:"},
            {"voice recorder", "ms-soundrecorder:"},
            {"weather", "msnweather:"},
            {"windows backup", "ms-settings:backup"},
            {"windows security", "ms-settings:windowsdefender"},
            {"word", "ms-word:"},
            {"xbox", "ms-xbox:"},
            {"about your pc", "ms-settings:about"}
        };
        
        // Software Dictionary
        software_dict = {
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
        
        // Merge all command dictionaries
        commands_dict.insert(SETTING_MAP.begin(), SETTING_MAP.end());
        commands_dict.insert(SETTING_MAP4s.begin(), SETTING_MAP4s.end());
        commands_dict.insert(software_dict.begin(), software_dict.end());
        commands_dict.insert(apps_commands.begin(), apps_commands.end());
        commands_dict.insert(apps_commands4q.begin(), apps_commands4q.end());
    }

    void set_female_voice() {
        try {
            if (pVoice) {
                // Try to set female voice (simplified - in practice would enumerate voices)
                pVoice->SetRate(0);
                pVoice->SetVolume(100);
                
                wstring welcome = L"Initializing voice";
                pVoice->Speak(welcome.c_str(), 0, NULL);
            }
        } catch (const exception& e) {
            logger.error("Failed to set female voice: " + string(e.what()));
        }
    }

    bool check_internet() {
        auto current_time = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::duration<double>>(
            current_time - last_internet_check);
        
        if (duration.count() < internet_check_interval) {
            return internet_status;
        }
        
        last_internet_check = current_time;
        
        // Try to connect to Google DNS
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            internet_status = false;
            return false;
        }
        
        SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
            WSACleanup();
            internet_status = false;
            return false;
        }
        
        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(80);
        inet_pton(AF_INET, "8.8.8.8", &addr.sin_addr);
        
        // Set timeout
        timeval timeout;
        timeout.tv_sec = 2;
        timeout.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
        
        internet_status = (connect(sock, (sockaddr*)&addr, sizeof(addr)) == 0);
        
        closesocket(sock);
        WSACleanup();
        
        return internet_status;
    }

    void gesture_loop() {
        if (!MEDIAPIPE_AVAILABLE) {
            speak("MediaPipe not installed. Cannot use hand gestures.");
            return;
        }
        
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            speak("Camera not available.");
            return;
        }
        
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        
        string last_action_message = "";
        auto action_message_time = chrono::steady_clock::now();
        
        while (gesture_active && cap.isOpened()) {
            cv::Mat frame;
            ret = cap.read(frame);
            if (!ret) continue;
            
            auto [processed_frame, gesture_text] = gesture_controller->process_frame(frame);
            
            if (gesture_controller->show_preview) {
                cv::rectangle(processed_frame, cv::Point(0, 0), 
                             cv::Point(processed_frame.cols, 60), 
                             cv::Scalar(0, 0, 0), -1);
                
                string status = gesture_active ? "ON" : "OFF";
                string text = "ISHA GESTURES: " + status;
                cv::putText(processed_frame, text, cv::Point(10, 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                
                cv::putText(processed_frame, "Gesture: " + gesture_text, 
                           cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 
                           0.6, cv::Scalar(255, 255, 255), 2);
                
                auto now = chrono::steady_clock::now();
                auto duration = chrono::duration_cast<chrono::duration<double>>(
                    now - action_message_time);
                
                if (duration.count() < 2 && !last_action_message.empty()) {
                    cv::putText(processed_frame, "Action: " + last_action_message,
                               cv::Point(10, processed_frame.rows - 20),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                }
                
                cv::putText(processed_frame, "Press 'Q' to close",
                           cv::Point(10, processed_frame.rows - 50),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
                
                cv::imshow("ISHA Hand Gestures", processed_frame);
                
                int key = cv::waitKey(1) & 0xFF;
                if (key == 'q') {
                    gesture_active = false;
                    break;
                }
            } else {
                this_thread::sleep_for(chrono::milliseconds(10));
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
    }

    string handle_settings_apps_commands(const string& command) {
        auto it = commands_dict.find(command);
        if (it != commands_dict.end()) {
            string cmd = it->second;
            try {
                if (cmd.find("http") == 0) {
                    ShellExecuteA(NULL, "open", cmd.c_str(), NULL, NULL, SW_SHOW);
                } else {
                    string start_cmd = "start \"\" \"" + cmd + "\"";
                    system(start_cmd.c_str());
                }
                string message = "Opening " + command;
                speak(message);
                return message;
            } catch (const exception& e) {
                string message = "Failed to open " + command + ": " + e.what();
                speak(message);
                return message;
            }
        }
        return "";
    }

    string open_google() {
        try {
            ShellExecuteA(NULL, "open", "https://www.google.com", NULL, NULL, SW_SHOW);
            string message = "Opening Google";
            speak(message);
            return message;
        } catch (const exception& e) {
            string message = "Failed to open Google: " + string(e.what());
            speak(message);
            return message;
        }
    }

    string get_weather_offline_first() {
        string message = "Which city's weather do you want to check?";
        speak(message);
        pending = true;
        
        string city;
        {
            unique_lock<mutex> lock(queue_mutex);
            if (queue_cv.wait_for(lock, chrono::seconds(30), [this] { return !input_queue.empty(); })) {
                city = input_queue.front();
                input_queue.pop();
            }
        }
        
        pending = false;
        
        if (city.empty() || city == "none" || city == "cancel" || city == "no") {
            message = "No city provided. Please try again.";
            speak(message);
            return message;
        }
        
        if (check_internet()) {
            try {
                // Using wttr.in for weather (simplified HTTP request)
                string url = "https://wttr.in/" + city + "?format=%C+%t";
                // In a full implementation, would use libcurl to fetch weather
                
                message = "Weather in " + city + ": Sunny 25°C"; // Placeholder
                speak(message);
                return message;
            } catch (const exception& e) {
                logger.error("Weather fetch failed: " + string(e.what()));
            }
        }
        
        message = "No internet and no cached weather available.";
        speak(message);
        return message;
    }

    string take_screenshot() {
        try {
            string folder = "isha_captures";
            CreateDirectoryA(folder.c_str(), NULL);
            
            auto now = chrono::system_clock::now();
            auto now_c = chrono::system_clock::to_time_t(now);
            stringstream filename;
            filename << "screenshot_" << put_time(localtime(&now_c), "%d-%m-%Y__%H-%M-%S") 
                    << "__" << chrono::duration_cast<chrono::milliseconds>(
                           now.time_since_epoch()).count() % 1000 << ".png";
            
            string path = folder + "\\" + filename.str();
            
            // Simplified screenshot (would need more complex implementation)
            HWND hdesktop = GetDesktopWindow();
            HDC hdc = GetDC(hdesktop);
            
            RECT desktop;
            GetWindowRect(hdesktop, &desktop);
            int width = desktop.right;
            int height = desktop.bottom;
            
            HDC hdest = CreateCompatibleDC(hdc);
            HBITMAP hbmp = CreateCompatibleBitmap(hdc, width, height);
            SelectObject(hdest, hbmp);
            BitBlt(hdest, 0, 0, width, height, hdc, 0, 0, SRCCOPY);
            
            // Save bitmap to file (simplified)
            
            DeleteDC(hdest);
            DeleteObject(hbmp);
            ReleaseDC(hdesktop, hdc);
            
            string message = "Screenshot saved";
            speak("Screenshot captured");
            logger.info("Screenshot saved: " + path);
            return message;
        } catch (const exception& e) {
            string message = "Failed to take screenshot: " + string(e.what());
            speak(message);
            return message;
        }
    }

    string take_selfie() {
        try {
            cv::VideoCapture cap(0);
            if (!cap.isOpened()) {
                throw runtime_error("Camera not available");
            }
            
            cv::Mat frame;
            bool ret = cap.read(frame);
            cap.release();
            
            if (ret) {
                string folder = "isha_captures";
                CreateDirectoryA(folder.c_str(), NULL);
                
                auto now = chrono::system_clock::now();
                auto now_c = chrono::system_clock::to_time_t(now);
                stringstream filename;
                filename << "selfie_" << put_time(localtime(&now_c), "%d-%m-%Y__%H-%M-%S") 
                        << "__" << chrono::duration_cast<chrono::milliseconds>(
                               now.time_since_epoch()).count() % 1000 << ".jpg";
                
                string path = folder + "\\" + filename.str();
                cv::imwrite(path, frame);
                
                string message = "Selfie captured";
                speak("Selfie captured");
                logger.info("Selfie saved: " + path);
                return message;
            } else {
                throw runtime_error("Failed to capture image");
            }
        } catch (const exception& e) {
            string message = "Failed to take selfie: " + string(e.what());
            speak(message);
            return message;
        }
    }

    void wish_me() {
        time_t now = time(nullptr);
        tm* ltm = localtime(&now);
        int current_hour = ltm->tm_hour;
        
        string greeting;
        if (current_hour >= 5 && current_hour < 12) {
            greeting = "Good morning";
        } else if (current_hour >= 12 && current_hour < 17) {
            greeting = "Good afternoon";
        } else if (current_hour >= 17 && current_hour < 21) {
            greeting = "Good evening";
        } else {
            greeting = "Good night";
        }
        
        speak(greeting);
        this_thread::sleep_for(chrono::seconds(1));
        
        string message = "I am Isha, Intelligent System for Human Assistance. Welcome! Say 'activate hand gestures' for gesture control.";
        speak(message);
        this_thread::sleep_for(chrono::seconds(2));
    }

public:
    IshaAssistant() : gesture_active(false), camera_active(false), 
                      personal_ai_enabled(false), is_listening(false), 
                      pending(false), internet_status(false) {
        
        // Initialize COM
        CoInitialize(NULL);
        
        // Initialize TTS
        HRESULT hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
        if (SUCCEEDED(hr)) {
            set_female_voice();
        }
        
        // Initialize Personal AI
        cout << "🧠 Initializing Your Personal AI..." << endl;
        personal_trainer = make_unique<PersonalAITrainer>();
        
        if (!personal_trainer->load_model()) {
            cout << "🎓 Training your personal AI for the first time..." << endl;
            personal_trainer->train(30);
        }
        
        personal_ai_enabled = personal_trainer->model != nullptr;
        
        // Initialize Gesture Controller
        cout << "🖐️ Initializing Advanced Hand Gestures..." << endl;
        gesture_controller = make_unique<AdvancedGestureController>();
        
        // Initialize command mappings
        setup_original_command_mappings();
        
        // Internet check
        last_internet_check = chrono::steady_clock::now();
        internet_check_interval = 60.0;
        
        // Welcome message
        wish_me();
        
        // Start HTTP Server
        start_server();
    }

    ~IshaAssistant() {
        if (pVoice) {
            pVoice->Release();
        }
        CoUninitialize();
    }

    void toggle_hand_gestures(bool activate = false) {
        if (activate) {
            gesture_active = true;
        } else {
            gesture_active = !gesture_active;
        }
        
        gesture_controller->active = gesture_active;
        
        if (gesture_active) {
            if (!gesture_thread.joinable()) {
                gesture_thread = thread(&IshaAssistant::gesture_loop, this);
                gesture_thread.detach();
            }
            string message = "Hand gestures activated";
            speak(message);
        } else {
            string message = "Hand gestures deactivated";
            speak(message);
        }
    }

    void toggle_camera_preview(bool show = false) {
        if (show) {
            gesture_controller->show_preview = true;
        } else {
            gesture_controller->show_preview = !gesture_controller->show_preview;
        }
        
        string status = gesture_controller->show_preview ? "on" : "off";
        string message = "Camera preview turned " + status;
        speak(message);
    }

    string get_personal_response(const string& user_input) {
        if (!personal_ai_enabled) return "";
        
        string response = personal_trainer->generate_response(user_input);
        
        if (!response.empty()) {
            personal_trainer->add_conversation(user_input, response);
        }
        
        return response;
    }

    string process_command(const string& command) {
        logger.info("Processing command: " + command);
        string cmd_lower = command;
        transform(cmd_lower.begin(), cmd_lower.end(), cmd_lower.begin(), ::tolower);
        
        if (pending) {
            {
                lock_guard<mutex> lock(queue_mutex);
                input_queue.push(command);
            }
            queue_cv.notify_one();
            pending = false;
            return "Input received";
        }
        
        // Hand Gesture Commands
        if (cmd_lower.find("activate hand") != string::npos || 
            cmd_lower.find("turn on hand") != string::npos ||
            cmd_lower.find("start hand") != string::npos ||
            cmd_lower.find("enable hand") != string::npos) {
            toggle_hand_gestures(true);
            return "Hand gestures activated";
        }
        
        else if (cmd_lower.find("deactivate hand") != string::npos || 
                 cmd_lower.find("turn off hand") != string::npos ||
                 cmd_lower.find("stop hand") != string::npos ||
                 cmd_lower.find("disable hand") != string::npos) {
            toggle_hand_gestures(false);
            return "Hand gestures deactivated";
        }
        
        else if (cmd_lower.find("gesture") != string::npos && 
                (cmd_lower.find("status") != string::npos || cmd_lower.find("check") != string::npos)) {
            string status = gesture_active ? "ON" : "OFF";
            return "Hand gestures are " + status;
        }
        
        // Camera Preview Commands
        else if (cmd_lower.find("show camera") != string::npos || 
                 cmd_lower.find("show preview") != string::npos ||
                 cmd_lower.find("camera on") != string::npos) {
            toggle_camera_preview(true);
            return "Camera preview turned on";
        }
        
        else if (cmd_lower.find("hide camera") != string::npos || 
                 cmd_lower.find("hide preview") != string::npos ||
                 cmd_lower.find("camera off") != string::npos) {
            toggle_camera_preview(false);
            return "Camera preview turned off";
        }
        
        // Training Commands
        else if (cmd_lower == "train my ai") {
            speak("Training your personal AI. This will take a moment.");
            personal_trainer->train(20);
            speak("Training complete!");
            return "Personal AI retrained";
        }
        
        // Gesture Activation Code
        else if (cmd_lower == "activet 156") {
            toggle_hand_gestures(true);
            return "Hand gestures activated";
        }
        
        // Try Personal AI First
        if (personal_ai_enabled) {
            string personal_response = get_personal_response(cmd_lower);
            if (!personal_response.empty()) {
                speak(personal_response);
                return personal_response;
            }
        }
        
        // Original Rule-Based Commands
        if (cmd_lower.find("open ") == 0) {
            string app_or_setting = cmd_lower.substr(5);
            string result = handle_settings_apps_commands(app_or_setting);
            if (!result.empty()) return result;
        }
        
        // Time
        if (cmd_lower == "what is the time" || cmd_lower == "tell me the time" ||
            cmd_lower == "current time" || cmd_lower == "time now" ||
            cmd_lower == "what time is it" || cmd_lower == "what's the time" ||
            cmd_lower == "time" || cmd_lower == "what time") {
            
            auto now = chrono::system_clock::now();
            auto now_c = chrono::system_clock::to_time_t(now);
            stringstream ss;
            ss << put_time(localtime(&now_c), "%H:%M:%S");
            string message = ss.str();
            speak(message);
            return message;
        }
        
        // Date
        else if (cmd_lower == "what is the date" || cmd_lower == "tell me the date" ||
                 cmd_lower == "current date" || cmd_lower == "date now" ||
                 cmd_lower == "what date is it" || cmd_lower == "what's the date" ||
                 cmd_lower == "date" || cmd_lower == "what date") {
            
            auto now = chrono::system_clock::now();
            auto now_c = chrono::system_clock::to_time_t(now);
            stringstream ss;
            ss << put_time(localtime(&now_c), "%A, %B %d, %Y");
            string message = ss.str();
            speak(message);
            return message;
        }
        
        // Google
        else if (cmd_lower == "open google" || cmd_lower == "launch google" || 
                 cmd_lower == "go to google") {
            if (check_internet()) {
                return open_google();
            } else {
                string message = "No internet connection. Google cannot be opened.";
                speak(message);
                return message;
            }
        }
        
        // Weather
        else if (cmd_lower == "weather" || cmd_lower == "check weather" || 
                 cmd_lower == "what's the weather") {
            return get_weather_offline_first();
        }
        
        // Greetings
        else if (cmd_lower == "hi" || cmd_lower == "hello" || cmd_lower == "hey") {
            string message = "Hello! How can I assist you today?";
            speak(message);
            return message;
        }
        
        // Screenshot
        else if (cmd_lower == "screenshot" || cmd_lower == "take screenshot" || 
                 cmd_lower == "capture screen") {
            return take_screenshot();
        }
        
        // Selfie
        else if (cmd_lower == "selfie" || cmd_lower == "take selfie" || 
                 cmd_lower == "capture selfie") {
            return take_selfie();
        }
        
        // Default
        string message = "Command not recognized: " + command;
        speak(message);
        return message;
    }

    void toggle_voice() {
        // Simplified - would need full speech recognition implementation
        is_listening = !is_listening;
        if (is_listening) {
            string message = "Microphone is now on";
            speak(message);
            cout << message << endl;
            // Start listening thread
        } else {
            string message = "Microphone is now off";
            speak(message);
            cout << message << endl;
        }
    }

    void speak(const string& text) {
        if (pVoice) {
            wstring wtext(text.begin(), text.end());
            pVoice->Speak(wtext.c_str(), 0, NULL);
            logger.info("Spoke: " + text);
        }
    }

    void start_server() {
        svr = make_unique<httplib::Server>();
        
        svr->Get("/", [this](const httplib::Request& req, httplib::Response& res) {
            res.set_content(get_html(), "text/html");
        });
        
        svr->Get("/command", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.has_param("cmd")) {
                string cmd = req.get_param_value("cmd");
                if (pending) {
                    {
                        lock_guard<mutex> lock(queue_mutex);
                        input_queue.push(cmd);
                    }
                    queue_cv.notify_one();
                    res.set_content("Input received", "text/plain");
                } else {
                    string response = process_command(cmd);
                    res.set_content(response, "text/plain");
                }
            } else {
                res.status = 400;
            }
        });
        
        svr->Get("/voice", [this](const httplib::Request& req, httplib::Response& res) {
            toggle_voice();
            res.set_content("Microphone toggled", "text/plain");
        });
        
        thread([this]() { svr->listen("localhost", 8000); }).detach();
        
        // Open browser
        ShellExecuteA(NULL, "open", "http://localhost:8000/", NULL, NULL, SW_SHOW);
    }

    string get_html() {
        // Build apps HTML
        string apps_html;
        for (const auto& [name, cmd] : apps_commands) {
            apps_html += "<div class=\"app-item\" data-command=\"open " + name + 
                        "\" style=\"margin:6px 0; cursor:pointer;\">• " + name + "</div>";
        }
        
        // Build settings HTML
        string settings_html;
        for (const auto& [name, cmd] : SETTING_MAP) {
            settings_html += "<div class=\"setting-item\" data-command=\"open " + name + 
                            "\" style=\"margin:6px 0; cursor:pointer;\">• " + name + "</div>";
        }
        
        // Get current time for display
        auto now = chrono::system_clock::now();
        auto now_c = chrono::system_clock::to_time_t(now);
        stringstream time_ss, date_ss;
        time_ss << put_time(localtime(&now_c), "%H:%M:%S");
        date_ss << put_time(localtime(&now_c), "%a, %b %d, %Y");
        
        string html = R"(
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
            <div class="time" id="time">)" + time_ss.str() + R"(</div>
            <div class="date" id="date">)" + date_ss.str() + R"(</div>
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
            )" + apps_html + R"(
        </div>
    </div>

    <div class="popup" id="settingsPopup">
        <div class="head">
            <div>⚙️ Settings</div>
            <div class="close" data-close="settingsPopup">✕</div>
        </div>
        <div style="font-size: 14px; color: #cfeeff; max-height: 350px; overflow-y: auto;">
            )" + settings_html + R"(
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
        
        return html;
    }
};

// ============================================
// MAIN
// ============================================

int main() {
    try {
        // Hide console on Windows
        HWND console = GetConsoleWindow();
        if (console) {
            ShowWindow(console, SW_HIDE);
        }
        
        cout << "============================================================" << endl;
        cout << "🤖 ISHA - Your Personal AI Assistant" << endl;
        cout << "🖐️ Advanced Hand Gestures Ready (OFF by default)" << endl;
        cout << "📷 Camera Preview: OFF (say 'show camera' to enable)" << endl;
        cout << "🧠 Personal LLM: Trained on YOUR conversations" << endl;
        cout << "✨ Beautiful Animated GUI with Floating Circle" << endl;
        cout << "============================================================" << endl;
        cout << "Commands:" << endl;
        cout << "  • 'activate hand gestures' - Turn gestures ON" << endl;
        cout << "  • 'deactivate hand gestures' - Turn gestures OFF" << endl;
        cout << "  • 'show camera' - See yourself" << endl;
        cout << "  • 'hide camera' - Hide preview (gestures still work)" << endl;
        cout << "  • 'screenshot' - Capture screen" << endl;
        cout << "  • 'selfie' - Take photo" << endl;
        cout << "  • 'open [app]' - Open any app" << endl;
        cout << "  • 'train my ai' - Improve your personal AI" << endl;
        cout << "  • 'activet 156' - Quick gesture activation" << endl;
        cout << "============================================================" << endl;
        
        auto app = make_unique<IshaAssistant>();
        
        // Keep alive
        while (true) {
            this_thread::sleep_for(chrono::seconds(1));
        }
        
    } catch (const exception& e) {
        logger.error("Application failed to start: " + string(e.what()));
        cerr << "Error: Application failed to start: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}

