cmake_minimum_required(VERSION 3.10)
project(IshaAssistant)

set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})

# Add executable
add_executable(IshaAssistant main.cpp)

# Link libraries
target_link_libraries(IshaAssistant 
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    winmm
    ole32
    oleaut32
    sapi
    SpeechSDK
)

# Windows specific
if(WIN32)
    target_compile_definitions(IshaAssistant PRIVATE _WIN32_WINNT=0x0601)
    set_target_properties(IshaAssistant PROPERTIES WIN32_EXECUTABLE TRUE)
endif()
