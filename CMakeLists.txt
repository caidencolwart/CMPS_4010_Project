cmake_minimum_required(VERSION 3.15)
project(ASL_Interpreter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OpenCV_DIR "C:/OpenCV/opencv/build" CACHE PATH "Path to OpenCVConfig.cmake")
find_package(OpenCV REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_cmake_result
)
if(NOT pybind11_cmake_result EQUAL 0)
    message(FATAL_ERROR "Could not find pybind11 for ${Python3_EXECUTABLE}. Install it with: python -m pip install pybind11")
endif()

find_package(pybind11 REQUIRED)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ASL_Interpreter.py"
    "${CMAKE_CURRENT_BINARY_DIR}/asl_interpreter.py"
    COPYONLY
)

add_executable(ASL_App main.cpp)

target_compile_definitions(ASL_App PRIVATE
    ASL_APP_PYTHON_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(ASL_App PRIVATE 
    ${OpenCV_LIBS} 
    pybind11::embed
)