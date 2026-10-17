#include <iostream>
#include <stdexcept>
#include <string>
#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    return py::array_t<uint8_t>(
        { mat.rows, mat.cols, mat.channels() },
        { mat.step[0], mat.step[1], mat.elemSize1() },
        mat.data
    );
}

int main(int argc, char* argv[]) {
    int camera_index = 0;
    if (argc > 2) {
        std::cerr << "Usage: ASL_App [camera-index]" << std::endl;
        return 1;
    }

    if (argc == 2) {
        try {
            std::size_t parsed_characters = 0;
            camera_index = std::stoi(argv[1], &parsed_characters);
            if (parsed_characters != std::string(argv[1]).size() || camera_index < 0) {
                throw std::invalid_argument("Camera index must be a non-negative integer.");
            }
        }
        catch (const std::exception&) {
            std::cerr << "Camera index must be a non-negative integer." << std::endl;
            return 1;
        }
    }

    std::cout << "Starting application..." << std::endl;

    py::scoped_interpreter guard{};

    try {
        py::module_::import("sys").attr("path").attr("insert")(0, ASL_APP_PYTHON_DIR);
        std::cout << "Loading Python module 'asl_interpreter'..." << std::endl;
        py::module_ py_module = py::module_::import("asl_interpreter");
        py::function predict_func = py_module.attr("process_frame");

        std::cout << "Attempting to open camera..." << std::endl;
        cv::VideoCapture cap(camera_index, cv::CAP_DSHOW);

        if (!cap.isOpened()) {
            std::cerr << "ERROR: Could not open camera " << camera_index
                << ". Check that it is connected and not being used by another app."
                << std::endl;
            return 1;
        }

        std::cout << "Camera opened successfully! Starting video loop..." << std::endl;
        cv::Mat frame;
        while (true) {
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "ERROR: The camera stopped providing frames." << std::endl;
                break;
            }

            py::array_t<uint8_t> input_array = mat_to_numpy(frame);
            py::object python_result = predict_func(input_array);
            py::handle image_result = python_result;
            if (py::isinstance<py::tuple>(python_result)) {
                py::tuple result = py::cast<py::tuple>(python_result);
                if (result.size() >= 1) {
                    image_result = result[0];
                }
            }

            py::array_t<uint8_t> annotated_array = py::array_t<uint8_t>::ensure(image_result);
            if (annotated_array && annotated_array.ndim() == 3 &&
                annotated_array.shape(0) == frame.rows &&
                annotated_array.shape(1) == frame.cols &&
                annotated_array.shape(2) == 3) {
                cv::Mat annotated(
                    frame.rows,
                    frame.cols,
                    CV_8UC3,
                    annotated_array.mutable_data()
                );
                annotated.copyTo(frame);
            }

            cv::imshow("ASL Interpreter", frame);
            char key = (char)cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        }

        cap.release();
        cv::destroyAllWindows();
    }
    catch (py::error_already_set& e) {
        std::cout << "\n================ PYTHON ERROR ================\n";
        PyErr_Print();
        std::cout << e.what() << std::endl;
        std::cout << "==============================================\n\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cout << "\nC++ Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
