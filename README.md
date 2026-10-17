# ASL Interpreter Webcam Setup

This starter application captures webcam frames in C++ with OpenCV and passes
them to Python for MediaPipe hand detection. Drop a Teachable Machine ONNX
classifier into `python/models/` to show the predicted letter and a confidence
bar on the webcam window.

## Requirements

- Windows and Visual Studio with the C++ desktop workload
- CMake and Ninja (available through a Visual Studio Developer PowerShell)
- OpenCV's native C++ SDK. The default CMake path is
  `C:\OpenCV\opencv\build`; use `-DOpenCV_DIR=...` if yours is elsewhere.
- Python 3.14, plus the Python packages in `requirements.txt`

Install the Python packages using the same Python interpreter CMake will embed:

```powershell
py -3.14 -m pip install -r requirements.txt
```

## Add the ASL ONNX model

1. Export your Teachable Machine image model as **TensorFlow** / Keras, then
   convert it to ONNX (`tf2onnx`), or export / convert to `asl_classifier.onnx`.
2. Put these files in `python/models/`:
   - `asl_classifier.onnx` (or `model.onnx`)
   - `labels.txt` from Teachable Machine (`0 A`, `1 B`, ...)
3. Install the extra Python package if you have not already:

```powershell
py -3.14 -m pip install -r requirements.txt
```

4. Rebuild so CMake copies `python/models/` into `out/build/x64-debug/models/`.

Until those files are present, the app still runs and only shows hand tracking.

## Download the hand model

MediaPipe Tasks needs its hand-landmarker model file. From the project directory,
run:

```powershell
$modelDir = "out\build\x64-debug\models"
New-Item -ItemType Directory -Force $modelDir | Out-Null
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "$modelDir\hand_landmarker.task"
```

## Build and run

From a Visual Studio Developer PowerShell in this directory:

```powershell
cmake --preset x64-debug
cmake --build out\build\x64-debug --config Debug
$env:Path = "C:\OpenCV\opencv\build\x64\vc16\bin;$env:Path"
.\out\build\x64-debug\ASL_App.exe
```

The default camera index is `0`. To try another camera, pass its index, for
example `.\out\build\x64-debug\ASL_App.exe 1`. Press **Q** or **Esc** in the
webcam window to exit.
