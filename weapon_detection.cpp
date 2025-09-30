#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {
    // Load YOLO model
    Net net = readNet("yolov3-tiny.weights", "yolov3-tiny.cfg");
    vector<string> classes = {"Weapon"};

    // Open camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera." << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Resize frame for better performance
        resize(frame, frame, Size(640, 480));

        // Prepare input blob
        Mat blob = blobFromImage(frame, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Perform detection
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // Process detections
        for (const auto& out : outs) {
            for (int i = 0; i < out.rows; i++) {
                const float* detection = out.ptr<float>(i);
                float confidence = detection[4];
                if (confidence > 0.5) {  // Confidence threshold
                    cout << "Weapon detected!" << endl;
                }
            }
        }

        // Display frame
        imshow("Weapon Detection", frame);
        if (waitKey(1) == 27) break;  // Exit on ESC
    }

    cap.release();
    destroyAllWindows();
    return 0;
}