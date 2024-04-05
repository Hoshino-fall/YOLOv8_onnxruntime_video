#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

std::vector<std::string> labels;

std::vector<std::string> readClassNames(std::string file = std::string(""));
bool predictImg(cv::Mat frame);
bool predictVideo(const std::string file);

int main(int argc, char** argv) {
    predictVideo("2_test_yolo.mp4");
    return 0;
}

std::vector<std::string> readClassNames(std::string file) {
    string name = "COCONAMES.txt";
    std::vector<std::string> output;
    if (!(file == "")) {
        name = file;
    }
    ifstream fin;
    fin.open(name, ios::in);
    if (!fin.is_open()) {
        cout << "file open failed" << endl;
        return output;
    }
    string buff;
    while (getline(fin, buff)) {
        output.push_back(buff);
    }
    fin.close();

    return output;
}

bool predictImg(cv::Mat frame) {
    // Read Labelqq
    std::vector<std::string> labels = readClassNames("coconames.txt");
    // set device
    // GPU Mode, 0 - gpu device id
    std::string onnxpath = "yolov8n.onnx";
    Ort::SessionOptions session_options;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");

    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session session_(env, onnxpath.c_str(), session_options);

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    size_t numInputNodes = session_.GetInputCount();
    size_t numOutputNodes = session_.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // get input
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session_.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // get output
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1]; // 84
    output_w = output_dims[2]; // 8400
    std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session_.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

    // format frame
    int64 start = cv::getTickCount();
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    // fix bug, boxes consistence!
    float x_factor = image.cols / static_cast<float>(input_w);
    float y_factor = image.rows / static_cast<float>(input_h);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }

    // output data
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
    cv::Mat det_output = dout.t(); // 8400x84

    // post-process
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    for (int i = 0; i < det_output.rows; i++) {
        cv::Mat classes_scores = det_output.row(i).colRange(4, 4 + labels.size());
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        // filter
        if (score > 0.25) {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int idx = classIds[index];
        cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
            cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
        putText(frame, labels[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
    }

    cv::imshow("YOLOv8+ONNXRUNTIME", frame);
    cv::waitKey(1);

    return true;
}

bool predictVideo(const std::string file) {
    cv::VideoCapture cap(file);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return false;
    }

    // Loop through video frames
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cout << "End of video stream." << std::endl;
            break;
        }

        // Perform prediction on the frame
        predictImg(frame);

        // Exit loop if 'q' key is pressed
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return true;
}

