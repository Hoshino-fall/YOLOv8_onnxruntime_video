#include <iostream>
#include <string>
#include <cstring>
#include "onnxruntime_cxx_api.h"

const std::string onnxPath = "/home/tacom/Development/Mounts/Project/yolo_convert/SimpleOnnx/best.onnx";

std::pair<std::string, std::vector<int64_t>> readInputInfo(bool readInput,
                                                           Ort::Session &session,
                                                           Ort::AllocatorWithDefaultOptions &allocator){
    if(readInput){
        std::shared_ptr<char> inName = std::move(session.GetInputNameAllocated(0, allocator));
        std::cout << "input Name: " << inName;

        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        Ort::ConstTensorTypeAndShapeInfo tensorInfo =  inputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType dataType = tensorInfo.GetElementType();
        std::vector<int64_t> dataShape = tensorInfo.GetShape();

        std::cout << " type: "  << dataType << " [";
        for(long j : dataShape){
            std::cout << j << ", ";
        }
        std::cout << "]" << std::endl;
        return std::make_pair(std::string(inName.get(), strlen(inName.get())), dataShape);
    }else{
        std::shared_ptr<char> outName = std::move(session.GetOutputNameAllocated(0, allocator));
        std::cout << "Output Name: " << outName;

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
        Ort::ConstTensorTypeAndShapeInfo tensorInfo =  outputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType dataType = tensorInfo.GetElementType();
        std::vector<int64_t> dataShape = tensorInfo.GetShape();

        std::cout << " type: "  << dataType << " [";
        for(long j : dataShape){
            std::cout << j << ", ";
        }
        std::cout << "]" << std::endl;
        return std::make_pair(std::string(outName.get(), strlen(outName.get())), dataShape);
    }
}

int64_t multiplyVector(std::vector<int64_t> &vec){
    int64_t res = 1;
    for(int64_t num : vec){
        res *= num;
    }
    return res;
}

int main() {
    // 环境
    Ort::Env env;
    Ort::Session session{env, onnxPath.c_str(), Ort::SessionOptions{nullptr}};
    Ort::AllocatorWithDefaultOptions allocator;

    // 读取信息
    std::pair<std::string, std::vector<int64_t>> inInfo = readInputInfo(true, session, allocator);
    std::pair<std::string, std::vector<int64_t>> outInfo = readInputInfo(false, session, allocator);

    // 生成示例数据
    std::vector<float> inputImage(multiplyVector(inInfo.second), 1.0);
    std::vector<float> outputImage(multiplyVector(outInfo.second), 0.0);

    // 拷贝示例数据并创建tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                          inputImage.data(), inputImage.size(),
                                                          inInfo.second.data(), inInfo.second.size());
    // 此处绑定输出到outputImage
    Ort::Value outTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                           outputImage.data(), outputImage.size(),
                                                           outInfo.second.data(), outInfo.second.size());

    // 计算
    Ort::RunOptions runOptions;
    std::vector<const char *> inputNames = {inInfo.first.c_str()};
    std::vector<const char *> outputNames ={outInfo.first.c_str()};

    session.Run(runOptions, inputNames.data(), &inTensor, 1, outputNames.data(), &outTensor, 1);

    // 解析
    for(size_t i = 0; i < outputImage.size(); ++i){
        std::cout << "Out data: " << outputImage[i] << std::endl;
        if(i > 100) break;
    }
    return 0;
}
