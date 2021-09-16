//
// Created by wyl on 2021/8/12.
//
#ifndef TENSORRTXPRACTICE_HRNET_H
#define TENSORRTXPRACTICE_HRNET_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static const int INPUT_H = 512;
static const int INPUT_W = 1024;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W;
static const int NUM_CLASSES = 19;
static const int BATCH_SIZE = 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "output";

using namespace nvinfer1;

static Logger gLogger;

std::map<std::string, Weights> loadWeights(const std::string file);
void debug_print(ITensor *input_tensor, std::string head);
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);

IActivationLayer* BasicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname);
IActivationLayer* BottleBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              int midch, int expansion, int stride, const std::string& bname, bool downsample);
IActivationLayer* HRBranches(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int inch, int outch, const std::string& bname);
std::vector<IActivationLayer*> FuseLayer2(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input0, ITensor& input1, const std::string& fname, int width);
std::vector<IActivationLayer*> FuseLayer3(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input0, ITensor& input1, ITensor& input2, const std::string& fname, int width);
std::vector<IActivationLayer*> FuseLayer4(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input0, ITensor& input1, ITensor& input2, ITensor& input3, const std::string& fname, int width);

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string wtsPath, int width);
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, const std::string& wtsPath, int width);

void doInference(IExecutionContext& context, float* input, float* output, int batchSize);

#endif //TENSORRTXPRACTICE_HRNET_H
