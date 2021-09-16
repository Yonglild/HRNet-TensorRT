//
// Created by wyl on 2021/8/12.
//
#include "HRNet.h"
#define DEVICE 0 // GPU id

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
    std::cout<< "Loading weights done!" << std::endl;
    return weightMap;
}

void debug_print(ITensor *input_tensor, std::string head)
{
    std::cout << head << " : ";

    for (int i = 0; i < input_tensor->getDimensions().nbDims; i++)
    {
        std::cout << input_tensor->getDimensions().d[i] << " ";
    }
    std::cout << std::endl;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps){
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0; i < len; i++){
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;    // 将计算出的scale和shift等参数放进weightMap中
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

/**
 * @param network
 * @param input
 * @param outdims
 * @return
 */
ILayer* addUpsample(INetworkDefinition* network, ITensor& input, Dims outdims){
    IResizeLayer* upSample = network->addResize(input);
    upSample->setResizeMode(ResizeMode::kLINEAR);
    upSample->setOutputDimensions(outdims);
    upSample->setAlignCorners(true);
    return upSample;
}

ILayer* ConvBnRelu(INetworkDefinition* network, std::map<std::string, Weights> &weightMap,
                   ITensor& input, int outch, int ksize, int s, int p, std::string convname, std::string bnname,
                   bool relu = true, bool bias = false){
    Weights empty{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer *conv1;
    if(!bias){
        conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[convname + ".weight"], empty);
    }else{
        conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[convname +".weight"], weightMap[convname + ".bias"]);

    }
    assert(conv1);
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setStrideNd(DimsHW{s, s});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), bnname, 1e-5);
    if(relu){
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        return relu1;
    }else {
        return bn1;
    }
}


IActivationLayer* BasicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }

    IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

/**
 *
 * @param network
 * @param weightMap
 * @param input
 * @param midch 中间层的通道数
 * @param stride
 * @param bname  for example:'layer1.0', 'layer1.1', 'layer1.2'
 * @param downsample
 * @return
 */
IActivationLayer* BottleBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              int midch, int expansion, int stride, const std::string& bname, bool downsample){
    auto conv1 = ConvBnRelu(network, weightMap, input, midch, 1, 1, 0, bname +".conv1", bname + ".bn1", true, false);
    debug_print(conv1->getOutput(0), bname+".conv1");
    auto conv2 = ConvBnRelu(network, weightMap, *conv1->getOutput(0), midch, 3, 1, 1, bname + ".conv2", bname + ".bn2", true, false);
    debug_print(conv2->getOutput(0), bname+".conv2");
    auto conv3 = ConvBnRelu(network, weightMap, *conv2->getOutput(0), midch*expansion, 1, 1, 0, bname + ".conv3", bname + ".bn3", false, false);
    debug_print(conv3->getOutput(0), bname+".conv3");

    IElementWiseLayer* add1;
    if(downsample){
        Weights empty{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, midch*expansion, DimsHW{1, 1}, weightMap[bname + ".downsample.0.weight"], empty);
        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), bname + ".downsample.1", 1e-5);
        add1 = network->addElementWise(*conv3->getOutput(0), *bn1->getOutput(0), ElementWiseOperation::kSUM);
        debug_print(add1->getOutput(0), bname+".downsample");
    }else{
        add1 = network->addElementWise(input, *conv3->getOutput(0), ElementWiseOperation::kSUM);
    }
    return network->addActivation(*add1->getOutput(0), ActivationType::kRELU);
}

/**
 * @param network
 * @param weightMap
 * @param input
 * @param inch
 * @param outch
 * @param bname  for example:"stage2.0.branches.0.", "stage2.0.branches.1.", “stage3.1.branches.2.”
 * @return
 */
IActivationLayer* HRBranches(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
        int inch, int outch, const std::string& bname){
    auto relu1 = BasicBlock(network, weightMap, input, inch, outch, 1, bname + "0.");
    auto relu2 = BasicBlock(network, weightMap, *relu1->getOutput(0), inch, outch, 1, bname + "1.");
    auto relu3 = BasicBlock(network, weightMap, *relu2->getOutput(0), inch, outch, 1, bname + "2.");
    auto relu4 = BasicBlock(network, weightMap, *relu3->getOutput(0), inch, outch, 1, bname + "3.");
    return relu4;
}

std::vector<IActivationLayer*> FuseLayer2(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input0, ITensor& input1, const std::string& fname, int width){
    ILayer* l1 = ConvBnRelu(network, weightMap, input1, width, 1, 1, 0, fname + ".0.1.0", fname + ".0.1.1", false, false);
    Dims dim0 = input0.getDimensions();
    ILayer* up = addUpsample(network, *l1->getOutput(0), dim0);
    IElementWiseLayer* ew1 = network->addElementWise(input0, *up->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu0 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);

    ILayer* l2 = ConvBnRelu(network, weightMap, input0, width*2, 3, 2, 1, fname + ".1.0.0.0", fname + ".1.0.0.1", false, false);
    // upSample
    IElementWiseLayer* ew2 = network->addElementWise(*l2->getOutput(0), input1, ElementWiseOperation::kSUM);
    IActivationLayer* relu1 = network->addActivation(*ew2->getOutput(0), ActivationType::kRELU);
    std::vector<IActivationLayer*> vIA;
    vIA.push_back(relu0);
    vIA.push_back(relu1);
    debug_print(vIA[0]->getOutput(0), fname + "_0");
    debug_print(vIA[1]->getOutput(0), fname + "_1");
    return vIA;
}
/**
 * @param network
 * @param weightMap
 * @param input1
 * @param input2
 * @param input3
 * @param fname for example "stage3.0.fuse_layers", ”stage3.1.fuse_layers", "stage3.2.fuse_layers", "stage3.3.fuse_layers"
 * @return
 */
std::vector<IActivationLayer*> FuseLayer3(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input0, ITensor& input1, ITensor& input2, const std::string& fname, int width){
    ILayer* l1 = ConvBnRelu(network, weightMap, input1, width, 1, 1, 0, fname + ".0.1.0", fname + ".0.1.1", false, false);
    // upSample
    ILayer* up01 = addUpsample(network, *l1->getOutput(0), input0.getDimensions());
    ILayer* l2 = ConvBnRelu(network, weightMap, input2, width, 1, 1, 0, fname + ".0.2.0", fname + ".0.2.1", false, false);
    ILayer* up02 = addUpsample(network, *l2->getOutput(0), input0.getDimensions());

    IElementWiseLayer* ew1 = network->addElementWise(input0, *up01->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* ew2 = network->addElementWise(*ew1->getOutput(0), *up02->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu0 = network->addActivation(*ew2->getOutput(0), ActivationType::kRELU);

    ILayer* l3 = ConvBnRelu(network, weightMap, input0, width*2, 3, 2, 1, fname + ".1.0.0.0", fname + ".1.0.0.1", false, false);
    ILayer* l4 = ConvBnRelu(network, weightMap, input2, width*2, 1, 1, 0, fname + ".1.2.0", fname + ".1.2.1", false, false);
    ILayer* up12 = addUpsample(network, *l4->getOutput(0), input1.getDimensions());
    IElementWiseLayer* ew3 = network->addElementWise(*l3->getOutput(0), input1, ElementWiseOperation::kSUM);
    IElementWiseLayer* ew4 = network->addElementWise(*ew3->getOutput(0), *up12->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu1 = network->addActivation(*ew4->getOutput(0), ActivationType::kRELU);

    ILayer* l5 = ConvBnRelu(network, weightMap, input0, width, 3, 2, 1, fname + ".2.0.0.0", fname + ".2.0.0.1", true, false);
    ILayer* l6 = ConvBnRelu(network, weightMap, *l5->getOutput(0), width*4, 3, 2, 1, fname + ".2.0.1.0", fname + ".2.0.1.1", false, false);
    ILayer* l7 = ConvBnRelu(network, weightMap, input1, width*4, 3, 2, 1, fname + ".2.1.0.0", fname + ".2.1.0.1", false, false);
    IElementWiseLayer* ew5 = network->addElementWise(*l6->getOutput(0), *l7->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* ew6 = network->addElementWise(*ew5->getOutput(0), input2, ElementWiseOperation::kSUM);
    IActivationLayer* relu2 = network->addActivation(*ew6->getOutput(0), ActivationType::kRELU);
    std::vector<IActivationLayer*> vIA;
    debug_print(relu0->getOutput(0), fname);
    debug_print(relu1->getOutput(0), fname);
    debug_print(relu2->getOutput(0), fname);
    vIA.push_back(relu0);
    vIA.push_back(relu1);
    vIA.push_back(relu2);
    return vIA;
}

/**
 * @param network
 * @param weightMap
 * @param input1
 * @param input2
 * @param fname for example: stage4.0, stage4.1, stage4.2, stage4.3
 * @return
 */
std::vector<IActivationLayer*> FuseLayer4(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input0, ITensor& input1, ITensor& input2, ITensor& input3, const std::string& fname, int width){
    ILayer* l1 = ConvBnRelu(network, weightMap, input1, width, 1, 1, 0, fname + ".0.1.0", fname + ".0.1.1", false, false);
    ILayer* up01 = addUpsample(network, *l1->getOutput(0), input0.getDimensions());
    ILayer* l2 = ConvBnRelu(network, weightMap, input2, width, 1, 1, 0, fname + ".0.2.0", fname + ".0.2.1", false, false);
    ILayer* up02 = addUpsample(network, *l2->getOutput(0), input0.getDimensions());
    ILayer* l8 = ConvBnRelu(network, weightMap, input3, width, 1, 1, 0, fname + ".0.3.0", fname + ".0.3.1", false, false);
    ILayer* up03 = addUpsample(network, *l8->getOutput(0), input0.getDimensions());

    IElementWiseLayer* ew1 = network->addElementWise(input0, *up01->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* ew2 = network->addElementWise(*ew1->getOutput(0), *up02->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* ew7 = network->addElementWise(*ew2->getOutput(0), *up03->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu0 = network->addActivation(*ew7->getOutput(0), ActivationType::kRELU);


    ILayer* l3 = ConvBnRelu(network, weightMap, input0, width*2, 3, 2, 1, fname + ".1.0.0.0", fname + ".1.0.0.1", false, false);
    ILayer* l4 = ConvBnRelu(network, weightMap, input2, width*2, 1, 1, 0, fname + ".1.2.0", fname + ".1.2.1", false, false);
    ILayer* up12 = addUpsample(network, *l4->getOutput(0), input1.getDimensions());
    ILayer* l9 = ConvBnRelu(network, weightMap, input3, width*2, 1, 1, 0, fname + ".1.3.0", fname + ".1.3.1", false, false);
    ILayer* up13 = addUpsample(network, *l9->getOutput(0), input1.getDimensions());
    IElementWiseLayer* ew3 = network->addElementWise(*l3->getOutput(0), input1, ElementWiseOperation::kSUM);
    IElementWiseLayer* ew4 = network->addElementWise(*ew3->getOutput(0), *up12->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* ew8 = network->addElementWise(*ew4->getOutput(0), *up13->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu1 = network->addActivation(*ew8->getOutput(0), ActivationType::kRELU);

    ILayer* l5 = ConvBnRelu(network, weightMap, input0, width, 3, 2, 1, fname + ".2.0.0.0", fname + ".2.0.0.1", true, false);
    ILayer* l6 = ConvBnRelu(network, weightMap, *l5->getOutput(0), width*4, 3, 2, 1, fname + ".2.0.1.0", fname + ".2.0.1.1", false, false);
    ILayer* l7 = ConvBnRelu(network, weightMap, input1, width*4, 3, 2, 1, fname + ".2.1.0.0", fname + ".2.1.0.1", false, false);
    ILayer* l10 = ConvBnRelu(network, weightMap, input3, width*4, 1, 1, 0, fname + ".2.3.0", fname + ".2.3.1", false, false);
    ILayer* up23 = addUpsample(network, *l10->getOutput(0), input2.getDimensions());
    IElementWiseLayer* ew5 = network->addElementWise(*l6->getOutput(0), *l7->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* ew6 = network->addElementWise(*ew5->getOutput(0), input2, ElementWiseOperation::kSUM);
    IElementWiseLayer* ew9 = network->addElementWise(*ew6->getOutput(0), *up23->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu2 = network->addActivation(*ew9->getOutput(0), ActivationType::kRELU);

    ILayer* l11 = ConvBnRelu(network, weightMap, input0, width, 3, 2, 1, fname + ".3.0.0.0", fname + ".3.0.0.1", true, false);
    ILayer* l12 = ConvBnRelu(network, weightMap, *l11->getOutput(0), width, 3, 2, 1, fname + ".3.0.1.0", fname + ".3.0.1.1", true, false);
    ILayer* l13 = ConvBnRelu(network, weightMap, *l12->getOutput(0), width*8, 3, 2, 1, fname + ".3.0.2.0", fname + ".3.0.2.1", false, false);
    ILayer* l14 = ConvBnRelu(network, weightMap, input1, width*2, 3, 2, 1, fname + ".3.1.0.0", fname + ".3.1.0.1", true, false);
    ILayer* l15 = ConvBnRelu(network, weightMap, *l14->getOutput(0), width*8, 3, 2, 1, fname + ".3.1.1.0", fname + ".3.1.1.1", false, false);
    ILayer* l16 = ConvBnRelu(network, weightMap, input2, width*8, 3, 2, 1, fname + ".3.2.0.0", fname + ".3.2.0.1", false, false);
    IElementWiseLayer* ew10 = network->addElementWise(*l13->getOutput(0), *l15->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* ew11 = network->addElementWise(*ew10->getOutput(0), *l16->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* ew12 = network->addElementWise(*ew11->getOutput(0), input3, ElementWiseOperation::kSUM);
    IActivationLayer* relu3 = network->addActivation(*ew12->getOutput(0), ActivationType::kRELU);


    std::vector<IActivationLayer*> vIA;
    vIA.push_back(relu0);
    vIA.push_back(relu1);
    vIA.push_back(relu2);
    vIA.push_back(relu3);
    debug_print(relu0->getOutput(0), fname);
    debug_print(relu1->getOutput(0), fname);
    debug_print(relu2->getOutput(0), fname);
    debug_print(relu3->getOutput(0), fname);
    return vIA;
}

ITensor *MeanStd(INetworkDefinition *network, ITensor *input, float *mean, float *std, bool div255)
{
    if (div255)
    {
        Weights Div_225{DataType::kFLOAT, nullptr, 3};
        float *wgt = reinterpret_cast<float *>(malloc(sizeof(float) * 3));
        for (int i = 0; i < 3; ++i)
        {
            wgt[i] = 255.0f;
        }
        Div_225.values = wgt;
        IConstantLayer *d = network->addConstant(Dims3{3, 1, 1}, Div_225);
        input = network->addElementWise(*input, *d->getOutput(0), ElementWiseOperation::kDIV)->getOutput(0);
    }
    Weights Mean{DataType::kFLOAT, nullptr, 3};
    Mean.values = mean;
    IConstantLayer *m = network->addConstant(Dims3{3, 1, 1}, Mean);
    IElementWiseLayer *sub_mean = network->addElementWise(*input, *m->getOutput(0), ElementWiseOperation::kSUB);
    if (std != nullptr)
    {
        Weights Std{DataType::kFLOAT, nullptr, 3};
        Std.values = std;
        IConstantLayer *s = network->addConstant(Dims3{3, 1, 1}, Std);
        IElementWiseLayer *std_mean = network->addElementWise(*sub_mean->getOutput(0), *s->getOutput(0), ElementWiseOperation::kDIV);
        return std_mean->getOutput(0);
    }
    else
    {
        return sub_mean->getOutput(0);
    }
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string wtsPath, int width){
    INetworkDefinition* network = builder->createNetworkV2(0U); // 创建一个空的INetWork
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_H, INPUT_W, 3});
    assert(data);

    // hwc to chw
    auto ps = network->addShuffle(*data);
    ps->setFirstTranspose(nvinfer1::Permutation{2, 0, 1});
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
//    float mean[3] = {0, 0, 0};
//    float std[3] = {1, 1, 1};
    ITensor *preinput = MeanStd(network, ps->getOutput(0), mean, std, true);

    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    auto relu2 = ConvBnRelu(network, weightMap, *preinput, 64, 3, 2, 1, "conv1", "bn1", true, false);
    debug_print(relu2->getOutput(0), "conv1");

    auto relu5 = ConvBnRelu(network, weightMap, *relu2->getOutput(0), 64, 3, 2, 1, "conv2", "bn2", true, false);
    debug_print(relu5->getOutput(0), "conv2");

    // layer1
    auto relu6 = BottleBlock(network, weightMap, *relu5->getOutput(0), 64, 4, 1, "layer1.0", true);
    debug_print(relu6->getOutput(0), "layer1.0");
    auto relu7 = BottleBlock(network, weightMap, *relu6->getOutput(0), 64, 4, 1, "layer1.1", false);
    debug_print(relu7->getOutput(0), "layer1.1");
    auto relu8 = BottleBlock(network, weightMap, *relu7->getOutput(0), 64, 4, 1, "layer1.2", false);
    debug_print(relu8->getOutput(0), "layer1.2");
    auto relu9 = BottleBlock(network, weightMap, *relu8->getOutput(0), 64, 4, 1, "layer1.3", false);
    debug_print(relu9->getOutput(0), "layer1.3");

    // trainsition1
    auto trans1_0 = ConvBnRelu(network, weightMap, *relu9->getOutput(0), width, 3, 1, 1, "transition1.0.0", "transition1.0.1", true, false);
    auto trans1_1 = ConvBnRelu(network, weightMap, *relu9->getOutput(0), width*2, 3, 2, 1, "transition1.1.0.0", "transition1.1.0.1", true, false);
    debug_print(trans1_0->getOutput(0), "transition1.0.0");
    debug_print(trans1_1->getOutput(0), "transition1.1.0.0");

    // stage2.0.branches
    auto relu10 = HRBranches(network, weightMap, *trans1_0->getOutput(0), width, width, "stage2.0.branches.0.");
    auto relu11 = HRBranches(network, weightMap, *trans1_1->getOutput(0), width*2, width*2, "stage2.0.branches.1.");
    debug_print(relu10->getOutput(0), "stage2.0.branches.0.");
    debug_print(relu11->getOutput(0), "stage2.0.branches.1.");
    // stage2.0.fuse_layers
    std::vector<IActivationLayer*> vfuse2 = FuseLayer2(network, weightMap, *relu10->getOutput(0), *relu11->getOutput(0), "stage2.0.fuse_layers", width);


    // transition2
    auto trans2_0 = vfuse2[0];
    auto trans2_1 = vfuse2[1];
//    trans2_0->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*trans2_0->getOutput(0));
    auto trans2_2 = ConvBnRelu(network, weightMap, *trans2_1->getOutput(0), width*4, 3, 2, 1, "transition2.2.0.0", "transition2.2.0.1", true, false);

    // stage3.0.branches
    auto relu12 = HRBranches(network, weightMap, *trans2_0->getOutput(0), width, width, "stage3.0.branches.0.");
    auto relu13 = HRBranches(network, weightMap, *trans2_1->getOutput(0), width*2, width*2, "stage3.0.branches.1.");
    auto relu14 = HRBranches(network, weightMap, *trans2_2->getOutput(0), width*4, width*4, "stage3.0.branches.2.");
    // stage3.0.fuse_layers
    std::vector<IActivationLayer*> vfuse3_0 = FuseLayer3(network, weightMap, *relu12->getOutput(0), *relu13->getOutput(0), *relu14->getOutput(0), "stage3.0.fuse_layers", width);
//    vfuse3_0[0]->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*vfuse3_0[0]->getOutput(0));

    // stage3.1.branches
    auto relu27 = HRBranches(network, weightMap, *vfuse3_0[0]->getOutput(0), width, width, "stage3.1.branches.0.");
    auto relu28 = HRBranches(network, weightMap, *vfuse3_0[1]->getOutput(0), width*2, width*2, "stage3.1.branches.1.");
    auto relu29 = HRBranches(network, weightMap, *vfuse3_0[2]->getOutput(0), width*4, width*4, "stage3.1.branches.2.");
    // stage3.1.fuse_layers
    std::vector<IActivationLayer*> vfuse3_1 = FuseLayer3(network, weightMap, *relu27->getOutput(0), *relu28->getOutput(0), *relu29->getOutput(0), "stage3.1.fuse_layers", width);
//    vfuse3_1[0]->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*vfuse3_1[0]->getOutput(0));
    // stage3.2.branches
    auto relu30 = HRBranches(network, weightMap, *vfuse3_1[0]->getOutput(0), width, width, "stage3.2.branches.0.");
    auto relu31 = HRBranches(network, weightMap, *vfuse3_1[1]->getOutput(0), width*2, width*2, "stage3.2.branches.1.");
    auto relu32 = HRBranches(network, weightMap, *vfuse3_1[2]->getOutput(0), width*4, width*4, "stage3.2.branches.2.");
    // stage3.2.fuse_layers
    std::vector<IActivationLayer*> vfuse3_2 = FuseLayer3(network, weightMap, *relu30->getOutput(0), *relu31->getOutput(0), *relu32->getOutput(0), "stage3.2.fuse_layers", width);
//    vfuse3_2[0]->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*vfuse3_2[0]->getOutput(0));
    // stage3.3.branches
    auto relu33 = HRBranches(network, weightMap, *vfuse3_2[0]->getOutput(0), width, width, "stage3.3.branches.0.");
    auto relu34 = HRBranches(network, weightMap, *vfuse3_2[1]->getOutput(0), width*2, width*2, "stage3.3.branches.1.");
    auto relu35 = HRBranches(network, weightMap, *vfuse3_2[2]->getOutput(0), width*4, width*4, "stage3.3.branches.2.");
    // stage3.3.fuse_layers
    std::vector<IActivationLayer*> vfuse3_3 = FuseLayer3(network, weightMap, *relu33->getOutput(0), *relu34->getOutput(0), *relu35->getOutput(0), "stage3.3.fuse_layers", width);
//    vfuse3_3[0]->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*vfuse3_3[0]->getOutput(0));
    // transition3
    auto trans3_0 = vfuse3_3[0];
    auto trans3_1 = vfuse3_3[1];
    auto trans3_2 = vfuse3_3[2];
//    trans3_0->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*trans3_0->getOutput(0));
    auto trans3_3 = ConvBnRelu(network, weightMap, *trans3_2->getOutput(0), width*8, 3, 2, 1, "transition3.3.0.0", "transition3.3.0.1", true, false);

    // stage4.0.branches
    auto relu15 = HRBranches(network, weightMap, *trans3_0->getOutput(0), width, width, "stage4.0.branches.0.");
//    relu15->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*relu15->getOutput(0));
    auto relu16 = HRBranches(network, weightMap, *trans3_1->getOutput(0), width*2, width*2, "stage4.0.branches.1.");
//    relu16->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*relu16->getOutput(0));
    auto relu17 = HRBranches(network, weightMap, *trans3_2->getOutput(0), width*4, width*4, "stage4.0.branches.2.");

    auto relu18 = HRBranches(network, weightMap, *trans3_3->getOutput(0), width*8, width*8, "stage4.0.branches.3.");
//    relu18->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*relu18->getOutput(0));
    // stage.4.0 fuse_layer
    std::vector<IActivationLayer*> vfuse4_0 = FuseLayer4(network, weightMap, *relu15->getOutput(0), *relu16->getOutput(0), *relu17->getOutput(0), *relu18->getOutput(0), "stage4.0.fuse_layers", width);

    // stage4.1.branches
    auto relu19 = HRBranches(network, weightMap, *vfuse4_0[0]->getOutput(0), width, width, "stage4.1.branches.0.");
    auto relu20 = HRBranches(network, weightMap, *vfuse4_0[1]->getOutput(0), width*2, width*2, "stage4.1.branches.1.");
    auto relu21 = HRBranches(network, weightMap, *vfuse4_0[2]->getOutput(0), width*4, width*4, "stage4.1.branches.2.");
    auto relu22 = HRBranches(network, weightMap, *vfuse4_0[3]->getOutput(0), width*8, width*8, "stage4.1.branches.3.");
    // stage4.1.fuse_layer
    std::vector<IActivationLayer*> vfuse4_1 = FuseLayer4(network, weightMap, *relu19->getOutput(0), *relu20->getOutput(0), *relu21->getOutput(0), *relu22->getOutput(0), "stage4.1.fuse_layers", width);
//    vfuse4_1[0]->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*vfuse4_1[0]->getOutput(0));
    // stage4.2.branches
    auto relu23 = HRBranches(network, weightMap, *vfuse4_1[0]->getOutput(0), width, width, "stage4.2.branches.0.");
    auto relu24 = HRBranches(network, weightMap, *vfuse4_1[1]->getOutput(0), width*2, width*2, "stage4.2.branches.1.");
    auto relu25 = HRBranches(network, weightMap, *vfuse4_1[2]->getOutput(0), width*4, width*4, "stage4.2.branches.2.");
    auto relu26 = HRBranches(network, weightMap, *vfuse4_1[3]->getOutput(0), width*8, width*8, "stage4.2.branches.3.");
    // stage4.2.fuse_layer
    std::vector<IActivationLayer*> vfuse4_2 = FuseLayer4(network, weightMap, *relu23->getOutput(0), *relu24->getOutput(0), *relu25->getOutput(0), *relu26->getOutput(0), "stage4.2.fuse_layers", width);
//    vfuse4_2[1]->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*vfuse4_2[1]->getOutput(0));

    // Upsampling
    Dims outdim = vfuse4_0[0]->getOutput(0)->getDimensions();
    outdim.d[0] = vfuse4_2[1]->getOutput(0)->getDimensions().d[0];
    ILayer* up1 = addUpsample(network, *vfuse4_2[1]->getOutput(0), outdim);
    outdim.d[0] = vfuse4_2[2]->getOutput(0)->getDimensions().d[0];
    ILayer* up2 = addUpsample(network, *vfuse4_2[2]->getOutput(0), outdim);
    outdim.d[0] = vfuse4_2[3]->getOutput(0)->getDimensions().d[0];
    ILayer* up3 = addUpsample(network, *vfuse4_2[3]->getOutput(0), outdim);

//    vfuse4_0[0]->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*vfuse4_0[0]->getOutput(0));

    ITensor* concatTensors[4] =  {vfuse4_2[0]->getOutput(0), up1->getOutput(0), up2->getOutput(0), up3->getOutput(0)};
    auto concat1 = network->addConcatenation(concatTensors, 4);
    concat1->setAxis(0);

//    concat1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*concat1->getOutput(0));
    // last_layer
    auto relu36 = ConvBnRelu(network, weightMap, *concat1->getOutput(0), width*15, 1, 1, 0, "last_layer.0", "last_layer.1", true, false);
    auto conv = network->addConvolutionNd(*relu36->getOutput(0), NUM_CLASSES, DimsHW{1, 1}, weightMap["last_layer.3.weight"], weightMap["last_layer.3.bias"]);
    assert(conv);
    conv->setStrideNd(DimsHW{1, 1});
    conv->setPaddingNd(DimsHW{0, 0});
    debug_print(conv->getOutput(0), "last_year");

    outdim.d[0] = NUM_CLASSES;
    outdim.d[1] = INPUT_H;
    outdim.d[2] = INPUT_W;
    auto score = addUpsample(network, *conv->getOutput(0), outdim);
    debug_print(score->getOutput(0), "upsample");

    auto topk = network->addTopK(*score->getOutput(0), TopKOperation::kMAX, 1, 0X01);
    topk->getOutput(1)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*topk->getOutput(1));

    builder->setMaxWorkspaceSize(maxBatchSize);
    config->setMaxWorkspaceSize((1<<30));   // 1G

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build success!" << std::endl;
    network->destroy();
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    std::cout << "free mem sucess!" << std::endl;

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, const std::string& wtsPath, int width)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();    // 配置类指针，可以设置最大空间

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wtsPath, width);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}


void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, int batchSize)
{
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
}

cv::Mat createLTU(int len)
{
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.data;
    for (int j = 0; j < 256; ++j)
    {
        p[j] = (j * (256 / len) > 255) ? uchar(255) : (uchar)(j * (256 / len));
    }
    return lookUpTable;
}

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, int &width, std::string &img_dir)
{
    if (std::string(argv[1]) == "-s" && argc == 5)
    {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        width = std::stoi(argv[4]);
    }
    else if (std::string(argv[1]) == "-d" && argc == 4)
    {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    }
    else
    {
        return false;
    }
    return true;
}

int main(int argc, char** argv){
    cudaSetDevice(DEVICE);
    std::string wtsPath = "";
    std::string engine_name = "";
    int width;
    std::string img_dir;
    // parse args
    if (!parse_args(argc, argv, wtsPath, engine_name, width, img_dir))
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./hrnet -s [.wts] [.engine] [18 or 32 or 48]  // serialize model to plan file" << std::endl;
        std::cerr << "./hrnet -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    // create a model using the API directly and serialize it to a stream
    if (!wtsPath.empty())
    {
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream, wtsPath, width);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    // deserialize the .engine and run inference
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        std::cerr << "could not open plan file" << std::endl;
    }

    // prepare input data ---------------------------
    cudaSetDeviceFlags(cudaDeviceMapHost);
    float *data;
    int *prob; // using int. output is index
    CHECK(cudaHostAlloc((void **)&data, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&prob, BATCH_SIZE * OUTPUT_SIZE * sizeof(int), cudaHostAllocMapped));

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream;
    void *buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0)
    {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    for (size_t i = 0; i < file_names.size(); ++i)
    {
        std::cout << file_names[i] << std::endl;
    }

    for (int f = 0; f < (int)file_names.size(); f++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();

        std::cout << file_names[f] << std::endl;
        cv::Mat pr_img;
        cv::Mat img_BGR = cv::imread(img_dir + "/" + file_names[f], 1); // BGR
        cv::Mat img;
        cv::cvtColor(img_BGR, img, cv::COLOR_BGR2RGB);

        if (img.empty())
            continue;

        cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H));
        img = pr_img.clone(); // for img show
        pr_img.convertTo(pr_img, CV_32FC3);
        if (!pr_img.isContinuous())
        {
            pr_img = pr_img.clone();
        }
        std::memcpy(data, pr_img.data, BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float));

        cudaHostGetDevicePointer((void **)&buffers[inputIndex], (void *)data, 0);  // buffers[inputIndex]-->data
        cudaHostGetDevicePointer((void **)&buffers[outputIndex], (void *)prob, 0); // buffers[outputIndex] --> prob

        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        doInference(*context, stream, buffers, BATCH_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::Mat outimg(INPUT_H, INPUT_W, CV_8UC1);
        for (int row = 0; row < INPUT_H; ++row)
        {
            uchar *uc_pixel = outimg.data + row * outimg.step;
            for (int col = 0; col < INPUT_W; ++col)
            {
                uc_pixel[col] = (uchar)prob[row * INPUT_W + col];
            }
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        std::cout << "time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;

        cv::Mat im_color;
        cv::cvtColor(outimg, im_color, cv::COLOR_GRAY2RGB);
        cv::Mat lut = createLTU(NUM_CLASSES);
        cv::LUT(im_color, lut, im_color);
        // false color
        cv::cvtColor(im_color, im_color, cv::COLOR_RGB2GRAY);
        cv::applyColorMap(im_color, im_color, cv::COLORMAP_HOT);
        cv::imwrite(std::to_string(f) + "_false_color_map.png", im_color);
        //fusion
        cv::Mat fusionImg;
        cv::addWeighted(img, 1, im_color, 0.8, 1, fusionImg);
        cv::imwrite(std::to_string(f) + "_fusion_img.png", fusionImg);
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFreeHost(buffers[inputIndex]));
    CHECK(cudaFreeHost(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}


