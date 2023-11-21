#include <string> 
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

using namespace std;

std::string getLabel(int class_num){
    std::vector<std::string> fine_labels = {"apple",  "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "computer_keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"};

    return fine_labels[class_num];
}


torch::Tensor padTensor(const torch::Tensor& input_tensor) {
    int original_height = input_tensor.size(2);
    int original_width = input_tensor.size(3);

    // Calculate the padding needed on both sides 
    int padding_updown = (original_width - original_height)/2; // assumes landscape photo

    // Pad the tensor to achieve the target width
    torch::Tensor padded_tensor = torch::nn::functional::pad(
        input_tensor,
        torch::nn::functional::PadFuncOptions({0, 0, padding_updown, padding_updown}).mode(torch::kConstant).value(0)
    );

    return padded_tensor;
}

int main(int argc, char **argv) {
    if(argc != 3)
    {
        std::cerr << "usage: ./classify_clusters <model weight name> <image to be classified> \n";
        return 1;
    }

    // loading model
    const std::string model_path = "../pytorch_weights/" + string(argv[1])+ ".pt";
    torch::jit::script::Module model = torch::jit::load(model_path);

    // loading image(s)
    //torch::Tensor input_example = torch::randn({1, 3, 224, 224});
    cout << string(argv[2]) << endl;
    cv::Mat image = cv::imread(string(argv[2]), cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Error: Unable to load the image." << std::endl;
        return -1;
    }
    cv::imshow("image", image);

    torch::Tensor tensor = torch::from_blob(image.data, {1, image.rows, image.cols, image.channels()}, torch::kByte);
    tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat32).div(255);
    
    // resize and padding the image
    // int targetHeight = 224;
    // int targetWidth = 224;
    torch::Tensor paddedTensor = padTensor(tensor);


    // Perform inference
    at::Tensor output = model.forward({paddedTensor}).toTensor();

    std::string predicted_class = getLabel(output.argmax().item<int>());
    
    // int predicted_label = predicted_class.item<int>();

    // Print the shape
    torch::IntArrayRef shape = tensor.sizes();
    shape = paddedTensor.sizes();
    std::cout << "Tensor Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "Predicted Class Number: " << output.argmax().item<int>() << std::endl;
    std::cout << "Predicted Class Label: " << predicted_class << std::endl;
    
    return 0;
}