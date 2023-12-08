#include "cluster.h"


using namespace std;
Cluster::Cluster(std::string image_path, std::vector<int> image_dims, std::vector<std::vector<float>> values_2D, int cluster_id){
    this->image_path = image_path;
    this-> image_dims = image_dims;
    this->values_2D = values_2D;
    this->cluster_id = cluster_id;
    this->size = values_2D.size();

    if (cluster_id == 0){ // background or outlier clusters will be classified as 'wall' as default
        this->pred_class_id = 164;
        this->pred_class_ade20k = "wall";
        this->pred_classes_sunrgbd = {"wall"};
    }
    getCropDimsAndMask();
}

void Cluster::getCropDimsAndMask(){
    std::vector<int> downsized_dims{image_dims[0]/5, image_dims[1]/5};
    cv::Mat cv_mask(downsized_dims[0], downsized_dims[1], CV_8UC1, cv::Scalar(0));
    this->downsized_mask = cv_mask;
    int min_x = this->values_2D[0][0];
    int max_x = 0;
    int min_y = this->values_2D[0][1];
    int max_y = 0;

    for (int i=0;i<size;i++) {
        int y = this->values_2D[i][1];
        int x = this->values_2D[i][0];

        if (y > max_y){
            max_y = y;
        }
        if (y < min_y){
            min_y = y;
        }
        if (x > max_x){
            max_x = x;
        }
        if(x < min_x){
            min_x = x;
        }
        this->downsized_mask.at<uchar>(static_cast<int>(y/5), static_cast<int>(x/5)) = 1;
    }
    this->crop_dims = {min_x, min_y, max_x-min_x, max_y-min_y};
}

torch::Tensor Cluster::getProcessedTensor(){
    cv::Mat image = cv::imread(this->image_path, cv::IMREAD_UNCHANGED); 
    if (image.empty()) {
        std::cout << this->image_path << std::endl;
        std::cerr << "Error: Unable to load the image." << std::endl;
        return this->processed_tensor;
    }

    // upsize downsized_mask
    cv::Mat resized_mask; 
    cv::resize(this->downsized_mask, resized_mask, cv::Size(), 5, 5, cv::INTER_NEAREST);
    // apply mask to rgb image
    cv::Mat masked_image;
    cv::bitwise_and(image, image, masked_image, resized_mask);
    // crop masked image
    this->processed_image = masked_image(cv::Rect(crop_dims[0], crop_dims[1], crop_dims[2], crop_dims[3]));
    this->processed_image = paddingAndResize(this->processed_image, cv::Size(240, 240));

    cv::Mat rgb_image;
    cv::cvtColor(this->processed_image, rgb_image, cv::COLOR_BGR2RGB);
    this->processed_tensor = torch::from_blob(rgb_image.data, {1, rgb_image.rows, rgb_image.cols, 3}, torch::kByte);
    this->processed_tensor = this->processed_tensor.permute({0, 3, 1, 2}).div(255);;
    //this->processed_tensor = padTensor(this->processed_tensor);

    return this->processed_tensor;
}