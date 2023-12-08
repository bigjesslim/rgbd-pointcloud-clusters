#ifndef CLUSTER_H
#define CLUSTER_H

#include <opencv2/opencv.hpp>
#include "../dbscan/dbscan.hpp"
#include "utils.h"
#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <string>
#include <system_error>
#include <vector>
#include <utility>
#include <fstream>
#include <charconv>
#include <cassert>
#include <tuple>
#include <cstring>
#include <cmath>
#include <time.h>
#include <chrono>

using namespace std;
class Cluster
{

    public:
        Cluster(std::string image_path, std::vector<int> image_dims, std::vector<std::vector<float>> values_2D, int cluster_id); 
        // values = (2D) array of 3D coords
        // clusters = 1D array of matching cluster assignments
 
        void getCropDimsAndMask(); // get crop dimensions and downsized mask

        torch::Tensor getProcessedTensor(); // get cropped and masked rgb image

    public: 
        std::string image_path;
        std::vector<int> image_dims; // h,w
        std::vector<vector<float>> values_2D; // xyz coordinates in 2d - i.e., list of 3d coordinates
        int size;
        int cluster_id; 
        int pred_class_id; // output by classification model
        std::string pred_class_ade20k;
        std::vector<std::string> pred_classes_sunrgbd; // sunrgbd classes mapped to fromo ade20k class

        std::vector<int> crop_dims; // min_x, min_y, width, height for crop
        cv::Mat downsized_mask; // 5x5 downsized mask of coordinates
        cv::Mat processed_image;
        torch::Tensor processed_tensor;

};
#endif // CLUSTER_H