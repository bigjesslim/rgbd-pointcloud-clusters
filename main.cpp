#include <GL/glew.h>
#include <GL/gl.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "dbscan/dbscan.hpp"
#include "src/utils.h"
#include "src/cluster.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

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

// noise will be labelled as 0

int main(int argc, char **argv) {
    if(argc != 5)
    {
        std::cerr << "usage: main <image name> <z weight> <epsilon> <min points> \n";
        return 1;
    }
    // LOAD IMAGES into cv::Mat
    std::string depth_map_path = "../tumrgbd_slam_data/images/depth/" + string(argv[1]) + ".png";
    std::string rgb_image_path =  "../tumrgbd_slam_data/images/rgb/" + string(argv[1]) + ".png";
    
    cv::Mat rgb_image = cv::imread(rgb_image_path, cv::IMREAD_UNCHANGED); 
        if (rgb_image.empty()) {
        std::cout << rgb_image_path << std::endl;
        std::cerr << "Error: Unable to load the image." << std::endl;
        return 1;
    }

    cv::Mat depth_map = cv::imread(depth_map_path, cv::IMREAD_ANYDEPTH);
    if (depth_map.empty()) {
        std::cout << depth_map_path << std::endl;
        std::cerr << "Error: Could not open or find the depth map" << std::endl;
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> image_dims{depth_map.rows, depth_map.cols};

    // 1oad depth map into 1D vector (x,y,z)
    const std::vector<float> values = toXYZ(depth_map, 5);

    // convert depth map into 2D vector (x,y,z) [H,W]
    auto values_2D = convertTo2DVector(values, 3);
    auto points = values_2D;

    auto end_time_1 = std::chrono::high_resolution_clock::now();
    // normalize and CLUSTER data (DBSCAN) 
    std::vector<size_t> labels = normalizeAndCluster(argv[2], argv[3], argv[4], values_2D);

    std::set<int> cluster_set(labels.begin(), labels.end());
    std::vector<int> cluster_ids(cluster_set.begin(), cluster_set.end());
    int num_clusters = cluster_ids.size();

    auto end_time_2 = std::chrono::high_resolution_clock::now();

    // get xyz values for each cluster

    std::map<int, cv::Mat> downsized_mask_by_cluster;
    std::map<int, std::vector<int>> xcoords_by_cluster;
    std::map<int, std::vector<int>> ycoords_by_cluster;
    int x;
    int y;
    std::vector<int> downsized_dims{rgb_image.cols/5, rgb_image.rows/5};

    // instantiation of maps
    for (const auto& id : cluster_ids) {
        xcoords_by_cluster[id]; 
        ycoords_by_cluster[id];

        cv::Mat cv_mask(downsized_dims[1], downsized_dims[0], CV_8UC1, cv::Scalar(0));
        downsized_mask_by_cluster[id] = cv_mask;
    }

    // populating maps of cluster data
    int num_points = values_2D.size();
    for (int i = 0; i < num_points; i++){
        x = values_2D[i][0];
        y = values_2D[i][1];
        xcoords_by_cluster[labels[i]].push_back(x);
        ycoords_by_cluster[labels[i]].push_back(y);

        downsized_mask_by_cluster[labels[i]].at<uchar>(static_cast<int>(y/5), static_cast<int>(x/5)) = 1;
    }

    // final variables required
    std::vector<torch::Tensor> tensor_list; 
    std::vector<int> new_cluster_ids;

    // loop variables
    int cluster_id;
    cv::Mat resized_mask;
    cv::Mat masked_image;
    cv::Mat cropped_image;
    cv::Mat processed_image;
    cv::Mat processed_rgb_image;
    torch::Tensor processed_tensor;

    std::vector<int> vec;
    int min_x; 
    int max_x; 
    int min_y;
    int max_y;

    for (int i = 0; i < num_clusters; i++){
        cluster_id = cluster_ids[i];
        if (cluster_id != 0){ // if not background cluster
            vec = xcoords_by_cluster[cluster_id];

            if (vec.size()>=50){ // if cluster size is at least 50
                cout << "cluster size: " << to_string(vec.size()) << endl;
                new_cluster_ids.push_back(cluster_id);

                // get masked image
                cv::resize(downsized_mask_by_cluster[cluster_id], resized_mask, cv::Size(), 5, 5, cv::INTER_NEAREST);
                cout << "Resized mask width: " << to_string(resized_mask.cols) << endl;
                cout << "Resized mask height: " << to_string(resized_mask.rows) << endl;

                cv::Mat masked_image;
                cv::bitwise_and(rgb_image, rgb_image, masked_image, resized_mask);

                // get crop dimensions
                min_x = *std::min_element(vec.begin(), vec.end());
                max_x = *std::max_element(vec.begin(), vec.end());

                vec = ycoords_by_cluster[cluster_id];
                min_y = *std::min_element(vec.begin(), vec.end());
                max_y = *std::max_element(vec.begin(), vec.end());

                cropped_image = masked_image(cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y));

                // padding and conversion to tensor
                processed_image = paddingAndResize(cropped_image, cv::Size(240, 240));
            
                // cv::namedWindow("cluster id: " + to_string(cluster_id), cv::WINDOW_NORMAL);
                // cv::imshow("cluster id: " + to_string(cluster_id), processed_image);
                // cv::waitKey(0);
                
                cv::cvtColor(processed_image, processed_rgb_image, cv::COLOR_BGR2RGB); // bgr (opencv) to rgb (pytorch)
                processed_tensor = torch::from_blob(processed_rgb_image.data, {1, processed_rgb_image.rows, processed_rgb_image.cols, 3}, torch::kByte);
                processed_tensor = processed_tensor.permute({0, 3, 1, 2}).div(255); // to_tensor

                tensor_list.push_back(processed_tensor);

            // }
            // else{ // merge small cluster masks with background cluster 
            //     cv::bitwise_or(downsized_mask_by_cluster[0], downsized_mask_by_cluster[cluster_ids[i]], downsized_mask_by_cluster[0]);
            }
        }
    }

    std::cout << "Total number of final class instances: " << to_string(new_cluster_ids.size()) << endl;

    // consolidation
    cluster_ids = new_cluster_ids; // exclude small clusters
    torch::Tensor batch_tensor = torch::cat(tensor_list, 0);

    auto end_time_3 = std::chrono::high_resolution_clock::now();

    auto duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_time_1 - start_time);
    float ms_duration_1 = duration_1.count()/1000;
    std::cout << "Time taken for pre-processing: " << to_string(ms_duration_1) << " ms." << std::endl;

    auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time_2 - end_time_1);
    float ms_duration_2 = duration_2.count()/1000;
    std::cout << "Time taken for clustering: " << to_string(ms_duration_2) << " ms." << std::endl;

    auto duration_3 = std::chrono::duration_cast<std::chrono::microseconds>(end_time_3 - end_time_2);
    float ms_duration_3 = duration_3.count()/1000;
    std::cout << "Time taken for post-processing: " << to_string(ms_duration_3) << " ms." << std::endl;

    torch::IntArrayRef shape = batch_tensor.sizes();
    std::cout << "Batch input shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;


    // if (torch::cuda::is_available()) {
    //     // Move the tensor to CUDA device (GPU)
    //     batch_tensor = batch_tensor.to(torch::kCUDA);
    // } else {
    //     std::cerr << "CUDA is not available. Unable to move tensor to GPU.\n";
    // }

    // // classification
    // torch::jit::FusionStrategy strat = {{torch::jit::FusionBehavior::DYNAMIC, 1}};
    // torch::jit::setFusionStrategy(strat);
    // const std::string model_path = "../trained_weights_fixed/efficientnetb1_50000_block7_mc_epoch5.pt";
    // torch::jit::script::Module model = torch::jit::load(model_path);
    // model.to(torch::kCUDA);

    // start_time = std::chrono::high_resolution_clock::now();
    // torch::Tensor output = model.forward({batch_tensor}).toTensor();
    // end_time = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // ms_duration = duration.count()/1000;
    // std::cout << "Time taken for classification: " << to_string(ms_duration) << " ms." << std::endl;

    // // torch::IntArrayRef shape = output.sizes();
    // // std::cout << "Output shape: [";
    // // for (size_t i = 0; i < shape.size(); ++i) {
    // //     std::cout << shape[i];
    // //     if (i < shape.size() - 1) {
    // //         std::cout << ", ";
    // //     }
    // // }
    // // std::cout << "]" << std::endl;

    // // get class_id to class_name mapping
    // const std::string filePath = "../label_mappings/labels_ade20kmodel.txt";

    // std::ifstream inputFile(filePath);
    // if (!inputFile.is_open()) {
    //     std::cerr << "Error opening the file: " << filePath << std::endl;
    //     return -1;
    // }

    // // Variables for reading from the file
    // char discard;
    // int labelIndex;
    // std::string line;
    // std::map<int, std::string> ade20k_model_labels;

    // // Read each line in the format {index: 'label',}
    // while (std::getline(inputFile, line, '\n')) {
    //     std::stringstream ss(line);
    //     ss >> labelIndex >> discard >> discard;
    //     std::string label;
    //     std::getline(ss >> std::ws, label, '\'');
    //     ade20k_model_labels[labelIndex] = label;
    // }

    // cout << to_string(ade20k_model_labels.size()) << endl;

    // cv::Mat processed_image;
    // std::string pred_class;
    // int pred_class_id;

    // // Font settings
    // int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    // double fontScale = 1.0;
    // int thickness = 2;
    // cv::Scalar textColor(255, 0, 0);  // BGR color

    // for(int i=0;i<cluster_ids.size();i++){
        
    //     pred_class_id = output[i].argmax().item<int>();
    //     pred_class = ade20k_model_labels[pred_class_id];

    //     auto it = clusters_objects.find(cluster_ids[i]);
    //     if (it != clusters_objects.end()) {
    //         processed_image = it->second.processed_image;
            
    //         // place label
    //         cv::Point textPosition(10, processed_image.rows - 10);
    //         cv::putText(processed_image, pred_class, textPosition, fontFace, fontScale, textColor, thickness);

    //         cv::namedWindow("Cluster ID " + std::to_string(cluster_ids[i]), cv::WINDOW_NORMAL);
    //         cv::imshow("Cluster ID " + std::to_string(cluster_ids[i]), processed_image);
    //         while (true) {
    //             int key = cv::waitKey(500); 

    //             if (key == 13) {  // ASCII value for Enter key
    //                 break;
    //             }
    //         }


    //     } else {
    //         std::cerr << "Key not found in the map." << std::endl;
    //     }
    // }

    return 0;

}