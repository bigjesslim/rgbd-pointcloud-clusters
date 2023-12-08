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
    cv::Mat depth_map = cv::imread(depth_map_path, cv::IMREAD_ANYDEPTH);
    if (depth_map.empty()) {
        std::cout << depth_map_path << std::endl;
        std::cerr << "Error: Could not open or find the depth map" << std::endl;
        return 1;
    }
    std::vector<int> image_dims{depth_map.rows, depth_map.cols};

    // 1oad depth map into 1D vector (x,y,z)
    const std::vector<float> values = toXYZ(depth_map, 5);

    // convert depth map into 2D vector (x,y,z) [H,W]
    auto values_2D = convertTo2DVector(values, 3);
    auto points = values_2D;

    // normalize and CLUSTER data (DBSCAN) 
    std::vector<size_t> labels = normalizeAndCluster(argv[2], argv[3], argv[4], values_2D);

    std::set<int> cluster_set(labels.begin(), labels.end());
    std::vector<int> cluster_ids(cluster_set.begin(), cluster_set.end());
    int num_clusters = cluster_ids.size();

    // get xyz values for each cluster

    auto start_time = std::chrono::high_resolution_clock::now();
    std::map<int, std::vector<std::vector<float>>> values_2D_by_cluster;
    for (const auto& id : cluster_ids) {
        values_2D_by_cluster[id]; 
    }
    int num_points = values_2D.size();
    for (int i = 0; i < num_points; i++){
        values_2D_by_cluster[labels[i]].push_back(values_2D[i]);
    }

    // create cluster classes
    std::map<int, Cluster> clusters_objects;
    std::vector<int> new_cluster_ids;
    std::vector<torch::Tensor> batched_tensor;
    torch::Tensor processed_tensor;

    for (int i = 0; i < num_clusters; i++){
        if (cluster_ids[i] != 0){ // if not background cluster
            std::vector<std::vector<float>> cluster_values_2D = values_2D_by_cluster[cluster_ids[i]];
            if (cluster_values_2D.size()>=50){
                clusters_objects.emplace(std::piecewise_construct,
                                            std::forward_as_tuple(cluster_ids[i]),
                                            std::forward_as_tuple(rgb_image_path, image_dims, cluster_values_2D, cluster_ids[i]));
                new_cluster_ids.push_back(cluster_ids[i]);
                auto it = clusters_objects.find(cluster_ids[i]);
                if (it != clusters_objects.end()) {
                    processed_tensor = it->second.getProcessedTensor();
                    batched_tensor.push_back(processed_tensor);
                } else {
                    std::cerr << "Key not found in the map." << std::endl;
                }
            }
            else{ // insert small clusters with size < 30 into background cluster
                values_2D_by_cluster[0].insert(values_2D_by_cluster[0].end(), cluster_values_2D.begin(), cluster_values_2D.end());
            }
        }
    }
    // get background cluster (automatically classified as 'wall')
    std::vector<std::vector<float>> cluster_values_2D = values_2D_by_cluster[0];
    Cluster bg_cluster(rgb_image_path, image_dims, cluster_values_2D, 0);

    // consolidation
    cluster_ids = new_cluster_ids; // exclude small clusters
    torch::Tensor batch_tensor = torch::cat(batched_tensor, 0);

    torch::IntArrayRef shape = batch_tensor.sizes();
    std::cout << "Batch input shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;


    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float ms_duration = duration.count()/1000;
    std::cout << "Time taken for cluster processing: " << to_string(ms_duration) << " ms." << std::endl;

    if (torch::cuda::is_available()) {
        // Move the tensor to CUDA device (GPU)
        batch_tensor = batch_tensor.to(torch::kCUDA);
    } else {
        std::cerr << "CUDA is not available. Unable to move tensor to GPU.\n";
    }

    // classification
    torch::jit::FusionStrategy strat = {{torch::jit::FusionBehavior::DYNAMIC, 1}};
    torch::jit::setFusionStrategy(strat);
    const std::string model_path = "../trained_weights_fixed/efficientnetb1_50000_block7_mc_epoch5.pt";
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.to(torch::kCUDA);

    start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor output = model.forward({batch_tensor}).toTensor();
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    ms_duration = duration.count()/1000;
    std::cout << "Time taken for classification: " << to_string(ms_duration) << " ms." << std::endl;

    // torch::IntArrayRef shape = output.sizes();
    // std::cout << "Output shape: [";
    // for (size_t i = 0; i < shape.size(); ++i) {
    //     std::cout << shape[i];
    //     if (i < shape.size() - 1) {
    //         std::cout << ", ";
    //     }
    // }
    // std::cout << "]" << std::endl;

    // get class_id to class_name mapping
    const std::string filePath = "../label_mappings/labels_ade20kmodel.txt";

    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file: " << filePath << std::endl;
        return -1;
    }

    // Variables for reading from the file
    char discard;
    int labelIndex;
    std::string line;
    std::map<int, std::string> ade20k_model_labels;

    // Read each line in the format {index: 'label',}
    while (std::getline(inputFile, line, '\n')) {
        std::stringstream ss(line);
        ss >> labelIndex >> discard >> discard;
        std::string label;
        std::getline(ss >> std::ws, label, '\'');
        ade20k_model_labels[labelIndex] = label;
    }

    cout << to_string(ade20k_model_labels.size()) << endl;

    cv::Mat processed_image;
    std::string pred_class;
    int pred_class_id;

    // Font settings
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    cv::Scalar textColor(255, 0, 0);  // BGR color

    for(int i=0;i<cluster_ids.size();i++){
        
        pred_class_id = output[i].argmax().item<int>();
        pred_class = ade20k_model_labels[pred_class_id];

        auto it = clusters_objects.find(cluster_ids[i]);
        if (it != clusters_objects.end()) {
            processed_image = it->second.processed_image;
            
            // place label
            cv::Point textPosition(10, processed_image.rows - 10);
            cv::putText(processed_image, pred_class, textPosition, fontFace, fontScale, textColor, thickness);

            cv::namedWindow("Cluster ID " + std::to_string(cluster_ids[i]), cv::WINDOW_NORMAL);
            cv::imshow("Cluster ID " + std::to_string(cluster_ids[i]), processed_image);
            while (true) {
                int key = cv::waitKey(500); 

                if (key == 13) {  // ASCII value for Enter key
                    break;
                }
            }


        } else {
            std::cerr << "Key not found in the map." << std::endl;
        }
    }

    return 0;

}