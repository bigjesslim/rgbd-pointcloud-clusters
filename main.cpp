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
#include <algorithm>

using namespace std;


// noise will be labelled as 0

int main(int argc, char **argv) {
    if(argc != 6)
    {
        std::cerr << "usage: main <z weight> <epsilon> <min points> <weights path> <data folder>\n";
        return 1;
    }

    // Set all cmd line arguments
    const std::string z_weight = string(argv[1]);
    const std::string epsilon = string(argv[2]);
    const std::string min_points = string(argv[3]);
    const std::string model_path = string(argv[4]);
    std::string data_dir = string(argv[5]);

    // Load label mappings
    // id to label mappings for sunrgbd
    std::string sunrgbd_labels_path = "../label_mappings/labels_sunrgbd_NYUdata.txt"; // Replace with your file path
    std::vector<std::string> sunrgbd_labels = read_txt_to_vector(sunrgbd_labels_path);
    cout << "Number of SUNRGBD NYU classes: " << to_string(sunrgbd_labels.size()) << endl;
    // ade20k to sunrgbd label mapping
    std::string adeid_to_sunid_path = "../label_mappings/labels_adeid_to_sunid.txt"; // Replace with your file path
    std::vector<vector<int>> adeid_to_sunid_mapping = load_ade_to_sunrgbd_mapping(adeid_to_sunid_path);
    // get class_id to class_name mapping
    const std::string ade20k_labels_path = "../label_mappings/labels_ade20kmodel.txt";
    std::vector<std::string> ade20k_labels = read_txt_to_vector(ade20k_labels_path);

    // Metric variables
    const int num_sunrgbd_classes = sunrgbd_labels.size();
    torch::Tensor totalTP = torch::zeros({num_sunrgbd_classes});
    torch::Tensor totalFP = torch::zeros({num_sunrgbd_classes});
    torch::Tensor totalFN = torch::zeros({num_sunrgbd_classes});

    torch::Tensor totalIOU = torch::zeros({num_sunrgbd_classes});
    torch::Tensor totalDICE = torch::zeros({num_sunrgbd_classes});

    // std::vector<int> totalTP(num_sunrgbd_classes,0); 
    // std::vector<int> totalFP(num_sunrgbd_classes,0);
    // std::vector<int> totalFN(num_sunrgbd_classes,0);

    // std::vector<double> totalIOU(num_sunrgbd_classes,0); 
    // std::vector<double> totalDICE(num_sunrgbd_classes,0);

    // Load classification model
    torch::jit::FusionStrategy strat = {{torch::jit::FusionBehavior::DYNAMIC, 1}};
    torch::jit::setFusionStrategy(strat);
    //"../weights/trained_weights_vpad/efficientnetb1_50000_block7_mc_epoch4.pt";
    torch::jit::script::Module model = torch::jit::load(model_path);
    //model.to(torch::kCUDA);

    // Get filenames
    std::string filenames_path = data_dir + "filenames.txt";
    std::vector<std::string> filenames = read_txt_to_vector(filenames_path);
    bool binary_classifier = false; 

    for (std::string filename : filenames) {
        cout << filename << endl;

        std::string rgb_image_path = data_dir + "image/" + filename + ".jpg";
        std::string depth_map_path = data_dir + "depth/" + filename + ".png";
        std::string seglabel_path = data_dir + "seglabel/" + filename + ".xml";
        std::string seginstances_path = data_dir + "seginstances/" + filename + ".xml";
        
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

        // get seglabel matrix
        cv::FileStorage fs(seglabel_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Unable to open " << seglabel_path << std::endl;
            return -1;
        }

        cv::Mat seg_label(rgb_image.rows, rgb_image.cols, CV_32S, cv::Scalar(0));
        fs["Matrix"] >> seg_label;
        fs.release();

        // Display the matrix
        std::cout << "Matrix read from XML file:" << std::endl;
        std::cout << "Rows: " << seg_label.rows << std::endl;
        std::cout << "Cols: " << seg_label.cols << std::endl;

        // get seginstances matrix
        cv::FileStorage fs2(seginstances_path, cv::FileStorage::READ);
        if (!fs2.isOpened()) {
            std::cerr << "Error: Unable to open " << seginstances_path << std::endl;
            return -1;
        }

        cv::Mat seg_instances(rgb_image.rows, rgb_image.cols, CV_32S, cv::Scalar(0));
        fs2["Matrix"] >> seg_instances;
        fs2.release();

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<int> image_dims{depth_map.rows, depth_map.cols};

        // 1oad depth map into 1D vector (x,y,z)
        const std::vector<float> values = toXYZ(depth_map, 5);

        // convert depth map into 2D vector (x,y,z) [H,W]
        auto values_2D = convertTo2DVector(values, 3);
        auto points = values_2D;

        auto end_time_1 = std::chrono::high_resolution_clock::now();
        // normalize and CLUSTER data (DBSCAN) 
        std::vector<size_t> labels = normalizeAndCluster(z_weight, epsilon, min_points, values_2D);

        std::set<int> cluster_set(labels.begin(), labels.end());
        std::vector<int> cluster_ids(cluster_set.begin(), cluster_set.end());
        int num_clusters = cluster_ids.size();

        auto end_time_2 = std::chrono::high_resolution_clock::now();

        // get xyz values for each cluster

        std::unordered_map<int, cv::Mat> downsized_mask_by_cluster;
        std::unordered_map<int, cv::Mat> mask_by_cluster;
        std::unordered_map<int, std::vector<int>> xcoords_by_cluster;
        std::unordered_map<int, std::vector<int>> ycoords_by_cluster;
        int x;
        int y;
        std::vector<int> downsized_dims{(rgb_image.cols/5), (rgb_image.rows/5)};

        cout << "downsized dims: " << to_string(downsized_dims[0]) << ", " << to_string(downsized_dims[1]) << endl;  

        //instantiation of maps
        for (int i = 0; i < cluster_ids.size(); i++){
            // std::vector<int> xcoords_by_cluster[cluster_ids[i]];
            // std::vector<int> ycoords_by_cluster[cluster_ids[i]];

            cv::Mat downsized_mask(downsized_dims[1], downsized_dims[0], CV_8UC1, cv::Scalar(0));
            downsized_mask_by_cluster[cluster_ids[i]] = downsized_mask.clone();
            cv::Mat cv_mask(seg_label.rows, seg_label.rows, CV_8UC1, cv::Scalar(0));
            mask_by_cluster[cluster_ids[i]] = cv_mask.clone();
        }

        // populating maps of cluster data
        int num_points = values_2D.size();
        for (int i = 0; i < num_points; i++){
            x = values_2D[i][0];
            y = values_2D[i][1];
            int cluster_id = static_cast<int>(labels[i]);

            std::vector<int>& x_coords = xcoords_by_cluster[cluster_id];
            x_coords.push_back(x);
            std::vector<int>& y_coords = ycoords_by_cluster[cluster_id];
            y_coords.push_back(y);

            // safety assertions for downsized_mask_by_cluster
            auto it = downsized_mask_by_cluster.find(cluster_id);
            assert(it != downsized_mask_by_cluster.end() && "Invalid key in downsized_mask_by_cluster");

            assert(static_cast<int>(y/5) >= 0 && static_cast<int>(y/5) < downsized_dims[1]);
            assert(static_cast<int>(x/5) >= 0 && static_cast<int>(x/5) < downsized_dims[0]);

            cv::Mat& downsized_mask =  downsized_mask_by_cluster[cluster_id];
            downsized_mask.at<uchar>(static_cast<int>(y/5), static_cast<int>(x/5)) = 1;
        }

        // for(std::map<int,vector<int>>::iterator it = xcoords_by_cluster->begin(); it != xcoords_by_cluster->end(); ++it) {
        //     std::cout << "Key: " << it->first << std::endl;
        // }

        // final variables required
        std::vector<torch::Tensor> tensor_list; 
        std::vector<cv::Mat> processed_image_list; 
        std::vector<int> new_cluster_ids;

        cv::Mat cv_mask(downsized_dims[1], downsized_dims[0], CV_8UC1, cv::Scalar(0));
        cv::Mat rowsToAdd = cv::Mat::zeros(2, 560, cv_mask.type());
        cv::Mat colsToAdd = cv::Mat::zeros(427, 1, cv_mask.type());

        for (int i = 0; i < num_clusters; i++){
            int cluster_id = cluster_ids[i];
            if (cluster_id != 0){ // if not background cluster
                //cout << "current cluster: " << to_string(cluster_id) << endl;
                std::vector<int>& vec_x = xcoords_by_cluster[cluster_id];
                std::vector<int>& vec_y = ycoords_by_cluster[cluster_id];
                
                if (vec_x.size()>=50){ // if cluster size is at least 50
                    //cout << "cluster size: " << to_string(vec_x.size()) << endl;

                    // get masked image
                    cv::Mat resized_mask;
                    cv::Mat& downsized_mask = downsized_mask_by_cluster[cluster_id];
                    
                    
                    cv::Size targetSize(560, 425);
                    cv::resize(downsized_mask, resized_mask, targetSize, cv::INTER_NEAREST);
                    cv::vconcat(resized_mask, rowsToAdd, resized_mask);
                    cv::hconcat(resized_mask, colsToAdd, resized_mask);

                    mask_by_cluster[cluster_id] = resized_mask.clone();

                    cv::Mat masked_image;
                    cv::bitwise_and(rgb_image, rgb_image, masked_image, resized_mask);

                    // get crop dimensions
                    int min_x = *std::min_element(vec_x.begin(), vec_x.end());
                    int max_x = *std::max_element(vec_x.begin(), vec_x.end());

                    int min_y = *std::min_element(vec_y.begin(), vec_y.end());
                    int max_y = *std::max_element(vec_y.begin(), vec_y.end());

                    if (max_x > 561){
                        max_x = 561;
                    }

                    if (max_y > 427){
                        max_y = 427;
                    }

                    cv::Mat cropped_image = masked_image(cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y));
                    if ((cropped_image.cols == 0)|| (cropped_image.rows == 0)){
                        continue;
                    }

                    new_cluster_ids.push_back(cluster_id);

                    paddingAndResize(cropped_image, cv::Size(240, 240));
                    processed_image_list.push_back(cropped_image);

                    cv::Mat processed_image = cropped_image.clone();

                    cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB); // bgr (opencv) to rgb (pytorch)
                    torch::Tensor processed_tensor = torch::from_blob(processed_image.data, {1, processed_image.rows, processed_image.cols, 3}, torch::kByte);
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

        // torch::IntArrayRef shape = batch_tensor.sizes();
        // std::cout << "Batch input shape: [";
        // for (size_t i = 0; i < shape.size(); ++i) {
        //     std::cout << shape[i];
        //     if (i < shape.size() - 1) {
        //         std::cout << ", ";
        //     }
        // }
        // std::cout << "]" << std::endl;


        // if (torch::cuda::is_available()) {
        //     // Move the tensor to CUDA device (GPU)
        //     batch_tensor = batch_tensor.to(torch::kCUDA);
        // } else {
        //     std::cerr << "CUDA is not available. Unable to move tensor to GPU.\n";
        // }

        start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor output = model.forward({batch_tensor}).toTensor();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        float ms_duration = duration.count()/1000;
        std::cout << "Time taken for classification: " << to_string(ms_duration) << " ms." << std::endl;


        // get gt object instances
        std::vector<std::tuple<int, int>> merged_vector;
        for (int i = 0; i < seg_label.rows; ++i) {
            for (int j = 0; j < seg_label.cols; ++j) {

                if (seg_label.at<int>(i, j) > 894){
                    std::cout << "invalid value "  <<  to_string(seg_label.at<int>(i, j)) << endl;
                }
                // assert(seg_label.at<int>(i, j) <= 894);
                // assert(seg_label.at<int>(i, j) >= 0);
                merged_vector.push_back(std::make_tuple(seg_label.at<int>(i, j), seg_instances.at<int>(i, j)));
            }
        }
        std::set<std::tuple<int, int>> unique_tuples_set;

        // Iterate over the vector and insert into the set (which automatically discards duplicates)
        for (const auto& tuple : merged_vector) {
            unique_tuples_set.insert(tuple);
        }

        std::cout << "Number of unique gt instances: " << unique_tuples_set.size() << std::endl;

        std::vector<int> gt_class_ids;
        std::vector<int> gt_instance_ids;
        for (std::tuple<int, int> instance: unique_tuples_set){
            int class_id = std::get<0>(instance);
            if (class_id > 0){ // 0 in seg mat means unknown/background 
                //cout << to_string(class_id) << ": ";
                //cout << sunrgbd_labels[class_id-1] << endl;
                gt_class_ids.push_back(std::get<0>(instance));
                gt_instance_ids.push_back(std::get<1>(instance));
            }
        }
        // // Font settings
        // int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        // double fontScale = 1.0;
        // int thickness = 2;
        // cv::Scalar textColor(255, 0, 0);  // BGR color
        if (output[0].sizes()[0] == 1) { binary_classifier = true;}

        for(int i=0;i<cluster_ids.size();i++){
            int pred_class_id;
            
            // if binary model is used
            //cout << output[i].sizes()[0] << endl;
            if (binary_classifier) {
                cout << "output probability: " << output[i].item<float>() << endl;
                if (output[i].item<float>()>0.10){
                    pred_class_id = 74;
                }else{
                    pred_class_id = 164;
                }
            } else {
                pred_class_id = output[i].argmax().item<int>();
            }
            
            // cout << "final prediction score: " << to_string(output[i][pred_class_id].item<float>()) << endl;
            // cout << "person prediction score: " << to_string(output[i][74].item<float>()) << endl;
            std::string pred_class = ade20k_labels[pred_class_id];

            // corresponding sunrgbd labels
            std::vector<int> sunrgbd_ids = adeid_to_sunid_mapping[pred_class_id];
            int max_indice = -1;
            double max_iou = 0;
            double dice_at_max_iou = 0;


            // for each of the corresponding sunrgbd class ids
            for (int sunrgbd_id: sunrgbd_ids){
                auto it = std::find(gt_class_ids.begin(), gt_class_ids.end(), sunrgbd_id);
                std::vector<int> matched_indices;

                // getting gt indices which match the current crop classification
                while (it != gt_class_ids.end()) {
                    int index = static_cast<int>(std::distance(gt_class_ids.begin(), it));
                    matched_indices.push_back(index);
                    it = std::find(std::next(it), gt_class_ids.end(), sunrgbd_id);
                }

                cv::Mat all_class_id(seg_label.rows, seg_label.cols, CV_32S, cv::Scalar(sunrgbd_id));

                // checking iou for each matched instance
                for (int indice: matched_indices){
                    int instance_id = gt_instance_ids[indice];
                    cv::Mat all_instance_id(seg_instances.rows, seg_instances.cols, CV_32S, cv::Scalar(instance_id));

                    // get instance mask
                    cv::Mat class_mask;
                    cv::Mat instance_mask;
                    cv::Mat gt_mask(seg_label.rows, seg_label.cols, CV_8UC1, cv::Scalar(0));

                    cv::compare(seg_label, all_class_id, class_mask, cv::CMP_EQ);
                    cv::compare(seg_instances, all_instance_id, instance_mask, cv::CMP_EQ);
                    cv::bitwise_and(class_mask, instance_mask, gt_mask);

                    double iou = calculateIoU(gt_mask, mask_by_cluster[cluster_ids[i]]);
                    double dice = (2*iou)/(1+iou);

                    // // VISUALIZATION
                    // cv::namedWindow("Matched class " + sunrgbd_labels[sunrgbd_id-1] + ", iou = " + to_string(iou), cv::WINDOW_NORMAL);
                    // cv::imshow("Matched class " + sunrgbd_labels[sunrgbd_id-1] + ", iou = " + to_string(iou), gt_mask);
                    // while (true) {
                    //     int key = cv::waitKey(500); 

                    //     if (key == 13) {  // ASCII value for Enter key
                    //         cv::destroyWindow("Matched class " + sunrgbd_labels[sunrgbd_id-1] + ", iou = " + to_string(iou));
                    //         break;
                    //     }
                    // }

                    if (iou > max_iou){
                        max_iou = iou;
                        dice_at_max_iou = dice;
                        max_indice = indice;
                    }

                }

            }

            assert(gt_class_ids.size() == gt_instance_ids.size() && "gt class and instance vectors of different sizes.");

            // TP MATCH! 
            // change based on mAP 0.25 or mAP 0.5
            cout << "max iou: " << to_string(max_iou) << endl;

            float iou_cutoff = 0.25;
            if (max_iou > iou_cutoff){

                // get matched sunrgbd class id and class name
                int matched_class_id = gt_class_ids[max_indice]; 
                std::string matched_class_name = sunrgbd_labels[matched_class_id-1];

                std::cout << "TP MATCHED!" << " CLASS = " << matched_class_name << ", IOU = " << to_string(max_iou) << std::endl;

                // remove the matched indice from the lookup vectors
                if (max_indice < gt_class_ids.size()) {
                    gt_class_ids.erase(gt_class_ids.begin() + max_indice);
                    gt_instance_ids.erase(gt_instance_ids.begin() + max_indice);
                } else {
                    std::cerr << "Index out of bounds." << std::endl;
                }
                // update aggregate metrics
                int matched_class_id_key = matched_class_id-1;
                totalTP[matched_class_id_key]+=1; // +1 to TP count 
                totalIOU[matched_class_id_key]+= max_iou; // add iou and dice into totalIOU and totalDICE
                totalDICE[matched_class_id_key]+= dice_at_max_iou;
            } else {
                // +1 to FP count of first mapped sunrgbd class (if any sunrgbd classes are mapped)
                if (sunrgbd_ids.size() > 0){
                    totalFP[sunrgbd_ids[0]-1]+=1; 
                }

                cout << "NO TP MATCH! MAX IOU = " << to_string(max_iou) << " < " << to_string(iou_cutoff) << endl;
            }

            // // VISUALIZATION 
            // cv::Mat processed_image = processed_image_list[i].clone();
            
            // // place label
            // cv::Point textPosition(10, processed_image.rows - 10);
            // cv::putText(processed_image, pred_class, textPosition, fontFace, fontScale, textColor, thickness);

            // cv::namedWindow("Cluster ID " + std::to_string(new_cluster_ids[i]), cv::WINDOW_NORMAL);
            // cv::imshow("Cluster ID " + std::to_string(new_cluster_ids[i]), processed_image);
            // while (true) {
            //     int key = cv::waitKey(500); 

            //     if (key == 13) {  // ASCII value for Enter key
            //         cv::destroyWindow("Cluster ID " + std::to_string(new_cluster_ids[i]));
            //         break;
            //     }
            // }
        }

        // update totalFN using remaining unmatched gt class instances
        assert(gt_class_ids.size() == gt_instance_ids.size()  && "gt class and instance vectors of different sizes.");
        for (int gt_class_id: gt_class_ids){
            totalFN[gt_class_id-1]+=1;
        }

        // calculate metrics
        cout << "Number of unmatched GT labels = " << to_string(gt_class_ids.size()) << endl;
    }
    // Open logging txt
    std::istringstream ss(model_path);
    std::string part;
    std::string model_name;
    while (std::getline(ss, part, '/')) {}
    if (part.length() >= 3) {
        model_name = part.substr(0, part.length() - 3);
        std::cout << "Model name: " << model_name << std::endl;
    } else {
        std::cerr << "Model name cannot be split from model path correctly." << std::endl;
    }

    if (binary_classifier){model_name = "binary_" + model_name;}
    std::ofstream loggingFile("../log/" + model_name + "_results.txt");

    if (loggingFile.is_open()) {
        // Log all classes metrics -> mAP@0.25 or mAP@0.5
        torch::Tensor v_small_tensor = torch::full({num_sunrgbd_classes}, 0.0001);

        torch::Tensor tp_fp = torch::add(totalTP, totalFP);
        torch::Tensor prec_indices = torch::nonzero(tp_fp).squeeze();
        torch::Tensor prec_selected_TP = torch::index_select(totalTP, 0, prec_indices);
        torch::Tensor prec_selected_FP = torch::index_select(totalFP, 0, prec_indices);
        torch::Tensor prec_all = torch::div(prec_selected_TP, torch::add(prec_selected_TP, prec_selected_FP));

        torch::Tensor tp_fn = torch::add(totalTP, totalFN);
        torch::Tensor rec_indices = torch::nonzero(tp_fn).squeeze();
        torch::Tensor rec_selected_TP = torch::index_select(totalTP, 0, rec_indices);
        torch::Tensor rec_selected_FN = torch::index_select(totalFN, 0, rec_indices);
        torch::Tensor rec_all = torch::div(rec_selected_TP, torch::add(rec_selected_TP, rec_selected_FN));
        
        //torch::Tensor f1_all = 2 * torch::div(torch::mul(prec_all, rec_all), torch::max(torch::add(prec_all, rec_all), v_small_tensor));

        float mean_prec =  torch::mean(prec_all).item<float>();
        float mean_rec =  torch::mean(rec_all).item<float>();
        //float mean_f1 =  torch::mean(f1_all).item<float>();

        loggingFile << "<Overall class metrics>" << endl;
        loggingFile << "Classification: " << endl;
        loggingFile << "mean precision: "  << to_string(mean_prec) << endl;
        loggingFile << "mean recall: " << to_string(mean_rec) << endl;
        //loggingFile << "mean f1: " << to_string(mean_f1) << endl;

        torch::Tensor tp_indices = torch::nonzero(totalTP).squeeze();
        torch::Tensor iou_all = torch::div(torch::index_select(totalIOU, 0, tp_indices), torch::index_select(totalTP, 0, tp_indices));
        torch::Tensor dice_all = torch::div(torch::index_select(totalDICE, 0, tp_indices), torch::index_select(totalTP, 0, tp_indices));

        float mean_iou =  torch::mean(iou_all).item<float>();
        float mean_dice =  torch::mean(dice_all).item<float>();

        loggingFile << endl << "Segmentation: " << endl;
        loggingFile << "mIOU: " << to_string(mean_iou) << endl;
        loggingFile << "mDICE: " << to_string(mean_dice) << endl;


        // Log person metrics
        loggingFile << endl <<  endl << "<Person vs non-person metrics>" << endl;
        loggingFile << "Classification: " << endl;

        float tp_person = totalTP[330].item<float>();
        float fp_person = totalFP[330].item<float>();
        float fn_person = totalFN[330].item<float>();

        loggingFile << "tp for persons: " << to_string(tp_person) << endl;
        loggingFile << "fp for persons: " << to_string(fp_person) << endl;
        loggingFile << "fn for persons: " << to_string(fn_person) << endl;

        float prec_person = tp_person/(tp_person + fp_person);
        float rec_person = tp_person/(tp_person + fn_person);
        float f1_person = (2*prec_person*rec_person)/(prec_person+rec_person);
        loggingFile << "precision for persons: "  << to_string(prec_person) << endl;
        loggingFile << "recall for persons: " << to_string(rec_person) << endl;
        loggingFile << "f1 for persons: " << to_string(f1_person) << endl;


        // segmentation metrics
        float miou_person = totalIOU[330].item<float>()/tp_person;
        float mdice_person = totalDICE[330].item<float>()/tp_person;

        loggingFile << endl << "Segmentation: " << endl;
        loggingFile << "mIOU for persons: "  << to_string(miou_person) << endl;
        loggingFile << "mDICE for persons: " << to_string(mdice_person) << endl;

        loggingFile.close();

    } else {
        std::cerr << "Error opening the file." << std::endl;
    }


    return 0;

}