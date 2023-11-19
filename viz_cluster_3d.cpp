#include <GL/glew.h>
#include <GL/gl.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
using namespace std;


int main(int argc, char **argv) {

    if (argc < 2){
        cerr << "format: viz_cluster.exe (point cloud coords csv) (cluster data txt)" << endl;
        return 1;
    }

	std::string csvname = argv[1];
	std::string txtname = argv[2];

    // saving points to dataset variable
    std::ifstream file(csvname);

    if (!file.is_open()) {
        std::cerr << "Failed to open the CSV file." << std::endl;
        return 1;
    }

    // Create a vector of vectors to store the data
    std::vector<std::vector<std::string>> dataset;
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::istringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }

        dataset.push_back(row);
    }
    file.close();

    //saving cluster indices
    std::ifstream cluster_file(txtname);

    if (!cluster_file.is_open()) {
        std::cerr << "Failed to open the text file." << std::endl;
        return 1;
    }

    std::string cluster_line;
    std::vector<int> clusters;

    // Read a line from the text file
    while (std::getline(cluster_file, cluster_line)) {
        // Create a stringstream to tokenize the line
        std::istringstream lineStream(cluster_line);
        int cluster;

        // Read and convert each token to an integer
        while (lineStream >> cluster) {
            clusters.push_back(cluster);
        }
    }

    cluster_file.close();


    // making vectors of eigen vectors for visualization later
    std::vector<Eigen::Vector3f> colors;
    std::vector<Eigen::Vector3d> colors_bgr;
	std::vector<Eigen::Vector3d> points;

	// random colours for each cluster (assume max 80 cluster labels)
	for(int i=0;i<80;i++){
		Eigen::Vector3f eigen_colour;
		eigen_colour << (rand() % 256), (rand() % 256), (rand() % 256);
		colors.push_back(eigen_colour);
	}

    // dataset has RGBD and coords
    if (dataset[0].size()==6){
        // actual colours
        for(int j=0;j<dataset.size();j++){
            Eigen::Vector3d eigen_bgr;
            double b = std::stod(dataset[j][0]); // Convert the string to double
            double g = std::stod(dataset[j][1]);
            double r = std::stod(dataset[j][2]);
            eigen_bgr << b, g, r;
            colors_bgr.push_back(eigen_bgr);
        }

        for(int j=0;j<dataset.size();j++){
            Eigen::Vector3d eigen_point;
            double x = std::stod(dataset[j][3]); // Convert the string to double
            double y = std::stod(dataset[j][4]);
            double z = std::stod(dataset[j][5])/100;
            eigen_point << x, y, z;
            points.push_back(eigen_point);
        }
    }
    else{ // dataset only has coords
        for(int j=0;j<dataset.size();j++){
            Eigen::Vector3d eigen_point;
            double x = std::stod(dataset[j][1]); // Convert the string to double
            double y = std::stod(dataset[j][0]);
            double z = std::stod(dataset[j][2])/100;
            eigen_point << x, y, z;
            points.push_back(eigen_point);
        }

    }

    cout << "number of points: " << points.size() << endl;
    cout << "number of coord: " << points[0].size() << endl;


    // visualization with pangolin
	pangolin::CreateWindowAndBind("Point Cloud Viewer", 640, 480);
	glEnable(GL_DEPTH_TEST);
	// Create an OpenGlRenderState
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000), // Set your projection matrix
        pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisNegY) // Set the camera's initial view
    );

    // Create a view with the Handler3D using the OpenGlRenderState
    pangolin::View& view3d = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam)); // Associate the Handler3D with the render state

    // Add the view to your display
    pangolin::Display("RGBD Display")
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .AddDisplay(view3d);


	// pangolin::View &d_cam = pangolin::CreateDisplay()
    // .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f/768.0f)
    // .SetHandler(new pangolin::Handler3D(s_cam));
    

	while (!pangolin::ShouldQuit()) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		// Clear the display
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        view3d.Activate(s_cam);

        glPointSize(3.0f);

		// Your rendering code goes here
        glBegin(GL_POINTS);
        int color_i;

        for (size_t i = 0; i < points.size(); ++i) {
            color_i = clusters[i];
            glColor3ub(colors[color_i](0), colors[color_i](1), colors[color_i](2));
            //glColor3ub(colors_bgr[i][2], colors_bgr[i][1], colors_bgr[i][0]);
            glVertex3d(points[i][1], points[i][0], points[i][2]);
        }
        glEnd();

		pangolin::FinishFrame();
	}

    pangolin::DestroyWindow("Point Cloud Viewer");

    return 0;
}