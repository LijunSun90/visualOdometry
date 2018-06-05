// visualOdometry.cpp
// Calling convention:
// ./visualOdometry [frames_directory]
// ./visualOdometry /home/lijun/data/KITTI_odometry_color/dataset/

#include <istream>
#include <ostream>
#include <opencv2/opencv.hpp>

using namespace std;

void help(char* argv[]){
}

/**
 * @brief Obtain the 3x4 projection matrix after rectification
 *        from the KITTI odometry dataset.
 * @param file_name: the calibration file to be parsed.
 * @param PO: for camera0, the left gray camera.
 * @param P1: for camera1, the right gray camera.
 * @param P2: for camera2, the left color camera.
 * @param P3: for camera3, the right gray camera.
 */
void parserCalib(string file_name , cv::Mat & P0, cv::Mat & P1, cv::Mat & P2, cv::Mat & P3){
    ifstream loaded_projection_matrices(file_name.c_str());
    if(!loaded_projection_matrices.is_open()){
        cout << "ERROR: cannot open the file " + file_name << endl;
        exit(-1);
    }
    string lines, entry;
    istringstream linestream;
    // P0.
    //
    getline(loaded_projection_matrices, lines);
    linestream.str(lines);
    linestream >> entry;
    cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            // cout << entry << endl;
            P0.at<double>(row, col) = atof(entry.c_str());
        }
    }
    cout << P0 << endl;
    // P1.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P1.at<double>(row, col) = atof(entry.c_str());
        }
    }
    cout << P1 << endl;
    // P2.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P2.at<double>(row, col) = atof(entry.c_str());
        }
    }
    cout << P2 << endl;
    // P3.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P3.at<double>(row, col) = atof(entry.c_str());
        }
    }
    cout << P3 << endl;

} // END OF void parserCalib().


/**
 * Find features in the left and right images of the stereo rig.
 * Correspond the found features in the stereo images.
 * @return
 * @param image_1, the former frame.
 * @param image_2, the latter frame.
 * @param keypoints_1, features in image_1.
 * @param keypoints_2, features in image_2.
 * @param matches, correspondence between keypoints_1 and keypoints_2.
*/
void trackFeatures(
    cv::Mat & image_1,
    cv::Mat & image_2,
    vector< cv::KeyPoint > & keypoints_1,
    vector< cv::KeyPoint > & keypoints_2, 
    vector < cv::DMatch > & matches) {

    cv::Mat image_out;

    // Detector definition.
    cv::Ptr<cv::Feature2D> f2d;
    // Step 1: Detect the keypoints.
    // Step 2: Extract descriptors (feature vectors).
    cv::Mat descriptors_1, descriptors_2;
    // Step 3: Matching descriptor vectors.
    cv::FlannBasedMatcher matcher_flann;
    //

    // ORB detector and extractor + FLANN.
    //
    f2d = cv::ORB::create();
    // Step 1 and 2: Detect the keypoints & Extract descriptors.
    f2d->detectAndCompute(image_1, cv::noArray(), keypoints_1, descriptors_1);
    f2d->detectAndCompute(image_2, cv::noArray(), keypoints_2, descriptors_2);
    // cout << "\nkeypoints_1: " << keypoints_1.at(0).pt << endl;
    // cout << "keypoints_2: " << keypoints_2.at(0).pt << endl;
    // cout << "descriptors_1: " << descriptors_1 << endl;
    // Step 3: Matching descriptor vectors.
    if(descriptors_1.type()!=CV_32F) {
        descriptors_1.convertTo(descriptors_1, CV_32F);
    }
    if(descriptors_2.type()!=CV_32F) {
        descriptors_2.convertTo(descriptors_2, CV_32F);
    }
    matcher_flann.match(descriptors_1, descriptors_2, matches);
    // cout << "matches queryIdx: " << matches.at(0).queryIdx
    // << "\nmatches trainIdx: " << matches.at(0).trainIdx
    // << "\nmatches imgIdx: " << matches.at(0).imgIdx
    // << endl;
    // Display results.
    // cv::drawMatches(
    //     image_1,
    //     keypoints_1,
    //     image_2,
    //     keypoints_2,
    //     matches,
    //     image_out,
    //     cv::Scalar::all(-1),
    //     cv::Scalar::all(-1),
    //     vector<char>(),
    //     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // cv::imshow("ORB matches between the stereo images.", image_out);
    // if(cv::waitKey(50) == 27) exit(0);

} // End of extractCorrespondeces.




int main(int argc, char* argv[]){

    // Read the input parameters.
    if(argc < 2){
        cout << "ERROR: Wrong number of parameters!" << endl;
        help(argv);
    }

    string dir_name = argv[1];

    // Obtain the 3x4 projection matrix after rectification.
    // PO: for camera0, the left gray camera.
    // P1: for camera1, the right gray camera.
    // P2: for camera2, the left color camera.
    // P3: for camera3, the right gray camera.
    //
    cv::Mat P0 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P3 = cv::Mat::zeros(3, 4, CV_64F);

    string file_name = dir_name + "/sequences/00/calib.txt";
    parserCalib(file_name, P0, P1, P2, P3);


    // Obtain the N frames of stereo images: image_1, image_2.
    // image_1 is for the left image, image_2 is for the right image.
    //
    string image_1_names_file = dir_name + "/sequences/00/image_2.txt";
    string image_2_names_file = dir_name + "/sequences/00/image_3.txt";

    ifstream image_1_names(image_1_names_file.c_str());
    if(!image_1_names.is_open()){
      cout << "ERROR: Cannot open the file "
        << image_1_names_file << endl;
      exit(-1);
    }

    ifstream image_2_names(image_2_names_file.c_str());
    if(!image_2_names.is_open()){
      cout << "ERROR: Cannot open the file "
        << image_2_names_file << endl;
      exit(-1);
    }

    // The number of image_1 is the same to that of the image_2.
    // So is the names of image_left and image_2.
    string image_1_name, image_2_name, image_1_name_former;
    cv::Mat image_1, image_2;
    //
    // Detector definition.
    cv::Ptr<cv::Feature2D> f2d;
    // Step 1: Detect the keypoints.
    vector< cv::KeyPoint > keypoints_1, keypoints_2, keypoints_1_former;
    // Step 2: Extract descriptors (feature vectors).
    cv::Mat descriptors_1, descriptors_2, descriptors_1_former;
    // cv::Ptr<cv::DescriptorExtractor> extractor;
    // Step 3: Matching descriptor vectors using BFMatcher.
    // cv::BFMatcher matcher_bf;
    cv::FlannBasedMatcher matcher_flann;
    vector < cv::DMatch > matches;
    //
    // Get the most first frame.
    //
    // Obtain the relative affine transformation (R | t)
    // between the former and the latter frames.
    //
    vector< cv::Mat > R, t;
    //
    // Step 1: Obtain the 4x4 reprojection matrix.
    //
    // Step 2: Obtain the disparity map of the left stereo image.
    //
    // Step 3: Obtain the depth map of the left stereo image.
    //
    while(getline(image_1_names, image_1_name)){
        // Obtain the aboslute data path from the relative path.
        image_1_name =  dir_name + "/sequences/00/" + image_1_name;

        // Read in the correspondent left and right image of the stereo rig.
        //
        getline(image_2_names, image_2_name);
        image_2_name = dir_name + "sequences/00/" +   image_2_name;
        image_1 = cv::imread(image_1_name, cv::IMREAD_GRAYSCALE);
        image_2 = cv::imread(image_2_name, cv::IMREAD_GRAYSCALE);

        // Find the features to track.
        //
        // trackFeatures(image_1, image_2, keypoints_1, keypoints_2, matches);
        

    } // END OF while(getline(image_1_names, image_1_name)).

    // Ground truth poses.
    //
    vector< cv::Mat > Rs_truth, ts_truth;
    // File name.
    string poses_groundTruth_file;
    poses_groundTruth_file = dir_name + "/poses/" + "00.txt";
    // Load the file.
    ifstream loaded_poses(poses_groundTruth_file.c_str());
    if(!loaded_poses.is_open()){
        cout << "ERROR: cannot open the file " + file_name << endl;
        exit(-1);
    }
    //
    string lines, entry;
    istringstream linestream;
    // trajectory: Horizontal: x; Vertical: z.
    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    // cv::Point3f origin = cv::Point3f(300, 300, 300);
    // cout << origin.x << origin.y << endl;
    cv::Point3f origin[2] = {
        cv::Point3f(300, 0, 100),
        cv::Point3f(316, 0, 127)
    };
    vector< cv::Mat> origin_homogeneous;
    cv::Mat dst = cv::Mat::zeros(4, 1, CV_64F);
    cv::convertPointsToHomogeneous(cv::Point3f(300, 0, 100), dst);
    origin_homogeneous.push_back(dst);
    cv::convertPointsToHomogeneous(cv::Point3f(316, 0, 127), dst);
    origin_homogeneous.push_back(dst);
    
    cv::Point3f pose_current;
    //
    while(getline(loaded_poses, lines)){
        linestream.clear();
        linestream.str(lines);
        // cv::Mat R_truth = cv::Mat::zeros(3, 3, CV_64F);
        // cv::Mat t_truth = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat transformation = cv::Mat::zeros(3, 4, CV_64F);
        // // R_truth.
        // //
        // for(int row = 0; row < 3; row++){
        //     for(int col = 0; col < 3; col++){
        //         linestream >> entry;
        //         // cout << entry << endl;
        //         R_truth.at<double>(row, col) = atof(entry.c_str());
        //     }
        // } 
        // Rs_truth.push_back(R_truth);
        // // t_truth.
        // //
        // for(int row = 0; row < 3; row++){
        //     linestream >> entry;
        //     t_truth.at<double>(row) = atof(entry.c_str());
        // }
        // ts_truth.push_back(t_truth);
        // 
        // pose_current = R_truth * origin + t_truth;

        for(int row = 0; row < 3; row++){
            for(int col = 0; col < 4; col++){
                linestream >> entry;
                // cout << entry << endl;
                transformation.at<double>(row, col) = atof(entry.c_str());
            }
        } 

        cv::transform(
            origin,
            dst,
            transformation
        )

    } // while(getline(loaded_poses, lines)).
    
   






    cv::destroyAllWindows();
    return 0;
}
