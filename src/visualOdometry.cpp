// visualOdometry.cpp
// Calling convention:
// ./visualOdometry [frames_directory]
// ./visualOdometry ~/data/KITTI_odometry_color/dataset/

#include <istream>
#include <ostream>
#include <math.h>
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
void parserCalib(string file_name , cv::Mat & Q){

    // Obtain the 3x4 projection matrix after rectification.
    // PO: for camera0, the left gray camera.
    // P1: for camera1, the right gray camera.
    // P2: for camera2, the left color camera.
    // P3: for camera3, the right gray camera.
    
    cv::Mat P0 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P3 = cv::Mat::zeros(3, 4, CV_64F);

    ifstream loaded_projection_matrices(file_name.c_str());
    if(!loaded_projection_matrices.is_open()){
        cout << "ERROR: cannot open the file " + file_name << endl;
        exit(-1);
    }
    string lines, entry;
    istringstream linestream;
    // P0. 0 represents the left grayscale camera.
    //
    getline(loaded_projection_matrices, lines);
    linestream.str(lines);
    linestream >> entry;
    // cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            // cout << entry << endl;
            P0.at<double>(row, col) = atof(entry.c_str());
        }
    }
    // cout << P0 << endl;
    // P1. 1 represents the right grayscale camera.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    // cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P1.at<double>(row, col) = atof(entry.c_str());
        }
    }
    // cout << P1 << endl;
    // P2. 2 represents the left color camera.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    // cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P2.at<double>(row, col) = atof(entry.c_str());
        }
    }
    // cout << P2 << endl;
    // P3. 3 represents the right color camera.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    // cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P3.at<double>(row, col) = atof(entry.c_str());
        }
    }
    // cout << P3 << endl;
    //
    // Get the camera intrinsics (cx, cy, f)
    // and the extrinsics tx in (tx, ty, tz).
    // Here, we took the baseline between the left and right
    // stereo cameras as tx.
    double cx, cy, f, baseline, baselinePoint2, baselinePoint3;
    
    cx = P2.at<double>(0, 2);
    cy = P2.at<double>(1, 2);
    f = P2.at<double>(0, 0);
    baselinePoint2 = P2.at<double>(0, 3) / (-f);
    baselinePoint3 = P3.at<double>(0, 3) / (-f);
    baseline = baselinePoint3 - baselinePoint2;

    // cout << "\ncx: " << cx << "\n"
    // << "cy: " << cy << "\n"
    // << "f: " << f << "\n"
    // << "baselinePoint2: " << baselinePoint2 << "\n"
    // << "baselinePoint3: " << baselinePoint3 << "\n"
    // << "baseline: " << baseline << endl;

    // Build the reprojection matrix Q.
    Q = cv::Mat::eye(4, 4, CV_64F);
    Q.at<double>(0, 3) = -cx;
    Q.at<double>(1, 3) = -cy;
    Q.at<double>(2, 3) = f;
    Q.at<double>(3, 3) = 0;
    Q.at<double>(3, 2) = -1/f;

    cout << "\nQ: \n" << Q << endl;

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
void matchFeatures(
    cv::Mat image_1,
    cv::Mat image_2,
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
    cv::drawMatches(
        image_1,
        keypoints_1,
        image_2,
        keypoints_2,
        matches,
        image_out,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("ORB matches between the stereo images.", image_out);
    if(cv::waitKey(50) == 27) exit(0);

} // End of matchFeatures.


/**
 * @param image_1, the left RECTIFIED stereo image.
 * @param image_2, the right RECTIFIED stereo image.
 * @param Q, the 4x4 reprojection matrix, which maps the 2-D coordiantes
 *           withe the disparity (x, y, d) to the 3-D 3-D coordinates 
 *           in the physical world (X/W, Y/W, Z/W).
 * @param image_3D, (X/W, Y/W, Z/W),3-D coordinates in the physical world.
 */
void triangulateStereo(
    string image_1_name,
    string image_2_name,
    cv::Mat Q,
    cv::Mat & image_1,
    cv::Mat & image_3D){

    // cout << "triangulateStereo" << endl;

    cv::Mat image_2;
    image_1 = cv::imread(image_1_name, cv::IMREAD_GRAYSCALE);
    image_2 = cv::imread(image_2_name, cv::IMREAD_GRAYSCALE);

    // Step 1: Disparity.
    //         using the stero correspondence.
    //
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        -64, 128, 11, 0, 0, 12, 
        0, 15, 1000, 16,
        cv::StereoSGBM::MODE_SGBM // cv::StereoSGBM::MODE_HH
    );
    // 
    cv::Mat disparity, vdisparity;
    stereo->compute(image_1, image_2, disparity);
    // cout << disparity << endl;
    // This is for displaying the disparity image only.
    cv::normalize(disparity, vdisparity, 0, 256, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity_map", vdisparity);
    cv::moveWindow("Disparity_map", 0, 0);

    // Step 2: Depth map.
    cv::reprojectImageTo3D(
        disparity,
        image_3D,
        Q,
        false,
        -1
    );
    // cout << "\nimage_3D: \n" << image_3D << endl;
    cv::imshow("Depth_map", image_3D);
    cv::moveWindow("Depth_map", 0, vdisparity.size().height + 60);

    if(cv::waitKey(1) == 27) exit(0);

} // END OF triangulateStereo().


/**
 * @param keypoints
 * @param image_3D, (X/W, Y/W, Z/W),3-D coordinates in the physical world.
 * @param keypoints3D, the collection of (X/W, Y/W, Z/W) in the order of
 *                     the keypoints.
 */
void get3Dkeypoints(
    vector< cv::KeyPoint > keypoints,
    cv::Mat image_3D,
    vector< cv::Vec3f > & keypoints3D){
    
    cv::Vec3f point3D;
    for(size_t ix = 0; ix < keypoints.size(); ix++){
        // Find the corresponding 3D coordinates of the specific
        // keypoints[ix].pt
        // and store it in keypoints3D.
        point3D = image_3D.at<cv::Vec3f>(keypoints[ix].pt);
        keypoints3D.push_back(point3D);
        // cout << "point3D: " << point3D << endl;
    } // END OF for(size_t ix = 0; ix < keypoints.size(); ix++).

} // END OF get3Dkeypoints().


/**
 * Sort keypoints3D_former and keypoints3D_new in the order
 * according to @param matches.
 */
void sortKeypoints3D(
    vector< cv::Vec3f > keypoints3D_former, 
    vector< cv::Vec3f > & keypoints3D_new,
    vector < cv::DMatch > matches){

    vector< cv::Vec3f > keypoints3D_temp;
    for(size_t ix = 0; ix < matches.size(); ix++){

        keypoints3D_temp.push_back(keypoints3D_new[matches[ix].trainIdx]);

    } // END OF for(size_t ix = 0; ix < matches.size(); ix++).

    // Update.
    //
    keypoints3D_new = keypoints3D_temp;

} // END OF sortKeypoints3D().


void getRelativeTransform(
    vector< cv::KeyPoint > keypoints_former,
    vector< cv::KeyPoint > keypoints_new,
    vector < cv::DMatch > matches,
    cv::Mat image_3D_former,
    cv::Mat image_3D_new,
    cv::Mat & R_relative,
    cv::Mat & t_relative
){
    R_relative = cv::Mat::eye(3, 3, CV_64F);
    t_relative = cv::Mat::zeros(3, 1, CV_64F);
    // cout << "\nR_relative: \n" << R_relative
    // << "\nt_relative: \n" << t_relative << endl;

    vector< cv::Vec3f > keypoints3D_former, keypoints3D_new;
    get3Dkeypoints(keypoints_former, image_3D_former, keypoints3D_former);
    get3Dkeypoints(keypoints_new, image_3D_new, keypoints3D_new);
    // Sort keypoints3D_former and keypoints3D_new in the order
    // according to matches.
    sortKeypoints3D(keypoints3D_former, keypoints3D_new, matches);

    

}

/**
 * @param dir_name, the name of the directory containing the
 *                  RECTIFIED stereo images.
 * @param Q, the 4x4 reprojection matrix, which maps the 2-D coordiantes
 *           withe the disparity (x, y, d) to the 3-D 3-D coordinates 
 *           in the physical world (X/W, Y/W, Z/W).
 * @param Rs, the collection of the rotation matrix.
 * @param ts, the collection of the translation matrix.
 */
void getTrajectory(
    string dir_name,
    cv::Mat Q,
    vector< cv::Mat > & Rs,
    vector< cv::Mat > & ts){

    // Open the stereo left and right RECTIFIED images files and check.
    // image_1 correspond to the left RECTIFIED stereo image.
    // image_2 correspond to the right RECTIFIED stereo image.
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

    string image_1_name, image_2_name;
    cv::Mat image_1_former, image_1_new;
    cv::Mat image_3D_former, image_3D_new;
    
    // Get the most first frame.
    //
    getline(image_1_names, image_1_name);
    image_1_name =  dir_name + "/sequences/00/" + image_1_name;

    getline(image_2_names, image_2_name);
    image_2_name = dir_name + "sequences/00/" +   image_2_name;

    triangulateStereo(image_1_name, image_2_name, Q, image_1_former, image_3D_former);

    // Get the relative transformation between frames.
    // 
    while(getline(image_1_names, image_1_name)){

        // Obtain the aboslute data path from the relative path.
        image_1_name =  dir_name + "/sequences/00/" + image_1_name;
        getline(image_2_names, image_2_name);
        image_2_name = dir_name + "sequences/00/" +   image_2_name;
        triangulateStereo(image_1_name, image_2_name, Q, image_1_new, image_3D_new);

        // Find the features, and match them between 
        // the former and the new frames of the left camera.
        // 
        vector< cv::KeyPoint > keypoints_former, keypoints_new;
        vector < cv::DMatch > matches;
        matchFeatures(image_1_former, image_1_new, keypoints_former, keypoints_new, matches);

        // Calculate the relatvie transformation between the former and the new frames.
        //
        cv::Mat R_relative, t_relative;
        getRelativeTransform(
            keypoints_former, 
            keypoints_new, 
            matches, 
            image_3D_former, 
            image_3D_new,
            R_relative, 
            t_relative);


        // Update.
        image_1_former = image_1_new;
        image_3D_former = image_3D_new;

    } // END OF while(getline(image_1_names, image_1_name)).



}



int main(int argc, char* argv[]){

    // Read the input parameters.
    if(argc < 2){
        cout << "ERROR: Wrong number of parameters!" << endl;
        help(argv);
    }

    string dir_name = argv[1];

    // Parse the calib.txt file and build the reprojection matrix Q.
    //
    string file_name = dir_name + "/sequences/00/calib.txt";
    // Reprojection matrix Q.
    cv::Mat Q;
    parserCalib(file_name, Q);
    

    // Obtain the relative affine transformation (R | t)
    // between the former and the latter frames.
    //
    vector< cv::Mat > R, t;

    getTrajectory(
        dir_name, 
        Q,
        R, t);

    // Over.
    cv::destroyAllWindows();
    return 0;
}
