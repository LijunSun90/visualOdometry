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
    cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P2.at<double>(row, col) = atof(entry.c_str());
        }
    }
    cout << P2 << endl;
    // P3. 3 represents the right color camera.
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
    // cout << "P3.type(): " << P3.type() << endl;    
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

    cout << "\ncx: " << cx << "\n"
    << "cy: " << cy << "\n"
    << "f: " << f << "\n"
    << "baselinePoint2: " << baselinePoint2 << "\n"
    << "baselinePoint3: " << baselinePoint3 << "\n"
    << "baseline: " << baseline << endl;

    // Build the reprojection matrix Q.
    Q = cv::Mat::eye(4, 4, CV_64F);
    Q.at<double>(0, 3) = -cx;
    Q.at<double>(1, 3) = -cy;
    Q.at<double>(2, 3) = f;
    Q.at<double>(3, 3) = 0;
    Q.at<double>(3, 2) = -1/baseline;

    cout << "\nQ: \n" << Q << "\n" << endl;

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
    vector < cv::DMatch > & matches
) {

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
    cv::Mat & image_3D
){

    // cout << "triangulateStereo" << endl;

    cv::Mat image_2;
    image_1 = cv::imread(image_1_name, cv::IMREAD_GRAYSCALE);
    image_2 = cv::imread(image_2_name, cv::IMREAD_GRAYSCALE);

    // Step 1: Disparity.
    //         using the stero correspondence.
    //
    // cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
    //     -64, 128, 11, 0, 0, 12, 
    //     0, 15, 1000, 16,
    //     cv::StereoSGBM::MODE_SGBM // cv::StereoSGBM::MODE_HH
    // );
    //
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0, // minDisparity
        128, // numDisparities, total number of distinct possible disparities
             // to be returned. 
        5, // blockSize, usually 3 or 5 would be enough. Always be odd.
            // the larger the value is, the fewer false matches you 
            // are likely to find. However, not only the computational
            // cost scale with the area of the window, but it implicitly
            // assume that the disparity is actually the same over the
            // area of the windlow. Also, with larger blockSize, it is
            // likely to get vaguer depth maps.
        0, // P1
        0, // P2
           // Leave P1 and P2 as zeros and make the implementation 
           // compute some optimal values for them based on the image
           // resolution and blockSize.
        0, // disp12MaxDiff
        0, // preFilterCap, a positive numeric limit.
        15, // uniquessRatio, typical values: 5 ~ 15.
            // speckleWindowSize and speckleRange work together.
        21, // speckleWindowSize, the size of any small, isolated blobs 
            // that are substantially different from their surrounding values.
        2,  // speckleRange, the largest difference between disparities
            // that will include in the same blob.
            // This value is compared directly to the values of the disparity,
            // then this value will, in effect, be multiplied by 16.
            // Typically, a small number, 1 or 2 work most of the time,
            // though values as large as 4 are not uncommon.
        cv::StereoSGBM::MODE_SGBM // mode
                                  // the five-directional version.
    );
    
    // 
    cv::Mat disparity, vdisparity, disparity_true;
    // Output 16-bit fixed-point disparity map.
    //
    stereo->compute(image_1, image_2, disparity);
    // cout << "disparity1: " <<  disparity.at<double>(0, 0) << endl;
    // cout << "disparity2: " <<  disparity.at<double>(300, 100) << endl;
    // cout << "disparity: " << disparity << endl;

    //Get the true disparity values.
    // cout << "disparity type: " << disparity.type() << endl;
    disparity.convertTo(disparity_true, CV_32F, 1.0/16.0, 0.0);

    // This is for displaying the disparity image only.
    // cv::normalize(disparity_true, vdisparity, 0, 256, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("Disparity_map", vdisparity);
    // cv::moveWindow("Disparity_map", 670, 0);

    // Step 2: Depth map.
    cv::reprojectImageTo3D(
        disparity_true,
        image_3D,
        Q,
        true,
        -1
    );

    // Mark the outliers.
    double count = 0;
    for(int row = 0; row < disparity_true.size().height; row++){
        for(int col = 0; col < disparity_true.size().width; col++){
            if(disparity_true.at<double>(row, col) < 0){
                // cout << "Original outlier depth: "
                // << image_3D.at<cv::Vec3f>(row, col)(2) << endl;
                image_3D.at<cv::Vec3f>(row, col)(2) = 10000;
                count++;
            }
        }
    }
    cout << "The ratio of negative disparities is: " 
    << count/(disparity_true.rows * disparity_true.cols) 
    << "\n" << endl;

    // if(cv::waitKey(1) == 27) exit(0);

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
    vector< cv::Vec3f > & keypoints3D
){
    
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
    vector < cv::DMatch > matches
){
    // cout << "keypoints3D_new Before: " << keypoints3D_new[30] << endl;

    vector< cv::Vec3f > keypoints3D_temp;
    for(size_t ix = 0; ix < matches.size(); ix++){

        keypoints3D_temp.push_back(keypoints3D_new[matches[ix].trainIdx]);

    } // END OF for(size_t ix = 0; ix < matches.size(); ix++).

    // Update.
    //
    keypoints3D_new = keypoints3D_temp;
    // cout << "keypoints3D_temp: " << keypoints3D_temp[30] << endl;
    // cout << "keypoints3D_new: " << keypoints3D_new[30] << endl;

} // END OF sortKeypoints3D().


/**
 * Filter out the outliers whose depth value equals to 10000.
 */
void filter3DPoints(
    vector< cv::Vec3f > & keypoints3D_former,
    vector< cv::Vec3f > & keypoints3D_new
){

    // cout << "Before: " << keypoints3D_former.size() << endl;

    for(vector< cv::Vec3f >::iterator iter_former = keypoints3D_former.begin(), iter_new = keypoints3D_new.begin(); 
        iter_former < keypoints3D_former.end() || iter_new < keypoints3D_new.end();
        iter_former++, iter_new++){

        if ((*iter_former)(2) == 10000 || (*iter_new)(2) == 10000){
            // cout << "iter_former: " << *iter_former << endl;
            // cout << "iter_new: " << *iter_new << endl;
            keypoints3D_former.erase(iter_former);
            keypoints3D_new.erase(iter_new);
            iter_former--;
            iter_new--;
        } // END OF if ((*iter_former)(2) == 10000 || (*iter_new)(2) == 10000).

    } // END OF for();

    // cout << "After: " << keypoints3D_former.size() << endl;

} // END OF filter3DPoints().


void getRelativeTransform(
    vector< cv::KeyPoint > keypoints_former,
    vector< cv::KeyPoint > keypoints_new,
    vector < cv::DMatch > matches,
    cv::Mat image_3D_former,
    cv::Mat image_3D_new,
    cv::Mat & R_relative,
    cv::Mat & t_relative
){

    vector< cv::Vec3f > keypoints3D_former, keypoints3D_new;
    get3Dkeypoints(keypoints_former, image_3D_former, keypoints3D_former);
    get3Dkeypoints(keypoints_new, image_3D_new, keypoints3D_new);
    
    // Sort keypoints3D_former and keypoints3D_new in the order
    // according to matches.
    //
    sortKeypoints3D(keypoints3D_former, keypoints3D_new, matches);
    
    // Filter out the outliers with depth value equal to 10000.
    //
    filter3DPoints(keypoints3D_former, keypoints3D_new);
    cout << "\nThe effective keypoints pair is: " 
    << keypoints3D_former.size() << endl;

    // Calculate the relative 3D affine transformation.
    //
    cv::Mat affine_relative_3D;
    cv::estimateAffine3D(keypoints3D_former, keypoints3D_new, affine_relative_3D, cv::noArray());
    // cout << "\naffine_relative_3D: \n" << affine_relative_3D << endl;
    R_relative = affine_relative_3D.colRange(0, 3);
    t_relative = affine_relative_3D.col(3);
    // cout << "\nR_relative: \n" << R_relative
    // cout << "\nt_relative: \n" << t_relative
    // << endl;

}


/**
 * @param Rs, the collection of the total rotation matrix to the reference.
 * @param ts, the collection of the total translation matrix to the reference.
 */
void visualizeTrajectoryXZ(cv::Mat trajectory, vector< cv::Mat > Rs, vector< cv::Mat > ts){
   
    // trajectory: Horizontal: x; Vertical: z.  
    cv::Point pose_current;
    //
    cv::Mat R, t;
    for(size_t ix = 0; ix < Rs.size(); ix++){
        R = Rs[ix];
        t = ts[ix];

        pose_current = cv::Point(t.at<double>(0) + 300, t.at<double>(2) + 100);
        cv::circle(trajectory, pose_current, 1, cv::Scalar(0, 0, 255), 1, CV_FILLED);
        
        // cout << "Show trajectory" << endl;
        cv::imshow("trajectory", trajectory);
        cv::moveWindow("trajectory", 0, 0);
        if(cv::waitKey(1) == 27){
            cv::imwrite("trajectory.jpg", trajectory);
            exit(0);
        }
    } // END OF for(size_t ix = 0; ix < Rs.size(); ix++).

} // END OF visualizeTrajectory().



/**
 * @param dir_name, the name of the directory containing the
 *                  RECTIFIED stereo images.
 * @param Q, the 4x4 reprojection matrix, which maps the 2-D coordiantes
 *           withe the disparity (x, y, d) to the 3-D 3-D coordinates 
 *           in the physical world (X/W, Y/W, Z/W).
 * @param Rs, the collection of the total rotation matrix to the reference.
 * @param ts, the collection of the total translation matrix to the reference.
 */
void getTrajectory(
    string dir_name,
    cv::Mat Q,
    vector< cv::Mat > & Rs,
    vector< cv::Mat > & ts
){

    cv::FileStorage fs("poses.xml", cv::FileStorage::WRITE);
    int frame = 0;

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
    image_2_name = dir_name + "/sequences/00/" +   image_2_name;

    triangulateStereo(image_1_name, image_2_name, Q, image_1_former, image_3D_former);
    cout << "Image size: " << image_1_former.size() << endl;

    // Get the total transformation with respect to the reference.
    // 
    cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat R_relative_new, t_relative_new;
    cv::Mat affine_3D = cv::Mat::zeros(3, 4, CV_64F);
    // Save.
    R_total.copyTo(affine_3D.colRange(0, 3));
    t_total.copyTo(affine_3D.col(3));
    // cout << "R_total: \n" << R_total << endl;
    // cout << "t_total: \n" << t_total << endl;
    // cout << "affine_3D: \n" << affine_3D << endl;
    fs << "frame" + to_string(frame) << affine_3D;
    
    // trajectory: Horizontal: x; Vertical: z.
    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3); 

    while(getline(image_1_names, image_1_name)){
        frame++;

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
        getRelativeTransform(
            keypoints_former, 
            keypoints_new, 
            matches, 
            image_3D_former, 
            image_3D_new,
            R_relative_new, 
            t_relative_new);
        
        // Obtain the total transformation from the relative transformation.
        //
        R_total = R_relative_new * R_total;
        t_total = R_relative_new * t_total + t_relative_new;
        // t_total = t_relative_former + t_relative_new;
        // cout << "\nR_relative_former: \n" << R_relative_former << endl;
        // cout << "R_relative_new: \n" << R_relative_new << endl;
        // cout << "R_total: \n" << R_total << endl;
        // cout << "t_relative_former: \n" << t_relative_former << endl;
        cout << "\nt_relative_new: \n" << t_relative_new << endl;
        cout << "t_total: \n" << t_total << endl;
        
        // Save.
        //
        Rs.push_back(R_total);
        ts.push_back(t_total);
        // Save.
        R_total.copyTo(affine_3D.colRange(0, 3));
        t_total.copyTo(affine_3D.col(3));
        // cout << "affine_3D: \n" << affine_3D << endl;
        fs << "frame" + to_string(frame) << affine_3D;

        // Visualize the trajectory.
        //
        visualizeTrajectoryXZ(trajectory, Rs, ts);

        // Update.
        //
        // cout << "\nimage_1_former: \n" << image_1_former.at<double>(0, 0) << endl;
        // cout << "image_1_new: \n" << image_1_new.at<double>(0, 0) << endl;  
        // cout << "image_3D_former: " << image_3D_former.at<cv::Vec3f>(300, 100) << endl;
        // cout << "image_3D_new: " << image_3D_new.at<cv::Vec3f>(300, 100) << endl;       
        image_1_former = image_1_new.clone();
        image_3D_former = image_3D_new.clone();

    } // END OF while(getline(image_1_names, image_1_name)).

    // Save the trajectory.
    //
    cv::imwrite("trajectory.jpg", trajectory);

    fs.release();

} // END OF getTrajectory().





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

    // Get the trajectory.
    vector< cv::Mat > Rs, ts;
    getTrajectory(
        dir_name, 
        Q,
        Rs, 
        ts);

    

    // Over.
    cv::destroyAllWindows();
    return 0;
}
