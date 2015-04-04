#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

void compareImages (Mat& img1, Mat& img2);
void compareGrayImages (Mat& img1Gray, Mat& img2Gray);
void sobelImages (Mat& img1, Mat& img2, Mat& img1Sobel, Mat& img2Sobel);
void flattenImages (Mat& img1Gray, Mat& img2Gray);
void subtraction (Mat& compared, Mat& img1);
void detectBlobs(Mat& img2, Mat& result);
void alignImage (Mat& image);

int main()
{
    Mat img1;
    Mat img2;
    Mat img1Orig;
    Mat img2Orig;
    Mat img1Sobel;
    Mat img2Sobel;

    img1 = imread("/home/toTest/test48.JPG", CV_LOAD_IMAGE_UNCHANGED);
    img2 = imread("/home/toTest/test56.JPG", CV_LOAD_IMAGE_UNCHANGED);

    if(img1.empty())
    {
        cout << "Could not load image 1." << endl;
        return -1;
    }
    else if (img2.empty())
    {
        cout << "Could not load image 2." << endl;
        return -1;
    }

    ///Resize both images
    resize(img1, img1, Size(), 0.2, 0.2, CV_INTER_AREA);
    resize(img2, img2, Size(), 0.2, 0.2, CV_INTER_AREA);

    ///Align and crop both images
    alignImage(img1);
    alignImage(img2);

    ///Create a copy of img2 for final display
    Mat displayImage = img2;

    namedWindow("Perfect", CV_WINDOW_AUTOSIZE);
    namedWindow("Comparison", CV_WINDOW_AUTOSIZE);

    imshow("Perfect", img1);
    imshow("Comparison", img2);

    waitKey(0);

    ///Apply the Sobel filering to both images
    sobelImages(img1, img2, img1Sobel, img2Sobel);

    ///Show the results of the sobel filtering
    imshow("Perfect", img1Sobel);
    imshow("Comparison", img2Sobel);

    waitKey(0);

    ///Compare the images without applying the filtering
    //compareImages(img1, img2);

    ///Flatten the grayscale images
    flattenImages(img1Sobel, img2Sobel);

    imshow("Perfect", img1Sobel);
    imshow("Comparison", img2Sobel);

    waitKey(0);

    ///Compare the images after having applied the Sobel filtering
    compareGrayImages(img1Sobel, img2Sobel);

    ///Subtract the "Perfect" flattened image from the result of comparison
    subtraction(img2Sobel, img1Sobel);

    namedWindow("Post Subtraction", CV_WINDOW_AUTOSIZE);
    imshow("Post Subtraction", img2Sobel);
    waitKey(0);

    ///Detect errors as blobs
    detectBlobs(displayImage, img2Sobel);

    ///Display the results
    //imshow("Comparison", img2);
    namedWindow("Sobel Comparison", CV_WINDOW_AUTOSIZE);
    imshow("Sobel Comparison", img2Sobel);

    waitKey(0);

    destroyAllWindows();

    return 0;
}

///This function takes four images and applies the Sobel filter to the first two, placing the results into the second 2.
void sobelImages(Mat& img1, Mat& img2, Mat& img1Sobel, Mat& img2Sobel)
{
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    Mat img1Gray;
    Mat img2Gray;

    ///Apply blur to reduce noise
    GaussianBlur(img1, img1, Size(3,3), 0, 0, BORDER_DEFAULT);
    GaussianBlur(img2, img2, Size(3,3), 0, 0, BORDER_DEFAULT);

    /// Convert images to gray
    cvtColor(img1, img1Gray, CV_RGB2GRAY);
    cvtColor(img2, img2Gray, CV_RGB2GRAY);

    ///++++++++++++++FOR IMG1+++++++++++++++++++++++++++++++++

    /// Generate gradX and gradY
    Mat gradX1, gradY1;
    Mat absGradX1, absGradY1;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel(img1Gray, gradX1, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(gradX1, absGradX1);

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel(img1Gray, gradY1, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(gradY1, absGradY1);

    /// Total Gradient (approximate)
    addWeighted( absGradX1, 0.5, absGradY1, 0.5, 0, img1Sobel);

    ///++++++++++++++FOR IMG2+++++++++++++++++++++++++++++++++

    /// Generate gradX and gradY
    Mat gradX2, gradY2;
    Mat absGradX2, absGradY2;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel(img2Gray, gradX2, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(gradX2, absGradX2);

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel(img2Gray, gradY2, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(gradY2, absGradY2);

    /// Total Gradient (approximate)
    addWeighted( absGradX2, 0.5, absGradY2, 0.5, 0, img2Sobel);

    return;
}

///This function iterates through two RGB images and changes any different pixels to green.
void compareImages (Mat& img1, Mat& img2)
{

    /*
    Mat img1Final;
    Mat img2Final;

    cvtColor(img1, img1Final, CV_GRAY2RGB);
    cvtColor(img2, img2Final, CV_GRAY2RGB);
    */

    //const int channels = img2.channels();

    MatIterator_<Vec3b> it1, end1, it2, end2;
    for (it1 = img1.begin<Vec3b>(), end1 = img1.end<Vec3b>(), it2 = img2.begin<Vec3b>(), end2 = img2.end<Vec3b>(); it1!=end1, it2!=end2; ++it1, ++it2)
    {
        if ((*it1)[0] != (*it2)[0])
        {
            //cout << "Different Detected at:" << endl;
            (*it2)[0] = 0;
            (*it2)[1] = 255;
            (*it2)[2] = 0;
        }
    }
    return;
}

///This function iterates through two grayscale images and marks differences as white pixels and changes the rest black.
void compareGrayImages(Mat& img1Gray, Mat& img2Gray)
{
    MatIterator_<uchar> it1, end1, it2, end2;
    for (it1 = img1Gray.begin<uchar>(), end1 = img1Gray.end<uchar>(), it2 = img2Gray.begin<uchar>(), end2 = img2Gray.end<uchar>(); it1!=end1, it2!=end2; ++it1, ++it2)
    {
        if (*it1 != *it2)
        {
            //cout << *it1 << endl;
            *it2 = 255;
        }
        else
        {
            *it2 = 0;
        }
    }

    return;
}

///This function takes two grayscale images and converts them to purely black and white, with no shades of gray in between.
void flattenImages(Mat& img1Gray, Mat& img2Gray)
{
    ///Apply blur to reduce noise
    GaussianBlur(img1Gray, img1Gray, Size(3,3), 0, 0, BORDER_DEFAULT);
    GaussianBlur(img2Gray, img2Gray, Size(3,3), 0, 0, BORDER_DEFAULT);

    MatIterator_<uchar> it1, end1, it2, end2;
    for (it1 = img1Gray.begin<uchar>(), end1 = img1Gray.end<uchar>(), it2 = img2Gray.begin<uchar>(), end2 = img2Gray.end<uchar>(); it1!=end1, it2!=end2; ++it1, ++it2)
    {
        if (*it1 < 10)
        {
            *it1 = 0;
        }
        else
        {
            *it1 = 255;
        }

        if (*it2 < 10)
        {
            *it2 = 0;
        }
        else
        {
            *it2 = 255;
        }
    }

    return;
}

///This function subtracts the original good file from the output of the comparison.
void subtraction(Mat& compared, Mat& img1)
{
    MatIterator_<uchar> it1, end1, it2, end2;
    for (it1 = compared.begin<uchar>(), end1 = compared.end<uchar>(), it2 = img1.begin<uchar>(), end2 = img1.end<uchar>(); it1!=end1, it2!=end2; ++it1, ++it2)
    {
        *it1 = *it1 - *it2;
        if (*it1 < 0)
        {
            *it1 = 0;
        }
    }
    return;
}

///This function attempts to identify true errors by looking for blobs
void detectBlobs(Mat& img2, Mat& result)
{
    ///Setup the blob detector
    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    /// Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    /// Filter by Area.
    params.filterByArea = true;
    params.minArea = 100;

    ///Turn off extra parameters
    params.filterByCircularity = false;
    params.filterByConvexity = false;
    params.filterByInertia = false;

    ///Declare the blob detector
    SimpleBlobDetector detector(params);

    ///Create a vector for the blob keypoints
    std::vector<KeyPoint> keypoints;

    ///Detect blobs
    detector.detect(result, keypoints);

    ///Draw circles around the keypoints of the blobs
    drawKeypoints(img2, keypoints, img2, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    ///List the keypoint coordinates - for debugging purposes
    /*
    for (int p = 0; p < keypoints.size(); p++)
    {
        cout << "keypoint: " << keypoints[p].pt.x << ", " << keypoints[p].pt.y << endl;
    }
    */

    ///Show the original image with the errors circled
    namedWindow("BLOB", CV_WINDOW_AUTOSIZE);
    imshow("BLOB", img2);
    waitKey(0);

    return;
}

///This function takes an image, detects the corners of the largest rectangle and crops/rotates the image to be just that rectangle
void alignImage(Mat& image)
{
    Mat imageGray;
    namedWindow("Test", CV_WINDOW_AUTOSIZE);
    imshow("Test", image);
    waitKey(0);

    ///Convert image to grayscale
    cvtColor(image, imageGray, CV_RGB2GRAY);

    ///Apply blur to smooth edges and use adapative thresholding -- CAUSES PROBLEMS!
    cv::Size size(3,3);
    cv::GaussianBlur(imageGray,imageGray,size,0);
    //adaptiveThreshold(imageGray, imageGray,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);
    //cv::threshold(imageGray,imageGray, 100, 255, CV_THRESH_TOZERO); //- See more at: http://en.efreedom.net/Question/1-7263621/Find-Corners-Image-Using-OpenCv#sthash.fHgOd4bq.dpuf
    cv::bitwise_not(imageGray, imageGray);

    imshow("Test", imageGray);
    waitKey(0);


    ///+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ///Harris Corner Detection - http://docs.opencv.org/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html
    ///+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    int thresh = 180;
    //int max_thresh = 255;

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( image.size(), CV_32FC1 );

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    //cout << "Detecting Corners" << endl;
    cornerHarris( imageGray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

    /// Normalizing
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    ///Preparing the points for the warp

    ///Declaring variables needed for sorting points
    double centreX = (double)(image.cols/2);
    double centreY = (double)(image.rows/2);
    double D; ///Distance to centre

    int pointsIndex = -1; ///Tracks the array index
    int pts = 0; ///Tracks the number of points found

    ///Arrays to hold the points and distances from the centre
    Point2f allPoints[1000];
    double distances[1000];

    /// Drawing a circle around corners and placing all unique points in an array along with a
    for( int j = 0; j < dst_norm.rows ; j++ )
    {
        for( int i = 0; i < dst_norm.cols; i++ )
        {
            if( (int) dst_norm.at<float>(j,i) > thresh )
            {
                ///Create a Point2f for the point
                Point2f pt;
                pt.x = i;
                pt.y = j;

                double x = (double)i;
                double y = (double)j;

                ///For debugging purposes
                //cout << "pts: " << pts << endl;
                //cout << "pointsIndex: " << pointsIndex << endl;

                if (pts >= 1)
                {
                    if ((i > (allPoints[pointsIndex].x + 5) || i < (allPoints[pointsIndex].x - 5)) && (j > (allPoints[pointsIndex].y + 5) || j < (allPoints[pointsIndex].y - 5)))
                    {
                        ///Point is not a duplicate
                        ///Increment count of points
                        pts++;

                        ///For debugging purposes
                        //cout << "Adding: " << pt << endl;
                        //cout << "The previous point" << allPoints[pointsIndex] << "\n" << endl;


                        ///Calculate the distance from the centre
                        D = sqrt(pow((centreX-x), 2) + pow((centreY-y),2));

                        ///Add the point and its distance from the centre to two vectors
                        int q = pts - 1;
                        allPoints[q] = pt;
                        distances[q] = D;

                        circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );

                        //distances.push_back(D);
                        pointsIndex++;
                    }
                    else
                    {
                        ///Point is a duplicate
                        ///For debugging purposes
                        //cout << "Skipping: " << pt << "\n" << endl;
                    }
                }
                else
                {
                    ///The first point
                    ///Increment count of points
                    pts++;

                    ///For debugging purposes
                    //cout << "Adding: " << pt << endl;
                    //cout << "The previous point" << allPoints[pointsIndex] << "\n" << endl;


                    ///Calculate the distance from the centre
                    D = sqrt(pow((centreX-x), 2) + pow((centreY-y),2));

                    ///Add the point and its distance from the centre to two vectors
                    int q = pts - 1;
                    allPoints[q] = pt;
                    distances[q] = D;
                    circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
                    //distances.push_back(D);
                    pointsIndex++;
                }

                ///Circle the point
                //circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }

    ///Circle the centre
    //circle( dst_norm_scaled, Point( (image.cols/2), (image.rows/2) ), 5,  Scalar(100), 2, 8, 0 );

    ///For debugging purposes
    /*
    cout << "The centre is: " << image.cols/2 << ", " << image.rows/2 << "\n" << endl;
    for (int f = 0; f < pts; f++)
    {
        cout << "index: " << f << " point: " << allPoints[f] << " distance: " << distances[f] << endl;
    }
    */

    ///Add all the points to a vector and print it (for debugging purposes)
    /*
    std::vector<cv::Point2f> pointsVector;
    for (int p = 0; p < pts; p++)
    {
        pointsVector.push_back(allPoints[y]);
    }
    cout << pointsVector << endl;
    */

    ///Show the image with the corners circled
    namedWindow("corners_window", CV_WINDOW_AUTOSIZE);
    imshow("corners_window", dst_norm_scaled);
    waitKey(0);

    Point2f points[4];
    int index = 0;
    int count;
    ///Determing, based on the distance from the centre, the four corners.
    for (int p = 0; p < pts; p++)
    {
        count = 0;
        for (int q = 0; q < pts; q++)
        {
            if (distances[q] > distances[p])
            {
                count++;
            }
        }

        if (count <= 3)
        {
            ///For debugging purposes
            //cout << "Adding:  " << allPoints[p] << "  with distance:  " << distances[p] << "\n" << endl;

            ///Add the corner point into the array of corners
            points[index] = allPoints[p];
            index++;
        }
    }

    ///For debugging purposes
    //for (int h = 0; h < 4; h++)
    //{
    //  cout << points[h] << endl;
    //}

    ///Ordering the Points
    Point2f topLeft;
    Point2f topRight;
    Point2f bottomLeft;
    Point2f bottomRight;

    for (int c = 0; c < 4; c++)
    {
        if (points[c].x < (image.cols/2) && points[c].y < (image.rows/2))
        {
            topLeft = points[c];
        }
        else if (points[c].x > (image.cols/2) && points[c].y < (image.rows/2))
        {
            topRight = points[c];
        }
        else if (points[c].x < (image.cols/2) && points[c].y > (image.rows/2))
        {
            bottomLeft = points[c];
        }
        else
        {
            bottomRight = points[c];
        }
    }

    ///Printing the coords of the points (for debugging purposes)
    /*
    for (int u = 0; u < 4; u++)
    {
        cout << points[u].x << "   " << points[u].y << endl;
    }
    */

    ///Create a vector with the ordered corners
    std::vector<cv::Point2f> corners;
    corners.push_back(topLeft);
    corners.push_back(topRight);
    corners.push_back(bottomRight);
    corners.push_back(bottomLeft);

    /// Define the destination image
    cv::Mat quad = cv::Mat::zeros(600, 500, CV_8UC3);

    /// Corners of the destination image
    std::vector<cv::Point2f> quad_pts;
    quad_pts.push_back(cv::Point2f(0, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
    quad_pts.push_back(cv::Point2f(0, quad.rows));

    /// Get transformation matrix
    cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);

    /// Apply perspective transformation
    cv::warpPerspective(image, quad, transmtx, quad.size());
    cv::imshow("Test", quad);

    waitKey(0);

    image = quad;

    destroyAllWindows();

    return;
}
