#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void correctPerpective(Mat& inputImg);
cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b);
void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center);
void compareImages (Mat& img1, Mat& img2);

int main()
{
    Mat img1;
    Mat img2;

    img1 = imread("/home/toTest/blank.png", CV_LOAD_IMAGE_UNCHANGED);
    img2 = imread("/home/toTest/test.png", CV_LOAD_IMAGE_UNCHANGED);

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

    namedWindow("Perfect", CV_WINDOW_AUTOSIZE);
    namedWindow("Comparison", CV_WINDOW_AUTOSIZE);

    imshow("Perfect", img1);
    imshow("Comparison", img2);

    waitKey(0);

    /*
    int i;

    do
    {
        cout << "Enter a 1 to compare images" << endl;
        i = 0;
        cin >> i;
    }
    while(i != 1);
    */

    correctPerpective(img2);

    compareImages(img1, img2);

    imshow("Comparison", img2);

    waitKey(0);

    destroyAllWindows();

    return 0;

}

void correctPerpective(Mat& inputImg)
{
    Mat bw;

    ///Convert input to grayscale
    cv::cvtColor(inputImg, bw, CV_RGB2GRAY);
    cout << "Grayscale" << endl;

    ///Apply Canny edge detection
    cv::Canny(bw, bw, 100, 100, 3);
    cout << "Canny" << endl;

    ///Apply Hough transform to detect straight lines
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(bw, lines, CV_PI/180, 70, 30, 10);
    cout << "Hough" << endl;

    ///Identify the Corners
    std::vector<cv::Point2f> corners;
    for (int i = 0; i < lines.size(); i++)
    {
        for (int j = i+1; j < lines.size(); j++)
        {
            cv::Point2f pt = computeIntersect(lines[i], lines[j]);
            if (pt.x >= 0 && pt.y >= 0)
                corners.push_back(pt);
        }
    }
    cout << "Corners" << endl;

    /*
    ///Check if the polygon is a quadralateral
   // std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f> approx;
    cv::approxPolyDP(cv::Mat(corners), approx, cv::arcLength(cv::Mat(corners), true) * 0.02, true);
    if (approx.size() != 4)
    {
        std::cout << "The object is not quadrilateral!" << std::endl;
        return;
    }
    cout << "Check if quad" << endl;
    */
    /// Get mass center
    cv::Point2f center(0,0);
    for (int i = 0; i < corners.size(); i++)
        center += corners[i];

    center *= (1. / corners.size());
    cout << "Get mass center" << endl;

    ///Sort the corners and check if successful
    sortCorners(corners, center);
    if (corners.size() == 0)
    {
        std::cout << "The corners were not sorted correctly!" << std::endl;
        return;
    }
    cout << "Sort Corners" << endl;

    ///Map the corners to the output image
    cv::Mat quad = cv::Mat::zeros(300, 220, CV_8UC3);

    std::vector<cv::Point2f> quad_pts;
    quad_pts.push_back(cv::Point2f(0, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
    quad_pts.push_back(cv::Point2f(0, quad.rows));

    cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);
    cv::warpPerspective(inputImg, quad, transmtx, quad.size());

    //cv::imshow("image", dst);
    cv::imshow("quadrilateral", quad);
    cv::waitKey();
}

cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    float denom;
    if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
    {
        cv::Point2f pt;
        pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
        pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
        return pt;
    }
    else
        return cv::Point2f(-1, -1);
}

void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
    std::vector<cv::Point2f> top, bot;

    for (int i = 0; i < corners.size(); i++)
    {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }

    cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
    cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
    cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
    cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

    corners.clear();
    corners.push_back(tl);
    corners.push_back(tr);
    corners.push_back(br);
    corners.push_back(bl);
}

void compareImages (Mat& img1, Mat& img2)
{
    const int channels = img2.channels();

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








