#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void compareImages (Mat& img1, Mat& img2);
void compareGrayImages (Mat& img1Gray, Mat& img2Gray);
void sobelImages (Mat& img1, Mat& img2, Mat& img1Sobel, Mat& img2Sobel);

int main()
{
    Mat img1;
    Mat img2;
    Mat img1Sobel;
    Mat img2Sobel;

    img1 = imread("/home/zefan/Documents/toTest/blank.png", CV_LOAD_IMAGE_UNCHANGED);
    img2 = imread("/home/zefan/Documents/toTest/test.png", CV_LOAD_IMAGE_UNCHANGED);
    img1Sobel = imread("/home/zefan/Documents/toTest/blank.png", CV_LOAD_IMAGE_UNCHANGED);
    img2Sobel = imread("/home/zefan/Documents/toTest/test.png", CV_LOAD_IMAGE_UNCHANGED);

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

    int i;

    do
    {
        cout << "Enter a 1 to compare images" << endl;
        i = 0;
        cin >> i;
    }
    while(i != 1);

    sobelImages(img1, img2, img1Sobel, img2Sobel);

    imshow("Perfect", img1Sobel);
    imshow("Comparison", img2Sobel);

    waitKey(0);

    //compareImages(img1Sobel, img2Sobel);
    compareGrayImages(img1Sobel, img2Sobel);

    imshow("Comparison", img2Sobel);

    waitKey(0);

    destroyAllWindows();

    return 0;

}

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

void compareGrayImages(Mat& img1Gray, Mat& img2Gray)
{
    MatIterator_<uchar> it1, end1, it2, end2;
    for (it1 = img1Gray.begin<uchar>(), end1 = img1Gray.end<uchar>(), it2 = img2Gray.begin<uchar>(), end2 = img2Gray.end<uchar>(); it1!=end1, it2!=end2; ++it1, ++it2)
    {
        if (*it1 != *it2)
        {
            *it2 = 255;
        }
        else
        {
            *it2 = 0;
        }
    }

    return;
}



