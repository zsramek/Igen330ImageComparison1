#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

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

    int i;

    do
    {
        cout << "Enter a 1 to compare images" << endl;
        i = 0;
        cin >> i;
    }
    while(i != 1);

    compareImages(img1, img2);

    imshow("Comparison", img2);

    waitKey(0);

    destroyAllWindows();

    return 0;

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




