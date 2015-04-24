#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "asmlib-opencv/asmmodel.h"
#include <string>
#include <vector>

using std::vector;
using std::string;
using cv::Mat;
using cv::VideoCapture;

int main(int argc, char** argv )
{
    if(argc != 3)
    {
        return 1;
    }

    string arg = argv[1];
    cv::namedWindow("Image", CV_WINDOW_NORMAL);
    vector<cv::Mat> faces;
    vector<cv::Mat> test_faces;

    Mat test_img;
    for (int i = 1; i < 16; i++)
    {
        string path = arg + cv::format("subject%02d", i) + ".happy";
        VideoCapture cap(path);
        Mat image;
        cap >> image;

        test_img = image;
        if(i > 10) {
            test_faces.push_back(image);
        } else
            faces.push_back(image);
    }

    for (int i = 1; i < 16; i++)
    {
        string path = arg + cv::format("subject%02d", i) + ".sad";
        VideoCapture cap(path);
        Mat image;
        cap >> image;

        test_img = image;
        if(i > 10) {
            test_faces.push_back(image);
        } else
            faces.push_back(image);
    }

    for (int i = 1; i < 16; i++)
    {
        string path = arg + cv::format("subject%02d", i) + ".normal";
        VideoCapture cap(path);
        Mat image;
        cap >> image;

        test_img = image;
        if(i > 10) {
            test_faces.push_back(image);
        } else
            faces.push_back(image);
    }

    for (int i = 1; i < 16; i++)
    {
        string path = arg + cv::format("subject%02d", i) + ".sleepy";
        VideoCapture cap(path);
        Mat image;
        cap >> image;

        test_img = image;
        if(i > 10) {
            test_faces.push_back(image);
        } else
            faces.push_back(image);
    }

    for (int i = 1; i < 16; i++)
    {
        string path = arg + cv::format("subject%02d", i) + ".wink";
        VideoCapture cap(path);
        Mat image;
        cap >> image;

        test_img = image;
        if(i > 10) {
            test_faces.push_back(image);
        } else
            faces.push_back(image);
    }

    for (int i = 1; i < 16; i++)
    {
        string path = arg + cv::format("subject%02d", i) + ".surprised";
        VideoCapture cap(path);
        Mat image;
        cap >> image;

        test_img = image;
        if(i > 10) {
            test_faces.push_back(image);
        } else
            faces.push_back(image);
    }


    int num_files = 60;
    int num_test = 30;
    int col = test_img.cols;
    int row = test_img.rows;
    int img_area = col * row;

    Mat training_mat(num_files,img_area,CV_32FC1);
    for (int p = 0; p < num_files; p++) {
        int ii = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                training_mat.at<float>(p,ii++) = faces[p].at<uchar>(i,j);
            }
        }
    }

    Mat test_mat(num_files,img_area,CV_32FC1);
    for (int p = 0; p < num_test; p++) {
        int ii = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                test_mat.at<float>(p,ii++) = test_faces[p].at<uchar>(i,j);
            }
        }
    }


    Mat labels;
    for (int i = 0; i< num_files; i++) {
        if (i > 9)
            labels.push_back(-1);
        else
            labels.push_back(1);
    }

    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::SIGMOID;
    params.nu = 100;
    //params.degree = 6;
    params.C = 0.001;

    CvSVM svm;
    svm.train(training_mat, labels, Mat(), Mat(), params);


    int correct = 0;
    for (int i = 0; i < num_test; i++) {
        if (i < 5 && svm.predict(test_mat.row(i)) > 0)
            correct++;
        if (i > 5 && svm.predict(test_mat.row(i)) < 0)
            correct++;

    }
    std::cout << (float)correct / num_test << std::endl;

    return 0;
}

/*
    StatModel::ASMModel model(argv[2]);
    vector<vector<Point_<int>>> landmarks;
    vector<Mat>::iterator iter;
    for (iter = faces.begin(); iter != faces.end(); iter++) {
        StatModel::ASMFitResult res = model.fit(*iter);
        vector<Point_<int>> face_landmarks;
        res.toPointList(face_landmarks);
        landmarks.push_back(face_landmarks);
        std::cout << face_landmarks.size() << std::endl;

        vector<Point>::iterator piter;
        for (piter = face_landmarks.begin(); piter != face_landmarks.end(); piter++) {
            Point center = Point(piter->x, piter->y);
            circle(test_img, center, 1, CV_RGB(255,255,0),3);
        }
    }

    cv::imshow("Image", test_img);
*/