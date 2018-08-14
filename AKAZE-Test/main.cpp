#include "stdafx.h"

namespace neno
{
    double getMinDistance(const std::vector<cv::DMatch>& matches)
    {
        double mindistance = DBL_MAX;

        for (auto& match : matches)
        {
            if (match.distance < mindistance)
            {
                mindistance = match.distance;
            }
        }

        return mindistance;
    }

    //�o�����}�b�`���O���Ȃ��Ƃ܂Ƃ��Ȑ��x���o�Ȃ������B�o�����}�b�`���O�����܂��ł������distance���݂Ă�萸�x���グ��悤�ɂ��Ă邯�Ǒo�����}�b�`���O�Œe���ꂽ�狗���������������ǂǂ��Ȃ񂾂�B
    void extractGoogPoint(const std::vector<cv::DMatch>& matchesSrc2Dst,const std::vector<cv::DMatch> matchesDst2Src,std::vector<cv::DMatch>& gootPoints, double minDistance)
    {
        for(auto& src2Dst : matchesSrc2Dst)
        {
            cv::DMatch dst2src = matchesDst2Src[src2Dst.trainIdx];
            if(src2Dst.queryIdx == dst2src.trainIdx)
            {
                if (src2Dst.distance < minDistance * 2)
                {
                    gootPoints.push_back(src2Dst);
                }
            }
        }
    }

    //VideoCapture��const�œn����>>���g���Ȃ��Ȃ�
    void takePicture(cv::VideoCapture& capture,cv::Mat& img)
    {
        while (true)
        {
            cv::Mat captureimg;
            capture >> captureimg;
            cv::imshow("photoImg", captureimg);
            char key = cv::waitKey(1);
            if(key == 't')
            {
                captureimg.copyTo(img);
                break;
            }
        }
        cv::destroyAllWindows();
    }
}


int main()
{
    cv::VideoCapture capture(0);
    cv::Mat srcImg;
    neno::takePicture(capture, srcImg);
    cv::resize(srcImg, srcImg, cv::Size(srcImg.cols*0.5, srcImg.rows*0.5));
    cv::imshow("srcImg", srcImg);


    //�g���b�L���O�������摜�̓����_�����߂�B
    cv::Ptr<cv::Feature2D> feature2 = cv::AKAZE::create();

    std::vector<cv::KeyPoint> srcKeypoints;
    cv::Mat srcDescriptors;
    
    feature2->detectAndCompute(srcImg, cv::noArray(),srcKeypoints,srcDescriptors);


    //�����ʂ̃}�b�`���[�����BFlann���g���Ȃ��B�Ȃ��Ȃ̂��B
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");


    char inputKey = 'a';
    while (inputKey != 'q')
    {
        cv::Mat dstImg;
        capture >> dstImg;
        std::vector<cv::KeyPoint> dstKeypoints;
        cv::Mat dstDescriptors;

        feature2->detectAndCompute(dstImg, cv::noArray(),dstKeypoints,dstDescriptors);
        if(dstDescriptors.data == nullptr)
        {
            inputKey == cv::waitKey(1);
            continue;
        }
        cv::imshow("cameraImg", dstImg);
        std::vector<cv::DMatch> matchesSrc2dst,matchesDst2src;

        matcher->match(srcDescriptors, dstDescriptors, matchesSrc2dst);
        matcher->match(dstDescriptors, srcDescriptors, matchesDst2src);


        if(matchesSrc2dst.size() > 0)
        {
            //�ŏ����������߂�B
            double minDistance = neno::getMinDistance(matchesSrc2dst);
            std::vector<cv::DMatch> goodMatches;
            neno::extractGoogPoint(matchesSrc2dst,matchesDst2src, goodMatches, minDistance);

            if (goodMatches.size() > 10)
            {
                std::vector<cv::Point2f> srcPoints, dstPoints;
                for (auto& point : goodMatches)
                {
                    srcPoints.push_back(srcKeypoints[point.queryIdx].pt);
                    dstPoints.push_back(dstKeypoints[point.trainIdx].pt);
                }

                cv::Mat h = cv::findHomography(srcPoints, dstPoints, cv::RANSAC);

                if (!h.empty())
                {
                    cv::Mat srcImgCornerMat = (cv::Mat_<double>(3, 4) <<
                        0, srcImg.cols, srcImg.cols, 0,
                        0, 0, srcImg.rows, srcImg.rows,
                        1, 1, 1, 1
                        );
                    std::cout << srcImgCornerMat.type() << h.type() << std::endl;
                    
                    cv::Mat dstCornerMat = h * srcImgCornerMat;
                    cv::Mat drawImg;
                    cv::drawMatches(srcImg, srcKeypoints, dstImg, dstKeypoints, goodMatches, drawImg);
                    double* cornerPtr = reinterpret_cast<double*>(dstCornerMat.data);
                    cv::Point2f corner0(cornerPtr[0]/cornerPtr[8], cornerPtr[4]/ cornerPtr[8]); //[x,y,w]��w�Ŋ��鑀��Ƃ���perspectiveTransform�g���Ɨv��Ȃ��Ȃ�݂����B
                    cv::Point2f corner1(cornerPtr[1]/ cornerPtr[9], cornerPtr[5]/ cornerPtr[9]);
                    cv::Point2f corner2(cornerPtr[2] / cornerPtr[10], cornerPtr[6]/ cornerPtr[10]);
                    cv::Point2f corner3(cornerPtr[3]/cornerPtr[11], cornerPtr[7]/cornerPtr[11]);

                    cv::line(drawImg, corner0 + cv::Point2f(srcImg.cols, .0f), corner1 + cv::Point2f(srcImg.cols, .0f), cv::Scalar(0, 255, 0), 4);
                    cv::line(drawImg, corner1 + cv::Point2f(srcImg.cols, .0f), corner2 + cv::Point2f(srcImg.cols, .0f), cv::Scalar(0, 255, 0), 4);
                    cv::line(drawImg, corner2 + cv::Point2f(srcImg.cols, .0f), corner3 + cv::Point2f(srcImg.cols, .0f), cv::Scalar(0, 255, 0), 4);
                    cv::line(drawImg, corner3 + cv::Point2f(srcImg.cols, .0f), corner0 + cv::Point2f(srcImg.cols, .0f), cv::Scalar(0, 255, 0), 4);

                    cv::imshow("drawImg", drawImg);

                    //std::vector<cv::Point2f> obj_corners(4);
                    //obj_corners[0] = cv::Point2f(.0f, .0f);
                    //obj_corners[1] = cv::Point2f(static_cast<float>(srcImg.cols), .0f);
                    //obj_corners[2] = cv::Point2f(static_cast<float>(srcImg.cols), static_cast<float>(srcImg.rows));
                    //obj_corners[3] = cv::Point2f(.0f, static_cast<float>(srcImg.rows));
                    //cv::Mat drawImg;
                    //cv::drawMatches(srcImg, srcKeypoints, dstImg, dstKeypoints, goodMatches, drawImg);

                    //// �V�[���ւ̎ˉe�𐄒�
                    //std::vector<cv::Point2f> scene_corners(4);
                    //cv::perspectiveTransform(obj_corners, scene_corners, h);

                    //// �R�[�i�[�Ԃ���Ō��� ( �V�[�����̃}�b�v���ꂽ�Ώە��� - �V�[���摜 )
                    //float w = static_cast<float>(srcImg.cols);
                    //cv::line(drawImg, scene_corners[0] + cv::Point2f(w, .0f), scene_corners[1] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 4);
                    //cv::line(drawImg, scene_corners[1] + cv::Point2f(w, .0f), scene_corners[2] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 4);
                    //cv::line(drawImg, scene_corners[2] + cv::Point2f(w, .0f), scene_corners[3] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 4);
                    //cv::line(drawImg, scene_corners[3] + cv::Point2f(w, .0f), scene_corners[0] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 4);

                    //cv::imshow("drawImg", drawImg);

                }
            }
        }

        inputKey = cv::waitKey(1);
    }

    return 0;
}

