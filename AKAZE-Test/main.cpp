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

    //双方向マッチングしないとまともな精度が出なかった。双方向マッチングがうまくできた上でdistanceをみてより精度を上げるようにしてるけど双方向マッチングで弾かれたら距離小さそうだけどどうなんだろ。
    void extractGoogPoint(const std::vector<std::vector<cv::DMatch>>& matchesSrc2Dst,const std::vector<std::vector<cv::DMatch>> matchesDst2Src,std::vector<cv::DMatch>& gootPoints)
    {
        for(auto& src2Dst : matchesSrc2Dst)
        {
            cv::DMatch src2dstMatch = src2Dst[0];
            float dist1 = src2Dst[0].distance;
            float dist2 = src2Dst[1].distance;

            if(dist1 < 0.8 * dist2)
            {
                std::vector<cv::DMatch> dst2src = matchesDst2Src[src2dstMatch.trainIdx];
                cv::DMatch dst2srcMatch = dst2src[0];
                dist1 = dst2src[0].distance;
                dist2 = dst2src[1].distance;

                if(dist1 < 0.8 * dist2)
                {
                    if (src2dstMatch.queryIdx == dst2srcMatch.trainIdx)
                    {
                        gootPoints.push_back(src2dstMatch);
                    }
                }
            }
        }
    }

    //VideoCaptureはconstで渡すと>>が使えなくなる
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


    //トラッキングしたい画像の特徴点を求める。
    cv::Ptr<cv::Feature2D> feature2 = cv::AKAZE::create();

    std::vector<cv::KeyPoint> srcKeypoints;
    cv::Mat srcDescriptors;
    
    feature2->detectAndCompute(srcImg, cv::noArray(),srcKeypoints,srcDescriptors);


    //特徴量のマッチャーを作る。Flannが使えない。なぜなのか。
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
        std::vector<std::vector<cv::DMatch>> matchesSrc2dst,matchesDst2src;

        matcher->knnMatch(srcDescriptors, dstDescriptors, matchesSrc2dst,2);
        matcher->knnMatch(dstDescriptors, srcDescriptors, matchesDst2src,2);


        if(matchesSrc2dst.size() > 0)
        {
            //最小距離を求める。
            std::vector<cv::DMatch> goodMatches;
            neno::extractGoogPoint(matchesSrc2dst,matchesDst2src, goodMatches);

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
                    cv::Point2f corner0(cornerPtr[0]/cornerPtr[8], cornerPtr[4]/ cornerPtr[8]); //[x,y,w]でwで割る操作とかがperspectiveTransform使うと要らなくなるみたい。
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

                    //// シーンへの射影を推定
                    //std::vector<cv::Point2f> scene_corners(4);
                    //cv::perspectiveTransform(obj_corners, scene_corners, h);

                    //// コーナー間を線で結ぶ ( シーン中のマップされた対象物体 - シーン画像 )
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

