/*
 * main.cc
 *
 *      Author: Emad
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <fstream>
#include <iostream>
#include "Types.hh"
#include <stdlib.h>
#include <sstream>
#include "AdaBoost.hh"

using namespace std;

#define _objectWindow_width 121
#define _objectWindow_height 61

#define _searchWindow_width 61
#define _searchWindow_height 61


// use 30/15 for overlapping negative examples and 120/60 for non-overlapping negative examples
#define _displacement_overlap_x 30
#define _displacement_overlap_y 15

#define _displacement_x 120
#define _displacement_y 60


void computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram) {

	cv::Mat patch;
	cv::getRectSubPix(image, cv::Size(_objectWindow_width, _objectWindow_height), p, patch);

	s32 histSize = 256;
	f32 range[] = { 0, (f32)256 };
	const f32* histRange = {range};
	bool uniform = true; bool accumulate=false;

	cv::Mat hist;

	cv::calcHist(&patch, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	histogram.resize(hist.rows*hist.cols);
	if(hist.isContinuous())
		histogram = hist;
	else{
		for(u32 i=0; i<hist.rows ; i++)
			histogram[i] = hist.at<f32>(i);
	}

}


void createExample(const cv::Mat& image, cv::Point& point, Example& example, u32 label){

	Vector histogram;
	computeHistogram(image, point, histogram);
	example.attributes = histogram;
	example.label = label;
}



void generateTrainingData(std::vector<Example>& data, const std::vector<cv::Mat>& imageSequence, const std::vector<cv::Point>& referencePoints) {

	for(u32 i=0; i<imageSequence.size();i++){
		cv::Point refPoint = referencePoints.at(i);
		Example posExample;
		createExample(imageSequence[i], refPoint, posExample, 1);
		data.push_back(posExample);


		cv::Point displacedPoint(refPoint.x+_displacement_overlap_x, refPoint.y);
		Example negExample1;
		createExample(imageSequence[i], displacedPoint, negExample1, 0);
		data.push_back(negExample1);

        displacedPoint.x = refPoint.x;
        displacedPoint.y = refPoint.y +_displacement_overlap_y;
		Example negExample2;
		createExample(imageSequence[i], displacedPoint, negExample2, 0);
		data.push_back(negExample2);


		displacedPoint.x = refPoint.x + _displacement_x;
		displacedPoint.y = refPoint.y;
		Example negExample3;
		createExample(imageSequence[i], displacedPoint, negExample3, 0);
		data.push_back(negExample3);


        displacedPoint.x = refPoint.x;
        displacedPoint.y = refPoint.y + _displacement_y;
        Example negExample4;
        createExample(imageSequence[i], displacedPoint, negExample4, 0);
        data.push_back(negExample4);

	}

}



void loadImage(const std::string& imageFile, cv::Mat& image) {

}

void loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence,
					 std::vector<cv::Point>& referencePoints, std::string& imgDir) {

	std::ifstream inStream(trainDataFile);
	if(!inStream){
		std::cout << "Error in reading training file " << trainDataFile << std::endl;
		exit(-1);
	}
	u32 numInput;
	inStream >> numInput;

	std::string imgPath;
	u32 x,y;

	for(u32 i=0; i<numInput;i++){
		inStream >> imgPath;
		imgPath = imgDir + imgPath;
		cv::Mat frame = cv::imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);
		if(frame.empty()){
			cout << "Frame not found at path " << imgPath << endl;
			exit(-1);
		}
		imageSequence.push_back(frame);
		inStream >> x;
		inStream >> y;
		cv::Point refPoint(x,y);
		referencePoints.push_back(refPoint);
	}

}

void loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint, string& imgDir) {

	std::ifstream inStream(testDataFile);
	if(!inStream){
		std::cout << "Error in reading training file " << testDataFile << std::endl;
		exit(-1);
	}

	inStream >> startingPoint.x;
	inStream >> startingPoint.y;

	u32 numImg;
	inStream >> numImg;

    std::string imgPath;

    for(u32 i=0; i<numImg;i++){
        inStream >> imgPath;
        imgPath = imgDir + imgPath;
        cv::Mat frame = cv::imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);
        if(frame.empty()){
            cout << "Frame not found at path " << imgPath << endl;
            exit(-1);
        }
        imageSequence.push_back(frame);
    }

}

void findBestMatch(const cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost) {

    cv::Mat patch;
    cv::getRectSubPix(image, cv::Size(_searchWindow_width, _searchWindow_height), lastPosition, patch);

	cv::Mat img(image);
    // the last point is located in middle of the 61x61 patch
    cv::Point startPoint(lastPosition.x-(u32)_searchWindow_width/2, lastPosition.y-(u32)_searchWindow_height/2);

    cv::Mat confidenceVal(_searchWindow_width, _searchWindow_height, CV_32F, cv::Scalar(0));
	f32 maxConfidence=-1;


	u32 x_pos,y_pos;

    for(u32 i=0 ; i < _searchWindow_width ; i++) {
		for (u32 j=0; j < _searchWindow_height ; j++) {
			Vector histogram;
			cv::Point candidatePoint(startPoint.x+i, startPoint.y+j);

			computeHistogram(image, candidatePoint, histogram);
			f32 confidence = adaBoost.confidence(histogram, 1);

			if(confidence > maxConfidence){
				maxConfidence = confidence;
				x_pos = candidatePoint.x;
				y_pos = candidatePoint.y;
			}
		}
	}

	lastPosition.x = x_pos;
	lastPosition.y = y_pos;

}

void drawTrackedFrame(cv::Mat& image, cv::Point& position) {
	cv::rectangle(image, cv::Point(position.x - _objectWindow_width / 2, position.y - _objectWindow_height / 2),
			cv::Point(position.x + _objectWindow_width / 2, position.y + _objectWindow_height / 2), 0, 3);
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", image);
	//std::sleep(1);
	cv::waitKey(0);
}

int main( int argc, char** argv )
{
	//implement the functions above to make this work. Or implement your solution from scratch
	if(argc != 4) {
		std::cout <<" Usage: " << argv[0] << " <training-frame-file> <test-frame-file> <# iterations for AdaBoost>" << std::endl;
		return -1;
	}

	u32 adaBoostIterations = atoi(argv[3]);
    string imgDir = "nemo/";

    // load the training frames
	std::vector<cv::Mat> imageSequence;
	std::vector<cv::Point> referencePoints;
	loadTrainFrames(argv[1], imageSequence, referencePoints, imgDir);

	// generate gray-scale histograms from the training frames:
	// one positive example per frame (_objectWindow_width x _objectWindow_height window around reference point for object)
	// four negative examples per frame (with _displacement_{x/y} + small random displacement from reference point)
	std::vector<Example> trainingData;
	generateTrainingData(trainingData, imageSequence, referencePoints);

//	std::vector<Example> trainingData_1;
//	generateTrainingData_1(trainingData_1, imageSequence, referencePoints);

	// initialize AdaBoost and train a cascade with the extracted training data
	AdaBoost adaBoost(adaBoostIterations);
	adaBoost.initialize(trainingData);
	adaBoost.trainCascade(trainingData);

	// log error rate on training set
	u32 nClassificationErrors = 0;
	for (u32 i = 0; i < trainingData.size(); i++) {
		u32 label = adaBoost.classify(trainingData.at(i).attributes);
		nClassificationErrors += (label == trainingData.at(i).label ? 0 : 1);
	}
	std::cout << "Error rate on training set: " << (f32)nClassificationErrors / (f32)trainingData.size() << std::endl;

	// load the test frames and the starting position for tracking
	std::vector<Example> testImages;
	cv::Point lastPosition;
	loadTestFrames(argv[2], imageSequence, lastPosition, imgDir);

//	// for each frame...
	for (u32 i = 10; i < imageSequence.size()  ; i++) { //
		// ... find the best match in a window of size
//		// _searchWindow_width x _searchWindow_height around the last tracked position
		findBestMatch(imageSequence.at(i), lastPosition, adaBoost);

		//
//		// draw the result
		drawTrackedFrame(imageSequence.at(i), lastPosition);
	}

	return 0;
}
