/*
 * AdaBoost.cc
 *
 */

#include <limits>
#include "AdaBoost.hh"
#include <iostream>
#include <cmath>

using  namespace std;

AdaBoost::AdaBoost(u32 nIterations) :
	nIterations_(nIterations)
{}

void AdaBoost::normalizeWeights() {
	f32 sum = 0;
	for(f32 &weight: weights_) sum+=weight;
	for(f32 &weight: weights_) weight= weight/sum;
}

void AdaBoost::updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration) {
	for(u32 i=0; i<weights_.size() ; i++){
		s32 diff = classAssignments.at(i)-data.at(i).label;

        f32 newWeight = weights_.at(i)*pow(beta_.at(iteration), 1-abs(diff));
        weights_.at(i) = newWeight ;
	}
}

f32 AdaBoost::weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments) {
    f32 weightedError = 0;
    for(u32 i=0; i<data.size(); i++){
			if(data.at(i).label != classAssignments.at(i)){
				weightedError = weightedError+ weights_.at(i);
			}
    }
	return weightedError;
}

void AdaBoost::initialize(std::vector<Example>& data) {
	// initialize weak classifiers
	for (u32 iteration = 0; iteration < nIterations_; iteration++) {
		weakClassifier_.push_back(Stump());
	}
	// initialize classifier weights
	classifierWeights_.resize(nIterations_);
	// initialize weights
	weights_.resize(data.size());
	for (u32 i = 0; i < data.size(); i++) {
		weights_.at(i) = 1/(f32)data.size();
	}
}

void AdaBoost::trainCascade(std::vector<Example>& data) {

	beta_.resize(weakClassifier_.size());

    for(u32 iter=0; iter<nIterations_ ; iter++){
        weakClassifier_.at(iter).initialize(data.at(0).attributes.size());
        f32 error =  weakClassifier_.at(iter).train(data, weights_);
        std::vector<u32> classAssignments;
        weakClassifier_.at(iter).classify(data, classAssignments);

		//error bound of 0.5
        if(error < ERROR_BOUND && error>0){
			beta_.at(iter) = (error/(1.0-error));
			updateWeights(data, classAssignments, iter);
			normalizeWeights();
        }
    }
}

u32 AdaBoost::classify(const Vector& v) {
	u32 nClasses = 2;
	f32 maxConfidence = numeric_limits<f32>::min();
	u32 selectedLabel;

	for(u32 k=0; k<nClasses; k++){
		f32 confidenceVal = confidence(v, k);
		if(confidenceVal > maxConfidence){
			maxConfidence = confidenceVal;
			selectedLabel = k;
		}
	}
	return selectedLabel;
}

f32 AdaBoost::confidence(const Vector& v, u32 k) {

    f32 sum=0;
	for(u32 t=0;t<nIterations_;t++){
		u32 predLabel = weakClassifier_.at(t).classify(v);
        if(predLabel==k){
//            sum += alpha_.at(t);
            sum += log(1.0/beta_.at(t));
        }
	}

	return sum;
}
