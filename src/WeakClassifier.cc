/*
 * WeakClassifier.cc
 *
 *
 */

#include "WeakClassifier.hh"
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <algorithm>

using namespace std;

/*
 * Stump
 */

Stump::Stump() :
		dimension_(0),
		splitAttribute_(0),
		splitValue_(0),
		classLabelLeft_(0),
		classLabelRight_(0)
{}

void Stump::initialize(u32 dimension) {
	dimension_ = dimension;
}

/*
 * Computes the error for the given attribute and split value
 *
 * */
f32 Stump::weightedGain(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32 splitValue, u32& resultingLeftLabel) {


	f32 weightedError = 0;
	for(u32 i=0; i<data.size(); i++){
		u32 label = data.at(i).label;
		if(data.at(i).attributes.at(splitAttribute) > splitValue){
			if(data.at(i).label!=1){
				weightedError = weightedError+ weights.at(i);
			}
		}
		else{
            if(data.at(i).label!=0){
                weightedError = weightedError+ weights.at(i);
            }
		}
	}
	resultingLeftLabel = 0;

	return weightedError;

}

f32 Stump::train(const std::vector<Example>& data, const Vector& weights) {
	u32 splitValueIndex;

	f32 minError = 2;

	u32 bestAttributeIndex;
	f32 bestThreshold;
	u32 selectedLeftLabel=0;

	for(u32 attrIndex=0; attrIndex<dimension_; attrIndex++){

		// the index of a data point for threshold
		// threshold is a random value from the values in dataset
		splitValueIndex = rand()% data.size();
		u32 leftLabel;
		f32 error = weightedGain(data, weights, attrIndex, data.at(splitValueIndex).attributes.at(attrIndex), leftLabel);
		if(error < minError){
			minError = error;
			bestAttributeIndex = attrIndex;
			bestThreshold = data.at(splitValueIndex).attributes.at(attrIndex);
			selectedLeftLabel = leftLabel;
		}
	}

	splitAttribute_ = bestAttributeIndex;
	splitValue_ = bestThreshold;
	classLabelLeft_ = selectedLeftLabel;
	classLabelRight_ = 1-selectedLeftLabel;

	return minError;

}

u32 Stump::classify(const Vector& v) {
	if(v.at(splitAttribute_) > splitValue_)
		return classLabelRight_;
	else
		return classLabelLeft_;
}

void Stump::classify(const std::vector<Example>& data, std::vector<u32>& classAssignments) {
	classAssignments.resize(data.size());
	for(u32 i=0; i< data.size(); i++){
		if(data.at(i).attributes.at(splitAttribute_) > splitValue_)
			classAssignments.at(i) = classLabelRight_;
		else
			classAssignments.at(i) = classLabelLeft_;
	}
}
