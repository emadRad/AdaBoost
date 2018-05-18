/*
 * AdaBoost.hh
 *
 */

#ifndef ADABOOST_HH_
#define ADABOOST_HH_

#include <vector>
#include <string.h>
#include "Types.hh"
#include "WeakClassifier.hh"

#define ERROR_BOUND 0.5

class AdaBoost
{
private:
	u32 nIterations_;
	Vector weights_;
	std::vector<Stump> weakClassifier_;
	Vector classifierWeights_;
	Vector beta_;

	void normalizeWeights();
	void updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration);
	f32 weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments);
public:
	AdaBoost(u32 nIterations);
	void initialize(std::vector<Example>& data);
	void trainCascade(std::vector<Example>& data);
	u32 classify(const Vector& v);
	f32 confidence(const Vector& v, u32 k);
};


#endif /* ADABOOST_HH_ */
