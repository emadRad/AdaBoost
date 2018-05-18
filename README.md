#AdaBoost Algorithm for Object Tracking

This a C++ implementation of the AdaBoost  algorithm from [A desicion-theoretic generalization of on-line learning and an application to boosting](https://link.springer.com/chapter/10.1007/3-540-59119-2_166) by Yoav Freund and Robert E. Schapire. The algorithm is used  to track an object in a video sequence.

The task is to track Nemo in a sequence 32 frames. The first ten annotated frames are used for training and the remaining frames are used for testing.



### Requirements 

*  CMake 2.6+
* OpenCV 2.4.x
* C++11



### How to compile:

In Linux terminal :

`$ cmake CMakeLists.txt`

Then compile it with :

`$ make`

### How to run:

In Linux terminal :

`$ ./bin/tracking nemo/frames.train nemo/frames.test 50 `

After running the program you will see  22 frames with the bounding box around the object(Nemo).

