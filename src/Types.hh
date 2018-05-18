/*
 * Types.hh
 *
 */

#ifndef TYPES_HH_
#define TYPES_HH_

#include <vector>

typedef unsigned char u8;
typedef signed char s8;
typedef unsigned int u32;
typedef signed int s32;
typedef unsigned long int u64;
typedef signed long int s64;
typedef float f32;
typedef double f64;

typedef std::vector<f32> Vector;

struct Example {
	Vector attributes;
	u32 label;
};

#endif /* TYPES_HH_ */
