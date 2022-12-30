#ifndef ALPHAPOSE_HEADERS_H
#define ALPHAPOSE_HEADERS_H

#include <map>
#include <vector>
#include <cassert>
#include <locale.h>
#include <string>
#include <algorithm>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <type_traits>
#define _USE_MATH_DEFINES
#include <cmath>
#include "opencv2/opencv.hpp"

#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
# define ORT_CHAR wchar_t
#else
# define ORT_CHAR char
#endif

#ifdef ENABLE_DEBUG_STRING
#define POSE_DEBUG 1
#else
#define POSE_DEBUG 0
#endif

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#endif // !ALPHAPOSE
