#ifndef NNFRAMEWORK_COMMON_HPP
#define NNFRAMEWORK_COMMON_HPP

// Common.hpp -> place for defining common things among the NNFramework

#define NNFRAMEWORK_ZERO     (0L)
#define MATRIX_COL_INIT_VAL  (1L)

#define INPUT_LAYER_IDX      (0L)
#define OUTPUT_LAYER_IDX(noOfLayers) (noOfLayers - 1) // forgive me my ugly 'C' past. :) 

#define PREVIOUS_LAYER_IDX(currentIdx) (currentIdx - 1)
#define NEXT_LAYER_IDX(currentIdx) (currentIdx + 1)

#endif