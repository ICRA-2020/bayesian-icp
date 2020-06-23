/*
 Copyright (c) 2019
 
 Fahira Afzal Maken and Lionel Ott,
 The University of Sydney, Australia.
 
 All rights reserved.
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 
 1. Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in
 the documentation and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 */



#ifndef __UTILS_H__
#define __UTILS_H__


#include <Eigen/Core>

#include <pcl/common/common_headers.h>
#include <pcl/common/io.h>

#include "types.h"


/**
 * \brief Unroll transformation matrix into 6D pose parameters
 *
 * \param transformation matrix transformation matrix from which to extract
 *      individual 6D pose parameters
 * \return vector containing individual pose parameters
 *      {x,y,z,roll,pitch and yaw}
 */
std::vector<double> get_translation_roll_pitch_yaw(
        Eigen::Matrix4d const&          transformation_matrix
);

/**
 * \brief Normalizes an angle into [-Pi, PI] range.
 *
 * \param radians angle to be normalized in radian
 * \return normalized angle
 */
double normalizeAngle(double radians);

/**
 * \brief Converts a XYZ point cloud into matrix representation.
 *
 * \param cloud input cloud to convert
 * \return matrix representing the cloud as a 3xN matrix
 */
Eigen::MatrixXd cloud_to_matrix(Cloud_t::Ptr cloud);

/**
 * \brief Returns the RMSE of the alignment between the clouds given
 *      the transform.
 *
 * \param source source cloud to be transformed with the provided transform
 * \param target cloud representing the desired location for transformed source
 * \param transform transformation to be applied to source before compute the
 *      RMSE
 */
double compute_rmse(
        Cloud_t::Ptr                    source,
        Cloud_t::Ptr                    target,
        Eigen::Matrix4d const&          transform
);

/**
 * \brief Returns matching point cloud entries.
 *
 * Return a pair of clouds with matching points, as well as, an untransformed
 * batch from source cloud containing points at the same indices as transformed
 * batch after max_distance clipping.
 * If filter is true, duplicate correspondences from target to source will be filtered out.
 *
 * \param raw_batch untransformed batch from source cloud
 * \param transformed_batch points to find matches in target cloud
 * \param target cloud in which to find matches for batch
 * \param max_dist maximum distance between matching points
 * \param filter true or false
 * \return pair of matching points in batch and target along with raw cloud of same size
 */
std::tuple<Cloud_t::Ptr, Cloud_t::Ptr, Cloud_t::Ptr> find_correspondences(
           Cloud_t::Ptr                    raw_batch,
           Cloud_t::Ptr                    transformed_batch,
           Cloud_t::Ptr                    target,
           double                          max_dist,
           bool                            filter
);

/**
 * \brief Return indices of duplicate correspondence entries.
 *
 * This find duplicate correspondence indices based on distance information
 * which should be removed.
 *
 * \param correspondences point correspondence information
 * \return vector containing indices to be removed
 */
std::vector<int> erase_duplicate_correspondences(
        pcl::CorrespondencesPtr         correspondences
);

/**
 * \brief Returns maximum absolute value from x,y,z dimensions from both
 *      source and target clouds.
 *
 * \param source source cloud to process
 * \param target target cloud to process
 * \return maximum value
 */
float get_absolute_max(
        Cloud_t::Ptr const&             source,
        Cloud_t::Ptr const&             target
);

/**
 * \brief Normalize cloud entries using the provided maximum value.
 *
 * This normalizes all distances in the point cloud into the range [-1, 1]
 * across all dimensions while keeping the aspect ratio between the different
 * dimensions intact.
 *
 * \param cloud the point cloud to normalize the entries of
 * \param max_absolute normalization value
 * \return normalised cloud
 */

Cloud_t::Ptr normalise_clouds(Cloud_t::Ptr const& cloud, float max_absolute);

/**
 * \brief Rescale the translation part of the transformation matrix.
 *
 * Scales the translation component of the transformation matrix while keeping
 * the rotational aspect unchanged.
 *
 * \param transformation_matrix the matrix to apply the scaling transform to
 * \param max_absolute normalization value
 */
void rescale_transformation_matrix(
        Eigen::Matrix4d &               transformation_matrix,
        float                           max_absolute
);
/**
 * \brief Rescale the translation part of the transformation matrix overlao function that accepts std::array.
 *
 * Scales the translation component of the transformation matrix while keeping
 * the rotational aspect unchanged.
 *
 * \param transformation_matrix the matrix to apply the scaling transform to
 * \param max_absolute normalization value
 */
Array6_d  rescale_transformation_parameters(
        
        Array6_d                        params,
        float                           max_absolute
);
/**
 * \brief converts array of pose params into homogeneous transformation matrix
 * \param array of pose params
 * \return 4x4 homogeneous transformation matrix
 */

Eigen::Matrix4d get_transform(Array6_d const& params) ;



// function to print array

template <typename T>
    void print_array(std::array<T,6> vec)
    {
        for (int i=0; i<vec.size(); i++)
        {
            std::cout<<vec[i]<<" ";
        }
        std::cout<<std::endl;
    }

#endif /* __UTILS_H__ */
