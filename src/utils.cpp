
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


#include <cmath>
#include <random>

#include <pcl/registration/registration.h>

#include "utils.h"


std::vector<double> get_translation_roll_pitch_yaw(
        Eigen::Matrix4d const&          transformation_matrix
)
{
    double x = transformation_matrix(0, 3);
    double y = transformation_matrix(1, 3);
    double z = transformation_matrix(2, 3);

    double roll = std::atan2(
            transformation_matrix(2, 1),
            transformation_matrix(2, 2)
    );
    double pitch = std::asin(-transformation_matrix(2, 0));
    double yaw = std::atan2(
            transformation_matrix(1, 0),
            transformation_matrix(0, 0)
    );

    std::cout << "[x, y, z, roll, pitch, yaw] = ["
              << x << "," << y << "," << z << ","
              << roll << "," << pitch << "," << yaw << "]" <<std::endl;

    return std::vector<double>{x, y, z, roll, pitch, yaw};
}

double normalizeAngle(double radians)
{
    if(std::isnan(radians))
    {
        return 0.0;
    }

    radians = std::fmod(radians + M_PI, 2*M_PI);
    if(radians < 0.0)
    {
        radians += 2*M_PI;
    }
    radians -= M_PI;

    assert(radians >= -M_PI && radians <= M_PI);

    return radians;
}


Eigen::MatrixXd cloud_to_matrix(Cloud_t::Ptr cloud)
{
    Eigen::MatrixXd mat(3, cloud->size());
    auto point_data = cloud->points;
    for(int i=0; i<cloud->size(); ++i)
    {
        mat(0, i) = point_data[i].x;
        mat(1, i) = point_data[i].y;
        mat(2, i) = point_data[i].z;
    }

    return mat;
}

double compute_rmse(
        Cloud_t::Ptr                    source,
        Cloud_t::Ptr                    target,
        Eigen::Matrix4d const&          transform
)
{
    Cloud_t::Ptr transformed(new Cloud_t);
    pcl::transformPointCloud<Point_t>(*source, *transformed, transform);

    auto correspondences = find_correspondences(source,transformed,target,0.1,true);

    auto transformed_paired = std::get<1>(correspondences);
    auto target_paired = std::get<2>(correspondences);

    auto target_paired_matrix = cloud_to_matrix(target_paired);
    auto transformed_paired_matrix = cloud_to_matrix(transformed_paired);

    // Check that sufficient matching pairs have been found
    if(target_paired_matrix.cols() < 3)
    {
        std::cout << "No correspondence found. SGD Pose is not relaible.\n";
    }

    double error = 0.0;
    for(int i=0; i<transformed_paired_matrix.cols(); ++i)
    {
        error += (
                target_paired_matrix.col(i) -
                transformed_paired_matrix.col(i)
        ).norm();
    }

    return std::sqrt(error / transformed_paired_matrix.cols());
}


std::tuple<Cloud_t::Ptr, Cloud_t::Ptr , Cloud_t::Ptr> find_correspondences(
          Cloud_t::Ptr                  raw_batch,
          Cloud_t::Ptr                  transformed_batch,
          Cloud_t::Ptr                  target,
          double                        max_dist,
          bool                          filter
)
{
    // Find correspondences between batch and target cloud
    pcl::registration::CorrespondenceEstimation<Point_t, Point_t> estimator;
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
    estimator.setInputSource(transformed_batch);
    estimator.setInputTarget(target);
    estimator.determineCorrespondences(*correspondences, max_dist);

    // Extract matching correspondences
    Cloud_t::Ptr raw_batch_resized (new Cloud_t);
    Cloud_t::Ptr tr_batch_paired(new Cloud_t);
    Cloud_t::Ptr target_paired(new Cloud_t);

    if (filter)
{
    // Remove duplicate correspondences
    auto skip_indices = erase_duplicate_correspondences(correspondences);
    for(size_t i=0; i<correspondences->size(); ++i)
    {
        // If the current correspondence should be skipped abort
        // further processing
        if(std::find(skip_indices.begin(), skip_indices.end(), i) != skip_indices.end())
        {
            continue;
        }
        else
        {
            raw_batch_resized->points.push_back(
                    raw_batch->points[(*correspondences)[i].index_query]
            );
            tr_batch_paired->points.push_back(
                    transformed_batch->points[(*correspondences)[i].index_query]
            );
            target_paired->points.push_back(
                    target->points[(*correspondences)[i].index_match]
            );
        }
    }
}
    else
    {
        for(size_t i=0; i<correspondences->size(); ++i)
        {
            raw_batch_resized->points.push_back(
                                                raw_batch->points[(*correspondences)[i].index_query]
                                                );
            tr_batch_paired->points.push_back(
                                              transformed_batch->points[(*correspondences)[i].index_query]
                                              );
            target_paired->points.push_back(
                                            target->points[(*correspondences)[i].index_match]
                                            );
        }
    }
    // Set cloud properties
    raw_batch_resized->width = raw_batch_resized->points.size ();
    raw_batch_resized->height = 1;
    raw_batch_resized->is_dense = true;

    tr_batch_paired->width = tr_batch_paired->points.size ();
    tr_batch_paired->height = 1;
    tr_batch_paired->is_dense = true;

    target_paired->width = target_paired->points.size ();
    target_paired->height = 1;
    target_paired->is_dense = true;

    return std::make_tuple(raw_batch_resized,tr_batch_paired, target_paired);
}

std::vector<int> erase_duplicate_correspondences(
        pcl::CorrespondencesPtr         correspondences
)
{
    struct DuplicateCorrespondencesences
    {
        std::vector<int>                duplicate_match;
        std::vector<int>                duplicate_query;
        std::vector<double>             duplicate_distance;
    };
    struct MultipleCorrespondencesences
    {
        std::vector<int>                source;
        std::vector<int>                target;
        std::vector<double>             distances;
    };


    DuplicateCorrespondencesences dup;
    for(int i=0; i<correspondences->size(); i++)
    {
        dup.duplicate_match.push_back((*correspondences)[i].index_match);
        dup.duplicate_query.push_back((*correspondences)[i].index_query);
        dup.duplicate_distance.push_back((*correspondences)[i].distance);
    }

    // Storage for indices that should be skipped due to duplication
    std::vector<int> indices_to_skip;

    int count = 0;
    // Stores the number being considered as multiple match
    std::vector<double> multiple_match;
    for(int i=0; i<correspondences->size(); ++i)
    {
        int num_items = std::count(
                dup.duplicate_match.begin(),
                dup.duplicate_match.end(),
                dup.duplicate_match[i]
        );

        if(num_items > 1)
        {
            // Skip points that have already been considered for multiple entries
            bool already_processed = std::find(
                    multiple_match.begin(),
                    multiple_match.end(),
                    dup.duplicate_match[i]
            ) != multiple_match.end();
            if(already_processed)
            {
                continue;
            }

            // Add vector being consisdred for multple entry
            multiple_match.push_back(dup.duplicate_match[i]);

            MultipleCorrespondencesences mul;
            for(int j=0; j<correspondences->size(); ++j)
            {
                if((*correspondences)[j].index_match == dup.duplicate_match[i])
                {
                    mul.source.push_back((*correspondences)[j].index_query);
                    mul.target.push_back((*correspondences)[j].index_match);
                    mul.distances.push_back((*correspondences)[j].distance);
                    count++;
                }
            }

            // Get the minimum element from distance vector
            auto min_element_itr = std::min_element(
                    std::begin(mul.distances),
                    std::end(mul.distances)
            );
            auto minimum_index =
                std::distance(std::begin(mul.distances), min_element_itr);

            // Save indices of correspondences to be removed
            for(int k=0; k<mul.source.size(); ++k)
            {
                if(k != minimum_index)
                {
                    indices_to_skip.push_back(mul.source[k]);
                }
            }
        }
    }

    return indices_to_skip;
}

float get_absolute_max(
        Cloud_t::Ptr const&             source,
        Cloud_t::Ptr const&             target
)
{
    Eigen::Vector4f min_in(Eigen::Vector4f::Zero());
    Eigen::Vector4f max_in(Eigen::Vector4f::Zero());
    pcl::getMinMax3D(*source, min_in, max_in);

    Eigen::Vector4f min_out(Eigen::Vector4f::Zero());
    Eigen::Vector4f max_out(Eigen::Vector4f::Zero());
    pcl::getMinMax3D(*target, min_out, max_out);

    float x_min = std::min(min_in[0], min_out[0]);
    float y_min = std::min(min_in[1], min_out[1]);
    float z_min = std::min(min_in[2], min_out[2]);

    float x_max = std::max(std::abs(max_in[0]), std::abs(max_out[0]));
    float y_max = std::max(std::abs(max_in[1]), std::abs(max_out[1]));
    float z_max = std::max(std::abs(max_in[2]), std::abs(max_out[2]));

    std::vector<float> pool{
        std::abs(x_min),
        std::abs(y_min),
        std::abs(z_min),
        std::abs(x_max),
        std::abs(y_max),
        std::abs(z_max)
    };
    return *std::max_element(std::begin(pool), std::end(pool));
}

Cloud_t::Ptr normalise_clouds(Cloud_t::Ptr const& cloud, float max_absolute)
{
    Cloud_t::Ptr normalised_cloud(new Cloud_t);
    *normalised_cloud = *cloud;

    for(int i=0; i<cloud->size(); ++i)
    {
        normalised_cloud->points[i].x = (cloud->points[i].x ) / (max_absolute);
        normalised_cloud->points[i].y = (cloud->points[i].y ) / (max_absolute);
        normalised_cloud->points[i].z = (cloud->points[i].z ) / (max_absolute);
    }

    return normalised_cloud;
}

void rescale_transformation_matrix(
        Eigen::Matrix4d &               transformation_matrix,
        float                           max_absolute
)
{
    double x_rescaled = (transformation_matrix(0,3) * (max_absolute));
    double y_rescaled = (transformation_matrix(1,3) * (max_absolute));
    double z_rescaled = (transformation_matrix(2,3) * (max_absolute));

    transformation_matrix(0, 3) = x_rescaled;
    transformation_matrix(1, 3) = y_rescaled;
    transformation_matrix(2, 3) = z_rescaled;
}

Array6_d  rescale_transformation_parameters(
        
        Array6_d                        params,
        float                           max_absolute
)
{

    double a = (params[0] * max_absolute);
    double b = (params[1] * max_absolute);
    double c = (params[2]* max_absolute);
    params[0]=a;
    params[1]=b;
    params[2]=c;
    return params;
}

Eigen::Matrix4d get_transform(Array6_d const& params) 
{
    
    Eigen::Affine3d transform(Eigen::Affine3d::Identity());
    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(params[5], Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(params[4], Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(params[3], Eigen::Vector3d::UnitX());
    
    transform.translate(Eigen::Vector3d(params[0], params[1], params[2]));
    transform.rotate(rot);
    
    return transform.matrix();
}

