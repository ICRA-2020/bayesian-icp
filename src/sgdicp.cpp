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


#include <Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include "sgdicp.h"
#include "point_cloud_batch.h"
#include "utils.h"


SGDICP::SGDICP(std::unique_ptr<AbstractSgdOptimizer> sgd_optimizer)
    :   m_sgd_optimizer(std::move(sgd_optimizer))
{
    
}

SGDICP::SGDICP(std::unique_ptr<AbstractSgldOptimizer> sgld_optimizer)
    :   m_sgld_optimizer(std::move(sgld_optimizer))
{
    
}


Eigen::Matrix4d SGDICP::align_clouds(
        Cloud_t::Ptr                    cloud_in,
        Cloud_t::Ptr                    cloud_out,
        Parameters const&               parameters
)
{
    // Mini batch generator
   
    auto batcher = PointCloudBatch(*cloud_in, parameters.batch_size);

    // Convergence monitoring
    Eigen::VectorXd reference_state = Eigen::VectorXd::Zero(6);
    update_state(reference_state);
    auto has_converged = false;
    auto convergence_count = 0;
    auto iterations = 0;
    int diverge_count = 0;
    // SGD iteration loop
    while(iterations < parameters.iterations && !has_converged)
    {
        // Pick batch from the input cloud
        auto raw_initial_batch = batcher.next_batch();
        Cloud_t::Ptr transformed_batch(new Cloud_t);
        pcl::transformPointCloud<Point_t>(
                *raw_initial_batch,
                *transformed_batch,
                current_transform()
        );

        // Find correspondences between input batch and target cloud
        auto correspondences = find_correspondences(
                raw_initial_batch,
                transformed_batch,
                cloud_out,
                parameters.max_matching_distance,
                parameters.filter
        );

        auto raw_batch_resized = std::get<0>(correspondences);
        auto transformed_batch_paired = std::get<1>(correspondences);
        auto cloud_out_paired = std::get<2>(correspondences);

        // If correspondences are not found for 100 mini-batches, return
        // the current diverged transform
        if (diverge_count==100) {
            std::cout << "ICP is diverging ..." << std::endl;
            return current_transform();
        }

        // Abort this iteration if not enough matches are found
        if(cloud_out_paired->size() == 0 || cloud_out_paired->size() < 3)
        {
            std::cout << "No correspondences between clouds found" << std::endl;
            diverge_count++;
            continue;
        }

        // Perform a single SGD step
        auto gradient_terms = compute_gradient_terms(
                raw_batch_resized,
                transformed_batch_paired,
                cloud_out_paired
        );
        m_sgd_optimizer->update_parameters(gradient_terms);

        // Check for convergence
        bool translation_converged = is_translation_converged(
                reference_state,
                parameters.translational_convergence_threshold
        );
        bool rotation_converged = is_rotation_converged(
                reference_state,
                parameters.rotational_convergence_threshold
        );
        if(translation_converged && rotation_converged)
        {
            convergence_count++;
        }
        else
        {
            update_state(reference_state);
            convergence_count = 0;
        }

        if(convergence_count >= parameters.convergence_steps)
        {
            has_converged = true;
            std::cout << "Convergence achieved" << std::endl;
        }

        iterations++;
    }

    std::cout << "Iterations: " << iterations
              << " converged: " << has_converged << std::endl;

    return current_transform();
}

 std::vector<Array6_d> SGDICP::align_clouds_with_bayesian_icp(
                Cloud_t::Ptr            cloud_in,
                Cloud_t::Ptr            cloud_out,
                Parameters const&       parameters,
                std::vector<double>      prior_mean,
                float                    prior_variance
        )
 {
  
     m_pose_samples.reserve(parameters.iterations);
      int cloud_size = cloud_in->size();
    
     // Mini batch generator
    auto batcher = PointCloudBatch(*cloud_in, parameters.batch_size);

    // Convergence monitoring
    Eigen::VectorXd reference_state = Eigen::VectorXd::Zero(6);
    update_state(reference_state,1);
    auto has_converged = false;
    auto convergence_count = 0;
    auto iterations = 0;
    int diverge_count = 0;
   
    // SGD iteration loop
    while(iterations < parameters.iterations && !has_converged)
    {
        // Pick batch from the input cloud
        auto raw_initial_batch = batcher.next_batch();
        Cloud_t::Ptr transformed_batch(new Cloud_t);
        pcl::transformPointCloud<Point_t>(
                *raw_initial_batch,
                *transformed_batch,
                current_transform(1)
        );

        // Find correspondences between input batch and target cloud
        auto correspondences = find_correspondences(
                raw_initial_batch,
                transformed_batch,
                cloud_out,
                parameters.max_matching_distance,
                parameters.filter
        );

        auto raw_batch_resized = std::get<0>(correspondences);
        auto transformed_batch_paired = std::get<1>(correspondences);
        auto cloud_out_paired = std::get<2>(correspondences);

        // If correspondences are not found for 100 mini-batches, return
        // the current diverged transform
        if (diverge_count==100) {
            std::cout << "ICP is diverging ..." << std::endl;
            m_pose_samples.resize(iterations);
            return m_pose_samples;
        }

        // Abort this iteration if not enough matches are found
        if(cloud_out_paired->size() == 0 || cloud_out_paired->size() < 3)
        {
            std::cout << "No correspondences between clouds found" << std::endl;
            diverge_count++;
            continue;
        }

        // get the gradient of likelihood
        auto gradient_terms = compute_gradient_terms(
                raw_batch_resized,
                transformed_batch_paired,
                cloud_out_paired,
                1
        );
        
        //add the gradient of prior
     auto  param= m_sgld_optimizer->get_parameters();
        for (size_t j=0; j<gradient_terms.size(); j++)
            
        {
        
        if (j<=2)
        {
           
          gradient_terms[j] =    (cloud_size*gradient_terms[j]) + ((1/prior_variance)*(param[j] - prior_mean[j])) ;
        }
        else
        {
            gradient_terms[j]= (cloud_size*gradient_terms[j])+ ((1/prior_variance)*(std::sin(param[j]-prior_mean[j])));
           
        }
        }
        
        m_sgld_optimizer->update_parameters(gradient_terms);
        m_pose_samples.push_back(m_sgld_optimizer->get_parameters());

        // Check for convergence
        bool translation_converged = is_translation_converged(
                reference_state,
                parameters.translational_convergence_threshold,1
        );
        bool rotation_converged = is_rotation_converged(
                reference_state,
                parameters.rotational_convergence_threshold,1
        );
        if(translation_converged && rotation_converged)
        {
            convergence_count++;
        }
        else
        {
            update_state(reference_state,1);
            convergence_count = 0;
        }

        if(convergence_count >= parameters.convergence_steps)
        {
            has_converged = true;
            std::cout << "Convergence achieved" << std::endl;
        }

        iterations++;
    }

    std::cout << "Iterations: " << iterations<<std::endl;

    return m_pose_samples;
 }



std::vector<double> SGDICP::compute_gradient_terms(
           Cloud_t::Ptr                    raw_cloud_resized,
           Cloud_t::Ptr                    transformed_batch_paired,
           Cloud_t::Ptr                    cloud_out_paired,
           int                              method
)
{
    auto raw_cloud_matrix = cloud_to_matrix(raw_cloud_resized);
    auto batch_paired_matrix = cloud_to_matrix(transformed_batch_paired);
    auto cloud_out_paired_matrix = cloud_to_matrix(cloud_out_paired);
    std::vector<double> gradient_cost = {0,0,0,0,0,0};
    
    if (method==0)
    {
    auto partial_derivatives = get_partial_derivative_terms();

    Eigen::RowVector3d error;
    for(int i=0; i<batch_paired_matrix.cols(); ++i)
    {
        error = (batch_paired_matrix.col(i) - cloud_out_paired_matrix.col(i))
                .transpose();

        gradient_cost[0] += (error * (std::get<0>(partial_derivatives)));
        gradient_cost[1] += (error * (std::get<1>(partial_derivatives)));
        gradient_cost[2] += (error * (std::get<2>(partial_derivatives)));

        gradient_cost[3] += (error * ((std::get<3>(partial_derivatives))
                    * raw_cloud_matrix.col(i)));
        gradient_cost[4] +=  (error * ((std::get<4>(partial_derivatives))
                    * raw_cloud_matrix.col(i)));
        gradient_cost[5] += (error * ((std::get<5>(partial_derivatives))
                    * raw_cloud_matrix.col(i)));
    }
    }
    
    else
    {
        auto partial_derivatives = get_partial_derivative_terms(1);

    Eigen::RowVector3d error;
    for(int i=0; i<batch_paired_matrix.cols(); ++i)
    {
        error = (batch_paired_matrix.col(i) - cloud_out_paired_matrix.col(i))
                .transpose();

        gradient_cost[0] += (error * (std::get<0>(partial_derivatives)));
        gradient_cost[1] += (error * (std::get<1>(partial_derivatives)));
        gradient_cost[2] += (error * (std::get<2>(partial_derivatives)));

        gradient_cost[3] += (error * ((std::get<3>(partial_derivatives))
                    * raw_cloud_matrix.col(i)));
        gradient_cost[4] +=  (error * ((std::get<4>(partial_derivatives))
                    * raw_cloud_matrix.col(i)));
        gradient_cost[5] += (error * ((std::get<5>(partial_derivatives))
                    * raw_cloud_matrix.col(i)));
    }
    }

    for(int i=0; i<gradient_cost.size(); ++i)
    {
        gradient_cost[i] /= batch_paired_matrix.cols();
    }

    return gradient_cost;
}

Eigen::Matrix4d SGDICP::current_transform( int method) const
{
    if (method==0)
    {
    auto params = m_sgd_optimizer->get_parameters();
    Eigen::Affine3d transform(Eigen::Affine3d::Identity());
    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(params[5], Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(params[4], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(params[3], Eigen::Vector3d::UnitX());

    transform.translate(Eigen::Vector3d(params[0], params[1], params[2]));
    transform.rotate(rot);

    return transform.matrix();
    }
    else
       {
           
    auto params = m_sgld_optimizer->get_parameters();
    Eigen::Affine3d transform(Eigen::Affine3d::Identity());
    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(params[5], Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(params[4], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(params[3], Eigen::Vector3d::UnitX());

    transform.translate(Eigen::Vector3d(params[0], params[1], params[2]));
    transform.rotate(rot);

    return transform.matrix();
    } 
    
}


std::tuple<
        Eigen::Vector3d,
        Eigen::Vector3d,
        Eigen::Vector3d,
        Eigen::Matrix<double, 3, 3>,
        Eigen::Matrix<double, 3, 3>,
        Eigen::Matrix<double, 3, 3>
>
SGDICP::get_partial_derivative_terms(int method) const //int method=0) const
{

    
    if (method==0)//default sgd optimizer
    {
    auto params = m_sgd_optimizer->get_parameters();
    
    double A = std::cos(params[5]);
    double B = std::sin(params[5]);
    double C = std::cos(params[4]);
    double D = std::sin(params[4]);
    double E = std::cos(params[3]);
    double F = std::sin(params[3]);
    double DE = D * E;
    double DF = D * F;
    double AC = A * C;
    double AF = A * F;
    double AE = A * E;
    double ADE = A * DE;
    double ADF = A * DF;
    double BC = B * C;
    double BE = B * E;
    double BF = B * F;
    double BDE = B * DE;

    Eigen::Matrix3d partial_roll;
    Eigen::Matrix3d partial_pitch;
    Eigen::Matrix3d partial_yaw;

    partial_roll(0, 0) = 0.0;
    partial_roll(0, 1) = ADE - (-BF);
    partial_roll(0, 2) = BE - ADF;
    partial_roll(1, 0) = 0.0;
    partial_roll(1, 1) = -AF + BDE;
    partial_roll(1, 2) = B * -DF - AE;
    partial_roll(2, 0) = 0.0;
    partial_roll(2, 1) = C * E;
    partial_roll(2, 2) = C * -F;

    partial_pitch(0, 0) = A * -D;
    partial_pitch(0, 1) = AC * F;
    partial_pitch(0, 2) = AC * E;
    partial_pitch(1, 0) = B * -D;
    partial_pitch(1, 1) = BC * F;
    partial_pitch(1, 2) = BC * E;
    partial_pitch(2, 0) = -C;
    partial_pitch(2, 1) = -DF;
    partial_pitch(2, 2) = -DE;

    partial_yaw(0, 0) = -BC;
    partial_yaw(0, 1) = -B * DF - AE;
    partial_yaw(0, 2) = AF - BDE;
    partial_yaw(1, 0) = AC;
    partial_yaw(1, 1) = -BE + ADF;
    partial_yaw(1, 2) = ADE - -BF;
    partial_yaw(2, 0) = 0.0;
    partial_yaw(2, 1) = 0.0;
    partial_yaw(2, 2) = 0.0;

    Eigen::Vector3d  partial_x(1.0, 0.0, 0.0);
    Eigen::Vector3d  partial_y(0.0, 1.0, 0.0);
    Eigen::Vector3d  partial_z(0.0, 0.0, 1.0);

    return std::make_tuple(
            partial_x,
            partial_y,
            partial_z,
            partial_roll,
            partial_pitch,
            partial_yaw
    );
    }
    
    else
    { 
    auto params = m_sgld_optimizer->get_parameters();
    double A = std::cos(params[5]);
    double B = std::sin(params[5]);
    double C = std::cos(params[4]);
    double D = std::sin(params[4]);
    double E = std::cos(params[3]);
    double F = std::sin(params[3]);
    double DE = D * E;
    double DF = D * F;
    double AC = A * C;
    double AF = A * F;
    double AE = A * E;
    double ADE = A * DE;
    double ADF = A * DF;
    double BC = B * C;
    double BE = B * E;
    double BF = B * F;
    double BDE = B * DE;

    Eigen::Matrix3d partial_roll;
    Eigen::Matrix3d partial_pitch;
    Eigen::Matrix3d partial_yaw;

    partial_roll(0, 0) = 0.0;
    partial_roll(0, 1) = ADE - (-BF);
    partial_roll(0, 2) = BE - ADF;
    partial_roll(1, 0) = 0.0;
    partial_roll(1, 1) = -AF + BDE;
    partial_roll(1, 2) = B * -DF - AE;
    partial_roll(2, 0) = 0.0;
    partial_roll(2, 1) = C * E;
    partial_roll(2, 2) = C * -F;

    partial_pitch(0, 0) = A * -D;
    partial_pitch(0, 1) = AC * F;
    partial_pitch(0, 2) = AC * E;
    partial_pitch(1, 0) = B * -D;
    partial_pitch(1, 1) = BC * F;
    partial_pitch(1, 2) = BC * E;
    partial_pitch(2, 0) = -C;
    partial_pitch(2, 1) = -DF;
    partial_pitch(2, 2) = -DE;

    partial_yaw(0, 0) = -BC;
    partial_yaw(0, 1) = -B * DF - AE;
    partial_yaw(0, 2) = AF - BDE;
    partial_yaw(1, 0) = AC;
    partial_yaw(1, 1) = -BE + ADF;
    partial_yaw(1, 2) = ADE - -BF;
    partial_yaw(2, 0) = 0.0;
    partial_yaw(2, 1) = 0.0;
    partial_yaw(2, 2) = 0.0;

    Eigen::Vector3d  partial_x(1.0, 0.0, 0.0);
    Eigen::Vector3d  partial_y(0.0, 1.0, 0.0);
    Eigen::Vector3d  partial_z(0.0, 0.0, 1.0);

    return std::make_tuple(
            partial_x,
            partial_y,
            partial_z,
            partial_roll,
            partial_pitch,
            partial_yaw
    );
  
    }

}

void SGDICP::update_state(Eigen::VectorXd & storage, int method)//, int method=0)
{
    if (method==0)
    {
    auto param= m_sgd_optimizer->get_parameters();
     storage[0] = param[0];
     storage[1] = param[1];
     storage[2] = param[2];
     storage[3] = param[3];
     storage[4] = param[4];
     storage[5] = param[5];
    }
    else
    {
        auto param= m_sgld_optimizer->get_parameters();
     storage[0] = param[0];
     storage[1] = param[1];
     storage[2] = param[2];
     storage[3] = param[3];
     storage[4] = param[4];
     storage[5] = param[5];
    }
     
}

bool SGDICP::is_translation_converged(
        Eigen::VectorXd const&          reference_state,
        double                          threshold,
        int 			                 method
)
{
    
    if (method==0)
    {
    auto param = m_sgd_optimizer->get_parameters();
    auto delta = std::sqrt(
            std::pow(reference_state[0] - param[0], 2) +
            std::pow(reference_state[1] - param[1], 2) +
            std::pow(reference_state[2] - param[2], 2)
    );

    return delta < threshold;
    }
    else
    {
        auto param = m_sgld_optimizer->get_parameters();
        auto delta = std::sqrt(
            std::pow(reference_state[0] - param[0], 2) +
            std::pow(reference_state[1] - param[1], 2) +
            std::pow(reference_state[2] - param[2], 2)
    );

    return delta < threshold;
    }
}

bool SGDICP::is_rotation_converged(
        Eigen::VectorXd const&          reference_state,
        double                          threshold,
        int 			                 method
)

{
    if (method==0)
    {
    auto param= m_sgd_optimizer->get_parameters();
    auto delta =
        std::abs(normalizeAngle(reference_state[3] - param[3])) +
        std::abs(normalizeAngle(reference_state[4] - param[4])) +
        std::abs(normalizeAngle(reference_state[5] - param[5]));

    return delta < threshold;
    }
    else
    {
        auto param= m_sgld_optimizer->get_parameters();
    auto delta =
        std::abs(normalizeAngle(reference_state[3] - param[3])) +
        std::abs(normalizeAngle(reference_state[4] - param[4])) +
        std::abs(normalizeAngle(reference_state[5] - param[5]));

    return delta < threshold;
    }
}
