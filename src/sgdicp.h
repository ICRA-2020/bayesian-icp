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




#include "abstract_sgd_optimizer.h"
#include "types.h"

/**
 * \brief Stochastic gradient descent based ICP implementation.
 */
class SGDICP
{
    public:
        /**
         * \brief Holds the parameters of the ICP algorithm.
         */
        struct Parameters
        {
            //! Maximum number of iterations to run
            int iterations = 50;
            //! Number of points to use per batch
            int batch_size = 50;
            //! Maximum point matching distance
            double max_matching_distance = 1.0;
            //! Number of consecutive converged steps to terminate
            int convergence_steps = 5;
            //! Translation change threshold for convergence
            double translational_convergence_threshold = 0.05;
            //! Rotation change threshold for convergence
            double rotational_convergence_threshold = 0.05;
            //! filter duplicate correspondences
            bool filter = true;

            /**
             * \brief Creates a new default instance.
             */
            Parameters() = default;

            /**
             * \brief Creates a new instance with the specified parameters
             */
            Parameters(
                    int                 iterations_,
                    int                 batch_size_,
                    double              max_matching_distance_,
                    int                 convergence_steps_,
                    double              translational_convergence_threshold_,
                    double              rotational_convergence_threshold_,
                    bool                filter_
            )   :   iterations(iterations_)
                  , batch_size(batch_size_)
                  , max_matching_distance(max_matching_distance_)
                  , convergence_steps(convergence_steps_)
                  , translational_convergence_threshold(translational_convergence_threshold_)
                  , rotational_convergence_threshold(rotational_convergence_threshold_)
                  , filter (filter_)
            {}
        };


    public:
        /**
         * \brief Creates a new instance with the specified SGD optimizer.
         *
         * \param sgd_optimizer optimizer instance to use
         */
        SGDICP(std::unique_ptr<AbstractSgdOptimizer> sgd_optimizer);

        /**
         * \brief Computes the transform between the two clouds using
         *      the provided parameters.
         *
         * \param cloud_in source cloud to be transformed onto the target cloud
         * \param cloud_out target cloud onto which the source cloud should be
         *      transformed
         * \return transformation matrix which moves cloud_in onto cloud_out
         */
        Eigen::Matrix4d align_clouds(
                Cloud_t::Ptr            cloud_in,
                Cloud_t::Ptr            cloud_out,
                Parameters const&       parameters
        );


    private:
        /**
         * \brief Computes the gradient terms needed for the SGD step.
         *
         * \param raw_cloud_resized
         * \param transformed_batch_paired
         * \param cloud_out_paired
         * \return gradient information
         */
        std::vector<double> compute_gradient_terms(
                Cloud_t::Ptr            raw_cloud_resized,
                Cloud_t::Ptr            transformed_batch_paired,
                Cloud_t::Ptr            cloud_out_paired
        );

        /**
         * \brief Computes the partial derivates.
         *
         * \return partial derivative terms
         */
        std::tuple<
            Eigen::Vector3d,
            Eigen::Vector3d,
            Eigen::Vector3d,
            Eigen::Matrix<double, 3, 3>,
            Eigen::Matrix<double, 3, 3>,
            Eigen::Matrix<double, 3, 3>
        >
        get_partial_derivative_terms() const;

        /**
         * \brief Returns the current transformation matrix.
         *
         * \return current transformation matrix
         */
        Eigen::Matrix4d current_transform() const;

        /**
         * \brief Updates the state in the provided storage.
         *
         * \param storage container into which state information is saved
         */
        void update_state(Eigen::VectorXd & storage);

        /**
         * \brief Returns whether or not translational convergence was achieved.
         *
         * \param reference_state: state to compare against
         * \param threshold convergence threshold to use
         * \return true if convergence is achieved, false otherwise
         */
        bool is_translation_converged(
                Eigen::VectorXd const&  reference_state,
                double                  threshold
        );

        /**
         * \brief Returns whether or not rotational convergence was achieved.
         *
         * \param reference_state state to compare against
         * \param threshold convergence threshold to use
         * \return true if convergence is achieved, false otherwise
         */
        bool is_rotation_converged(
                Eigen::VectorXd const&  reference_state,
                double                  threshold
        );


    private:
        //! Optimizer for the transform parameters
        std::unique_ptr<AbstractSgdOptimizer> m_sgd_optimizer;
};
