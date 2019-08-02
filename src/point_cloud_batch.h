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



#ifndef __POINT_CLOUD_BATCH_H__
#define __POINT_CLOUD_BATCH_H__


#include <random>

#include "types.h"


/**
 * \brief Provides random batches of points from a point cloud.
 */
class PointCloudBatch
{
    public:
        /**
         * \brief Creates a new point cloud batcher instance.
         *
         * \param point_cloud the cloud to sample batches from
         * \param batch_size number of points per batch
         */
        PointCloudBatch(Cloud_t point_cloud, int batch_size);

        /**
         * \brief Returns a new batch of points from the cloud.
         *
         * \return new batch of points
         */
        Cloud_t::Ptr next_batch();


    private:
        /**
         * \brief Randomizes the data to provide random batches.
         */
        void shuffle_data();


    private:
        //! Number of points per batch
        int                             m_batch_size;
        //! Current offset from the beginning of the cloud
        int                             m_current_offset;
        //! Point cloud which is the source of the batch data
        Cloud_t                         m_cloud;
        //! Random number generator
        std::mt19937                    m_generator;
};


#endif /* __POINT_CLOUD_BATCH_H__ */
