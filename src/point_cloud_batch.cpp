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



#include <iostream>

#include "point_cloud_batch.h"


PointCloudBatch::PointCloudBatch(Cloud_t point_cloud, int batch_size)
    :   m_cloud(point_cloud)
      , m_current_offset(0)
      , m_batch_size(batch_size)
{
    // Initialize the random number generator
    std::random_device rand_dev;
    m_generator = std::mt19937(rand_dev());

    shuffle_data();
}

Cloud_t::Ptr PointCloudBatch::next_batch()
{
    Cloud_t::Ptr batch(new Cloud_t);

    auto target_offset = m_current_offset + m_batch_size;

    // If we require more data then left or then present copy the points
    // and then reshuffle (as often as needed) the sequence
    while(target_offset >= m_cloud.points.size())
    {
        while(m_current_offset < m_cloud.points.size())
        {
            batch->points.push_back(m_cloud.points[m_current_offset++]);
        }

        shuffle_data();
        m_current_offset = 0;
        target_offset = target_offset - m_cloud.points.size();
    }

    // Copy the reamining points needed
    while(m_current_offset < target_offset)
    {
        batch->points.push_back(m_cloud.points[m_current_offset++]);
    }

    return batch;
}

void PointCloudBatch::shuffle_data()
{
    std::shuffle(m_cloud.points.begin(), m_cloud.points.end(), m_generator);
}
