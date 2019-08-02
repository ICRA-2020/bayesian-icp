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
#include <stddef.h>

#include "adam.h"


Adam::Adam(
        std::vector<double>         initial_values,
        double                      step_size,
        double                      decay_rate_a,
        double                      decay_rate_b
)   :   AbstractSgdOptimizer(initial_values)
      , m_step_size(step_size)
      , m_decay_rate_a(decay_rate_a)
      , m_decay_rate_b(decay_rate_b)
      , m_timestep(0)
      , m_decay_rate_a_t(1.0)
      , m_decay_rate_b_t(1.0)
      , m_first_moment(initial_values.size(), 0.0)
      , m_second_moment(initial_values.size(), 0.0)
{}

void Adam::do_perform_update(std::vector<double> const& gradients)
{
    // Update time step related variables
    m_timestep++;
    m_decay_rate_a_t *= m_decay_rate_a;
    m_decay_rate_b_t *= m_decay_rate_b;

    // Precompute values
    auto rate_a_inv = 1.0 - m_decay_rate_a;
    auto rate_b_inv = 1.0 - m_decay_rate_b;

    for(size_t i=0; i<gradients.size(); ++i)
    {
        m_first_moment[i] = m_decay_rate_a * m_first_moment[i] +
            rate_a_inv * gradients[i];
        m_second_moment[i] = m_decay_rate_b * m_second_moment[i] +
            rate_b_inv * gradients[i] * gradients[i];

        auto fm_unbiased = m_first_moment[i] / (1.0 - m_decay_rate_a_t);
        auto sm_unbiased = m_second_moment[i] / (1.0 - m_decay_rate_b_t);

        m_parameters[i] -= m_step_size * fm_unbiased / (std::sqrt(sm_unbiased) + 1e-8);
    }
}
