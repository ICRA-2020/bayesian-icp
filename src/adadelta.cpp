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

#include "adadelta.h"


AdaDelta::AdaDelta(
        std::vector<double>             initial_values,
        double                          decay_rate,
        double                          preconditioner
)
    :   AbstractSgdOptimizer(initial_values)
      , m_decay_rate(decay_rate)
      , m_preconditioner(preconditioner)
      , m_acc_gradient(initial_values.size(), 0.0)
      , m_acc_updates(initial_values.size(), 0.0)
{}

void AdaDelta::do_perform_update(std::vector<double> const& gradients)
{
    auto decay_inv = 1.0 - m_decay_rate;

    for(size_t i=0; i<gradients.size(); ++i)
    {
        // Accumulate gradient via exponential decay
        m_acc_gradient[i] = m_decay_rate * m_acc_gradient[i] +
            decay_inv * gradients[i] * gradients[i];

        // Compute RMS values
        double rms_gradient = std::sqrt(m_acc_gradient[i] + m_preconditioner);
        double rms_update = std::sqrt(m_acc_updates[i] + m_preconditioner);

        // Compute update value
        double update = -(rms_update / rms_gradient) * gradients[i];

        // Accumulate change via exponential decay
        m_acc_updates[i] = m_decay_rate * m_acc_updates[i] +
            decay_inv * update * update;

        // Update parameter value
        m_parameters[i] += update;
    }
}
