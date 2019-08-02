#ifndef __ABSTRACT_SGD_OPTIMIZER_H__
#define __ABSTRACT_SGD_OPTIMIZER_H__


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



#include <vector>


/**
 * \brief Base class for SGD optimization methods.
 */
class AbstractSgdOptimizer
{
    public:
        /**
         * \brief Creates a new instance with initial parameter values.
         *
         * \param initial_values initial values of the parameters
         *      to be optimized
         */
        AbstractSgdOptimizer(std::vector<double> initial_values);

        /**
         * \brief Virtual destructor.
         */
        virtual ~AbstractSgdOptimizer();

        /**
         * \brief Updates the parameters using the provided gradient.
         *
         * \param gradients gradient information for each of the parameters
         * \return new parmeter values after update
         */
        std::vector<double> update_parameters(
                std::vector<double> const& gradients
        );

        /**
         * \brief Returns the current parameter values.
         *
         * \return current values of the parameters
         */
        std::vector<double> get_parameters() const;


    protected:
        /**
         * \brief Virtual function performing the parameter update.
         *
         * \param gradients gradient information for each of the parameters
         */
        virtual
        void do_perform_update(std::vector<double> const& gradients) = 0;


    protected:
        //! Parameters to optimise
        std::vector<double>             m_parameters;
};



#endif /* __ABSTRACT_SGD_OPTIMIZER_H__ */
