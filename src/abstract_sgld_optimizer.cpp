#include <cassert>
#include <iostream>


#include "abstract_sgld_optimizer.h"

/*
AbstractSgdOptimizer::AbstractSgdOptimizer(std::unique_ptr<SGDICP> sgdicp)
:   m_sgd(std::move(sgdicp))
{}*/
//------------------
AbstractSgldOptimizer::AbstractSgldOptimizer(std::vector<double> initial_values)
    :   m_parameters{{initial_values[0],initial_values[1],initial_values[2],initial_values[3],initial_values[4],initial_values[5]}}
{}


AbstractSgldOptimizer::~AbstractSgldOptimizer()
{}


Array6_d AbstractSgldOptimizer::update_parameters(
        std::vector<double> const&      gradients
)
{
    assert(gradients.size() == m_parameters.size());

    do_perform_update(gradients);
    return m_parameters;
}
double AbstractSgldOptimizer::sample_distribution (float mean, float sigma)
{
    std::random_device rand_dev;
    m_generator = std::mt19937(rand_dev());
    // m_default_generator = std::default_random_engine;
    std::normal_distribution<double> distribution (mean,sigma);
    //return distribution(m_default_generator);
    return distribution(m_generator);
}

Array6_d AbstractSgldOptimizer::get_parameters() const
{
    return m_parameters;
}
