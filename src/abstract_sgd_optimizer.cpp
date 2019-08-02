#include <cassert>
#include <iostream>


#include "abstract_sgd_optimizer.h"

/*
AbstractSgdOptimizer::AbstractSgdOptimizer(std::unique_ptr<SGDICP> sgdicp)
:   m_sgd(std::move(sgdicp))
{}*/
//------------------
AbstractSgdOptimizer::AbstractSgdOptimizer(std::vector<double> initial_values)
    :   m_parameters(initial_values)
{}


AbstractSgdOptimizer::~AbstractSgdOptimizer()
{}


std::vector<double> AbstractSgdOptimizer::update_parameters(
        std::vector<double> const&      gradients
)
{
    assert(gradients.size() == m_parameters.size());

    do_perform_update(gradients);
    return m_parameters;
}
double AbstractSgdOptimizer::sample_distribution (float mean, float sigma)
{
    std::random_device rand_dev;
    m_generator = std::mt19937(rand_dev());
    // m_default_generator = std::default_random_engine;
    std::normal_distribution<double> distribution (mean,sigma);
    //return distribution(m_default_generator);
    return distribution(m_generator);
}

std::vector<double> AbstractSgdOptimizer::get_parameters() const
{
    return m_parameters;
}
