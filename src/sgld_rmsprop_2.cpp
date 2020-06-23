#include <cmath>
#include <stddef.h>
#include <iostream>
#include "sgld_rmsprop_2.h"
#include<fstream>
#include <Eigen/Core>
SGLD_Preconditioned2::SGLD_Preconditioned2(
                                         std::vector<double>         initial_values,
                                         double                      step_size,
                                         double                      decay_rate_a,
                                         double                      max_range,
                                         int                         open_mp_id,
                                         float                       adjust_noise
                                         
                                         )   :   AbstractSgldOptimizer(initial_values)
, m_step_size(step_size)
, m_decay_rate_a(decay_rate_a)
, m_second_moment(initial_values.size(), 0.0)
, m_max_range(max_range)
, m_open_mp_id (open_mp_id)
, m_adjust_noise(adjust_noise)
{}

void SGLD_Preconditioned2::do_perform_update(std::vector<double> const& gradients)
{
    
    
    // Precompute values
    auto rate_a_inv = 1.0 - m_decay_rate_a;
    
    m_iteration_count +=1;
    
    
    //---------------
    /*

    if (m_iteration_count >1900) ==0) 
    {
         if (m_iteration_count%10==0)
        {
        m_step_size = m_step_size*0.8;
            
            m_adjust_noise = m_adjust_noise/6;
        
       }
    }
    
     
     */
    //--------------------------
    for(size_t i=0; i<gradients.size(); ++i)
    {
        
        m_second_moment[i] = m_decay_rate_a * m_second_moment[i] +
        rate_a_inv * gradients[i] * gradients[i];
        
        
         double precond = 1/(1e-8+ std::sqrt(m_second_moment[i]));
        //double precond = 1/(std::sqrt(m_second_moment[i] + 1e-8) );
       
        double step = (m_step_size*precond);
    
         double var = (m_adjust_noise*step);
        double noise=0;
        
        if ((m_iteration_count % 1==0) || (m_iteration_count==1))//change 1 in first condition to n to inject noise every n iteration
        {
            noise = sample_distribution(0,var);
          
            
            
        }
           
            
    m_parameters[i] = m_parameters[i] -  0.5*step*gradients[i] + noise;
        

    }
    if (m_iteration_count % 1 ==0)// change 1 to m to save sample every m iterations instead of getting samples every iteration
    {
       
        std::string sfile = "/tmp/sgld" + std::to_string(m_open_mp_id)+ ".txt";
        std::ofstream ofile (sfile,std::ios::out|std::ios::app);
        
        ofile<< (m_max_range*m_parameters[0])<<" "<<(m_max_range* m_parameters[1])<<" "<< (m_max_range*m_parameters[2])<<" "<< normalizeAngle(m_parameters[3])<<" "<< normalizeAngle(m_parameters[4])<<" "<< normalizeAngle(m_parameters[5])<<std::endl;
        ofile.close();
    }
}


double SGLD_Preconditioned2:: normalizeAngle(double radians)
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


