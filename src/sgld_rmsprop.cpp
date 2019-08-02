#include <cmath>
#include <stddef.h>
#include <iostream>
#include "sgld_rmsprop.h"
#include<fstream>
#include"utils.h"
SGLD_Preconditioned::SGLD_Preconditioned(
                                         std::vector<double>         initial_values,
                                         double                      step_size,
                                         double                      decay_rate_a,
                                         double                      max_range,
                                         int                         open_mp_id,
                                         double                      cloud_size,
                                         std::vector<double>         prior_mean,
                                         float                       prior_variance
                                         
                                         )   :   AbstractSgdOptimizer(initial_values)
, m_step_size(step_size)
, m_decay_rate_a(decay_rate_a)
, m_second_moment(initial_values.size(), 0.0)
, m_max_range(max_range)
, m_open_mp_id (open_mp_id)

, m_cloud_size (cloud_size)
, m_prior_mean (prior_mean)
, m_prior_variance(prior_variance)
{}

void SGLD_Preconditioned::do_perform_update(std::vector<double> const& gradients)
{
    
    
    // Precompute values
    auto rate_a_inv = 1.0 - m_decay_rate_a;
    
    m_iteration_count +=1;
    
    
    //---------------
    /*//-- reduce step size every 500 epoch to reduce effect of noise in final iterations
    double step_for_noise = m_step_size;
    if (((m_iteration_count < 1500) ==0) )
    //if (((m_iteration_count % 300) ==0) && (m_iteration_count < 210))
    {
        std::cout<<"iteration # "<<m_iteration_count<<std::endl;
        m_step_size = m_step_size/2;
        //step_for_noise = step_for_noise/2;
        // std::cout<<"step "<<m_step_size<<std::endl;
        // std::cout<<"step noi"<<step_for_noise<<std::endl;
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
    
         double var = (12*step);
        double noise=0;
        
        if ((m_iteration_count % 1==0) || (m_iteration_count==1))//change 1 in first condition to n to inject noise every n iteration
        {
            noise = sample_distribution(0,var);
            // noise = sample_distribution(0,var2);
            
        }
        
        /* if (noise>0.1)
         {
         std::cout<<"noise "<<noise<<std::endl;
         }*/
       
        if (i<=2)// for translation (0,1,2) priro distribution is gaussian and for rotation it is von mises
        {
           
            
            m_parameters[i] = m_parameters[i] -  0.5*step*(m_cloud_size*gradients[i] + ((1/m_prior_variance)*(m_parameters[i] - m_prior_mean[i]))) + noise;
        }
        else
        {
            m_parameters[i] = m_parameters[i] -  0.5*step*(m_cloud_size*gradients[i]+ ((1/m_prior_variance)*(std::sin(m_parameters[i]-m_prior_mean[i]))))+ noise ;
           
        }

    }
    if (m_iteration_count % 1 ==0)// change 1 to m to save sample every m iterations instead of getting samples every iteration
    {
       
        std::string sfile = "/tmp/sgld" + std::to_string(m_open_mp_id)+ ".txt";
        std::ofstream ofile (sfile,std::ios::out|std::ios::app);
        
        ofile<< (m_max_range*m_parameters[0])<<" "<<(m_max_range* m_parameters[1])<<" "<< (m_max_range*m_parameters[2])<<" "<< normalizeAngle(m_parameters[3])<<" "<< normalizeAngle(m_parameters[4])<<" "<< normalizeAngle(m_parameters[5])<<std::endl;
        ofile.close();
    }
}
}


