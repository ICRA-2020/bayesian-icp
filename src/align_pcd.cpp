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

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <pcl/common/transforms.h>
#include <pcl/console/time.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>

#include "adadelta.h"
#include "adam.h"
#include "fixed_sgd.h"
#include "momentum.h"
#include "rmsprop.h"
#include "sgdicp.h"
#include "types.h"
#include "utils.h"

#include "sgld_rmsprop.h"
#include <omp.h>


namespace po = boost::program_options;
namespace pt = boost::property_tree;


/**
 * \brief Saves a point cloud with color information specified by the user.
 *
 * \param cloud the cloud xyz information to store
 * \param fname name of the file in which to store the cloud
 * \param color the color to use with values between 0 and 255
 */
void write_colorized_cloud(Cloud_t::Ptr cloud, std::string fname, Eigen::Vector3i const& color)
{
    ColorCloud_t::Ptr color_cloud(new ColorCloud_t);
    for(auto const& pt : *cloud)
    {
        ColorPoint_t cpt;
        cpt.x = pt.x;
        cpt.y = pt.y;
        cpt.z = pt.z;
        cpt.r = color[0];
        cpt.g = color[1];
        cpt.b = color[2];
        color_cloud->push_back(cpt);
    }

    pcl::io::savePCDFile(fname, *color_cloud);
}


int main(int argc, char * argv[])
{
    // +------------------------------------------------------------------------
    // | Parse command line arguments
    // +------------------------------------------------------------------------

    po::options_description desc("Options");
    desc.add_options()
        ("help", "Show the help message")
        ("source", po::value<std::string>(), "Source point cloud")
        ("target", po::value<std::string>(), "Target point cloud")
        ("config", po::value<std::string>(), "Configuration")
    ;

    // Configure positional arguments
    po::positional_options_description pos_args;
    pos_args.add("source", 1);
    pos_args.add("target", 1);
    pos_args.add("config", 1);

    po::variables_map vm;
    po::store(
            po::command_line_parser(argc, argv)
                .options(desc)
                .positional(pos_args)
                .run(),
            vm
    );
    po::notify(vm);

    if(vm.count("help") || !vm.count("source") || !vm.count("target")
            || !vm.count("config")
    )
    {
        std::cout << desc << std::endl;
        return 1;
    }


    // +------------------------------------------------------------------------
    // | Processing of inputs via SGD ICP
    // +------------------------------------------------------------------------
    //pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    Cloud_t::Ptr cloud_in(new Cloud_t);
    Cloud_t::Ptr cloud_in2(new Cloud_t);
    Cloud_t::Ptr cloud_out(new Cloud_t);

    Cloud_t::Ptr result(new Cloud_t);

    // Load point clouds and clean them
    if (pcl::io::loadPCDFile<Point_t>(vm["source"].as<std::string>(), *cloud_in)==-1)
    {
        std::cout << "Could not read source file" << std::endl;
        return -1;
    }
    if (pcl::io::loadPCDFile<Point_t>(vm["target"].as<std::string>(), *cloud_out) ==-1)
    {
        std::cout << "Could not read target file" << std::endl;
        return -1;
    }

    pcl::io::loadPCDFile<Point_t>(vm["source"].as<std::string>(), *cloud_in2);

    auto indices = std::vector<int>();
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);
    pcl::removeNaNFromPointCloud(*cloud_in2, *cloud_in, indices);
    pcl::removeNaNFromPointCloud(*cloud_out,*cloud_out, indices);


    // +------------------------------------------------------------------------
    // | Read configuration and setup accordingly
    // +------------------------------------------------------------------------
    auto config = pt::ptree{};
    pt::read_json(vm["config"].as<std::string>(), config);


    auto max_range = get_absolute_max(cloud_in, cloud_out);
    if(config.get<bool>("normalize-cloud"))
    {
        cloud_in = normalise_clouds(cloud_in, max_range);
        cloud_out = normalise_clouds(cloud_out, max_range);
    }

    std::vector<double> initial_guess = {
        config.get<double >("initial-guess.x"),
        config.get<double >("initial-guess.y"),
        config.get<double >("initial-guess.z"),
        config.get<double >("initial-guess.roll"),
        config.get<double >("initial-guess.pitch"),
        config.get<double >("initial-guess.yaw")
    };

    // +------------------------------------------------------------------------
    // | Parallel Alignment
    // +------------------------------------------------------------------------
 
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
 bool abort =0;
   
#pragma omp parallel private(transformation_matrix, abort) num_threads(4) //shared (cloud_in)
  
    {
  //  while(!abort)
//{
//make sure each core has its own copy of cloud in beacuse it needs to write/change it internally
Cloud_t::Ptr cloud_in_copy (new Cloud_t);
*cloud_in_copy =*cloud_in;
        
Cloud_t::Ptr cloud_out_copy (new Cloud_t);
*cloud_out_copy =*cloud_out;

//total =omp_get_num_threads();
int id = omp_get_thread_num();//pass it to sgld to save independ file for each core to manage burnin.

        
    std::unique_ptr<SGDICP> sgd_icp;
    
    if(config.get<std::string>("method") == "fixed")
    {
        sgd_icp.reset(new SGDICP(
            std::unique_ptr<FixedSgd>(
                new FixedSgd(
                    initial_guess,
                    config.get<double>("fixed.step-size")
                )
            )
        ));
    }
    else if(config.get<std::string>("method") == "adadelta")
    {
        sgd_icp.reset(new SGDICP(
            std::unique_ptr<AdaDelta>(
                new AdaDelta(
                    initial_guess,
                    config.get<double>("adadelta.decay-rate"),
                    config.get<double>("adadelta.preconditioner")
                )
            )
        ));
    }
    else if(config.get<std::string>("method") == "adam")
    {
        sgd_icp.reset(new SGDICP(
            std::unique_ptr<Adam>(
                new Adam(
                    initial_guess,
                    config.get<double>("adam.step-size"),
                    config.get<double>("adam.decay-rate-a"),
                    config.get<double>("adam.decay-rate-b")
                )
            )
        ));
    }
    else if (config.get<std::string>("method") == "momentum")
    {
        sgd_icp.reset(new SGDICP(
            std::unique_ptr<MomentumSgd>(
                new MomentumSgd(
                    initial_guess,
                    config.get<double>("momentum.step-size")
                )
            )
        ));
    }
    else if(config.get<std::string>("method") == "rmsprop")
    {
        sgd_icp.reset(new SGDICP(
            std::unique_ptr<Rmsprop>(
                new Rmsprop(
                    initial_guess,
                    config.get<double>("rmsprop.step-size"),
                    config.get<double>("rmsprop.decay-rate")
                )
            )
        ));
    }
    
    else if (config.get<std::string>("method")=="preconditioned_sgld")
    {
        std::vector<double> mean_prior ={config.get<double>("prior_mean.x"),
                                        config.get<double>("prior_mean.y"),
                                        config.get<double>("prior_mean.z"),
                                        config.get<double>("prior_mean.roll"),
                                        config.get<double>("prior_mean.pitch"),
                                        config.get<double>("prior_mean.yaw")
            
                                        };
     //double sgld_step = 0.06147/double(cloud_in->size());
        sgd_icp.reset(new SGDICP
                (std::unique_ptr<SGLD_Preconditioned>
                 (new SGLD_Preconditioned
                  (initial_guess,
                   config.get<double>("rmsprop.step-size"),
                   config.get<double>("rmsprop.decay-rate"),
                   max_range,
                   id,
                   cloud_in->size(),
                   mean_prior,
                   config.get<float>("prior_variance")
                    )
                   )
                 )
                      );
    }
    
    else
    {
        std::cout << "Invalid optimizer specified, valid optiosn are: "
                  << "adadelta, adam, fixed, momentum, and rmsprop" << std::endl;
       // return 1;
        abort =1;
    }

//}
    // +------------------------------------------------------------------------
    // | Perform ICP alignment
    // +------------------------------------------------------------------------
   
    
    
    auto time = pcl::console::TicToc();
    time.tic();
     transformation_matrix = sgd_icp->align_clouds(
            cloud_in_copy,
            cloud_out_copy,
            SGDICP::Parameters(
                //1,
                config.get<int>("icp.max-iterations"),
                config.get<int>("icp.batch-size"),
                config.get<double>("icp.max-matching-distance"),
                config.get<int>("icp.convergence-steps"),
                config.get<double>("icp.translational-convergence"),
                config.get<double>("icp.rotational-convergence"),
                config.get<bool>("icp.filter")
            )
    );
    std::cout << "ICP Duration: " << time.toc() << " ms" << std::endl;
    
}
    //std::cout << "ICP Duration: " << time.toc() << " ms" << std::endl;
    auto rmse = compute_rmse(cloud_in, cloud_out, transformation_matrix);

    // If the clouds were normalized undo this to obtain the true transformation
    if(config.get<bool>("normalize-cloud"))
    {
        rescale_transformation_matrix(transformation_matrix, max_range);
    }

    std::cout << "RMSE: "<< rmse << "\n";
    std::cout << "Transformation_matrix:\n"
              << transformation_matrix << std::endl;

    auto parameters = get_translation_roll_pitch_yaw(transformation_matrix);

    // Save resulting point cloud
    pcl::transformPointCloud<Point_t>(
            *cloud_in2,
            *result,
            transformation_matrix
    );

    write_colorized_cloud(cloud_in, "/tmp/cloud_source.pcd", {196, 98, 33});
    write_colorized_cloud(cloud_out, "/tmp/cloud_target.pcd", {89, 96, 99});
    write_colorized_cloud(result, "/tmp/cloud_aligned.pcd", {29, 156, 229});

    return 0;
}
