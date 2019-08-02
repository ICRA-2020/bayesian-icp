#include "types.h"

#include <boost/format.hpp>

#include <pcl/common/transforms.h>
#include <pcl/console/time.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>


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



int main(int argc, char const* argv[])
{
    Cloud_t::Ptr cloud_in(new Cloud_t);
    Cloud_t::Ptr cloud_out(new Cloud_t);
    Cloud_t::Ptr cloud_final(new Cloud_t);


    pcl::io::loadPCDFile(argv[1], *cloud_in);
    pcl::io::loadPCDFile(argv[2], *cloud_out);

    auto time = pcl::console::TicToc();
    time.tic();
    auto icp = pcl::IterativeClosestPoint<Point_t, Point_t>();
    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_out);
    icp.setRANSACIterations(10);
    icp.setMaxCorrespondenceDistance(1.0);
    icp.setMaximumIterations(80);

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    icp.align(*cloud_final, transform);
    //transform = icp.getFinalTransformation();
    std::cout << "Time: " << time.toc() << " ms" << std::endl;


    write_colorized_cloud(cloud_in, "/tmp/cloud_source.pcd", {196, 98, 33});
    write_colorized_cloud(cloud_out, "/tmp/cloud_target.pcd", {89, 96, 99});
    write_colorized_cloud(cloud_final, "/tmp/cloud_pcl.pcd", {147, 38, 193});

    return 0;
}
