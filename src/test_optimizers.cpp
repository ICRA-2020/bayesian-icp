
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
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "adam.h"
#include "adadelta.h"
#include "fixed_sgd.h"
#include "rmsprop.h"

double f(double x)
{
    return 5*x*x + 2*x + 3;
}

double fdx(double x)
{
    return 10*x + 2;
}


TEST_CASE("adam", "")
{
    double param = 10.0;
    auto adam = Adam({param}, 0.1, 0.9, 0.999);
    for(int i=0; i<1000; ++i)
    {
        auto new_param = adam.update_parameters({fdx(param)});
        param = new_param[0];
    }
    REQUIRE(-0.2 == Approx(param).margin(0.00001));
}

TEST_CASE("adadelta", "")
{
    double param = 10.0;
    auto adadelta = AdaDelta({param}, 0.9, 1e-3);
    for(int i=0; i<1000; ++i)
    {
        auto new_param = adadelta.update_parameters({fdx(param)});
        param = new_param[0];
    }
    REQUIRE(-0.2 == Approx(param).margin(0.00001));
}

TEST_CASE("fixed", "")
{
    double param = 10.0;
    auto fixed = FixedSgd({param}, 0.1);
    for(int i=0; i<1000; ++i)
    {
        auto new_param = fixed.update_parameters({fdx(param)});
        param = new_param[0];
    }
    REQUIRE(-0.2 == Approx(param).margin(0.00001));
}

TEST_CASE("rmsprop", "")
{
    double param = 10.0;
    auto rmsprop = Rmsprop({param}, 0.1, 0.9);
    for(int i=0; i<1000; ++i)
    {
        auto new_param = rmsprop.update_parameters({fdx(param)});
        param = new_param[0];
    }
    REQUIRE(-0.2 == Approx(param).margin(0.00001));
}
