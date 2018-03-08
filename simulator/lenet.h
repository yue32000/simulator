#ifndef LENET_HPP_
#define LENET_HPP_

#include "layers.h"


class lenet{
public:
    int layers;
    ML::datasize inputs;
//    ML::convolution_layer conv1 ('conv1',
//                          1,//filter channels
//                          20,//NumofFilters
//                          5,//size
//                          1,//stride
//                          );
//
    ML::convolution_layer conv1;
    ML::Pool pool1;
    ML::convolution_layer conv2;
    ML::Pool pool2;
    ML::fullyconnected_layer ip1;
    ML::fullyconnected_layer ip2;
//    ML::fullyconnected_layer ip1("ip1",
//                                         1,//filter channels
//                                         32,//NumofFilters
//                                         1024,//size
//                                         1024//input size
//    );
//    lenet(){
//        layers=0;
//    }
};


#endif

