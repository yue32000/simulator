#ifndef LAYERS_HPP_
#define LAYERS_HPP_

#include <stdio.h>

namespace ML{
    
    
    typedef enum{
        CONV,
        FUL,
        POOL
    } layer_type;
    
//    class input{
//    public:
//        int batch;
//        int channels;
//        int size;
//        input(){
//            batch =0;
//            channels =0;
//            size =0;
//        }
//        void initialize (int _batch, int _channels, int _size){
//            batch =_batch;
//            channels = _channels;
//            size = _size;
//        }
//    };
    //datasize is the size of data between layers, start from input
    class datasize{
    public:
        int batch;
        int channels;
        int sizex;
        int sizey;
        datasize(){
            batch =0;
            channels =0;
            sizex =0;
            sizey=0;
        }
        void initialize (int _batch, int _channels, int _sizex, int _sizey){
            batch = _batch;
            channels = _channels;
            sizex = _sizex;
            sizey = _sizey;
        }
    };
    
    
    class convolution_layer{
    public:
        layer_type type;
        char* name;
        int channels;
        int NumofFilters;
        int size;

        int stride;
        int inputx;
        int inputy;
        datasize output;
        convolution_layer(){
            type = CONV;
            name = NULL;
            channels=0;
            NumofFilters=0;
            size=0;
            stride=0;
            
        }
        convolution_layer(char* _name,
                        int _channels,
                        int _NumofFilters,
                        int _size,
              
                        int _stride
                        ){
            type=CONV;
            name = _name;
            channels = _channels;
            NumofFilters = _NumofFilters;
            size = _size;
            stride = _stride;
            
        }
        
        void initialize(char* _name,
                        //int _channels,
                        int _NumofFilters,
                        int _size,
                        int _stride,
                        datasize input
                        ){
            name = _name;
            channels = input.channels;
            NumofFilters = _NumofFilters;
            size = _size;
            stride =_stride;
            output.batch = input.batch;
            output.channels = _NumofFilters;
            output.sizex= (input.sizex- _size)/stride+1;
            output.sizey= (input.sizey- _size)/stride+1;
            inputx = input.sizex;
            inputy = input.sizey; 
            //inputsize = _inputsize;
            printf("name of %s\n",name);
            
        }

        
    };
    
//    class pooling_layer{
//
//    };
    
    class fullyconnected_layer{
    public:
        layer_type type;
        char* name;
        int channels;
        int NumofFilters;
        int sizex;
        int sizey;
        int relu_flag; // 1 do relu, 0 not
        datasize output;
        fullyconnected_layer(){
            type = FUL;
            name=NULL;
            channels=0;
            NumofFilters=0;
            sizex=0;
            sizey=0;
            relu_flag = 0;
            
        }
        fullyconnected_layer(char* _name,
                        int _channels,
                        int _NumofFilters,
                        int _sizex,
                        int _sizey
                        
                        ){
            type =FUL;
            name = _name;
            channels = _channels;
            NumofFilters = _NumofFilters;
            sizex = _sizex;
            sizey = _sizey;
           

            
        }
        void initialize(char* _name,
                        int _NumofFilters,
                        
                        datasize input,
                        int _relu_flag
                        ){
            name = _name;
            channels = input.channels;
            NumofFilters = _NumofFilters;
            sizex = input.sizex;
            sizey = input.sizey;
            output.batch = input.batch;
            output.channels = _NumofFilters;
            output.sizex=1;
            output.sizey=1;
            relu_flag = _relu_flag;
            printf("name of %s\n",name);
            
        }
        
    
    };
    
    class Pool{
    public:
        typedef enum{
            MAX,
            MIN
        } Pool_type;
        
        layer_type type;
        Pool_type pool_type;
        char* name;
        int size;
        int stride;
        datasize output;
        Pool(){
            type = POOL;
            name =NULL;
            size=0;
            stride =0;
          
        }
        void initialize(char* _name,
                        Pool_type _pool_type,
                        int _size,
                        int _stride,
                        datasize input){
            name =_name;
            pool_type = _pool_type;
            size = _size;
            stride = _stride;
          
            output.batch = input.batch;
            output.channels = input.channels;
            output.sizex = (input.sizex - _size)/_stride +1;
            output.sizey = (input.sizey - _size)/_stride +1;
        }
        
        
        
    };
    

    
    
}


#endif
