//
//  main.cpp
//  simulator
//
//  Created by Yue Huang on 2018-03-06.
//  Copyright © 2018 Yue Huang. All rights reserved.
//

#include <stdlib.h>
#include <stdio.h>
//#include<string.h>
#include "layers.h"
#include "lenet.h"
#define AM_FETCH 16

static int sim_cycle =0 ;
static int multiply=0;

class memory{
    
public:
    typedef enum{
        LOAD,
        STORE
    }ACCESS_TYPE;
    
    char* name;
    int size;
    int linesize;
    int access_num_load;
    int access_num_store;
    int* mem;
    
    memory(){
        linesize=16;
        access_num_load=0;
        access_num_store=0;
        mem = NULL;
    }
    memory(char* Name, int Size){
        linesize=16;
        access_num_load=0;
        access_num_store=0;
        name = Name;
        size =Size;
        mem = (int*)malloc(size*sizeof(int));
    }
    void allocate(char* Name, int Size){
        name = Name;
        size =Size;
        mem = (int*)malloc(size*sizeof(int));
    }
    void access(unsigned int addr,
                int num,//how many blocks to access
                int* buf,
                ACCESS_TYPE access_type//access type
    
    ){
        if(access_type == LOAD){
            for(unsigned int i=addr; i<num; i++ )
                buf[i-addr]=mem[i];
            access_num_load++;
        }
        else{//STORE
            for (unsigned int i=addr; i<num ; i++){
                mem[i] = buf[i-addr];
                access_num_store++;
            }
        }
    }
};

class tile{
public:
    int tile_num;
    memory WM;//('WM of tile'+tile_num,2MB)
    memory NBin;
    //NBin;
    int NBout[16];
    //constructor
    tile(){
        tile_num=16;
        for (int i=0;i<16;i++){
            NBout[i]=0;
        }
        WM.allocate("WM",2*1024*1024);
    }
    void mul_add(int *a, int *b,int k,int access_num){
        for (int i=0;i<k;i++){
            for (int j=0;j<access_num;j++){
                NBout[i]=a[j]*b[i*16+j]+NBout[i];
                multiply++;
            }
        }
    }
    
};

class schedule_command{
public:
    ML::layer_type type;
    int AM_fetch;
    int AM_it;//this is used for one output, how many times activation has to be fetched in groups of 16
    int AM_it2;//this is used for the same set of 16 activation, how many times it needs to be fetched repetitively because of more than 256 filters
    int tile_num;
    int filtersize;//size * size * channels
    int NumofFilters;
    int next_fetch_addr;//next fetch address of AM
    int next_fetch_addr_outterloop;
    //for fully
    
    int progress;
    
    //for conv
    int channels;
    int stride;
    int inputx;
    int inputy;
    int outputx;
    int outputy;
    int size;
    int outputx_progress;
    int outputy_progress;
    int signal; //this is used for to refetch a new window of activation during convolution layer
    
    schedule_command(){
        
    }
    schedule_command(ML::layer_type _type,
                     int _AM_fetch,
                     int _AM_it,
                     int _AM_it2,
                     int _tile_num,
                     //int _stride,
                     int _filtersize,
                     int _NumofFilters,
                     int _stride,
                     int _inputx,
                     int _inputy,
                     int _outputx,
                     int _outputy,
                     int _size,
                     int _channels
                     ){
        type = _type;
        AM_fetch = _AM_fetch;//fetch 16 activations from AM every cycle;
        AM_it = _AM_it;//how many iterations for one neuron output
        AM_it2 = _AM_it2;
        tile_num = _tile_num;//how many tiles are actually used for last round, each tiles provides 16 filters; round number is it2
        
        //stride = _stride;
        filtersize = _filtersize;
        NumofFilters = _NumofFilters;
        if(_type == ML::FUL){
            next_fetch_addr=0;
            progress=0;
        }
        if(_type == ML::CONV){
            stride = _stride;
            inputx = _inputx;
            inputy = _inputy;
            outputx = _outputx;
            outputy = _outputy;
            size = _size;
            channels = _channels;
            next_fetch_addr =0;
            next_fetch_addr_outterloop=0;
            progress=0;
            outputx_progress=0;
            outputy_progress=0;
        }
    }
};
schedule_command schedule_conv(ML::convolution_layer layer){
    int size = layer.size;
    int channels = layer.channels;
    int NumofFilters = layer.NumofFilters;
    schedule_command SC(ML::CONV,
                        AM_FETCH,
                        ((size * size *channels)+AM_FETCH-1) /AM_FETCH,
                        (NumofFilters+255)/256,
                        NumofFilters%256==0?16:((NumofFilters%256)+15)/16,
                        size*size*channels,
                        NumofFilters,
                        layer.stride,
                        layer.inputx,
                        layer.inputy,
                        layer.output.sizex,
                        layer.output.sizey,
                        layer.size,
                        layer.channels
                        );
    
    return SC;
}

schedule_command schedule_pool(ML::Pool layer){
    schedule_command SC;
    SC.type = ML::POOL;
//    schedule_command SC(ML::POOL,
//                        AM_FETCH,
//
//                        );
    return SC;
}

schedule_command schedule_fully(ML::fullyconnected_layer layer){
    int channels=layer.channels;
    int NumofFilters = layer.NumofFilters;
    int sizex = layer.sizex;
    int sizey = layer.sizey;
   // int size = layer.size;
    //int stride = stride;
    //int input_size = layer.inputsize;
   // printf("size %d\n",size);
    schedule_command();
    schedule_command SC(ML::FUL,
                        AM_FETCH,//fetch 16 activations from AM every cycle
                        ((sizex * sizey *channels)+AM_FETCH-1) /AM_FETCH,//how many iterations for one neuron output);
                        (NumofFilters+255)/256,
                        NumofFilters%256==0?16:((NumofFilters%256)+15)/16,//how many tiles are actually used, each tiles provides 16 filters
                        //stride,
                        sizex*sizey*channels,//filtersize
                        NumofFilters,//num of filters
                        0, //stride
                        0,
                        0,
                        layer.output.sizex,
                        layer.output.sizey,
                        0,
                        layer.channels
                        );
    //printf("in SC_it %d\n", SC.AM_it);
    //    SC.AM_fetch = AM_FETCH;//fetch 16 activations from AM every cycle;
    //    SC.AM_it = ((size * size *channels)+AM_fetch-1) /AM_fetch;//how many iterations for one neuron output
    //    SC.tile_num = NumofFilters+15/16;//how many tiles are actually used, each tiles provides 16 filters
    //    SC.stride = stride;
    //    SC.filtersize = size*size*channels;
    //    SC.NumofFilters = size*size*channels;
    return SC;
    
}

//this should change to import value from extern memory
void initialize(){
    //create AM
    //memory AM("AM", 4*1024*1024);
    
    //create tiles...
    
}


int calculate(schedule_command &SC, memory &AM, tile* Tiles, memory &AM2){
    //calcalating addr first
    if (SC.type == ML::FUL){
        int a_addr = SC.next_fetch_addr*16;//start position to get activation
        int fs = SC.filtersize;
        int p = SC.progress;// which is used as part of indexing the weight
        int a[16]={0};//store 16 activations
        int b[256]={0};//store 256 for each tile
        //int result[16];
        int access_num =16;
        if(a_addr+16>fs){
            //printf("needs something to change");
            access_num = fs-a_addr;
        }
        AM.access (a_addr, access_num, a, memory::LOAD);//broadcast 16 activations from AM
        //the tiles are not fully used 16
        if(SC.AM_it2==1||p+1==SC.AM_it2){
    
            for(int i=0; i<SC.tile_num; i++){
        
                for (int j=0; j<16&&p*256+(16*i+j)<SC.NumofFilters;j++){
            
            
                    Tiles[i].WM.access(p*(256*fs)+16*i*fs+j*fs+a_addr,
                               access_num,
                               b+16*(16*i+j), //base+how many filters/tile*i+j*AM_fetch
                               memory::LOAD);
       
                }
        //if it is the last tile, which the filters less than 16
                if(i==SC.tile_num){
                    Tiles[i].mul_add(a,b,SC.NumofFilters%16==0?16:SC.NumofFilters%16,access_num);
                }
                else
                {
                    Tiles[i].mul_add(a, b,16,access_num);
                }
            }
            
    }
        else{
            for(int i=0; i<16; i++){
                for (int j=0; j<16;j++){
                    Tiles[i].WM.access(p*(256*fs)+16*i*fs+j*fs+a_addr,
                                       access_num,
                                       b+16*(16*i+j), //base+how many filters/tile*i+j*AM_fetch
                                       memory::LOAD);
                    
                }
                
                
                Tiles[i].mul_add(a,b,16,access_num);
                
            }
        }
        //schedule next cycle command
        SC.next_fetch_addr++;
        if(SC.next_fetch_addr>=SC.AM_it&&SC.progress+1<SC.AM_it2){
            SC.progress++;
            SC.next_fetch_addr=0;
        }
        if(SC.next_fetch_addr>=SC.AM_it&&SC.progress+1>=SC.AM_it2){
            return 1; //this layer calculation is finished!!
        }
        else
            return 0;//this layer is not finished yet
        
    }
    
    
    if(SC.type == ML::CONV){
        //int outloop = SC.outputsize;
        //broadcast the AM to NBin
        if(SC.signal==1){
            for (int i=0; i<SC.size;i++){
                for(int j=0; j<SC.size; j++){
                    AM.access((SC.next_fetch_addr_outterloop+SC.inputx*i+j)*SC.channels, SC.channels, AM2.mem+(i*SC.size+j)*SC.channels, memory::LOAD);
          
                }
            }
            SC.signal =0;
        
          
        }
        
        int a_addr = SC.next_fetch_addr*16;//start position to get activation
        int fs = SC.filtersize;
        int p = SC.progress;// which is used as part of indexing the weight
        int a[16]={0};//store 16 activations
        int b[256]={0};//store 256 for each tile
        //int result[16];
        int access_num=16;
        if(a_addr+16>fs){
            //printf("needs something to change");
            access_num = fs-a_addr;
        }
        AM2.access (a_addr, access_num, a, memory::LOAD);//broadcast 16 activations from AM2
        //the tiles are not fully used 16
        if(SC.AM_it2==1||p+1==SC.AM_it2){
            
            for(int i=0; i<SC.tile_num; i++){
                
                for (int j=0; j<16&&p*256+(16*i+j)<SC.NumofFilters;j++){
                    Tiles[i].WM.access(p*(256*fs)+16*i*fs+j*fs+a_addr,
                                       access_num,
                                       b+16*(16*i+j), //base+how many filters/tile*i+j*AM_fetch
                                       memory::LOAD);
                    
                }
                //if it is the last tile, which the filters may be less than 16
                if(i==SC.tile_num){
                    Tiles[i].mul_add(a,b,SC.NumofFilters%16==0?16:SC.NumofFilters%16,access_num);
                }
                else
                {
                    Tiles[i].mul_add(a, b,16,access_num);
                }
            }
            
        }
        else{
            for(int i=0; i<16; i++){
                for (int j=0; j<16;j++){
                    Tiles[i].WM.access(p*(256*fs)+16*i*fs+j*fs+a_addr,
                                       access_num,
                                       b+16*(16*i+j), //base+how many filters/tile*i+j*AM_fetch
                                       memory::LOAD);
                    
                }
                Tiles[i].mul_add(a,b,16,access_num);
            }
        }
        //schedule next cycle command
        SC.next_fetch_addr+=1;
        if(SC.next_fetch_addr>=SC.AM_it&&SC.progress+1<SC.AM_it2){
            //printf("next_fetch_addr %d\n",SC.next_fetch_addr);
            SC.progress++;
            SC.next_fetch_addr=0;
        }
        if(SC.next_fetch_addr>=SC.AM_it&&SC.progress+1>=SC.AM_it2){
            //return 1; //this iteration convolution is finished!!
            //printf("next_fetch_addr %d\n",SC.next_fetch_addr);
            SC.signal=1;
            SC.next_fetch_addr=0;
            SC.progress=0;
            
            SC.next_fetch_addr_outterloop += SC.stride;
            SC.outputx_progress++;
            if (SC.outputx_progress>= SC.outputx){
                SC.outputy_progress ++;
                SC.next_fetch_addr_outterloop =SC.outputy_progress*SC.inputx*SC.stride;
                SC.outputx_progress=0;
    
            }
            if (SC.outputy_progress>=SC.outputy)
                return 1;//all convolution is finished!!
            else return 0;
        }
        else
            return 0;
            //return 0;//this layer is not finished yet
        
    
        
        
    }
    
    
    if(SC.type== ML::POOL){
        return 1;
    }
    return 0;
}



int main(){
 
    memory AM("AM", 4*1024*1024);
    memory AM2("AM2", 4*1024*1024);
    tile Tiles[16];
    //initialize();
    //int weight[500]={0};
    //int activations[400]={0};
    printf("initialized finished\n");
    //pass activations to AM, pass weights to ...
    lenet Lenet;
    Lenet.layers=6;
    Lenet.inputs.initialize(1, 1, 28,28);
    Lenet.conv1.initialize("conv1",20, 5, 1,Lenet.inputs);
    Lenet.pool1.initialize("pool1",ML::Pool::MAX, 2, 2,Lenet.conv1.output);
    Lenet.conv2.initialize("conv2", 50, 5, 1, Lenet.pool1.output);
    Lenet.pool2.initialize("pool2", ML::Pool::MAX, 2, 2, Lenet.conv2.output);
    Lenet.ip1.initialize("ip1", 500, Lenet.pool2.output,1);
    Lenet.ip2.initialize("ip2", 10, Lenet.ip1.output, 0);
    
    
    
    
    schedule_command  SC[Lenet.layers];
    SC[0]=schedule_conv(Lenet.conv1);
    SC[1]= schedule_pool(Lenet.pool1);
    SC[2]= schedule_conv(Lenet.conv2);
    SC[3]=schedule_pool(Lenet.pool2);
    SC[4]=schedule_fully(Lenet.ip1);
    SC[5]= schedule_fully(Lenet.ip2);
    

    //assume starting from each layer, weights and activations start from addr 0 in AM and WM
 
    printf("calculating\n");
    int result=0;
    int i=0;
    for (;;){
        
        result= calculate(SC[i], AM, Tiles,AM2);
        //result=1 means this layer finished calculation, move to next layer
        if (result){
            i++;
            printf("layer %d finished, cycle:%d\n",i,sim_cycle);
            printf("multiply %d\n",multiply);
            multiply=0;
        }
        sim_cycle++;
        if(i>=Lenet.layers)break;
    }
    //finally write the results of each tile's NBout back...
    printf("finished   cycle:%d\n", sim_cycle);

}

