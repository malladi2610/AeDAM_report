# Resolve the error in the calculation of final ouput.

Hi, I need your help to resolve the error in the count of the final output that needs to be achieved after accumulation of the result.

Here is the context.

I am using event driven data flow exploration using the zigzag exploration tool.

I modelled seneca event driven architecture as follows

name: Event_Driven_Accelerator

operational_array:
  unit_energy: 0.04  # for ANN MAC operations
  unit_area: 1
  dimensions: [D1]
  sizes: [8]  # 8 NPEs

memories: # 
  register:
    size: 1024  # 1024 bits for NPE register files
    r_bw: 48  # high bandwidth for fast access
    w_bw: 48
    r_cost: 0.008  # low energy cost, e.g., 8-12 fJ
    w_cost: 0.008
    area: 0.01  # small area per instance
    r_port: 2
    w_port: 2
    rw_port: 1
    latency: 1  # <1 ns (Has to been an integer as per zigzag modelling, So 1ns)
    operands: [I1,I2,O]
    ports:
      - tl: r_port_1
        fh: w_port_1
      - tl: r_port_2
        fh: w_port_2
      - tl: r_port_1
        th: rw_port_1
        fh: rw_port_1
        
    served_dimensions: []

  # Original
  sram:
    size: 2097152  # 2 Mbits Data Memory
    r_bw: 128
    w_bw: 128
    r_cost: 0.18  # medium energy cost, e.g., 180-220 pJ, current unit is pj/bits
    w_cost: 0.18
    area: 1
    r_port: 1
    w_port: 2
    rw_port: 1
    latency: 2  # 2 ns
    operands: [I2, O]
    ports:
        - tl: r_port_1
          fh: w_port_1
        - fl: w_port_2          
          th: rw_port_1
    served_dimensions: [D1]

  shared_memory_Weights:
    size: 1677721600  # 1600 Mbits, e.g., for STT-MRAM
    r_bw: 128  
    w_bw: 128
    r_cost: 2  
    w_cost: 2
    area: 2
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 25  
    operands: [I2]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]

  NOC_inputs_outputs:
    size: 1073741824   # 1024 Mbits
    r_bw: 64  
    w_bw: 64
    r_cost: 0.03  
    w_cost: 0.03  
    area: 2
    r_port: 1
    w_port: 0
    rw_port: 1
    latency: 10  
    operands: [I1, O]
    ports:
      - tl: r_port_1
      - fl: rw_port_1
    served_dimensions: [D1]

I attached the paper of seneca for you to verify this model

I am finding the exploration of the VGGnet model on this above architecture, as I am trying to debug this error I am sharing the first layer of the VGGnet that I modelled

- id: 0 # Conv1 Stride 1
  name: layer_0
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 64, 3, 224, 224, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 32
    O_final: 16
  operand_source:
    I: 0
    W: 0


Here W = weights, I = inputs, O = partial sums, O_final = final ouputs

As its's an event driven architecture I have a defined path that the data follows to perform achieve the behaviour

# Input path
# NoC → NPE register file
# Events stream in, to the RF after pre processing .

# Weight path
# SRAM (DMEM) → NPE register file → MAC units
# Each event triggers a direct SRAM read of its associated weight into the NPEs .

# Depending on the size of the weights
# Shared memeory -> SRAM -> NPE register file -> MAC units

# Partial-sum path
# NPE register file accumulates → SRAM
# Updated neuron states (partial sums) are written back immediately into SRAM .

# Output path
# NPE registers → NoC


The zigzag upon exploration follows this defined path and for the above mentioned workload and architecture the explored best mapping is as follows

"Loop ordering for layer_0
========================================================================================================
Temporal Loops                     O                      W                      I                      
========================================================================================================
for IX in [0, 224):                NOC_inputs_outputs     sram                   NOC_inputs_outputs     
--------------------------------------------------------------------------------------------------------
  for IY in [0, 224):              sram                   sram                   NOC_inputs_outputs     
--------------------------------------------------------------------------------------------------------
    for C in [0, 3):               sram                   sram                   register               
--------------------------------------------------------------------------------------------------------
      for K in [0, 8):             sram                   sram                   register               
--------------------------------------------------------------------------------------------------------
        for FX in [0, 3):          register               register               register               
--------------------------------------------------------------------------------------------------------
          for FY in [0, 3):        register               register               register               
--------------------------------------------------------------------------------------------------------
========================================================================================================
Spatial Loops                                                                                           
========================================================================================================
            parfor K in [0, 8):                                                                         
--------------------------------------------------------------------------------------------------------
"

Now, When the stats are observed

{
    "outputs": {
        "memory": {
            "utilization": {
                "O": [
                    0.46875,
                    0.67529296875,
                    0.050952911376953125
                ],
                "W": [
                    0.46875,
                    0.67529296875,
                    1.64794921875e-05
                ],
                "I": [
                    0.46875,
                    0.050952911376953125
                ]
            },
            "word_accesses": {
                "O": [
                    {
                        "rd ^": 57802752,
                        "wr v": 57802752,
                        "rd v": 86704128,
                        "wr ^": 86704128
                    },
                    {
                        "rd ^": 2429952,
                        "wr v": 2429952,
                        "rd v": 21676032,
                        "wr ^": 21676032
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 4859904,
                        "wr ^": 4859904
                    }
                ],
                "W": [
                    {
                        "rd ^": 0,
                        "wr v": 28901376,
                        "rd v": 86704128,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 216,
                        "rd v": 10838016,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 216,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0,
                        "wr v": 401408,
                        "rd v": 1204224,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 50176,
                        "wr ^": 0
                    }
                ]
            }
        },
        "energy": {
            "energy_total": 17640531.808,
            "operational_energy": 3468165.12,
            "memory_energy": 14172366.688,
            "memory_energy_breakdown_per_level": {
                "O": [
                    2312110.08,
                    8678154.24,
                    291594.24
                ],
                "W": [
                    924844.032,
                    1950881.7599999998,
                    432
                ],
                "I": [
                    12845.056,
                    1505.28
                ]
            },
            "memory_energy_breakdown_per_level_per_operand": {
                "O": [
                    {
                        "rd ^": 462422.016,
                        "wr v": 462422.016,
                        "rd v": 693633.024,
                        "wr ^": 693633.024
                    },
                    {
                        "rd ^": 437391.36,
                        "wr v": 437391.36,
                        "rd v": 3901685.76,
                        "wr ^": 3901685.76
                    },
                    {
                        "rd ^": 0.0,
                        "wr v": 0.0,
                        "rd v": 145797.12,
                        "wr ^": 145797.12
                    }
                ],
                "W": [
                    {
                        "rd ^": 0.0,
                        "wr v": 231211.008,
                        "rd v": 693633.024,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0.0,
                        "wr v": 38.879999999999995,
                        "rd v": 1950842.88,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 432,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0.0,
                        "wr v": 3211.264,
                        "rd v": 9633.792,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0.0,
                        "wr v": 0.0,
                        "rd v": 1505.28,
                        "wr ^": 0.0
                    }
                ]
            }
        },
        "latency": {
            "data_onloading": 11.0,
            "computation": 21676023.0,
            "data_offloading": 21714.0
        },
        "spatial": {
            "mac_utilization": {
                "ideal": 1.0,
                "stalls": 0.5000002076026585,
                "stalls_onloading": 0.4999999538660993,
                "stalls_onloading_offloading": 0.4994995794033556
            }
        }

Focussing on the Word access count as this effects the latency and the energy stats as showcased in the flow

Mapspace → (For a single mapping) Path creation (in Mapping object) → Memory utilisation → Memory word access → (Split) 
    → (For latency) → Double buffer detection → Transfer cycle calculation → Port activity calculation → Latency calculation
    → (For energy) → MAC energy + Memory energy calculation


So, When you observer for the workload details 

I = (B, C , IX, IY) = (1,3,224,224)
W = (K, C, FX, FY) = (64,3,3,3)
O = (B,K, OX, OY) = (1,64,224,224)

Now from here
The input dimensions are (1x3x224x224) = 150528
The weight elemets are = (3x3x3x8) = 1728
The Ouptut elements are = (1x6x64x224x224) = 3211264

On validating this with the above achieved from the word access 

Inputs 
At the RF stage: the access is 1204224 = (150528 * 8) becasuse of 8 NPE which matches the required answer - For event driven a single input comes it is reused completely and never accessed again

Weights
At the SRAM:
All the weights are first moved from the shared to the SRAM: 216

From SRAM:
All the weights are accessed again and again for all the computations: 224x224x3x3x3x8 = 10838016

Now they reach RF and are accessed again and again for all the MAC operation i.e 224x224x3x3x3x8x8 = 86704128

All these values are correctly reflected in the word access values

Outputs - This is where the problem is

THe partial sum count which is equal to no.of MAC operation is performed correcly i.e 86704128

but once the partial sum is over and it is written to SRAM for reaccessing is also correct i.e 2*10828016 = 21656032

THen the final ouput is written to the NOC as per the requirement, in this case the ouput should be equal to the total ouput i.e 3211264, but it is equal to 4859904 which is almost 50% more than the actual value of 3211264. 

And I know it is because of calcualtion that is being done by the zigzag at it's word access computation with a specific value. 

This problem dpesn't arrise when a normal frame based execution is done in zigag i.e when it is modelled as ouput stationary, I am attaching the output stationary map and the stats the way of calcuation remain the same as above

""outputs": {
        "memory": {
            "utilization": {
                "O": [
                    0.546875,
                    0.1380767822265625,
                    0.05013483762741089
                ],
                "W": [
                    0.1380767822265625,
                    1.64794921875e-05
                ],
                "I": [
                    0.546875,
                    0.1380767822265625,
                    0.05013483762741089
                ]
            },
            "word_accesses": {
                "O": [
                    {
                        "rd ^": 1204224,
                        "wr v": 0,
                        "rd v": 86704128,
                        "wr ^": 86704128
                    },
                    {
                        "rd ^": 401408,
                        "wr v": 0,
                        "rd v": 0,
                        "wr ^": 401408
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 0,
                        "wr ^": 802816
                    }
                ],
                "W": [
                    {
                        "rd ^": 0,
                        "wr v": 216,
                        "rd v": 10838016,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 216,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0,
                        "wr v": 3612672,
                        "rd v": 10838016,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 57120,
                        "rd v": 200704,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 114016,
                        "wr ^": 0
                    }
                ]
            }
        },
        "energy": {
            "energy_total": 7150404.384,
            "operational_energy": 3468165.12,
            "memory_energy": 3682239.2639999995,
            "memory_energy_breakdown_per_level": {
                "O": [
                    1396899.8399999999,
                    144506.88,
                    24084.48
                ],
                "W": [
                    1950881.7599999998,
                    432
                ],
                "I": [
                    115605.504,
                    46408.32,
                    3420.48
                ]
            },
            "memory_energy_breakdown_per_level_per_operand": {
                "O": [
                    {
                        "rd ^": 9633.792,
                        "wr v": 0.0,
                        "rd v": 693633.024,
                        "wr ^": 693633.024
                    },
                    {
                        "rd ^": 72253.44,
                        "wr v": 0.0,
                        "rd v": 0.0,
                        "wr ^": 72253.44
                    },
                    {
                        "rd ^": 0.0,
                        "wr v": 0.0,
                        "rd v": 0.0,
                        "wr ^": 24084.48
                    }
                ],
                "W": [
                    {
                        "rd ^": 0.0,
                        "wr v": 38.879999999999995,
                        "rd v": 1950842.88,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 432,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0.0,
                        "wr v": 28901.376,
                        "rd v": 86704.128,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0.0,
                        "wr v": 10281.6,
                        "rd v": 36126.72,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0.0,
                        "wr v": 0.0,
                        "rd v": 3420.48,
                        "wr ^": 0.0
                    }
                ]
            }
        },
        "latency": {
            "data_onloading": 518.0,
            "computation": 10988540.0,
            "data_offloading": 3592.0
        },
        "spatial": {
            "mac_utilization": {
                "ideal": 1.0,
                "stalls": 0.9863017288921003,
                "stalls_onloading": 0.986255236800097,
                "stalls_onloading_offloading": 0.9859329642988724
            }
        }"


        Here is the output stationary map

        Loop ordering for layer_0
==================================================================================================================================================
Temporal Loops                     O                                    W                                    I                                    
==================================================================================================================================================
for OX in [0, 224):                shared_memory_inputs_and_weights     sram                                 shared_memory_inputs_and_weights     
--------------------------------------------------------------------------------------------------------------------------------------------------
  for OY in [0, 224):              sram                                 sram                                 sram                                 
--------------------------------------------------------------------------------------------------------------------------------------------------
    for FX in [0, 3):              register                             sram                                 register                             
--------------------------------------------------------------------------------------------------------------------------------------------------
      for FY in [0, 3):            register                             sram                                 register                             
--------------------------------------------------------------------------------------------------------------------------------------------------
        for C in [0, 3):           register                             sram                                 register                             
--------------------------------------------------------------------------------------------------------------------------------------------------
          for K in [0, 8):         register                             sram                                 register                             
--------------------------------------------------------------------------------------------------------------------------------------------------
==================================================================================================================================================
Spatial Loops                                                                                                                                     
==================================================================================================================================================
            parfor K in [0, 8):                                                                                                                   
--------------------------------------------------------------------------------------------------------------------------------------------------




Now, I need your help to correct this value with the help of the correction value, by modifing this formula, which is present in the cost_model.py code that I shared for the word access which inturn needs correction to the latency and the similarly energy can also be changed.

So, here I atteched data_movement.py script, cost_model.py, port_activity.py script the seneca paper and I want you to go through all the information I shared and provide me on what correction can be applied that is general and will work whn I change the workload too.

