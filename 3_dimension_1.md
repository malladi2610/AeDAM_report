This exploration is all about the single dataflow exploration of a given workload on an event driven archecture using the tools Zigzag(Single core exploration) and Stream (Multi core exploration)

# Overall Todo list
- [ ] Taking details from the background study on how the Frame based execution and the Event based execution occurs and also the combination of the possible mapping to run the effect on latency and the energy and the EDP is as follows: (Validation process)

[From the background study, 1. Frame based execution, 2. Event based execution]

- [ ] Zigzag lenet 5 model implementation on the following architecture and possible mapping
    - [ ] Results of Latency, energy and EDP (Single bar graph)

- [ ] Stream lenet 5 model implementation on the following architecture and possible mapping
    - [ ] Results of latency, energy and EDP (Single bar graph)

- [ ] Upon validation of the Zigzag and Stream for the lenet 5 model the follwing results are supposed to be achieved
    - [ ]Results of latency, energy and EDP (double bar graph) - With comparision of the both and the change in the results

- [ ] Implication of this difference on the current solution and the exploration

- [ ] How to get all the possible mappings combination to do the test?

- [ ]Proposed solution implementation.

# [For 24/03 submision]
 - [ ] Write about the validation process using the simulator
        - [ ] Indicate about the execssive partial sums which are being generate that effects the latency which needs to be validated and compensated in the end
        - [ ] Take a simple exmple for now and show the reading by hand or python script
        - [ ] End with a need to verify this and will be developed within this week

 - [ ] Single core implementation using Zigzag
        - [ ] Lenet 1st layer readings tabular readings for different mappings (4 tables possible)
        - [ ] Present the information in the bar graph form for the latency, Energy and Energy Delay product
        - [ ] Lenet compete model in the 1 core implementation
            - [ ] Tabular readings for different mappings
            - [ ] Present the information in the graph form
    - [ ] End with a statement says that for event driven the value should be less as explained in the validation section will be getting that within this week

 - [ ] Multi-core implementation using stream (For kanishkan)
        - [ ] Explain about the two major challenges faced and currently being resolved
            - [ ] Stream doesn't accept the YAML file as input and only ONNX file. So, the input stationary results are difficult to get and need some modification
                - [ ] Work around being followed
            - [ ] Stream doesn't give the intra core loop ordering and the stats are mostly in terms in latency due to which there needs to be some work to get the required results
 - [ ] Plan for the this week (In a ppt)
        - [ ] Managing to make the stream work for the 2 core archtecture with the Lenet - 5 model
        - [ ] A validation program for the zigzag results
        - [ ] A final consolidated results from Zigzag with validation, Basic results from Stream for multi core exploration


---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Validation process 

Here are the steps followed for the validation of the results from the Zigzag and Stream on running the Lenet-5 Model

Frame based execution and the Event based execution

Building upon the basic of the execution the Frame based execution and the Event based execution of the custom CNN layer is as follows

Input - [1,1,5,5]
Weights(Kernels) - [8,1,3,3]
Outputs - [1,8,3,3]

The Frame based and event based execution is as follows


<!-- Now, On scaling the workload to be Lenet - 5 model and doing a comparison of the Zigzag with the event based execution

Input - [1,1,256,256]

Layer 1 :
    kernel - [8,1,129,129]
    Output - [1,8,128,128]
    Padding = 0
    Stride = 1

Layer 2:
    Input - [1,8,128,128]
    polling - [8,8,2,2]
    Output - [1,8,64,64]

Layer 3: 
    Input - [1,8,64,64]
    kernel - [24,8,15,15]
    Stride = 1
    Padding = 0
    Output - [1,24,48,48]

Layer 4:
    Input - [1,24,48,48]
    polling - [24,24,3,3]
    Output - [1,24,16,16]

Layer 5: (flatten)
    Input - [1,24,16,16]
    Output - 256 

Layer 6: (FCC)
    Input - 256
    Output - 128

Output: 
    Input - 128
    Output - 10 -->

The lenet - 5 configuration is as follows

Input: 32x32x1
C1: 6 filters, 5x5 kernel, output 28x28x6
S2: 2x2 average pooling, stride=2, output 14x14x6
C3: 16 filters, 5x5 kernel, output 10x10x16
S4: 2x2 average pooling, stride=2, output 5x5x16
C5: 120 filters, 5x5 kernel (or fully connected), output 120
F6: Fully connected, output 84
Output: Fully connected, output 10

On modeling the lenet -5 on to the Single core architecture using zigzag the results are as follows compared to the actual event driven computing (Generate a table and a bar graph to compare)


On modeling the lenet - 5 on to the Multi-core architecture using zigzag the results are as follows compared to the actual event driven computing


# Zigzag Results


# Stream Results

# Validation Results

# Implication of the differences in the results

# Proposed Solution


# Algorithm of the script that generates the possible mapping constraints





