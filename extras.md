! from -> 3_dimension_1.md

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