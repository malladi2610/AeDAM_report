#### Question 1:
I have a lenet-5 model configuration as shown below

#The model configuration is as follows
# C1: 6 filters, 5x5 kernel, output 28x28x6
# S2: 2x2 average pooling, stride=2, output 14x14x6
# C3: 16 filters, 5x5 kernel, output 10x10x16
# S4: 2x2 average pooling, stride=2, output 5x5x16
# C5: 120 filters, 5x5 kernel (or fully connected), output 120
# F6: Fully connected, output 84
# Output: Fully connected, output 10


- id: 0 # Conv1 Stride 1
  name: layer_1
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 6, 1, 32, 32, 5, 5]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 0

- id: 1 # Pool1 #Stride 2
  name: pooling_layer_1
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
  loop_dims: [B, G, IY, IX, FY, FX]
  loop_sizes: [1, 6, 28, 28, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 1

- id: 2 # conv2_1
  name: layer_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 16, 6, 14, 14, 5, 5]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 1
    W: 2

- id: 3
  name: pooling_layer_2
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
  loop_dims: [B, G, IY, IX, FY, FX]
  loop_sizes: [1, 16, 10, 10, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 2
    W: 3


- id: 4 #Flatenning
  name: layer_3
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 120, 16, 5, 5, 5, 5]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 3
    W: 4
  
- id: 5 #FC
  name: layer_4
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 84, 120, 1, 1, 1, 1]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 4
    W: 5

- id: 6 #FC
  name: layer_5
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1,10,84, 1, 1, 1, 1]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 5
    W: 6

I would like to scale this workload 20 times and all the subsequent layers with it the reason I am doing this because for my exploration tool when I am having a smaller workload the error is large and when I scale it the issue drops. So, I would like to scale the workload by the mentioned quantity.



#### Answer:

For the complete lenet 5 input stationary model
- id: 0 # Conv1 Stride 1
  name: layer_1
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 6, 1, 112, 112, 5, 5]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 0

- id: 1 # Pool1 #Stride 2
  name: pooling_layer_1
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
  loop_dims: [B, G, IY, IX, FY, FX]
  loop_sizes: [1, 6, 108, 108, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 1

- id: 2 # Conv2
  name: layer_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 16, 6, 53, 53, 5, 5]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 1
    W: 2

- id: 3 # Pool2
  name: pooling_layer_2
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
  loop_dims: [B, G, IY, IX, FY, FX]
  loop_sizes: [1, 16, 49, 49, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 2
    W: 3

- id: 4 # Flattening (adjusted kernel size)
  name: layer_3
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 120, 16, 23, 23, 23, 23]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 3
    W: 4

- id: 5 # FC
  name: layer_4
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 84, 120, 1, 1, 1, 1]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 4
    W: 5

- id: 6 # FC
  name: layer_5
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 10, 84, 1, 1, 1, 1]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 5
    W: 6

Now the same model which is used for the frame based execution:


- id: 0 # Conv1 Stride 1
  name: layer_1
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 6, 1, 108, 108, 5, 5]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 0

- id: 1 # Pool1 #Stride 2
  name: pooling_layer_1
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, G, OY, OX, FY, FX]
  loop_sizes: [1, 6, 53, 53, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 1

- id: 2 # Conv2
  name: layer_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 16, 6, 49, 49, 5, 5]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 1
    W: 2

- id: 3 # Pool2
  name: pooling_layer_2
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, G, OY, OX, FY, FX]
  loop_sizes: [1, 16, 23, 23, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 2
    W: 3

- id: 4 # Flattening (adjusted kernel size)
  name: layer_3
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 120, 16, 1, 1, 23, 23]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 3
    W: 4

- id: 5 # FC
  name: layer_4
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 84, 120, 1, 1, 1, 1]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 4
    W: 5

- id: 6 # FC
  name: layer_5
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 10, 84, 1, 1, 1, 1]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 5
    W: 6