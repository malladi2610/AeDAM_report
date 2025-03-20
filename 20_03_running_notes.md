#### Question 1:

Why is the below architecture cnfiguration and the model architecture able to create the mappings and stats and my architecture and model is not able to create. Where am I going wrong


Working Model architecture:
name: edge_tpu_like

memories:
  rf_1B:
    size: 8
    r_bw: 8
    w_bw: 8
    r_cost: 0.01
    w_cost: 0.01
    area: 0
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 1
    auto_cost_extraction: False
    operands: [I2]
    ports:
      - fh: w_port_1
        tl: r_port_1
    served_dimensions: [D3, D4]

  rf_2B:
    size: 16
    r_bw: 16
    w_bw: 16
    r_cost: 0.02
    w_cost: 0.02
    area: 0
    r_port: 2
    w_port: 2
    rw_port: 0
    latency: 1
    operands: [O]
    ports:
      - fh: w_port_1
        tl: r_port_1
        fl: w_port_2
        th: r_port_2
    served_dimensions: [D2]

  sram_32KB:
    size: 262144
    r_bw: 512
    w_bw: 512
    r_cost: 22.9
    w_cost: 52.01
    area: 0
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 1
    min_r_granularity: 64
    min_w_granularity: 64
    operands: [I2]
    ports:
      - fh: w_port_1
        tl: r_port_1
    served_dimensions: [D1, D2, D3, D4]

  sram_2MB:
    size: 16777216
    r_bw: 2048
    w_bw: 2048
    r_cost: 416.16
    w_cost: 378.4
    area: 0
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 1
    min_r_granularity: 64
    min_w_granularity: 64
    operands: [I1, O]
    ports:
      - fh: w_port_1
        tl: r_port_1
      - fh: w_port_1
        tl: r_port_1
        fl: w_port_1
        th: r_port_1
    served_dimensions: [D1, D2, D3, D4]

  dram:
    size: 10000000000
    r_bw: 64
    w_bw: 64
    r_cost: 700
    w_cost: 750
    area: 0
    r_port: 0
    w_port: 0
    rw_port: 1
    latency: 1
    operands: [I1, I2, O]
    ports:
      - fh: rw_port_1
        tl: rw_port_1
      - fh: rw_port_1
        tl: rw_port_1
      - fh: rw_port_1
        tl: rw_port_1
        fl: rw_port_1
        th: rw_port_1
    served_dimensions: [D1, D2, D3, D4]

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1, D2, D3, D4]
  sizes: [8, 8, 4, 4]

Working model mapping:
- name: default
  spatial_mapping:
    D1:
      - K, 8
    D2:
      - C, 8
    D3:
      - OX, 4
    D4:
      - OY, 4
  memory_operand_links:
    O: O
    W: I2
    I: I1

- name: Add
  spatial_mapping:
    D1:
      - G, 8
    D2:
      - C, 1
    D3:
      - OX, 1
    D4:
      - OY, 1
  memory_operand_links:
    O: O
    W: I2
    I: I1

- name: Pooling
  spatial_mapping:
    D1:
      - G, 8
    D2:
      - C, 1
    D3:
      - OX, 1
    D4:
      - OY, 1
  memory_operand_links:
    O: O
    W: I2
    I: I1

Working workload
- id: 0 # conv1 stride 2
  name: example_name_of_layer0
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 64, 3, 112, 112, 7, 7]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 0
    W: 0

- id: 1 # max pool, stride 2
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fx][fy]*I[b][g][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, G, OY, OX, FX, FY]
  loop_sizes: [1, 64, 56, 56, 3, 3]
  operand_precision:
    I: 8
    W: 0
    O: 16
    O_final: 8
  operand_source:
    I: 0
    W: 1 # Constant operand

- id: 2 # conv2_1
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 64, 64, 56, 56, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 1
    W: 2

- id: 3 # conv2_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 64, 64, 56, 56, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 2
    W: 3

- id: 4 # Addition of layer 1 (residual path) and layer 3 (main path)
  operator_type: Add
  equation: O[b][g][oy][ox]=W[b][g][oy][ox]+I[b][g][oy][ox]
  loop_dims: [B, G, OY, OX]
  loop_sizes: [1, 64, 56, 56]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    W: 1
    I: 3

- id: 5 # conv2_3
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 64, 64, 56, 56, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 4
    W: 5

- id: 6 # conv2_4
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 64, 64, 56, 56, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 5
    W: 6

- id: 7 # Addition of layer 4 (residual connection) and layer 6 (main path)
  operator_type: Add
  equation: O[b][g][oy][ox]=W[b][g][oy][ox]+I[b][g][oy][ox]
  loop_dims: [B, G, OY, OX]
  loop_sizes: [1, 64, 56, 56]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    W: 4
    I: 6

- id: 8 # conv3_1 stride 2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 128, 64, 28, 28, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 7
    W: 8

- id: 9 # conv3_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 128, 128, 28, 28, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 8
    W: 9

- id: 10 # conv downsample of layer 7
  operator_type: Conv_downsample
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 128, 64, 28, 28, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 7
    W: 10

- id: 11 # Addition of layer 10 (residual connection) and layer 9 (main path)
  operator_type: Add
  equation: O[b][g][oy][ox]=W[b][g][oy][ox]+I[b][g][oy][ox]
  loop_dims: [B, G, OY, OX]
  loop_sizes: [1, 128, 28, 28]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    W: 10
    I: 9

- id: 12 # conv3_3
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 128, 128, 28, 28, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 11
    W: 12

- id: 13 # conv3_4
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 128, 128, 28, 28, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 12
    W: 13

- id: 14 # Addition of layer 11 (residual connection) and layer 13 (main path)
  operator_type: Add
  equation: O[b][g][oy][ox]=W[b][g][oy][ox]+I[b][g][oy][ox]
  loop_dims: [B, G, OY, OX]
  loop_sizes: [1, 128, 28, 28]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    W: 11
    I: 13

- id: 15 # conv4_1 stride 2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 256, 128, 14, 14, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 14
    W: 15

- id: 16 # conv4_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 256, 256, 14, 14, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 15
    W: 16

- id: 17 # conv downsample of layer 14
  operator_type: Conv_downsample
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 256, 128, 14, 14, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 14
    W: 17

- id: 18 # Addition of layer 17 (residual connection) and layer 16 (main path)
  operator_type: Add
  equation: O[b][g][oy][ox]=W[b][g][oy][ox]+I[b][g][oy][ox]
  loop_dims: [B, G, OY, OX]
  loop_sizes: [1, 256, 14, 14]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    W: 17
    I: 16

- id: 19 # conv4_3
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 256, 256, 14, 14, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 18
    W: 19

- id: 20 # conv4_4
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 256, 256, 14, 14, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 19
    W: 20

- id: 21 # Addition of layer 18 (residual connection) and layer 20 (main path)
  operator_type: Add
  equation: O[b][g][oy][ox]=W[b][g][oy][ox]+I[b][g][oy][ox]
  loop_dims: [B, G, OY, OX]
  loop_sizes: [1, 256, 14, 14]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    W: 18
    I: 20

- id: 22 # conv5_1 stride 2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 512, 256, 7, 7, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 21
    W: 22

- id: 23 # conv5_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 512, 512, 7, 7, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 22
    W: 23

- id: 24 # conv downsample of layer 21
  operator_type: Conv_downsample
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 512, 256, 7, 7, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 21
    W: 24

- id: 25 # Addition of layer 24 (residual connection) and layer 23 (main path)
  operator_type: Add
  equation: O[b][g][oy][ox]=W[b][g][oy][ox]+I[b][g][oy][ox]
  loop_dims: [B, G, OY, OX]
  loop_sizes: [1, 512, 7, 7]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    W: 24
    I: 23

- id: 26 # conv5_3
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 512, 512, 7, 7, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 25
    W: 26

- id: 27 # conv4_4
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 512, 512, 7, 7, 3, 3]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 26
    W: 27

- id: 28 # Addition of layer 25 (residual connection) and layer 27 (main path)
  operator_type: Add
  equation: O[b][g][oy][ox]=W[b][g][oy][ox]+I[b][g][oy][ox]
  loop_dims: [B, G, OY, OX]
  loop_sizes: [1, 512, 7, 7]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    W: 25
    I: 27

- id: 29 # aver pool
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fx][fy]*I[b][g][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, G, OY, OX, FX, FY]
  loop_sizes: [1, 512, 1, 1, 7, 7]
  operand_precision:
    I: 8
    W: 0
    O: 16
    O_final: 8
  operand_source:
    I: 28
    W: 29

- id: 30 # fc
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 1000, 512, 1, 1, 1, 1]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 29
    W: 30

My non-working architecture
name: 8_NPE_Accelerator

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1]
  sizes: [8]

memories:  
  inputMem_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128 
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 1
    operands: [I1]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]

  sram_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 2
    w_port: 2
    rw_port: 0
    latency: 1
    operands: [I2, O]
    ports:
      - tl: r_port_1
        fl: w_port_1
      - tl: r_port_2
        fh: w_port_2
    served_dimensions: [D1]

Non working mapping:
- name: default
  spatial_mapping:
    D1:
      - K, 8
  memory_operand_links:
    O: O
    W: I2
    I: I1

- name: Pooling
  spatial_mapping:
    D1:
      - G, 8
  memory_operand_links:
    O: O
    W: I2
    I: I1

Non working workload

#The model configuration is as follows
# Conv1: Input 32x32x1, 3x3 filters, 16 output channels, stride 1, no padding.
# Pool1: Max pooling, 2x2 window, stride 2.
# Conv2: 3x3 filters, 32 output channels, stride 1, no padding.
# Pool2: Max pooling, 2x2 window, stride 2.
# Conv3: 3x3 filters, 64 output channels, stride 1, no padding.

- id: 0 # Conv1 Stride 1
  name: conv_layer
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 16, 1, 32, 32, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 0

- id: 1 # Pool1 #Stride 2
  name: pooling_layer
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
  loop_dims: [B, G, IY, IX, FY, FX]
  loop_sizes: [1, 16, 30, 30, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 1

- id: 2 #Conv2 # Stride 1
  name: conv_layer
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 32, 16, 15, 15, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 1
    W: 2

- id: 3 #Pool2
  name: pooling_layer
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
  loop_dims: [B, G, IY, IX, FY, FX]
  loop_sizes: [1, 32, 13, 13,2,2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 2
    W: 3

- id: 4 #Conv3
  name: conv_layer
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 64, 32, 6, 6, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 3
    W: 4

Here is the error I am getting
ain_ed.py
2025-03-20 16:30:02,767 - zigzag.parser.mapping_factory.__init__ +46 - WARNING - Operator Conv not defined in mapping. Using default mapping instead.
Traceback (most recent call last):
  File "/home/subhash/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/March_experiments/results-20-03/main_ed.py", line 39, in <module>
    result = get_hardware_performance_zigzag(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/api.py", line 145, in get_hardware_performance_zigzag
    cmes = mainstage.run()
           ^^^^^^^^^^^^^^^
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/main.py", line 18, in run
    for cme, extra_info in self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs).run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/parser/workload_parser.py", line 34, in run
    for cme, extra_info in sub_stage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/parser/accelerator_parser.py", line 24, in run
    for cme, extra_info in sub_stage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/results/save.py", line 84, in run
    for cme, extra_info in substage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/results/save.py", line 138, in run
    for cme, extra_info in substage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/results/reduce_stages.py", line 142, in run
    for cme, extra_info in substage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/exploit_data_locality_stages.py", line 116, in run
    self.update_top_mem_level()
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/exploit_data_locality_stages.py", line 224, in update_top_mem_level
    assert mem_level_in_info >= 0, "There are still layers with top mem levels not figured out."
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: There are still layers with top mem levels not figured out.


What is the major implementation difference and how to solve it   ?


[P.S. As I am doing an event driven computing the way I model the workload is a different that the working model but the logic of implementattion remains that same only the workload is passed differently to the zigzag tool]


#### Answer: Implement a DRAM to transmit to store the data and reuse the data


#### Question 2:

Now, Here is the modified Architecture with DRAM and the simple read and write ports as shown below, but I am getting error how to solve it

Accelerator architecture:

name: 8_NPE_Accelerator

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1]
  sizes: [8]

memories:
  dram:
    size: 10000000000
    r_bw: 64
    w_bw: 64
    r_cost: 700
    w_cost: 750
    area: 0
    r_port: 3
    w_port: 3
    rw_port: 0
    latency: 1
    operands: [I1, I2, O]
    ports:
      - tl: r_port_1
        fl: w_port_1
      - tl: r_port_2
        fh: w_port_2
      - tl: r_port_3
        fl: w_port_3
    served_dimensions: [D1]


  inputMem_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128 
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 1
    operands: [I1]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]

  sram_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 2
    w_port: 2
    rw_port: 0
    latency: 1
    operands: [I2, O]
    ports:
      - tl: r_port_1
        fl: w_port_1
      - tl: r_port_2
        fh: w_port_2
    served_dimensions: [D1]

Error:
main_ed.py
2025-03-20 16:53:11,865 - zigzag.parser.mapping_factory.__init__ +46 - WARNING - Operator Conv not defined in mapping. Using default mapping instead.
Traceback (most recent call last):
  File "/home/subhash/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/March_experiments/results-20-03/main_ed.py", line 39, in <module>
    result = get_hardware_performance_zigzag(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/api.py", line 145, in get_hardware_performance_zigzag
    cmes = mainstage.run()
           ^^^^^^^^^^^^^^^
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/main.py", line 18, in run
    for cme, extra_info in self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs).run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/parser/workload_parser.py", line 34, in run
    for cme, extra_info in sub_stage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/parser/accelerator_parser.py", line 24, in run
    for cme, extra_info in sub_stage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/results/save.py", line 84, in run
    for cme, extra_info in substage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/results/save.py", line 138, in run
    for cme, extra_info in substage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/results/reduce_stages.py", line 142, in run
    for cme, extra_info in substage.run():
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/exploit_data_locality_stages.py", line 116, in run
    self.update_top_mem_level()
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/exploit_data_locality_stages.py", line 199, in update_top_mem_level
    [
  File "/home/subhash/Thesis/zigzag/zigzag_env_1/lib/python3.11/site-packages/zigzag/stages/exploit_data_locality_stages.py", line 200, in <listcomp>
    self.each_layer_io_data_size[layer.id][0][operand]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
KeyError: I1

Now, With the previous context can you solve the error

#### Answer 2:
The issue is with the ports and can be solved by modifing them correctly

#### Question 3:

From the zigzag API code, Can a way be found to run the given worload on to the architecture without having a DRAM. The reason I am doing so is that I want to validate the claim that there is absolutely a need of a global memory to be present to communicate the data b/w the layers and lower memory levels can't be used at all

Here is the API code of Zigzag that you can refer to address on what features of the zigzag I can exploit to achieve what I mentioned above

import logging
from datetime import datetime
from typing import Any, Literal

from onnx import ModelProto

from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.exploit_data_locality_stages import (
    ExploitInterLayerDataLocalityStage,
    SearchInterLayerDataLocalityStage,
)
from zigzag.stages.main import MainStage
from zigzag.stages.mapping.salsa import SalsaStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.parser.accelerator_parser import AcceleratorParserStage
from zigzag.stages.parser.onnx_model_parser import ONNXModelParserStage
from zigzag.stages.parser.workload_parser import WorkloadParserStage
from zigzag.stages.results.reduce_stages import MinimalEDPStage, MinimalEnergyStage, MinimalLatencyStage, SumStage
from zigzag.stages.results.save import CompleteSaveStage, PickleSaveStage, SimpleSaveStage
from zigzag.stages.results.visualization import VisualizationStage
from zigzag.stages.stage import StageCallable
from zigzag.stages.workload_iterator import WorkloadStage


def get_hardware_performance_zigzag(
    workload: str | ModelProto,
    accelerator: str,
    mapping: str,
    *,
    temporal_mapping_search_engine: Literal["loma"] | Literal["salsa"] = "loma",
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str | None = None,
    lpf_limit: int = 6,
    nb_spatial_mappings_generated: int = 3,
    in_memory_compute: bool = False,
    exploit_data_locality: bool = False,
    enable_mix_spatial_mapping: bool = False,
) -> (
    tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]
    | tuple[float, float, float, float, list[tuple[CostModelEvaluationABC, Any]]]
):
    """! ZigZag API: estimates the cost of running the given workload on the given hardware architecture.
    @param workload Either a filepath to the workload ONNX or yaml file, an ONNX model.
    @param accelerator Filepath to accelerator yaml file.
    @param mapping Filepath to mapping yaml file.
    @param opt Optimization criterion: either `energy`, `latency` or `EDP`.
    @param dump_folder Folder where outputs will be saved.
    @param pickle_filename Filename of pickle dump.
    @param lpf_limit Determines the number of temporal unrollings that are evaluated.
    @param nb_spatial_mappings_generated Max nb of spatial mappings automatically generated (if not provided in
        mapping).
    @param in_memory_compute Optimizes the run for IMC architectures.
    @param exploit_data_locality Iff true, an attempt will be made to keep data in lower-level memory in between layers
    @param enable_mix_spatial_mapping Wether `mixed` spatial mappings will be generated, i.e. unrolling multiple Layer
        Dimensions in a single Operational Array Dimension.
    """
    pickle_filename = f"{dump_folder}/list_of_cmes.pickle" if pickle_filename is None else pickle_filename

    # Initialize the logger
    logging_level = logging.INFO
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

    match opt:
        case "energy":
            opt_stage = MinimalEnergyStage
        case "latency":
            opt_stage = MinimalLatencyStage
        case "EDP":
            opt_stage = MinimalEDPStage
        case _:
            raise NotImplementedError("Optimization criterion 'opt' should be either 'energy' or 'latency' or 'EDP'.")

    # Check workload format and based on it select the correct workload parser stage
    workload_parser_stage = (
        ONNXModelParserStage
        if isinstance(workload, ModelProto) or (workload.split(".")[-1] == "onnx")
        else WorkloadParserStage
    )

    # Add stages to keep whole layers in lower level memory instead of rewriting to DRAM, if possible
    do_exploint_inter_layer_locality = in_memory_compute or exploit_data_locality or enable_mix_spatial_mapping
    # Whether `mixed` mappings (e.g. `D1: {K:8, C:4}`) can be generated
    do_mix_spatial_mapping_generation = in_memory_compute or enable_mix_spatial_mapping
    # Select temporal mapping engine based on the function input
    temporal_mapping_engine = SalsaStage if temporal_mapping_search_engine == "salsa" else TemporalMappingGeneratorStage

    stages = [
        # Parse the ONNX Model into the workload
        workload_parser_stage,
        # Parse the accelerator module/passthrough given accelerator
        AcceleratorParserStage,
        # Save the summed CME energy and latency to a json
        SimpleSaveStage,
        # Save all received CMEs in a list to a pickle file
        PickleSaveStage,
        # Sum up the received best CME across all layers of the workload
        SumStage,
        # Search the lowest allowed memory level per operand per layer
        SearchInterLayerDataLocalityStage if do_exploint_inter_layer_locality else None,
        # Iterate through the different layers in the workload
        WorkloadStage,
        # Save the chosen loop ordering and memory hierarchy
        VisualizationStage,
        # Remove unused memories
        ExploitInterLayerDataLocalityStage if do_exploint_inter_layer_locality else None,
        # Save each processed layer to a json
        CompleteSaveStage,
        # Reduce all CMEs, returning minimal energy/latency one
        opt_stage,
        # Generate multiple spatial mappings (SM)
        SpatialMappingGeneratorStage,
        # Reduce all CMEs, returning minimal energy/latency one
        opt_stage,
        # Generate multiple temporal mappings (TM)
        temporal_mapping_engine,
        # Evaluate generated SM and TM through cost model
        CostModelStage,
    ]

    stage_callables: list[StageCallable] = [s for s in stages if s is not None]

    # Initialize the MainStage as entry point
    mainstage = MainStage(
        list_of_callables=stage_callables,
        accelerator=accelerator,
        workload=workload,
        mapping=mapping,
        dump_folder=dump_folder,
        pickle_filename=pickle_filename,
        loma_lpf_limit=lpf_limit,
        loma_show_progress_bar=True,
        nb_mappings_generated=nb_spatial_mappings_generated,
        enable_mix_spatial_mapping_generation=do_mix_spatial_mapping_generation,
        # If we need access the same input data multiple times from the innermost memory level and the data size is
        # smaller than the memory read bw, # take into account only one-time access cost (assume the data can stay at
        # the output pins of the memory as long as it is needed).
        access_same_data_considered_as_no_access=False,
    )

    # Launch the MainStage
    cmes = mainstage.run()
    energy_total: float = cmes[0][0].energy_total
    latency_total: float = cmes[0][0].latency_total2

    if in_memory_compute:
        tclk: float = cmes[0][1][0][0].tclk
        area: float = cmes[0][1][0][0].area_total
        return energy_total, latency_total, tclk, area, cmes  # type: ignore

    return energy_total, latency_total, cmes


def get_hardware_performance_zigzag_imc(
    *args: Any,
) -> tuple[float, float, float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """Overload with type hint"""
    return get_hardware_performance_zigzag(*args, in_memory_compute=True)  # type: ignore


Here is the workload
#The model configuration is as follows
# Conv1: Input 32x32x1, 3x3 filters, 16 output channels, stride 1, no padding.
# Pool1: Max pooling, 2x2 window, stride 2.
# Conv2: 3x3 filters, 32 output channels, stride 1, no padding.
# Pool2: Max pooling, 2x2 window, stride 2.
# Conv3: 3x3 filters, 64 output channels, stride 1, no padding.

- id: 0 # Conv1 Stride 1
  name: conv_layer
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 16, 1, 32, 32, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 0

- id: 1 # Pool1 #Stride 2
  name: pooling_layer
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
  loop_dims: [B, G, IY, IX, FY, FX]
  loop_sizes: [1, 16, 30, 30, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 1

- id: 2 #Conv2 # Stride 1
  name: conv_layer
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 32, 16, 15, 15, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 1
    W: 2

- id: 3 #Pool2
  name: pooling_layer
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
  loop_dims: [B, G, IY, IX, FY, FX]
  loop_sizes: [1, 32, 13, 13,2,2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 2
    W: 3

- id: 4 #Conv3
  name: conv_layer
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 64, 32, 6, 6, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 3
    W: 4

Here is the Archtecture
name: 8_NPE_Accelerator

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1]
  sizes: [8]

memories:
  inputMem_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128 
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 1
    operands: [I1]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]

  sram_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 2
    w_port: 2
    rw_port: 0
    latency: 1
    operands: [I2, O]
    ports:
      - tl: r_port_1
        fl: w_port_1
      - tl: r_port_2
        fh: w_port_2
    served_dimensions: [D1]


This still uses the same architecture and the workload about which we discussed in out previous chat. So, please use the context from the previous chat to solve the problem.


#### Answer 3:
To determine whether ZigZag can run the provided workload on the given architecture without a DRAM, and to validate the claim that a global memory is absolutely necessary for inter-layer data communication (implying lower memory levels alone cannot suffice), we need to analyze ZigZag’s API, the workload, and the architecture. The goal is to explore if features in ZigZag can be exploited to eliminate the need for DRAM, allowing lower memory levels like `inputMem_16KB` and `sram_16KB` to handle all data movement. Let’s proceed step-by-step.

---

### Understanding ZigZag’s Memory Hierarchy and Data Flow

ZigZag is a hardware cost estimation framework that models data movement through a defined memory hierarchy. In a typical setup, this hierarchy includes a top-level memory (e.g., DRAM) as the source of external operands (`I` and `W`) and a sink for final outputs (`O`), with lower-level memories closer to the compute array (e.g., SRAM) facilitating efficient data access. The provided architecture lacks a DRAM:

- **Architecture Overview**:
  - `inputMem_16KB`: 16 KB (131,072 bits), serves operand `I1` (inputs), read-only with one read port.
  - `sram_16KB`: 16 KB (131,072 bits), serves `I2` (weights) and `O` (outputs), has two read and two write ports.
  - Operational array: 8 processing elements along dimension `D1`.

- **Workload Overview**:
  - A five-layer neural network: Conv1 → Pool1 → Conv2 → Pool2 → Conv3.
  - Each layer specifies operand sources via `operand_source`:
    - Conv1: `I: 0`, `W: 0` (external inputs and weights).
    - Pool1: `I: 0` (external), `W: 1` (from Pool1’s own weights).
    - Conv2: `I: 1` (from Pool1’s `O`), `W: 2` (external).
    - Pool2: `I: 2` (from Conv2’s `O`), `W: 3` (external).
    - Conv3: `I: 3` (from Pool2’s `O`), `W: 4` (external).

The `operand_source` value `0` typically indicates an external source, which ZigZag expects to come from the top-level memory. For subsequent layers, inputs (`I`) are sourced from the previous layer’s outputs (`O`), suggesting potential for inter-layer data reuse in lower memory levels.

- **Key Question**: Can `inputMem_16KB` and `sram_16KB` handle all data needs—initial external inputs, weights, and inter-layer communication—without a DRAM?

---

### Analyzing ZigZag’s API for DRAM-Free Operation

ZigZag’s pipeline, defined in `get_hardware_performance_zigzag`, consists of stages that parse the workload and architecture, generate mappings, and evaluate costs. Relevant features and stages include:

1. **Memory Hierarchy Parsing**:
   - `AcceleratorParserStage` interprets the memory hierarchy. Without a DRAM, `inputMem_16KB` and `sram_16KB` are the only memory levels, with `inputMem_16KB` potentially acting as the top level for `I1`.

2. **Data Locality Exploitation**:
   - `SearchInterLayerDataLocalityStage` and `ExploitInterLayerDataLocalityStage` (enabled via `exploit_data_locality=True`) optimize data placement by keeping outputs (`O`) in lower memory levels for reuse as inputs (`I`) in subsequent layers, reducing reliance on a top-level memory for inter-layer transfers.
   - However, these stages assume a full hierarchy exists for initial data sourcing.

3. **Cost Model Evaluation**:
   - `CostModelStage` relies on the `ExploitDataLocalityStage` internally, which traces data paths from a top-level memory to the compute array. Operands must have a defined source, especially for the first layer.

4. **Operand Source Handling**:
   - For Conv1, `I: 0` and `W: 0` imply external sources. Without DRAM, ZigZag must source these from `inputMem_16KB` (for `I1`) and `sram_16KB` (for `I2`), but the framework expects a top-level memory to supply external data.

---

### Testing a DRAM-Free Configuration

Let’s assess whether the current setup can work without DRAM and identify necessary adjustments.

#### Data Requirements
- **Conv1 Inputs (`I`)**: 32 × 32 × 1 × 16 bits = 16,384 bits (2 KB). Fits in `inputMem_16KB`.
- **Conv1 Weights (`W`)**: 16 × 1 × 3 × 3 × 16 bits = 2,304 bits (< 1 KB). Fits in `sram_16KB`.
- **Conv1 Outputs (`O`)**: 1 × 16 × 30 × 30 × 16 bits = 230,400 bits (28.8 KB). Exceeds `sram_16KB`’s 16 KB capacity.

The output size immediately poses a problem—`sram_16KB` cannot store Conv1’s `O` fully, and `inputMem_16KB` is read-only (`w_port: 0`), preventing output storage there. Subsequent layers face similar issues, with outputs often exceeding 16 KB (e.g., Conv2: 1 × 32 × 13 × 13 × 16 bits ≈ 27 KB).

#### Inter-Layer Data Flow
- Pool1 needs Conv1’s `O` as `I`. If `O` cannot be stored in `sram_16KB` due to size, and `inputMem_16KB` cannot be written to, the data has nowhere to go without DRAM.
- Even if sizes fit, `inputMem_16KB` lacks write ports to accept `O` for reuse as `I` in later layers.

#### Initial Data Sourcing
- For Conv1, `I: 0` and `W: 0` require an external source. Without DRAM, ZigZag’s default behavior assumes these come from the top-level memory, leading to errors (e.g., `KeyError: I1`) if undefined.

---

### Modifying the Architecture

To make a DRAM-free setup plausible, let’s modify the architecture to allow writing to `inputMem_16KB`, enabling it to act as a pseudo-top-level memory and support inter-layer reuse:

```yaml
name: 8_NPE_Accelerator
operational_array:
  unit_energy: 0.04
  unit_area: 1
  dimensions: [D1]
  sizes: [8]
memories:
  inputMem_16KB:
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 1
    w_port: 1  # Added write port
    rw_port: 0
    latency: 1
    operands: [I1, O]  # Allow outputs to be stored
    ports:
      - tl: r_port_1
        fl: w_port_1
    served_dimensions: [D1]
  sram_16KB:
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 2
    w_port: 2
    rw_port: 0
    latency: 1
    operands: [I2, O]
    ports:
      - tl: r_port_1
        fl: w_port_1
      - tl: r_port_2
        fh: w_port_2
    served_dimensions: [D1]
```

- **Changes**:
  - Added a write port to `inputMem_16KB`.
  - Assigned `O` to `inputMem_16KB` alongside `I1`, allowing outputs to be stored and reused as inputs.

#### Revised Data Flow
- **Conv1**: `I` from `inputMem_16KB`, `W` from `sram_16KB`, `O` to `inputMem_16KB` (230,400 bits > 16 KB, still overflows).
- **Pool1**: `I` from `inputMem_16KB` (Conv1’s `O`), `W` from `sram_16KB`, `O` to `inputMem_16KB`.

The size issue persists—outputs exceed 16 KB, requiring a larger memory or partial storage strategies, which ZigZag may not support without DRAM.

---

### Exploiting ZigZag Features

- **Exploit Data Locality**: Setting `exploit_data_locality=True` activates `SearchInterLayerDataLocalityStage` and `ExploitInterLayerDataLocalityStage`. These could keep `O` in `inputMem_16KB` or `sram_16KB` for reuse, but they don’t eliminate the need for an initial external source or sufficient storage capacity.
- **Custom Operand Sources**: Adjusting `operand_source` to assume `I` is pre-loaded into `inputMem_16KB` (e.g., `I: inputMem_16KB`) isn’t directly supported—ZigZag expects numeric IDs mapping to layers or external memory.
- **Minimal Hierarchy**: Defining `inputMem_16KB` as the top-level memory might work if ZigZag accepts it as the external source, but the API’s design assumes a deeper hierarchy.

---

### Conclusion

ZigZag’s current implementation requires a top-level memory (like DRAM) for:
1. **Sourcing External Operands**: Conv1’s `I: 0` and `W: 0` need a defined origin, which defaults to DRAM.
2. **Handling Large Data**: Outputs exceeding 16 KB cannot be stored in `inputMem_16KB` or `sram_16KB` alone.

While lower memory levels can facilitate inter-layer communication (e.g., storing `O` in `sram_16KB` or a writable `inputMem_16KB` for reuse as `I`), they cannot:
- Replace the top-level memory for initial data sourcing, due to ZigZag’s internal assumptions.
- Accommodate the workload’s data sizes without overflow.

Thus, a global memory is necessary for this workload in ZigZag. Lower memory levels alone cannot fully substitute for DRAM given the framework’s design and the workload’s requirements.

**Answer**: No, ZigZag cannot run this workload on the given architecture without a DRAM. A global memory is required to source external operands and manage data flow, as lower memory levels like `inputMem_16KB` and `sram_16KB` cannot fully handle initial inputs and inter-layer communication due to size constraints and ZigZag’s hierarchy assumptions.

To validate whether a global memory (DRAM) is mandatory, you can modify your architecture and ZigZag configuration as follows:

### Key Issue & Solution
**Problem:** ZigZag requires a top-level memory hierarchy entry point for each operand. Your current setup lacks this, causing the `AssertionError`.

**Workaround:** Designate one of your existing memories as "top-level" by:
1. Adding write ports to `inputMem_16KB` (to accept initial data)
2. Configuring memory ports to act as terminal endpoints (no higher-level memory)

### Modified Architecture (No DRAM)
```yaml
name: 8_NPE_Accelerator

operational_array:
  unit_energy: 0.04
  unit_area: 1
  dimensions: [D1]
  sizes: [8]

memories:
  inputMem_16KB:
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 1
    w_port: 1  # Added write port
    latency: 1
    operands: [I1]
    ports:
      - fh: w_port_1  # Mark as terminal (no higher-level write)
        tl: r_port_1
    served_dimensions: [D1]

  sram_16KB:
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 2
    w_port: 2
    latency: 1
    operands: [I2, O]
    ports:
      - fh: w_port_1  # Terminal for writes
        tl: r_port_1  # Read to compute
      - fh: w_port_2  # Terminal for O writes
        tl: r_port_2  # Read to next layer
    served_dimensions: [D1]
```

### Critical Changes
1. **Write Ports:** Added `w_port:1` to `inputMem_16KB` to accept initial data loads
2. **Terminal Ports:** Used `fh` (from higher) designations without actual higher memories
3. **Operand Isolation:** Explicit terminal ports prevent looking for non-existent higher memories

### API Configuration
```python
result = get_hardware_performance_zigzag(
    workload="your_workload.yaml",
    accelerator="modified_arch.yaml",
    mapping="your_mapping.yaml",
    exploit_data_locality=True,  # Enable inter-layer data reuse
    opt="latency",
    dump_folder="outputs/no_dram_test",
)
```

### Why This Works
- **Terminal Ports:** `fh:w_port_*` marks these as endpoints for data loading
- **Data Locality:** `exploit_data_locality=True` keeps data in SRAM between layers
- **Workload Design:** Your workload uses sequential layers where output of layer N is input to N+1

### Validation Approach
1. **Check Layer 0:** Verify `I1` is loaded directly into `inputMem_16KB` (no prior source)
2. **Intermediate Layers:** Confirm `O` from layer 0 stays in `sram_16KB` as `I` for layer 1
3. **Final Layer:** Ensure `O` remains in `sram_16KB` without needing DRAM offload

### Limitations
- **Initial Data Loading:** Assumes weights/inputs are pre-loaded into SRAM
- **Model Capacity:** 16KB SRAM must hold all intermediate tensors (verify sizes):
  - Layer 0 Output: 16 channels × 30×30 = 14,400 elements (16-bit = 28.8KB) ❌
  
**Outcome:** This configuration will fail due to insufficient SRAM capacity, proving that:
1. DRAM-less operation is _technically possible_ architecturally
2. Practical implementations require either:
   - Larger on-chip buffers (not 16KB)
   - External memory (DRAM) for realistic workloads

This demonstrates ZigZag's ability to model DRAM-less systems while revealing practical constraints through memory capacity analysis.


#### Question 4:
Now, As in zigzag I am trying to emulate an architecture like SENECA which is an event driven architecture where the inputs are streamed from the NOC and the inter - core communications are also handeled by it, now If I want to get such behaviour on Zigzag how do you think I need to modify my current architecture further?

Here are the specific details about the SENECA architecture

Based on the paper, here are the details on NPEs, Memory Architecture and Interconnectivity used in the SENECA chip:

**Neural Processing Elements (NPEs)**

* The SENECA core includes an array of neuron processing elements (NPEs) that act as physical neurons. [cite: 137]
   
* Each NPE contains a small register-based memory and executes a category of instructions. [cite: 138]
   
* An array of NPEs forms a SIMD (Single Instruction Multiple Data) type architecture. [cite: 139]
   
* Instructions to be executed in NPEs come from the Loop Buffer. [cite: 140]
   
* NPEs can get their data from Data Memory (through a wide Data Memory port), RISC-V (by directly writing into their register file), and the Loop controller (broadcasting). [cite: 141]
   
* The register file inside the NPEs allows for reusing data as much as possible before reading/writing it into the Data Memory. [cite: 142]
   
* The SENECA core is synthesized with 8 NPEs. [cite: 143, 8]

**Memory Architecture**

SENECA employs a hierarchical memory structure with the flexibility to dynamically allocate different parameters to various memory blocks. [cite: 627, 628] The memory blocks used in SENECA include:

* **Register-File**: Located inside NPEs. [cite: 636]
   
   * Two sizes are used: 16W x 16b (256b) and 64W x 16b (1kb). [cite: 636]
   * Energy consumption: 8 pJ/b for 256b and 12 pJ/b for 1kb. [cite: 636]
   * Static power: 600 pW/b for both sizes. [cite: 636]
   * Area: 3.6 um²/b for both sizes. [cite: 636]
   * Latency: <1 ns for both sizes. [cite: 636]
* **SRAM Block (Instruction/Data Memory)**: 8KW x 32b (256Kb). [cite: 636]
   
   * Energy consumption: 180 pJ/b (Read), 220 pJ/b (Write). [cite: 636]
   * Static power: 10 pW/b. [cite: 636]
   * Area: 0.2 um²/b. [cite: 636]
   * Latency: 2 ns. [cite: 636]
* **STT-MRAM (Shared Memory)**: 256k x 144b (36.8Mb). [cite: 636]
   
   * Energy consumption: 2,000 pJ/b (Read). [cite: 636]
   * Static power: 0 pW/b. [cite: 636]
   * Area: 0.1 um²/b. [cite: 636]
   * Latency: 25 ns (Read). [cite: 636]
* **HBM (Shared Memory)**: 64 Gb. [cite: 636]
   
   * Energy consumption: 7000 pJ/b. [cite: 636]
   * Area: 0.003 um²/b. [cite: 636]
   * Latency: 135 ns. [cite: 636]

**Interconnectivity**

* SENECA uses a packet-switched Network on Chip (NoC) to connect the neuro-synaptic cores. [cite: 520]
   
* The packet-switched NoC is chosen for its higher performance and flexibility. [cite: 678]
   
* SENECA also supports a lightweight multi-casting NoC with a small routing table. [cite: 191, 56]


More details about the memory architecture

Here's a breakdown of SENECA's memory architecture and the rationale behind using each memory block:

SENECA employs a hierarchical memory structure to balance the trade-offs between speed, power consumption, area efficiency, and flexibility[cite: 594, 595, 596, 597]. This architecture allows for efficient data reuse and optimization of application mapping for energy and area efficiency[cite: 595, 596, 597]. The key memory blocks include:

* **Register File**:
   
   * Located inside the Neural Processing Elements (NPEs)[cite: 604].
   * Used for storing and reusing data within the NPEs before it is written to the Data Memory[cite: 198, 199, 200, 555, 556, 557, 558, 615].
   * Two sizes: 16W x 16b (256b) and 64W x 16b (1kb)[cite: 604].
   * Fastest memory with latency <1 ns[cite: 604].
   * Larger area and higher static power consumption compared to SRAM[cite: 582, 583, 584, 585, 604].
* **SRAM Block (Instruction/Data Memory)**:
   
   * Used for storing instructions and data[cite: 604, 615].
   * Capacity: 8KW x 32b (256Kb)[cite: 604].
   * Offers a balance of speed, area, and power consumption[cite: 585, 586, 604].
* **STT-MRAM (Shared Memory)**:
   
   * A non-volatile memory used as a shared memory block to extend local Data Memory[cite: 272, 273, 604, 615].
   * Larger capacity: 256k x 144b (36.8Mb)[cite: 604].
   * Used when local data memories are not enough to store the parameters[cite: 274, 275].
   * Allows for powering off volatile memories to reduce leakage power during low activity times[cite: 275].
   * Non-volatile nature helps in reducing static power consumption[cite: 593, 594, 599, 600, 604].
   * Higher latency and write power consumption compared to SRAM[cite: 593, 604].
* **HBM (Shared Memory)**:
   
   * High Bandwidth Memory used as a shared memory[cite: 604].
   * Very large capacity: 64 Gb[cite: 604].
   * Used for storing a large number of synaptic weights[cite: 587, 588, 589, 590].
   * Improves area efficiency and cost[cite: 273, 589].
   * Highest latency among all memory types[cite: 604].

I have also attached the SENECA paper. Can you take the information that I provided along with the details from the SENECA and suggest me the modification that can be done to the current architecture. That will fit my current needs of haivng to run models like Lenet-5, Resnet-18 and also Tiny YOLO.

For now the archtecture can be simple in terms of the memory size by the layout should be finalised and later on only the minor tweaks would be required in the sizes. 

With this information let's do some modifications

#### Answer

The architecture can be modified with multiple levels instead of just one flat level to achieve a single layer and multi-layer running in parallel with changing to multi core to run the models


#### Question 6:

How does this feedback from the zigzag team effect the modeling of the neural network

I asked the following question
"Hi, thank you for your kind words and interest. Sorry for getting back later.

Input-stationary dataflows with individual memories sharing inputs diagonally across PEs is currently not possible. The architecture is restricted to an individual memory sharing data across one or more dimensions, and these dimensions are always orthogonally-defined.

This can however be solved by defining a single memory level which feeds inputs to all the PEs (with a scaled bandwidth according tot he number of inputs needed per cycle). This should be correct for latency. Energy-wise it depends on how you define the read/write cost of the shared memory, and might need some postprocessing to adapt it to the individual-memory case."

I recieved the feedback as following
"Input-stationary dataflows with individual memories sharing inputs diagonally across PEs is currently not possible. The architecture is restricted to an individual memory sharing data across one or more dimensions, and these dimensions are always orthogonally-defined.

This can however be solved by defining a single memory level which feeds inputs to all the PEs (with a scaled bandwidth according tot he number of inputs needed per cycle). This should be correct for latency. Energy-wise it depends on how you define the read/write cost of the shared memory, and might need some postprocessing to adapt it to the individual-memory case."


#### Answer 6:

Let’s explore how the ZigZag team’s feedback impacts the modeling of your neural network within a SENECA-like architecture. The feedback points out a limitation in ZigZag’s ability to handle certain memory-sharing patterns and suggests a workaround. I’ll break this down step-by-step to explain its effects on your model, focusing on clarity and practical adjustments.

---

### Understanding the ZigZag Team’s Feedback

The ZigZag team highlighted two key points:

1. **Limitation in Memory Sharing**:
   - ZigZag cannot model *input-stationary dataflows* where individual memories share inputs *diagonally* across processing elements (PEs).
   - Instead, ZigZag restricts memory sharing to *orthogonal dimensions* (e.g., along rows or columns in a grid), not diagonal patterns.

2. **Suggested Workaround**:
   - Use a *single memory level* that supplies inputs to all PEs.
   - Scale the memory’s *bandwidth* to match the total input demand per cycle (e.g., number of PEs × inputs per PE).
   - This approach should accurately model *latency*.
   - For *energy*, you’ll need to tweak the read/write costs of this shared memory, possibly with post-processing, to reflect the behavior of individual memories.

**What This Means for Your Model**: If your neural network’s architecture (inspired by SENECA) relies on PEs sharing inputs diagonally—say, each PE accessing data from a diagonal neighbor—ZigZag can’t represent that directly. Instead, you’ll need to adapt your model to fit ZigZag’s constraints using the workaround.

---

### Your Original SENECA-Like Model

Let’s assume your model resembles a typical SENECA-like setup:

- **Processing Elements (PEs)**: 8 PEs, each with small local register files.
- **Memory Hierarchy**:
  - *Level 0*: Registers inside each PE (e.g., 256 bits per PE).
  - *Level 1*: SRAM (e.g., 256Kb) local to the core or shared in some pattern.
  - *Level 2*: DRAM (e.g., 1GB) as the top-level memory.
- **Dataflow**: Input-stationary, meaning inputs are held steady in memory while weights or outputs move. Inputs flow from DRAM to SRAM to registers, possibly with inter-PE communication.

If you intended for the SRAM or registers to share inputs diagonally across PEs, the feedback directly affects how you configure this memory hierarchy in ZigZag.

---

### How the Feedback Affects Your Model

Here’s how the limitation and workaround reshape your modeling approach:

#### 1. Memory Sharing Restriction
- **Diagonal Sharing Not Possible**: If your neural network depends on diagonal input sharing (e.g., PE1 shares with PE2, PE2 with PE3 along a diagonal), ZigZag can’t model that natively. It only supports sharing along defined, orthogonal dimensions (e.g., all PEs in a row or column).
- **Impact**: You can’t assign individual memories to PEs with diagonal connections. This might oversimplify or misrepresent your intended dataflow.

#### 2. Adopting the Single Shared Memory Workaround
- **New Approach**: Replace individual memories (or diagonally shared ones) with one shared memory—likely the SRAM level—that feeds inputs to all PEs.
- **Bandwidth Scaling**: If each PE needs, say, 16 bits of input per cycle, and you have 8 PEs, the shared memory needs a read bandwidth of at least 128 bits/cycle (8 × 16) to supply all PEs simultaneously.
- **Latency**: This setup should correctly capture latency, as the single memory’s access time will reflect the parallel input delivery to all PEs.

#### 3. Energy Modeling Challenges
- **Shared vs. Individual Memories**: A single shared SRAM might consume more energy per access than smaller, individual memories due to larger size or higher port counts. ZigZag will calculate energy based on the shared memory’s read/write costs, which may not match your design’s reality.
- **Adjustment Needed**: You’ll need to:
  - Set the shared memory’s read/write costs to approximate the average cost of individual memories in your architecture.
  - Or, run ZigZag with the shared memory setup and then post-process the energy results to scale them down, aligning with the efficiency of individual memories.

#### 4. Practical Implications for Neural Network Performance
- **Simplified Dataflow**: Using a single shared memory eliminates the complexity of diagonal sharing, which might affect how accurately ZigZag models your neural network’s parallelism or data reuse.
- **Scalability**: For small networks, this workaround might suffice. For larger ones (e.g., ResNet-18), the shared SRAM’s size (e.g., 256Kb) might be too small, forcing reliance on DRAM and potentially altering latency and energy profiles.

---

### Adjusting Your Model in ZigZag

Here’s how to modify your SENECA-like model based on the feedback:

#### Original Configuration (Hypothetical)
Imagine your initial setup looked like this:
```yaml
memories:
  registers:
    size: 256        # bits per PE
    r_bw: 16
    w_bw: 16
    operands: [I1, I2, O]
    served_dimensions: [D1]  # 8 PEs along dimension D1

  sram_individual:
    size: 32768      # 32Kb per PE or group, hypothetically
    r_bw: 16
    w_bw: 16
    operands: [I1, I2, O]
    served_dimensions: [D1]  # Attempted diagonal sharing not supported

  dram:
    size: 1000000000 # 1GB
    r_bw: 64
    w_bw: 64
    operands: [I1, I2, O]
```

#### Adjusted Configuration
Now, apply the workaround:
```yaml
name: SENECA_Like_Adjusted

operational_array:
  dimensions: [D1]
  sizes: [8]         # 8 PEs

memories:
  # Level 0: Registers (unchanged, local to each PE)
  registers:
    size: 256        # bits per PE
    r_bw: 16
    w_bw: 16
    r_cost: 0.01
    w_cost: 0.01
    r_port: 1
    w_port: 1
    latency: 0
    operands: [I1, I2, O]
    served_dimensions: [D1]

  # Level 1: Shared SRAM for all PEs
  sram_shared:
    size: 262144     # 256Kb total
    r_bw: 128        # 8 PEs × 16 bits/cycle
    w_bw: 128
    r_cost: 10.5     # Adjusted to approximate shared access
    w_cost: 12.8
    r_port: 8        # Multiple ports for parallel reads
    w_port: 2
    latency: 1
    operands: [I1, I2, O]
    served_dimensions: [D1]

  # Level 2: DRAM (unchanged)
  dram:
    size: 1000000000 # 1GB
    r_bw: 64
    w_bw: 64
    r_cost: 700
    w_cost: 750
    r_port: 1
    w_port: 1
    latency: 100
    operands: [I1, I2, O]
    served_dimensions: [D1]
```

**Key Changes**:
- **SRAM**: Now a single shared memory with scaled bandwidth (128 bits/cycle) to feed all 8 PEs.
- **Ports**: Increased read ports (e.g., 8) to support parallel access, though you could use a single high-bandwidth port depending on ZigZag’s flexibility.
- **Energy Costs**: Set to reasonable values for shared access; you may need to refine these or post-process based on your architecture’s specifics.

---

### Effects on Neural Network Modeling

1. **Accuracy of Dataflow**:
   - The shared SRAM simplifies your input-stationary dataflow, potentially losing some fidelity if diagonal sharing was critical to your design’s efficiency.
   - However, it still captures the parallelism across PEs, assuming bandwidth meets demand.

2. **Latency**:
   - The workaround should model latency correctly, as all PEs access inputs from the shared SRAM simultaneously, with latency tied to the memory’s access time (e.g., 1 cycle).

3. **Energy**:
   - Energy estimates might be higher due to the shared memory’s characteristics. Post-processing can adjust this by scaling energy costs to match individual memory behavior if needed.

4. **Scalability**:
   - For small neural networks, this model works fine. For larger ones, the 256Kb SRAM might not hold all required inputs, necessitating DRAM access and altering performance metrics.

---

### Conclusion

The ZigZag team’s feedback forces a shift from diagonal memory sharing to a single shared memory with scaled bandwidth. This affects your SENECA-like model by simplifying the memory hierarchy, potentially impacting dataflow accuracy and energy estimates. You can implement the workaround as shown, ensuring sufficient bandwidth (e.g., 128 bits/cycle for 8 PEs) and tweaking energy costs for realism. For larger neural networks, DRAM remains essential, but the adjusted model should still provide a reasonable approximation within ZigZag’s constraints. If you need further refinements, let me know!

So, the direction of how the modeling done till now is good enough and needn't make it more complicated

#### Question 7:

Now, WIth the above achitecture that you shared
name: SENECA_Like_Adjusted

operational_array:
  dimensions: [D1]
  sizes: [8]         # 8 PEs

memories:
  # Level 0: Registers (unchanged, local to each PE)
  registers:
    size: 256        # bits per PE
    r_bw: 16
    w_bw: 16
    r_cost: 0.01
    w_cost: 0.01
    r_port: 1
    w_port: 1
    latency: 0
    operands: [I1, I2, O]
    served_dimensions: [D1]

  # Level 1: Shared SRAM for all PEs
  sram_shared:
    size: 262144     # 256Kb total
    r_bw: 128        # 8 PEs × 16 bits/cycle
    w_bw: 128
    r_cost: 10.5     # Adjusted to approximate shared access
    w_cost: 12.8
    r_port: 8        # Multiple ports for parallel reads
    w_port: 2
    latency: 1
    operands: [I1, I2, O]
    served_dimensions: [D1]

  # Level 2: DRAM (unchanged)
  dram:
    size: 1000000000 # 1GB
    r_bw: 64
    w_bw: 64
    r_cost: 700
    w_cost: 750
    r_port: 1
    w_port: 1
    latency: 100
    operands: [I1, I2, O]
    served_dimensions: [D1]
```
```
Compared to my initial architecture
"name: 8_NPE_Accelerator

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1]
  sizes: [8]

memories:
  # dram:
  #   size: 10000000000
  #   r_bw: 64
  #   w_bw: 64
  #   r_cost: 700
  #   w_cost: 750
  #   area: 0
  #   r_port: 3
  #   w_port: 3
  #   rw_port: 0
  #   latency: 1
  #   operands: [I1, I2, O]
  #   ports:
  #     - tl: r_port_1
  #       fl: w_port_1
  #     - tl: r_port_2
  #       fl: w_port_2
  #     - tl: r_port_3
  #       fl: w_port_3
  #   served_dimensions: [D1]


  inputMem_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128 
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 1
    operands: [I1]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]

  sram_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 2
    w_port: 2
    rw_port: 0
    latency: 1
    operands: [I2, O]
    ports:
      - tl: r_port_1
        fl: w_port_1
      - tl: r_port_2
        fh: w_port_2
    served_dimensions: [D1]"
    ```

#### Answer 7:

Now the above question was about how the input data can be transferred from directly from the DRAM to the RF and get the solution. The following modifications are made as shown below question which achieves our goal


#### Question 8:

Here is the new modified accelerator with three stages of memory levels and different latencies and show be enough even when the multiple layers are passed sequenntially

```
name: 8_NPE_Accelerator

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1]
  sizes: [8]

memories:
  dram_inputs_outputs_1GB: # defined in bits (size: 1024*1024*1204*8)
    size: 8589934592 
    r_bw: 64
    w_bw: 64
    r_cost: 700
    w_cost: 750
    area: 0
    r_port: 2
    w_port: 1
    rw_port: 0
    latency: 100
    operands: [I1, O]
    ports:
      - tl: r_port_1
      - tl: r_port_2
        fl: w_port_1
    served_dimensions: [D1]

  sram_16KB: # define in bits (size: 16x1024x8)
    size: 131072
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 2
    w_port: 2
    rw_port: 0
    latency: 1
    operands: [I2, O]
    ports:
      - tl: r_port_1
        fh: w_port_1
      - tl: r_port_2
        fl: w_port_2
        th: r_port_3
      
    served_dimensions: [D1]

  inputRF_2B: # define in bits (size: 2*8)
    size: 16
    r_bw: 128 
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 1
    operands: [I1]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]
```

In the above architecture design, (From lower to higher), one 2Bytes RF that stores only the inputs as we are doing an event driven processing which recieves the inputs directly from the DRAM. The second level is SRAM which stores the weights and the partial sums the weighs are directly coming from the higher memory level and are streamed  from there to the SRAM where there are accessed directly by the PE units and the partial sums after computations are written in the SRAM. finally all the complete outputs are stored in the DRAM, which can be used for the next layer during computation and also can be passed on to the next core during inter core communications. The third level is the DRAM where the inputs and the outputs are present where the inputs are streamed and the outputs are passed on to the next layer.


How do you think this architecture is now.



