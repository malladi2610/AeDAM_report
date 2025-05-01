#### Question1:

Using the above generated table as reference and the definition of the zigzag latency cost model as shown below which include the following costs

### 4. Latency (`calc_latency`)

**Purpose**: Calculate the total execution time, including computation and data movement phases.

**Sub-Steps**:

#### a. Double Buffering Check (`calc_double_buffer_flag`)
- **Logic**: Determine if double buffering (overlapping data transfer and computation) is possible:
  - If effective shared utilization ≤ 50%, or individual utilization fits in free space, enable double buffering.
  - Update shared utilizations when one operand triggers double buffering.
- **Output**: `self.double_buffer_true`, a dictionary of boolean flags per operand/level.

#### b. Allowed and Real Data Transfer Cycles (`calc_allowed_and_real_data_transfer_cycle_per_DTL`)
- **Allowed Cycles**: Based on double buffering:
  - Full period if double buffering is true; otherwise, a fraction of the period.
- **Real Cycles**: Compute actual cycles needed per data movement, based on data amount, precision, and bandwidth.
- **Output**: `self.allowed_mem_updat_cycle` and `self.real_data_trans_cycle`.

#### c. Combine Data Transfer Rates (`combine_data_transfer_rate_per_physical_port`)
- **Port Activity**: Collect real and allowed cycles per port, considering sharing.
- **Stall/Slack (SS)**: Combine stalls or slacks per port, calculating a maximum SS value.
- **Output**: `self.SS_comb`, the maximum stall/slack across ports.

#### d. Data Loading/Offloading Latency (`calc_data_loading_offloading_latency`)
- **Loading**: Compute initial data loading cycles for input operands, handling shared/separate ports.
- **Offloading**: Compute final output offloading cycles (currently worst-case serial assumption).
- **Output**: `self.data_loading_cycle` and `self.data_offloading_cycle`.

#### e. Overall Latency (`calc_overall_latency`)
- **Components**:
  - **Ideal Cycle**: MAC count divided by processing units.
  - **Temporal Cycle**: Adjusted for spatial mapping.
  - **Total Latency**:
    - `latency_total0`: Computation + stalls.
    - `latency_total1`: Adds data loading.
    - `latency_total2`: Adds data offloading.
- **MAC Utilization**: Ratios of ideal cycles to total latencies.
- **Output**: `self.latency_total2`, `self.MAC_utilization2`, etc.

Can you help me calcalate the Latency_0 costs that include the computation costs and then the latency for data loading and data offloading by using this architecture configuration as shown below

"ame: Event_Driven_Accelerator

operational_array:
  unit_energy: 0.04  # for ANN MAC operations
  unit_area: 1
  dimensions: [D1]
  sizes: [8]  # 8 NPEs

memories:
  register:
    size: 64  # words, e.g., 64x16b for NPE register files
    r_bw: 128  # high bandwidth for fast access
    w_bw: 128
    r_cost: 0.1  # low energy cost, e.g., 8-12 pJ
    w_cost: 0.1
    area: 0.01  # small area per instance
    r_port: 4
    w_port: 3
    rw_port: 0
    latency: 1  # <1 ns (Has to been an integer as per zigzag modelling, So 1ns)
    operands: [I1,I2,O]
    ports:
      - tl: r_port_1
        fh: w_port_1
      - tl: r_port_2
        fh: w_port_2
      - th: r_port_3
        fh: w_port_3
        tl: r_port_4
    served_dimensions: []

  sram:
    size: 262144  # 256 KB, e.g., 2 Mb Data Memory
    r_bw: 128
    w_bw: 128
    r_cost: 10.5  # medium energy cost, e.g., 180-220 pJ
    w_cost: 12.8
    area: 1
    r_port: 3
    w_port: 3
    rw_port: 0
    latency: 2  # 2 ns
    operands: [I2, O]
    ports:
        - tl: r_port_1
          fh: w_port_1
        - fh: w_port_2
          tl: r_port_2
          fl: w_port_3
          th: r_port_3
    served_dimensions: [D1]

  shared_memory_Weights:
    size: 33554432  # 32 MB, e.g., for STT-MRAM
    r_bw: 64  # lower bandwidth
    w_bw: 64
    r_cost: 2000  # high energy cost, e.g., 2000 pJ
    w_cost: 2000
    area: 2
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 10  # higher latency, e.g., 2x SRAM
    operands: [I2]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]

  shared_memory_inputs_outputs:
    size: 33554432  # 32 MB, e.g., for STT-MRAM
    r_bw: 64  # lower bandwidth
    w_bw: 64
    r_cost: 2000  # high energy cost, e.g., 2000 pJ
    w_cost: 2000
    area: 2
    r_port: 2
    w_port: 1
    rw_port: 0
    latency: 10  # higher latency, e.g., 2x SRAM
    operands: [I1, O]
    ports:
      - tl: r_port_1
      - tl: r_port_2
        fl: w_port_1
    served_dimensions: [D1]"


Inputs -> Streamed one event at a time into the RF and resued for all it's computation and never seen again

They are streamed from shared_memory_inputs_outputs which has a latency of 10 cycles to RF where they stay until all their respective computations are performed

Weights -> Due to the smaller size right now they are stored in the SRAM along and are accessed again and again to RF to be used for the computation

Outputs -> Once the MAC operation is performed outputs are written to RF and then they are also streamed to the Shared_input_outputs 

Here I am trying to model the Seneca Event driven accelerator and performing event driven computation latency estimation using Zigzag 

THe script that you will be building will be a validataion script that I will use to compare the results with zigzag cost model how it should have behaved for an event driven architecture.

Your task is to read the above table you generated one input at a time, then perform the cost analysis as if each input is accessed, weight is access, output is calcaulated and then stremed to the Shared_input_output memory the above table clearly indicated which input  contributes to with output which is the important table for event driven cost estimation 