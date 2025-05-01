The ZigZag cost model assesses how a neural network layer executes on a specific accelerator core, considering both **spatial mapping** (how operations are distributed across processing elements) and **temporal mapping** (how operations are scheduled over time). It’s particularly useful for understanding energy and performance trade-offs in hardware accelerators, especially those with complex memory hierarchies.

### Key Inputs
- **Accelerator**: Defines the hardware, including the core, memory hierarchy, and bandwidth constraints.
- **Layer**: The neural network layer (e.g., a convolutional layer) to evaluate.
- **Spatial Mapping**: Describes how the layer’s operations are unrolled spatially across the accelerator’s processing elements.
- **Temporal Mapping**: Specifies the timing of operations and data movements.
- **Access Same Data Considered as No Access** (optional): A flag to optimize access counts by ignoring redundant data fetches.

### Key Outputs
- **Memory Utilization**: How efficiently memory levels are used.
- **Memory Word Accesses**: The number of read/write operations per memory level.
- **Energy Consumption**: Total energy, split into operational (MAC) and memory components.
- **Latency**: Execution time, including data loading, computation, and offloading phases.
- **MAC Utilization**: Efficiency of multiply-accumulate operations.

---

## The `CostModelEvaluation` Class

The `CostModelEvaluation` class encapsulates the ZigZag cost model algorithm. It initializes with the inputs, sets up necessary data structures, and runs the evaluation. Let’s break down its functionality.

### Initialization (`__init__`)

The constructor sets up the evaluation environment:

1. **Input Storage**:
   - Stores the accelerator, layer, spatial mapping (both fractional and integer forms), temporal mapping, and the access flag.
   - The integer spatial mapping (`spatial_mapping_int`) is used for latency calculations, while the fractional version aids energy computations.

2. **Core and Memory Setup**:
   - Identifies the core (`core_id`) from the layer’s allocation.
   - Extracts memory hierarchy details: levels, sizes, bandwidths (read/write, minimal/maximal), and sharing information.

3. **Mapping Object Creation**:
   - Creates two `Mapping` objects:
     - `self.mapping`: Uses fractional spatial mapping for energy calculations.
     - `self.mapping_int`: Uses integer spatial mapping for latency calculations.
   - These objects compute access patterns and data movements based on the mappings.

4. **Run the Evaluation**:
   - Calls `self.run()` to compute all metrics immediately after initialization.

### Running the Cost Model (`run`)

The `run` method orchestrates the evaluation by calling four key methods in sequence:

```python
def run(self):
    self.calc_memory_utilization()
    self.calc_memory_word_access()
    self.calc_energy()
    self.calc_latency()
```

Let’s dive into each.

---

## Step-by-Step Algorithm Breakdown

### 1. Memory Utilization (`calc_memory_utilization`)

**Purpose**: Assess how much of each memory level’s capacity is used by the layer’s operands (e.g., weights, inputs, outputs).

**Steps**:
- **Individual Utilization**: For each operand and memory level:
  - Compute utilization as the ratio of data size (in bits) to memory capacity.
  - Ensure utilization ≤ 1, raising an assertion error otherwise.
  - Calculate an "effective" utilization, excluding replicated data in parallel memories.
- **Shared Utilization**: Adjust for memory sharing:
  - Sum the individual utilizations of operands sharing a memory level.
  - Verify the total shared utilization ≤ 1.
- **Storage**: Save both individual and shared utilizations (effective and total) as attributes.

**Output**: Dictionaries like `self.mem_utili_shared` and `self.effective_mem_utili_individual`.

---

### 2. Memory Word Accesses (`calc_memory_word_access`)

**Purpose**: Calculate the number of memory read/write operations per operand and level, considering bandwidth and data movement patterns.

**Steps**:
- For each operand and memory level, compute four types of data movements:
  - **Write In by Low (`wr_in_by_low`)**: Data written from a lower level.
  - **Read Out to Low (`rd_out_to_low`)**: Data read to a lower level.
  - **Read Out to High (`rd_out_to_high`)**: Data read to a higher level.
  - **Write In by High (`wr_in_by_high`)**: Data written from a higher level.
- **Calculation**:
  - Use data movement amounts and periods from the `Mapping` object.
  - Adjust for bandwidth constraints (max and min bandwidths) and data precision.
  - Compute cycles per period, then scale by period count and spatial unrolling units.
  - Recent updates (June 2023) use minimal bandwidth (`min_bw`) for finer-grained access calculations.
- **Output**: Store results in `self.memory_word_access` as a dictionary of `FourWayDataMoving` objects.

---

### 3. Energy Consumption (`calc_energy`)

**Purpose**: Compute the total energy, split into MAC (operational) and memory components.

#### MAC Energy (`calc_MAC_energy_cost`)
- **Method**: Multiply the core’s single MAC energy cost by the layer’s total MAC count.
- **Output**: `self.MAC_energy`.

#### Memory Energy (`calc_memory_energy_cost`)
- **Steps**:
  - For each operand and memory level:
    - Retrieve read/write energy costs from the memory hierarchy.
    - Scale memory accesses (from `self.memory_word_access`) by these costs.
    - Compute four-way energy costs (to/from high/low) and sum them.
  - Aggregate total memory energy and add to MAC energy.
- **Outputs**:
  - `self.mem_energy_breakdown`: Energy per operand per level.
  - `self.mem_energy_breakdown_further`: Detailed four-way breakdown.
  - `self.energy_total`: Sum of `self.mem_energy` and `self.MAC_energy`.

---

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

---

## How It Works: A Summary

1. **Initialization**: Sets up the core, memory hierarchy, and mappings.
2. **Memory Utilization**: Evaluates memory usage, considering sharing and replication.
3. **Memory Accesses**: Quantifies data movements, adjusted for bandwidth.
4. **Energy**: Computes operational and memory energy costs.
5. **Latency**: Integrates computation, stalls, and data transfer times, leveraging double buffering where possible.

The algorithm excels at detailed modeling of memory hierarchies, port contention, and data movement, making it ideal for optimizing batch-oriented accelerators.

---

## Key Features in the Code

- **Memory Detail**: Handles sharing, bandwidth limits, and port contention.
- **Double Buffering**: Dynamically optimizes data transfer overlap.
- **Granular Outputs**: Breaks down energy and latency into components.
- **Flexibility**: Supports both fractional (energy) and integer (latency) mappings.

This implementation provides a robust tool for accelerator design, balancing complexity with actionable insights into performance and efficiency.
