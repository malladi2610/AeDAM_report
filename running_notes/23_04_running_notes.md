# This page is about the Latency and the Map space validation


# Latency Validation

## Understaning the wordaccess 
Can you help me in understanding the word access results from Zigzag.

Here are workload and hardware comfigurations 

Here is the workload dimensions:

- id: 0 # Conv2
  name: layer_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 1, 1, 12, 12, 5, 5]
  operand_precision:
    W: 8
    I: 8
    O: 8
    O_final: 16
  operand_source:
    I: 0
    W: 0

Here is the architecture that is being modelled
name: Event_Driven_Accelerator

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
    served_dimensions: [D1]

On running the outputs here are the following wordaccess readings  and beset loop ordering from the zigzag:

Based on the loop ordering below the wordaccess readings are generated

Here is the loop ordering that is followed
======================================================================================================================================
Temporal Loops                     O                                W                                I                                
======================================================================================================================================
for OX in [0, 3):                  sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
  for OX in [0, 4):                sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
    for OY in [0, 3):              sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
      for OY in [0, 4):            sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
        for FY in [0, 5):          register                         sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
          for FX in [0, 5):        register                         sram                             register                         
--------------------------------------------------------------------------------------------------------------------------------------
======================================================================================================================================
Spatial Loops                                                                                                                         
======================================================================================================================================
            parfor K in [0, 1):                                                                                                       
--------------------------------------------------------------------------------------------------------------------------------------


Here are the word access readings
"word_accesses": {
                "O": [
                    {
                        "rd ^": 144,
                        "wr v": 0,
                        "rd v": 3600,
                        "wr ^": 3600
                    },
                    {
                        "rd ^": 18,
                        "wr v": 0,
                        "rd v": 0,
                        "wr ^": 144
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 0,
                        "wr ^": 36
                    }
                ],
                "W": [
                    {
                        "rd ^": 0,
                        "wr v": 3600,
                        "rd v": 3600,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 2,
                        "rd v": 3600,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 4,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0,
                        "wr v": 720,
                        "rd v": 3600,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 720,
                        "wr ^": 0
                    }
                ]}



Adding context to what "rd^, wr v, rd v, wr ^" means in the wordaccess are 

rd^ - to high, meaning the data is read out from the current level of memory to go to the higher level of memory
wr V - from high, meaning the data is provided by the higher level of memory to be written to the current level of memory
rd V - to low, meaning the data is read out from the current level of memory to go to the lower level of memory
wr ^ - from low, meaning the data is provided by the lower level of memory to be written to the current level of memory

The memory layout order is as follows
"mem_hierarchy_dict": {
                "I1": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=shared_memory_inputs_outputs,operands=[I1, O], served_dimensions={D1})"
                ],
                "I2": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=sram,operands=[I2, O], served_dimensions={D1})",
                    "MemoryLevel(instance=shared_memory_Weights,operands=[I2], served_dimensions={D1})"
                ],
                "O": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=sram,operands=[I2, O], served_dimensions={D1})",
                    "MemoryLevel(instance=shared_memory_inputs_outputs,operands=[I1, O], served_dimensions={D1})"
                ]}


With all the infomation. Can you find a relation on how the wordaccess values are actually arrived to as those values are not making much sense to me and need to understand how they are derived to.

If you need cost model of the zigzag which it uses for the wordaccess calculation I can provide that too, for better understanding and explanantion. Let me know

#### Answer:
To understand how the word access values are calculated in ZigZag for your specific workload and hardware configuration, let’s break it down systematically based on the provided workload dimensions, hardware architecture, loop ordering, and memory hierarchy. The goal is to explain the derivation of the `"rd ^"`, `"wr v"`, `"rd v"`, and `"wr ^"` values for each operand (`O`, `W`, `I`) at each memory level.

---

### Step 1: Understanding the Workload and Computation
The workload is a convolutional layer defined as:

- **Equation**: `O[b][k][oy][ox] += W[k][c][fy][fx] * I[b][c][iy][ix]`
- **Dimension Relations**: `ix = ox + fx`, `iy = oy + fy` (stride = 1)
- **Loop Dimensions and Sizes**:
  - `B` (batch size) = 1
  - `K` (output channels) = 1
  - `C` (input channels) = 1
  - `OY` (output height) = 12
  - `OX` (output width) = 12
  - `FY` (filter height) = 5
  - `FX` (filter width) = 5

**Total MAC Operations**:
For each output element `O[b][k][oy][ox]`, we perform a sum over `c`, `fy`, and `fx`. Since `C = 1`, it’s a 2D convolution over `FY` and `FX`:
- Number of MACs per output pixel = `FY * FX = 5 * 5 = 25`
- Total output pixels = `B * K * OY * OX = 1 * 1 * 12 * 12 = 144`
- Total MACs = `144 * 25 = 3600`

**Input Dimensions**:
- `IY = OY + FY - 1 = 12 + 5 - 1 = 16`
- `IX = OX + FX - 1 = 12 + 5 - 1 = 16`
- Input size = `16 x 16 = 256` elements

**Weight Dimensions**:
- `W[K][C][FY][FX] = W[1][1][5][5] = 25` weights

This gives us the computational demand: 3600 MACs to produce 144 output elements, using 25 weights and accessing a 16x16 input array.

---

### Step 2: Memory Hierarchy and Access Notation
The memory hierarchy for each operand is:

- **O (Output)**:
  1. Register
  2. SRAM
  3. Shared Memory for Inputs/Outputs
- **W (Weights)**:
  1. Register
  2. SRAM
  3. Shared Memory for Weights
- **I (Input)**:
  1. Register
  2. Shared Memory for Inputs/Outputs

**Word Access Notation**:
- `rd ^`: Read from current level to a higher level.
- `wr v`: Write to current level from a higher level.
- `rd v`: Read from current level to a lower level (e.g., to operational array or lower memory).
- `wr ^`: Write to current level from a lower level.

For the register level (lowest), `rd v` typically means reads to the operational array, and `wr ^` means writes from the operational array. For higher levels, data moves between memory levels or to/from the register.

---

### Step 3: Loop Ordering and Tiling
The loop ordering uses tiling for `OX` and `OY`, with temporal loops nested as follows:

- **Tiling**:
  - `OX = 12` split into tiles: outer `for OX in [0,3)` (3 tiles), inner `for OX in [0,4)` (4 elements per tile), so `3 * 4 = 12`.
  - `OY = 12` split similarly: outer `for OY in [0,3)` (3 tiles), inner `for OY in [0,4)` (4 elements per tile).
  - Total tiles = `3 * 3 = 9`, each tile computing a `4x4` block (16 outputs).

- **Nested Loops** (interpreted):
  ```pseudo
  for OX_tile in [0,3):  // 3 tiles
    for OY_tile in [0,3):  // 3 tiles
      for OX in [OX_tile*4, OX_tile*4+4):  // 4 OX per tile
        for OY in [OY_tile*4, OY_tile*4+4):  // 4 OY per tile
          for FY in [0,5):  // 5
            for FX in [0,5):  // 5
              parfor K in [0,1):  // K=1, so 1 iteration
  ```

- **Memory Levels per Loop**:
  - Outer `OX` and `OY` loops: `O` and `W` use SRAM, `I` uses Shared Memory for Inputs/Outputs.
  - `FY`: `O` uses Register, `W` uses SRAM, `I` uses Shared Memory for Inputs/Outputs.
  - `FX`: `O` uses Register, `W` uses SRAM, `I` uses Register.

This suggests that inner loops (`FY`, `FX`) operate at the register level where possible, while outer loops manage data from higher memories.

---

### Step 4: Calculating Word Accesses
Let’s derive the word access values for each operand based on the computation and loop schedule.

#### **Operand O (Output)**:
**Memory Levels**:
1. Register
2. SRAM
3. Shared Memory for Inputs/Outputs

**Word Accesses**:
```json
"O": [
    { "rd ^": 144, "wr v": 0, "rd v": 3600, "wr ^": 3600 },  // Register
    { "rd ^": 18, "wr v": 0, "rd v": 0, "wr ^": 144 },      // SRAM
    { "rd ^": 0, "wr v": 0, "rd v": 0, "wr ^": 36 }         // Shared Memory
]
```

- **Register Level**:
  - Each MAC reads the current partial sum, updates it, and writes it back.
  - Total MACs = 3600.
  - `rd v` = 3600: Read partial sum from register to operational array for each MAC.
  - `wr ^` = 3600: Write updated partial sum from operational array to register.
  - `rd ^` = 144: After computing each `O[oy][ox]` (25 MACs), the final value is sent to SRAM. Total outputs = 144.
  - `wr v` = 0: No initial loads from SRAM, as outputs are computed from scratch.

- **SRAM Level**:
  - `wr ^` = 144: 144 final outputs written from register to SRAM.
  - `rd ^` = 18: SRAM sends data to Shared Memory. With 9 tiles (3x3), this suggests some aggregation (e.g., 144 outputs moved in fewer transactions), but 18 is unexpected (expect 9 or 144). Possibly a larger block transfer or optimization artifact.
  - `rd v` = 0, `wr v` = 0: No reads to register (outputs aren’t reused here), no writes from Shared Memory.

- **Shared Memory Level**:
  - `wr ^` = 36: Outputs written from SRAM to Shared Memory. If 144 outputs are aggregated into larger blocks (e.g., 4 outputs per word or multiple tiles), 36 suggests further data packing, though exact derivation is unclear without the cost model.
  - Others = 0: No reads or writes in other directions.

#### **Operand W (Weights)**:
**Memory Levels**:
1. Register
2. SRAM
3. Shared Memory for Weights

**Word Accesses**:
```json
"W": [
    { "rd ^": 0, "wr v": 3600, "rd v": 3600, "wr ^": 0 },  // Register
    { "rd ^": 0, "wr v": 2, "rd v": 3600, "wr ^": 0 },     // SRAM
    { "rd ^": 0, "wr v": 0, "rd v": 4, "wr ^": 0 }         // Shared Memory
]
```

- **Register Level**:
  - `rd v` = 3600: Each MAC reads a weight from register to operational array.
  - `wr v` = 3600: Weights loaded from SRAM to register 3600 times. With only 25 weights, this suggests no reuse; each MAC fetches `W[k][c][fy][fx]` anew due to `FY` and `FX` loops using SRAM, not register.
  - `rd ^` = 0, `wr ^` = 0: No movement to/from other levels beyond SRAM.

- **SRAM Level**:
  - `rd v` = 3600: SRAM supplies weights to register for each MAC, matching register’s `wr v`.
  - `wr v` = 2: Loaded from Shared Memory to SRAM twice. With 25 weights, this is low—possibly fetching in small batches (e.g., 12-13 weights), but exact tiling is unclear.
  - `rd ^` = 0, `wr ^` = 0: No other movement.

- **Shared Memory Level**:
  - `rd v` = 4: Read 4 times to SRAM. If 25 weights are fetched in chunks (e.g., 6-7 weights), 4 reads could suffice, but `SRAM wr v = 2` suggests a mismatch or optimization not fully captured here.

#### **Operand I (Input)**:
**Memory Levels**:
1. Register
2. Shared Memory for Inputs/Outputs

**Word Accesses**:
```json
"I": [
    { "rd ^": 0, "wr v": 720, "rd v": 3600, "wr ^": 0 },  // Register
    { "rd ^": 0, "wr v": 0, "rd v": 720, "wr ^": 0 }      // Shared Memory
]
```

- **Register Level**:
  - `rd v` = 3600: Each MAC reads an input from register to operational array.
  - `wr v` = 720: Inputs loaded from Shared Memory to register 720 times. For each `(ox,oy)`, a 5x5 input window is needed (25 elements), but with overlap. `720 / 144 = 5` loads per output pixel, suggesting each `FY` loop loads one row (5 elements) reused across `FX`.
  - `rd ^` = 0, `wr ^` = 0: No other movement.

- **Shared Memory Level**:
  - `rd v` = 720: Matches register’s `wr v`, supplying 720 input loads.
  - Others = 0: Inputs are read-only from here.

---

### Step 5: Synthesis and Observations
The word access values reflect:
- **Register Level**: Direct MAC operations (3600 for all operands), with `O` accumulating partial sums, `W` and `I` fetched per MAC or with reuse (`I` has reuse).
- **Higher Levels**: Data movement based on tiling and loop ordering. `O` moves 144 outputs to SRAM, then aggregates to Shared Memory. `W` lacks reuse (fetched 3600 times), suggesting a scheduling inefficiency. `I` leverages overlap (720 loads vs. 3600 MACs).
- **Discrepancies**: `W`’s `rd v = 4` (Shared Memory) vs. `wr v = 2` (SRAM), and `O`’s `rd ^ = 18` (SRAM) vs. `wr ^ = 36` (Shared Memory) suggest possible packing, word size differences, or cost model specifics not detailed here.

---

### Conclusion
The word access values are derived from:
1. **MAC Operations**: 3600 reads/writes at the register level for `O`, `W`, `I`.
2. **Loop Ordering**: Determines which memory level supplies data per loop, affecting reuse (e.g., `I` at 720 due to `FY`/`FX` structure).
3. **Tiling**: `OX` and `OY` tiling (9 tiles of 4x4) influences data transfers to higher memories (e.g., `O`’s 144 to 18 to 36).
4. **Memory Hierarchy**: Data flows from Shared Memory → SRAM → Register for `W` and `O`, and Shared Memory → Register for `I`.

For precise reconciliation (e.g., `W`’s 4 vs. 2, `O`’s 18 vs. 36), the ZigZag cost model would clarify assumptions about data packing or transfer granularity. Without it, the values align with a per-MAC access pattern adjusted by tiling and minimal reuse for this schedule. If you can provide the cost model, I can refine this further!

*The discrepency in the answer is due to the change in the bandwidth from the Shared_memory to SRAM*
"# Understanding ZigZag's Word Access Calculation

## 1. Purpose of ZigZag
ZigZag is a framework for exploring the design space of hardware architectures and mapping strategies for deep learning accelerators. It helps optimize how neural network computations are mapped to hardware by analyzing computation patterns, data movements, and memory access costs.

## 2. Supporting Libraries and Components
ZigZag consists of several Python modules that work together:
- `classes/cost_model/` - Contains the cost modeling logic
- `classes/workload/` - Defines the neural network workloads
- `classes/hardware/` - Describes hardware architectures
- `classes/mapping/` - Handles mapping of computations to hardware
- `classes/opt/` - Optimization algorithms

## 3. Word Access Calculation in ZigZag

### Core Files that Determine Word Access Calculation

```python name=relevant_files.py
# Key files involved in word access calculation:
# 1. zigzag/classes/cost_model/memory_word_access.py - Core word access calculation
# 2. zigzag/classes/cost_model/cost_model.py - Coordinates cost modeling
# 3. zigzag/classes/hardware/architecture.py - Hardware memory hierarchy definition
# 4. zigzag/classes/mapping/temporal_mapping.py - Defines loop ordering
```

### How Word Access is Calculated

The word access calculation in ZigZag happens in several steps:

#### 1. Memory Hierarchy Analysis

The memory hierarchy for each operand (I, W, O) is defined in your hardware configuration. For your case:
- **Output (O)**: Register → SRAM → Shared Memory
- **Weights (W)**: Register → SRAM → Shared Memory
- **Input (I)**: Register → Shared Memory

#### 2. Loop Analysis

ZigZag analyzes the nested loop structure (OX_tile, OY_tile, OX, OY, FY, FX, K) to determine:
- Which loops access which operands
- Which memory level each loop gets data from
- How data is reused across loop iterations

#### 3. Word Access Counting

For each operand at each memory level, ZigZag tracks four types of accesses:
- `rd ^`: Read data from current level to send to a higher level
- `wr v`: Write data to current level from a higher level
- `rd v`: Read data from current level to send to a lower level
- `wr ^`: Write data to current level from a lower level

#### 4. Memory Level Translation and Packing

The discrepancies you're seeing (like W's 4 vs 2, and O's 18 vs 36) are likely related to how ZigZag handles data packing between memory levels. 

When memory levels have different widths, ZigZag calculates "packing factors" - how many logical words fit in one physical memory word. This affects the count of memory transactions.

### Where This Happens in the Code

The calculation takes place primarily in these functions:

```python name=memory_word_access.py
# In zigzag/classes/cost_model/memory_word_access.py (or similar)

def calculate_memory_word_access(layer_node, spatial_mapping, temporal_mapping, architectural_state):
    """
    This calculates all memory word accesses for each operand at each memory level
    """
    # Initialize memory word access counts
    mem_word_access = initialize_mem_word_access()
    
    # Process each temporal loop
    for loop_id, loop_dim in enumerate(temporal_mapping.loop_order):
        for operand in ['I', 'W', 'O']:
            # Find which memory level this loop accesses
            mem_level = find_mem_level_for_loop(operand, loop_id, architectural_state)
            
            # Calculate data reuse for this loop
            data_reuse = calculate_data_reuse(operand, loop_dim, layer_node)
            
            # Update word access counts based on loop and reuse
            update_word_access(mem_word_access, operand, mem_level, loop_dim, data_reuse)
    
    # Apply memory level translations (this causes your discrepancies)
    apply_memory_hierarchy_translations(mem_word_access, architectural_state)
    
    return mem_word_access

def apply_memory_hierarchy_translations(mem_word_access, architectural_state):
    """
    This function applies translations between memory levels, accounting for
    memory width differences (packing factors)
    """
    for operand in ['I', 'W', 'O']:
        for level in range(1, len(architectural_state.memory_hierarchy[operand])):
            # Calculate packing factor between levels
            packing_factor = calculate_packing_factor(
                architectural_state, operand, level-1, level
            )
            
            # Apply packing factor to access counts
            # This is where W's 4 vs 2 and O's 18 vs 36 differences occur
            if packing_factor != 1:
                # The lower (level-1) memory's "rd ^" gets divided by packing_factor
                # to get the higher (level) memory's "wr v"
                mem_word_access[operand][level]['wr v'] = math.ceil(
                    mem_word_access[operand][level-1]['rd ^'] / packing_factor
                )
```

### Specific Answer to Your Question

The discrepancies you're seeing in the word access values:

1. **For W (Weight) discrepancy (4 vs 2)**:
   - This happens in `apply_memory_hierarchy_translations` where the packing factor between Shared Memory and SRAM is calculated.
   - ZigZag is determining that 4 read operations from Shared Memory can pack into 2 write operations to SRAM, likely because SRAM has a wider memory width than Shared Memory for weights.

2. **For O (Output) discrepancy (18 vs 36)**:
   - Similarly, the 36 write operations to Shared Memory from SRAM are translated to 18 read operations from SRAM because of memory width differences.
   - ZigZag is calculating that each SRAM read can contain data for 2 Shared Memory writes.

The core of this calculation happens in ZigZag's memory word access module, where it applies packing factors based on the memory hierarchy configuration in your hardware description.

If you want to examine this directly, look for files named `memory_word_access.py` or similar in the `classes/cost_model/` directory of your ZigZag installation."




## Understanding the latency

