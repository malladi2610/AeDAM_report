# ABout

THis section is deals with the hardware cost estimation which is nothing but the Dimension 1 i.e building the Event driven flow which is achieved through zigzag

# Understanding ZigZag's Hardware Cost Estimation Flow

## 1. Purpose and Problem Statement

You want to understand how ZigZag calculates hardware costs (energy and latency) in neural network accelerators. Specifically, you're looking to extract the flow and algorithms that process map spaces into energy and latency estimations.

Your description is mostly correct. Here's the complete flow:

1. Create the required mapping space (spatial and temporal)
2. Generate specific mappings (spatial + temporal) from this space
3. Use `data_movement.py` to model data movement patterns through memory hierarchy
4. Apply `port_activity.py` to track how data moves through memory ports 
5. Finally, the `cost_model.py` calculates memory word accesses, energy consumption, and latency

## 2. Supporting and Dependent Libraries

The key components in this flow are:

1. **Data Movement Structures** (`data_movement.py`):
   - `FourWayDataMoving`: Models data flow in four directions (read/write to higher/lower memory levels)
   - `MemoryAccesses`: Tracks number of accesses
   - `AccessEnergy`: Tracks energy for memory accesses
   - `DataMovePattern`: Tracks access patterns for each unit memory

2. **Port Activity Trackers** (`port_activity.py`):
   - `PortActivity`: Tracks data transfer rates during computation
   - `PortBeginOrEndActivity`: Tracks data loading/offloading activities

3. **Cost Model** (`cost_model.py`):
   - `CostModelEvaluation`: Main class that calculates costs
   - `CumulativeCME`: Aggregates costs across multiple layers

4. **Architecture Models** (referenced in code):
   - `Accelerator`: Defines the hardware architecture
   - `MemoryLevel`, `MemoryPort`: Define memory hierarchy and ports

## 3. Detailed Algorithm and Formulas

### Step 1: Calculate Memory Utilization

```python
# In calc_memory_utilization()
mem_utilization = data_bits_at_level / memory_size
```

This determines how much of each memory level is used, which affects double buffering possibilities.

### Step 2: Calculate Memory Word Access Patterns

```python
# In calc_memory_word_access()
memory_access = ceil((data_amount * precision) / min_bw) * (min_bw / max_bw) * period_count * spatial_units
```

For each memory level and data direction, calculate how many memory accesses are needed.

### Step 3: Energy Calculation

Energy calculation is split into MAC energy and memory energy:

**MAC Energy:**
```python
# In calc_mac_energy_cost()
mac_energy = single_mac_energy * total_mac_count
```

**Memory Energy:**
```python
# In calc_memory_energy_cost()
# For each memory level and direction
energy_per_access = memory_level.read_energy or memory_level.write_energy
access_energy = number_of_accesses * energy_per_access

# Total memory energy
mem_energy = sum of all access_energy
```

**Total Energy:**
```python
energy_total = mac_energy + mem_energy
```

### Step 4: Latency Calculation

Latency calculation has multiple sub-steps:

**1. Determine Double Buffer Potential:**
```python
# In calc_double_buffer_flag()
if effective_mem_utilization <= 0.5:
    double_buffer_possible = True
```

**2. Calculate Real vs. Allowed Data Transfer Cycles:**
```python
# For real cycles (how long transfers actually take)
real_cycles = ceil(data_amount * precision / memory_bandwidth)

# For allowed cycles (how long we can take without stalling)
allowed_cycles depends on double buffering and which attribute to use
```

**3. Calculate Port Activity and Stall/Slack:**
```python
# In combine_data_transfer_rate_per_physical_port()
stall_slack = (real_cycle - allowed_cycle) * (period_count - 1)
```

**4. Calculate Data Loading/Offloading Delays:**
```python
# Combines the loading/offloading delays from all ports
data_onloading_cycle = calculated in calc_onloading_combined()
data_offloading_cycle = calculated in calc_offloading_combined()
```

**5. Calculate Overall Latency:**
```python
# In calc_overall_latency()
ideal_cycle = ceil(total_mac_count / total_unit_count) * cycles_per_op
ideal_temporal_cycle = temporal_mapping.total_cycle * cycles_per_op

# Three latency metrics:
latency_total0 = ideal_temporal_cycle + stall_slack_comb  # Computation only
latency_total1 = latency_total0 + data_onloading_cycle    # With loading
latency_total2 = latency_total1 + data_offloading_cycle   # With loading & offloading
```

### The Complete Flow

1. A mapping (spatial + temporal) is generated
2. For this mapping, calculate memory utilization
3. Calculate memory word accesses for all operands at all memory levels
4. Calculate MAC and memory energy
5. Calculate latency components:
   - Determine double buffering opportunities
   - Calculate real vs. allowed transfer cycles
   - Calculate stalls/slacks
   - Calculate loading/offloading cycles
   - Combine all components for final latency

This is the complete algorithm ZigZag uses to estimate hardware costs. The final output is energy (in Joules) and latency (in cycles), which are the primary metrics for evaluating accelerator efficiency.

# Here is the final path 

Mapspace → (For a single mapping) Path creation (in Mapping object) → Memory utilisation → Memory word access → (Split) 
    → (For latency) → Double buffer detection → Transfer cycle calculation → Port activity calculation → Latency calculation
    → (For energy) → MAC energy + Memory energy calculation


# Understanding ZigZag Hardware Cost Estimation Flow

## 1. Purpose and Problem

You want to validate the complete flow of hardware cost estimation in ZigZag, particularly how data paths through the memory hierarchy are determined and used to calculate energy and latency metrics.

Your proposed flow is:
```
Mapspace -> (For a single mapping) determine the path creation -> Memory utilisation -> Memory word access -> (Split) 
    -> (For latency) -> Port activity calculation -> Latency calculation
    -> (For energy) -> Energy calculation
```

Let me validate this flow by examining the provided files.

## 2. Supporting and Dependent Libraries

The files you've provided show the key components involved in hardware cost estimation:

- **data_movement.py**: Defines data structures for representing data movement patterns
  - Dependencies: `zigzag.datatypes`, `zigzag.hardware.architecture.memory_port`

- **port_activity.py**: Models memory port activity during computation
  - Dependencies: `zigzag.datatypes`, `zigzag.hardware.architecture.memory_port`, `zigzag.utils`

- **cost_model.py**: Main cost model implementation
  - Dependencies: `zigzag.mapping`, `zigzag.hardware.architecture`, multiple utilities

## 3. Detailed Flow Analysis

Looking at the code, I can confirm your general flow is correct, with some minor adjustments:

### Step 1: Map Space → Path Creation
```python
# In CostModelEvaluation.__init__
self.mapping = Mapping(self.accelerator, self.spatial_mapping, self.temporal_mapping, 
                       self.layer, self.access_same_data_considered_as_no_access)
```
This creates the `Mapping` object which establishes the data movement patterns (the "path"). This happens FIRST, even before memory utilization calculation.

### Step 2: Memory Utilization Calculation
```python
# In CostModelEvaluation.run
self.calc_memory_utilization()
```
Calculates how much of each memory level is used by each operand, considering memory sharing.

### Step 3: Memory Word Access Calculation
```python
# In CostModelEvaluation.run
self.calc_memory_word_access()
```
Calculates the number of memory accesses for each operand at each memory level based on the data movement patterns.

### Step 4: Split into Energy and Latency Calculations

#### Step 4a: Energy Calculation Path
```python
# In CostModelEvaluation.run
self.calc_energy()
```
Which further calls:
```python
self.calc_mac_energy_cost()  # Calculates MAC operation energy
self.calc_memory_energy_cost()  # Calculates memory access energy
```
The memory energy calculation uses the memory word access data directly.

#### Step 4b: Latency Calculation Path
```python
# In CostModelEvaluation.run
self.calc_latency()
```
Which goes through several steps:
```python
self.calc_double_buffer_flag()  # Determines double buffering possibilities
self.calc_allowed_and_real_data_transfer_cycle_per_data_transfer_link()  # Calculates transfer cycles
self.combine_data_transfer_rate_per_physical_port()  # Creates PortActivity objects
self.calc_data_loading_latency()  # Calculates data loading/offloading latency
self.calc_overall_latency()  # Combines all components for final latency
```

The port activity calculation happens in `combine_data_transfer_rate_per_physical_port()` where `PortActivity` objects are created for each memory port and used to calculate stalls/slacks.

## Validation Result

Your flow is accurate with just a couple of minor refinements:

```
Mapspace → (For a single mapping) Path creation (in Mapping object) → Memory utilisation → Memory word access → (Split) 
    → (For latency) → Double buffer detection → Transfer cycle calculation → Port activity calculation → Latency calculation
    → (For energy) → MAC energy + Memory energy calculation
```

The key insight is that the data path (data movement patterns) is established first in the Mapping object creation, then memory utilization and word access are calculated. After that, the flow splits into separate energy and latency calculations, with the latency path involving port activity modeling.

So yes, your claim about the complete flow is valid with these small clarifications.

# 1. Data movement

# Understanding data_movement.py in ZigZag Framework

## 1. Purpose of This Program

**Purpose:** The `data_movement.py` file defines data structures for modeling data movement in memory hierarchies of deep learning accelerators. It provides the fundamental building blocks for tracking how data flows between different memory levels.

**Problem Context:** In your event-driven exploration, you need to understand how data moves through the memory hierarchy when an event arrives and is processed. This file provides the core data structures that keep track of data movement patterns, including the number of accesses, energy consumption, and bandwidth requirements.

## 2. Supporting and Dependent Libraries

```python
from enum import StrEnum                                  # For creating string-based enumerations
from typing import Generic, TypeVar                       # For generic type support
from zigzag.datatypes import LayerOperand                # Defines layer operands (weights, inputs, outputs)
from zigzag.hardware.architecture.memory_port import DataDirection  # Defines data movement directions
```

## 3. Detailed Explanation of Each Component

### 3.1 `FourWayDataMoving` Class

This generic class represents data movement in four directions within a memory hierarchy.

**Key Components:**
- **Data Directions:**
  - `RD_OUT_TO_HIGH`: Reading data out to a higher memory level
  - `WR_IN_BY_HIGH`: Writing data in from a higher memory level
  - `RD_OUT_TO_LOW`: Reading data out to a lower memory level
  - `WR_IN_BY_LOW`: Writing data in from a lower memory level

**Methods:**
- `__init__`: Initializes with a dictionary mapping directions to values
- `get`: Retrieves the value for a specific direction
- `set`: Updates the value for a specific direction
- `__add__`: Adds two FourWayDataMoving objects element-wise
- `__mul__`: Multiplies all values by a scalar

**Algorithm for Addition:**
```python
def add(a, b):
    result = {}
    for direction in DataDirection:
        result[direction] = a[direction] + b[direction]
    return FourWayDataMoving(result)
```

**Algorithm for Multiplication:**
```python
def multiply(a, scalar):
    result = {}
    for direction in DataDirection:
        result[direction] = a[direction] * scalar
    return FourWayDataMoving(result)
```

### 3.2 `MemoryAccesses` Class

A specialized version of `FourWayDataMoving` that represents the number of memory accesses in each direction.

**Formula:**
The number of memory accesses is calculated in `cost_model.py` using:
```
memory_accesses = ceil((data_elem_move_per_period * data_precision) / min_bw) * 
                  (min_bw / max_bw) * 
                  total_period_count * 
                  spatial_units
```
Where:
- `data_elem_move_per_period`: Number of data elements moved per period
- `data_precision`: Bit precision of the data
- `min_bw`: Minimum bandwidth of the memory
- `max_bw`: Maximum bandwidth of the memory
- `total_period_count`: Number of periods in the computation
- `spatial_units`: Number of spatial units accessing this memory level

### 3.3 `AccessEnergy` Class

Another specialized version of `FourWayDataMoving` that represents the energy consumed by memory accesses in each direction.

**Formula:**
For each direction:
```
energy = memory_accesses[direction] * energy_cost_per_access
```

Where `energy_cost_per_access` depends on whether it's a read or write operation, and on the specific memory level.

### 3.4 `DataMoveAttr` Enum

Defines attributes for data movement patterns:

- `DATA_ELEM_MOVE_COUNT`: Number of data elements moved
- `DATA_PRECISION`: Bit precision of data
- `REQ_MEM_BW_AVER`: Average required memory bandwidth
- `REQ_MEM_BW_INST`: Instantaneous required memory bandwidth
- `DATA_TRANS_PERIOD`: Data transfer period length
- `DATA_TRANS_PERIOD_COUNT`: Number of periods
- `DATA_TRANS_AMOUNT_PER_PERIOD`: Amount of data transferred per period
- `INST_DATA_TRANS_WINDOW`: Instantaneous data transfer window

### 3.5 `DataMovePattern` Class

This class collects memory access patterns for each unit memory (memory holding one operand at one level).

**Key Components:**
- `name`: Name of the operand + memory level
- `attributes`: Dictionary mapping attribute types to values in four directions

**Methods:**
- `__init__`: Initialize with an operand and memory level
- `set_attribute`: Set values for an attribute
- `get_attribute`: Get values for an attribute
- `update_single_dir_data`: Update a single direction across all attributes

**Algorithm for Creating a Data Movement Pattern:**
```python
def create_pattern(operand, mem_level):
    pattern = DataMovePattern(operand, mem_level)
    
    # Set attributes for each direction
    for attr in DataMoveAttr:
        values = calculate_values_for_attribute(attr, operand, mem_level)
        pattern.set_attribute(attr, values)
    
    return pattern
```

## Recreating for Event-Driven Exploration

For your event-driven exploration, you'll need to:

1. Create `DataMovePattern` objects for each operand at each memory level
2. For each event, update the patterns with relevant values for:
   - `DATA_TRANS_AMOUNT_PER_PERIOD`: How much data moves when an event is processed
   - `DATA_PRECISION`: Precision of the data for this event
   - `DATA_TRANS_PERIOD_COUNT`: How many periods this event spans
   - `REQ_MEM_BW_AVER` and `REQ_MEM_BW_INST`: Bandwidth requirements

3. Use these patterns to create `MemoryAccesses` objects that track how many accesses occur in each direction
4. Use the `MemoryAccesses` to calculate `AccessEnergy` and total energy consumption

The key difference in your event-driven approach might be that you'll focus on individual events rather than an entire layer's computation. This might require adjusting how `DATA_TRANS_PERIOD` and `DATA_TRANS_PERIOD_COUNT` are calculated to reflect the processing of individual events rather than full layer computations.



Hey, I need your help to understand a short explanation of each stage of the mapping exploration tool after the complete mapspace is generated

We will be discussing about the Zigzag exploration tool. After the map space is generated.

I will share the associated scripts responsible for the flow

For each stage I want to know the INPUTS, ASSOCIATED FORMULAS (SIMPLER REPRESENTATION) and OUTPUTS, that are inputs to the next stage.

Here is the flow

Mapspace → (For a single mapping) Path creation (in Mapping object) → Memory utilisation → Memory word access → (Split) 
    → (For latency) → Double buffer detection → Transfer cycle calculation → Port activity calculation → Latency calculation
    → (For energy) → MAC energy + Memory energy calculation

I already generated the mapspace and you can start with what happens after the Mapspace input us passed to the path creation.

I need a SHORT DESCRIPTION of each of the listed stages WITH THE FORMULAS used in the calculations.

I attached all the files from zigzag that needs to be used for cost estimation.

Go through these scripts and give me an explanation to this flow.

