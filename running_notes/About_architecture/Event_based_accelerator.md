Here the seneca is being modeled.

# Goal: Basic design principle of the Event driven accelerators

## Extra info
- Use Seneca as the base how is an event driven accelerator modelled as compared to the Frame based accelerator (Extract information from Seneca paper)
- What changes are there in terms of the zigzag cost model from the event based and frame based implementation 
- Design an Frame based accelerator and then modify it for the event based for the same workload

### Chain of question.

#### Question 1:
Like Seneca which is an event driven accelerator. What are the basic design principles that were followed during it's developement which are unique to event driven accelerators especially in terms of memory, processing elements as compared to the Frame based accelerators.

I attached the Seneca's paper and I want to know stricking differences b/w both the accelerator design especially in terms of memory and PE.


### Key Points
- Research suggests event-driven accelerators, like Seneca, use flexible memory and processing for sparse data, unlike frame-based ones.
- It seems likely that memory in event-driven designs is hierarchical, with fast registers for frequent data, while frame-based uses large buffers for batches.
- The evidence leans toward event-driven PEs being programmable for various models, while frame-based PEs are fixed for specific tasks.
- There is some debate on efficiency, with event-driven better for sparse, real-time tasks, but frame-based excelling in dense computations.

### Memory Design
Event-driven accelerators, such as Seneca, feature a hierarchical memory structure. This includes small, fast register files for frequently accessed data like neuron states and synaptic weights, larger SRAM for datasets that don't fit in registers, and optional shared memory (e.g., STT-MRAM or HBM) for scalability. This design minimizes power consumption and maximizes data reuse, crucial for handling sparse, asynchronous data accesses typical in event-driven processing.

In contrast, frame-based accelerators typically have memory systems optimized for batch processing. They use large buffers to hold entire frames or batches of data and caches to reduce latency for regular, predictable access patterns, designed for dense data computations.

### Processing Elements
For processing elements, event-driven accelerators like Seneca use programmable Neuron Processing Elements (NPEs). These can be configured to support various neuron models and data precisions, processing events as they occur and leveraging the sparsity of spiking neural networks (SNNs) for efficiency. Additional components, such as the Loop Controller and Event Generator, manage the timing and sequencing of event processing.

Frame-based accelerators, however, often incorporate fixed-function PEs optimized for specific operations like matrix multiplications or convolutions, common in traditional neural networks. These PEs are designed for high throughput on dense data and operate synchronously, processing entire layers or batches at once.

---
#### Answer 1 From grok, the difference b/w the Event based and Frame based accelerator
### Detailed Analysis of Event-Driven vs. Frame-Based Accelerator Design

This section provides a comprehensive examination of the design principles of event-driven accelerators, using Seneca as a case study, and contrasts them with frame-based accelerators, focusing on memory architecture and processing elements (PEs). The analysis is grounded in the provided Seneca paper and supplemented by additional research to ensure a thorough understanding.

#### Introduction to Event-Driven and Frame-Based Accelerators
Event-driven accelerators, particularly those designed for neuromorphic computing, are tailored for processing sparse, asynchronous data, such as spikes in spiking neural networks (SNNs). These are inspired by biological neural systems and are particularly efficient for tasks requiring low power and real-time processing, such as edge AI applications. Frame-based accelerators, on the other hand, are designed for traditional artificial neural networks (ANNs), including convolutional neural networks (CNNs) and recurrent neural networks (RNNs), which process data in fixed time steps or frames, suitable for dense, batch-oriented computations.

The comparison is critical as the choice between these architectures impacts energy efficiency, latency, and flexibility, especially in emerging applications like neuromorphic vision systems and edge intelligence.

#### Memory Architecture: A Comparative Analysis

**Event-Driven Accelerators (e.g., Seneca):**
The memory architecture of Seneca, as detailed in the provided paper, is hierarchical and optimized for event-driven processing. Key components include:
- **Register File (inside NPEs):** Small, fast, and energy-efficient memory (e.g., 16W x 16b or 64W x 16b) for storing neuron states, synaptic weights, and intermediate values. It is significantly more energy-efficient (20x) than accessing SRAM, with energy consumption as low as 8-12 pJ per access and latency less than 1 ns. This supports data reuse, reducing the need for frequent memory accesses to external or larger memory blocks.
- **SRAM (Data Memory and Instruction Memory):** Typically 2 MB per core for Data Memory and 256 KB for Instruction Memory, implemented using SRAM for a balance of area, performance, and power. Energy consumption is higher (180-220 pJ for read/write operations) with a latency of 2 ns, but it is efficient for larger memory requirements, dominating 70% of the core area and power consumption.
- **Shared Memory (STT-MRAM and HBM):** Optional, using denser technologies like Spin-Transfer Torque Magnetic Random Access Memory (STT-MRAM) or High Bandwidth Memory (HBM). For example, STT-MRAM offers high density (36.6 Mb) with lower static power (0 pW/bit) but higher read energy (2,000 pJ) and latency (2x). Shared memory is accessed via the Shared Memory Pre-fetch Unit (SMPU), which pre-fetches data to hide latency and reduce energy costs.
- **Hybrid Memory Approach:** Dynamically allocates parameters (e.g., neuron states, weights) to different memory blocks based on usage frequency and optimization needs, enabling efficient trade-offs between energy, area, and performance.

This hierarchical structure is designed to minimize power consumption and maximize data reuse, leveraging the sparsity of spikes through techniques like spike-grouping and event compression, as noted in the paper.

**Frame-Based Accelerators:**
Frame-based accelerators, such as those used in traditional DNN accelerators (e.g., Google's TPU, NVIDIA GPUs), are optimized for batch processing. Their memory architecture typically includes:
- Large on-chip buffers and caches to hold entire frames or batches of data, such as weights and activations for CNN layers. These are often implemented using SRAM or eDRAM for high bandwidth and low latency.
- Off-chip DRAM for storing larger datasets, accessed via high-bandwidth memory interfaces, which is less energy-efficient but necessary for large-scale models.
- Memory access patterns are regular and predictable, optimized for dense matrix operations, with data movement designed to minimize stalls during batch processing.

The key difference lies in the focus: event-driven memory is tailored for sparse, asynchronous accesses, while frame-based memory is optimized for dense, synchronous batch processing, often leading to higher baseline power consumption due to constant clocking and processing of all data, including zeros.

#### Processing Elements: Design and Functionality

**Event-Driven Accelerators (e.g., Seneca):**
The processing elements in Seneca are primarily the Neuron Processing Elements (NPEs), supported by other components for control and event management:
- **Neuron Processing Elements (NPEs):** Form a Single Instruction Multiple Data (SIMD) architecture, with 8 NPEs per core by default, each containing a small register-based memory (64 x 16b). They execute micro-codes provided by the Loop Buffer, performing operations like arithmetic (ADD, SUB, MUL), comparison (GTH, MAX, MIN), and data type conversions. NPEs support multiple data precisions (4, 8, and 16 bits), enabling flexibility for mixed-precision neural networks, and consume only 1.3% of the total core area, making them area-efficient while supporting parallel processing.
- **RISC-V Controller:** A low-power, open-source RISC-V processor (Ibex from lowRISC) handles higher-level control tasks, such as preprocessing incoming spikes, post-processing output events, and managing memory allocation. It consumes about 10% of the core area but is less energy-efficient (10x worse than NPEs) due to its general-purpose nature, performing per-spike operations to reduce energy overhead to less than 1% of total operations.
- **Loop Controller and Loop Buffer:** The Loop Controller orchestrates the time-multiplexing of NPEs and generates memory addresses for Data Memory access, executing micro-codes stored in the Loop Buffer (128 x 23b). It is highly energy-efficient (100x more efficient than RISC-V) and optimized for nested loops, typical in neural network computations, processing tasks queued in the Task FIFO for parallel execution.
- **Event Generator:** Converts neuron firing events into data packets using Address Event Representation (AER) for transmission via the Network-on-Chip (NoC), inspecting NPE registers and generating packets with source neuron IDs and optional values for graded spikes.

These components support event-driven processing by leveraging asynchronous operation (Globally Asynchronous, Locally Synchronous, GALS architecture), time-multiplexing, and sparsity exploitation, processing only when events occur to reduce idle power consumption.

**Frame-Based Accelerators:**
Frame-based accelerators typically feature:
- Fixed-function PEs, such as multiply-accumulate (MAC) units arranged in systolic arrays or tensor cores, optimized for specific operations like convolutions or matrix multiplications. These are designed for high throughput on dense data, often supporting operations on 16-bit or 8-bit floating-point numbers for efficiency.
- Controllers and schedulers that manage the synchronous processing of entire layers or batches, with dataflow architectures like dataflow engines to optimize memory bandwidth usage.
- PEs are less flexible, tailored for specific neural network layers (e.g., convolutional layers, fully connected layers), and operate at fixed clock rates, processing data in a frame-by-frame manner.

The key distinction is that event-driven PEs are programmable and flexible, supporting various neuron models and handling sparse, event-based data, while frame-based PEs are optimized for dense, batch-oriented computations with less adaptability.

#### Striking Differences and Design Implications

To highlight the striking differences, consider the following table summarizing the key aspects:

| **Aspect**               | **Event-Driven (e.g., Seneca)**                          | **Frame-Based**                                      |
|--------------------------|---------------------------------------------------------|-----------------------------------------------------|
| **Memory Hierarchy**      | Hierarchical (registers, SRAM, shared memory) for sparse access | Large buffers and caches for batch processing       |
| **Memory Access Pattern** | Asynchronous, sparse, optimized for data reuse          | Synchronous, dense, regular access for batches      |
| **Processing Elements**   | Programmable NPEs, flexible for various models          | Fixed-function PEs, optimized for specific operations |
| **Operation Mode**        | Asynchronous, event-triggered, GALS architecture        | Synchronous, frame-triggered, constant clocking     |
| **Power Efficiency**      | High, processes only on events, exploits sparsity       | Higher baseline, processes all data including zeros |
| **Flexibility**           | High, supports diverse neuron models and tasks          | Lower, optimized for specific neural network layers |

These differences stem from the fundamental processing paradigm: event-driven accelerators are designed for low-power, real-time tasks with sparse data, while frame-based accelerators excel in high-throughput, dense computations for traditional neural networks.

#### Supporting Context and Further Insights
The Seneca paper does not explicitly compare with frame-based accelerators but implicitly contrasts them, noting advantages in sparsity exploitation, power efficiency, and adaptability. For instance, Seneca's design supports low latency for real-time tasks, such as depth-first CNN processing, by processing events as they arrive rather than waiting for a complete frame, as seen in sections discussing convolutional and recurrent processing.

Additional research, such as [Event-Driven Graph Neural Network Accelerators for Low-Power Vision](https://repository.tudelft.nl/record/uuid:5bc8cb13-4147-4ff3-875f-fc3fd515c975), highlights that event-driven accelerators are particularly suited for neuromorphic vision systems, leveraging event-based sensors that capture only dynamic changes, contrasting with frame-based sensors that record all pixels at fixed rates, including static backgrounds.

Similarly, [Event-driven in-sensor spiking neural network](https://www.researchgate.net/figure/Event-driven-in-sensor-spiking-neural-network-a-Comparison-between-frame-and_fig1_375609518) provides a direct comparison, noting frame-based sensors record all pixels at a fixed frame rate, while event-based sensors asynchronously capture only light intensity changes, aligning with the accelerator designs discussed.

#### Conclusion
The design principles of event-driven accelerators like Seneca, particularly in memory and PEs, are tailored for sparse, asynchronous processing, offering flexibility and efficiency for neuromorphic computing tasks. In contrast, frame-based accelerators are optimized for dense, synchronous batch processing, suitable for traditional neural networks. These differences highlight the trade-offs between flexibility, power efficiency, and computational throughput, guiding the choice of accelerator based on application needs.

### Key Citations
- [Seneca neuromorphic architecture detailed analysis](attachment:0)
- [Event-Driven Graph Neural Network Accelerators for Low-Power Vision](https://repository.tudelft.nl/record/uuid:5bc8cb13-4147-4ff3-875f-fc3fd515c975)
- [Event-driven in-sensor spiking neural network comparison](https://www.researchgate.net/figure/Event-driven-in-sensor-spiking-neural-network-a-Comparison-between-frame-and_fig1_375609518)
- [An Event-driven Spiking Neural Network Accelerator with On-chip Sparse Weight](https://ieeexplore.ieee.org/document/9937521/)


#### Answer 1 from Google deep research

link to the converstion: [Link to the chat](https://g.co/gemini/share/a4a3f02696de)

 

#### Question 2:
Based on the about knowledge on how Event driven accelerators design is in terms of Memory and PE. How it would be modellied in an Exploration tool like Zigzag which models the architecture as follows

```yaml
name: Single_core_8_NPE_Accelerator

operational_array: -> This is the definition of the 1 Dimension Processing elements
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1]
  sizes: [8]

// Two seperate memory blocks for the inputs and (weights and partial sums), here the size of both the memory blocks are same but it is not correct as per the design principles of event driven accelerators.
memories:
  sram_buffer_inputs_256MB: 
    size: 2147483648
    # size: 2097152
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 1
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 1
    operands: [I1]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]

#This is used to store the weights and outputs and partial sums
  sram_256MB:
    size: 2147483648
    # size: 2097152
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 1
    r_port: 3
    w_port: 3
    rw_port: 0
    latency: 1
    operands: [I2, O]
    ports:
      - tl: r_port_1
        fl: w_port_1
      - fh: w_port_2
        tl: r_port_2
        fl: w_port_3
        th: r_port_3
    served_dimensions: [D1]
```

Now, How this about model would needs to be changed to effectively model the event driven accelerator

#### Answer 2:

### Key Points
- Research suggests event-driven accelerators like Seneca require a hierarchical memory model in ZigZag, unlike the single-level memory in frame-based models.
- It seems likely that memory should include small, fast register files local to processing elements (PEs) and larger shared SRAM, reflecting Seneca’s design.
- The evidence leans toward adjusting PE energy costs to match event-driven operations (e.g., additions, comparisons) rather than frame-based MAC operations.
- There is some uncertainty about ZigZag’s ability to model sparsity and asynchronous event processing, which may require workload or mapping adjustments.

### Memory Hierarchy Adjustments
To model an event-driven accelerator like Seneca in ZigZag, the memory architecture must reflect its hierarchical structure. Unlike the frame-based model, which uses a single SRAM level for inputs and weights/outputs, Seneca employs multiple memory levels: small register files within each Neuron Processing Element (NPE), a larger shared SRAM (Data Memory), and optionally a shared memory (e.g., STT-MRAM or HBM). In ZigZag, you can define multiple memory instances, with register files modeled as local to each PE and SRAM as shared across all PEs. This setup captures the fast, energy-efficient access of register files and the larger capacity of SRAM.

### Processing Elements (PEs) Modifications
The PEs in event-driven accelerators perform different operations compared to frame-based ones. While frame-based accelerators typically execute multiply-accumulate (MAC) operations for dense data, Seneca’s NPEs handle operations like additions and comparisons for sparse, event-driven data. In ZigZag, you should adjust the `unit_energy` parameter in the operational array to reflect these operations’ energy costs, ensuring the model accurately represents the computational behavior of event-driven processing.

### Workload and Sparsity Considerations
Event-driven accelerators process data only when events (e.g., spikes) occur, leading to sparse computations. ZigZag, designed for dense deep neural networks, may not natively support this asynchronous behavior. However, you can approximate sparsity by adjusting the workload definition or mapping to reduce the number of operations and memory accesses, reflecting the low activity rate (e.g., 10% of neurons active). This requires further exploration in ZigZag’s settings or mapping files, which are not fully detailed here.

---

```yaml
```yaml
# Memory pool defining different memory types
memory_pool:
  register:
    size: 64  # words, e.g., 64x16b for NPE register files
    r_bw: 128  # high bandwidth for fast access
    w_bw: 128
    r_cost: 0.1  # low energy cost, e.g., 8-12 pJ
    w_cost: 0.1
    area: 0.01  # small area per instance
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 0.5  # <1 ns
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
  shared_memory:
    size: 33554432  # 32 MB, e.g., for STT-MRAM
    r_bw: 64  # lower bandwidth
    w_bw: 64
    r_cost: 2000  # high energy cost, e.g., 2000 pJ
    w_cost: 2000
    area: 2
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 10  # higher latency, e.g., 2x SRAM

# Architecture definition
architecture:
  name: Event_Driven_Accelerator
  operational_array:
    unit_energy: 0.02  # adjusted for event-driven operations (e.g., ADD, CMP)
    unit_area: 1
    dimensions: [D1]
    sizes: [8]  # 8 NPEs
  memory_instances:
    - memory_instance: register
      bank_instances: 8  # one per PE
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
      served_dimensions: [D1]
    - memory_instance: sram
      bank_instances: 1  # shared across PEs
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
        - fh: w_port_2
          tl: r_port_2
          fl: w_port_3
          th: r_port_3
      served_dimensions: []
    - memory_instance: shared_memory
      bank_instances: 1  # shared
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
      served_dimensions: []
  memory_hierarchy:
    - register
    - sram
    - shared_memory
```
```

### Detailed Analysis of Modeling Event-Driven Accelerators in ZigZag

This section provides a comprehensive examination of how to adapt the ZigZag framework, originally designed for frame-based deep neural network (DNN) accelerators, to model event-driven accelerators like Seneca, focusing on memory hierarchy and processing elements (PEs). The analysis leverages insights from the Seneca neuromorphic architecture and ZigZag’s architectural modeling capabilities.

#### Background on ZigZag and Accelerator Types
ZigZag is a hardware architecture-mapping design space exploration (DSE) framework for deep learning accelerators, providing fast and accurate hardware cost estimation through analytical cost models ([ZigZag GitHub](https://github.com/KULeuven-MICAS/zigzag)). It supports multi-dimensional processing arrays, advanced memory structures, and detailed energy and latency analysis, primarily for frame-based accelerators that process dense data in fixed batches, such as convolutional neural networks (CNNs). Event-driven accelerators, like Seneca, are designed for neuromorphic computing, processing sparse, asynchronous data (e.g., spikes in spiking neural networks, SNNs) with high energy efficiency, particularly for edge AI applications ([Seneca Paper](attachment:0)).

The key challenge is adapting ZigZag’s frame-based modeling approach to capture the unique characteristics of event-driven accelerators, including their hierarchical memory, programmable PEs, and sparse computation patterns.

#### Seneca’s Architectural Features
Seneca’s design, as detailed in the provided paper, includes:
- **Memory Hierarchy**:
  - **Register Files**: Small (e.g., 64x16b), fast (<1 ns latency), and energy-efficient (8-12 pJ per access) memories within each Neuron Processing Element (NPE) for storing neuron states, synaptic weights, and intermediate values.
  - **Data Memory (SRAM)**: Larger (e.g., 2 Mb per core), shared memory for datasets, with higher energy costs (180-220 pJ) and 2 ns latency.
  - **Shared Memory**: Optional, high-capacity memory (e.g., 32 MB STT-MRAM or 64 Gb HBM) for scalability, with high energy costs (2000 pJ for STT-MRAM) and increased latency.
  - **Loop Buffer and Instruction Memory**: Small memories for control instructions, not directly relevant for data operands.
- **Processing Elements (NPEs)**: Programmable units in a SIMD architecture, performing operations like additions, subtractions, and comparisons, supporting multiple data precisions (4, 8, 16 bits).
- **Event Processing**: Events are preprocessed by a RISC-V controller, queued in a Task FIFO, executed by a Loop Controller, and computed by NPEs, with output events generated and post-processed.

These features contrast with frame-based accelerators, which use large buffers for batch processing, fixed-function PEs for MAC operations, and synchronous dataflows.

#### ZigZag’s Architectural Model
ZigZag’s architecture is defined in a YAML file (e.g., `architecture.yaml`), specifying:
- **Operational Array**: Characteristics of the PE array, including size, energy per operation (`unit_energy`), and dimensions.
- **Memory Instances**: Defined in a memory pool with attributes like size, bandwidth, energy costs, and ports, assigned to operands (I for inputs, W for weights, O for outputs).
- **Memory Hierarchy**: A list of memory instances ordered from lowest to highest level, with options for local (per-PE) or shared memories.
- **Additional Settings**: Parameters like `bank_instances` (number of memory instances) and `served_dimensions` (indicating which PEs the memory serves).

The provided frame-based model includes two SRAM instances: one for inputs (I1) and one for weights and outputs (I2, O), both shared across an 8-PE array. This is insufficient for event-driven accelerators, which require a more nuanced hierarchy.

#### Adapting ZigZag for Event-Driven Accelerators

##### Memory Hierarchy
To model Seneca’s memory, ZigZag’s single-level SRAM must be replaced with a multi-level hierarchy:
- **Register Files (L0)**: Modeled as small, fast memory instances local to each PE. In Seneca, each NPE has its own register file, so for an 8-PE array, there are 8 instances (`bank_instances: 8`). These store all operands (I, W, O) to support flexible data access during event processing.
- **SRAM (L1)**: A larger, shared memory instance (`bank_instances: 1`) for datasets that exceed register capacity, also storing all operands.
- **Shared Memory (L2, Optional)**: A high-capacity, shared instance for large models, using technologies like STT-MRAM or HBM, with higher latency and energy costs.

In ZigZag, memory instances are defined in a `memory_pool` with specific parameters, then instantiated in the `architecture` section. The `memory_hierarchy` lists these instances in order (e.g., [register, sram, shared_memory]). The `served_dimensions` parameter can indicate whether a memory is local (e.g., [D1] for per-PE) or shared (e.g., [] for all PEs).

**Example Parameters** (based on Seneca):
- **Register**: Size ~64 words, bandwidth 128 bits/cycle, read/write cost 0.1 pJ, latency 0.5 ns.
- **SRAM**: Size 256 KB (262144 words), bandwidth 128 bits/cycle, read/write cost 10.5/12.8 pJ, latency 2 ns.
- **Shared Memory**: Size 32 MB, bandwidth 64 bits/cycle, read/write cost 2000 pJ, latency 10 ns.

##### Processing Elements
Seneca’s NPEs perform operations like additions and comparisons, unlike the MAC-heavy PEs in frame-based accelerators. ZigZag’s `operational_array` assumes a uniform `unit_energy` for PE operations, typically calibrated for MACs (e.g., 0.04 pJ in the provided model). For event-driven accelerators, this must be adjusted to reflect the energy of typical operations (e.g., 0.02 pJ, assuming simpler operations). The array size (8 PEs) and dimensions ([D1]) can remain, as Seneca uses a similar SIMD structure.

##### Sparsity and Event-Driven Computation
Event-driven accelerators exploit sparsity, processing only active events (e.g., 10% neuron activity). ZigZag, designed for dense DNNs, may not natively model asynchronous event processing. However, sparsity can be approximated by:
- **Workload Definition**: Define loops over time steps or event groups, with reduced dimensions based on sparsity (e.g., 10% of neurons active).
- **Mapping Optimization**: Prioritize data reuse in register files to minimize SRAM accesses, reflecting Seneca’s spike-grouping techniques.
- **Settings**: If ZigZag supports sparsity parameters (e.g., weight or activation sparsity, as seen in ZigZag-IMC ([ZigZag-IMC GitHub](https://github.com/KULeuven-MICAS/zigzag-imc))), set high sparsity levels (e.g., 90% for activations).

Without explicit sparsity support, the mapping file must be carefully designed to reduce effective operations and memory accesses.

#### Proposed YAML Model
The following YAML model adapts the provided frame-based model for an event-driven accelerator, incorporating Seneca’s memory hierarchy and PE characteristics:

```yaml
```yaml
# Memory pool defining different memory types
memory_pool:
  register:
    size: 64  # words, e.g., 64x16b for NPE register files
    r_bw: 128  # high bandwidth for fast access
    w_bw: 128
    r_cost: 0.1  # low energy cost, e.g., 8-12 pJ
    w_cost: 0.1
    area: 0.01  # small area per instance
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 0.5  # <1 ns
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
  shared_memory:
    size: 33554432  # 32 MB, e.g., for STT-MRAM
    r_bw: 64  # lower bandwidth
    w_bw: 64
    r_cost: 2000  # high energy cost, e.g., 2000 pJ
    w_cost: 2000
    area: 2
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 10  # higher latency, e.g., 2x SRAM

# Architecture definition
architecture:
  name: Event_Driven_Accelerator
  operational_array:
    unit_energy: 0.02  # adjusted for event-driven operations (e.g., ADD, CMP)
    unit_area: 1
    dimensions: [D1]
    sizes: [8]  # 8 NPEs
  memory_instances:
    - memory_instance: register
      bank_instances: 8  # one per PE
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
      served_dimensions: [D1]
    - memory_instance: sram
      bank_instances: 1  # shared across PEs
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
        - fh: w_port_2
          tl: r_port_2
          fl: w_port_3
          th: r_port_3
      served_dimensions: []
    - memory_instance: shared_memory
      bank_instances: 1  # shared
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
      served_dimensions: []
  memory_hierarchy:
    - register
    - sram
    - shared_memory
```
```

#### Comparison with Frame-Based Model
The following table compares the frame-based and event-driven models in ZigZag:

| **Aspect**               | **Frame-Based Model**                          | **Event-Driven Model**                          |
|--------------------------|-----------------------------------------------|-----------------------------------------------|
| **Memory Levels**        | Single SRAM level per operand (I1, I2/O)      | Multiple levels: register (per-PE), SRAM, shared memory |
| **Memory Locality**      | Shared SRAM across PEs                        | Register files local to PEs, SRAM/shared memory shared |
| **PE Operations**        | MAC operations (e.g., 0.04 pJ)                | Additions, comparisons (e.g., 0.02 pJ)         |
| **Data Access Pattern**  | Dense, batch-oriented                         | Sparse, event-triggered                        |
| **Sparsity Handling**    | Limited, assumes dense data                   | Requires workload/mapping adjustments for sparsity |

#### Challenges and Limitations
- **Sparsity Modeling**: ZigZag’s focus on dense DNNs means it may not natively support event-driven sparsity. Approximations via workload or mapping adjustments are needed, but their accuracy is uncertain without explicit sparsity support.
- **Asynchronous Processing**: Seneca’s asynchronous event processing (Globally Asynchronous, Locally Synchronous) is not directly modeled in ZigZag’s loop-based framework, requiring a time-stepped or event-grouped approximation.
- **Validation**: The proposed model assumes ZigZag can handle per-PE memories and multi-level hierarchies, based on documentation ([ZigZag GitHub](https://github.com/KULeuven-MICAS/zigzag)). Validation with actual ZigZag runs is recommended.

#### Future Considerations
To fully support event-driven accelerators, ZigZag could be extended with:
- Explicit sparsity parameters for activations and weights, as seen in ZigZag-IMC ([ZigZag-IMC GitHub](https://github.com/KULeuven-MICAS/zigzag-imc)).
- Support for asynchronous or event-based computation models, possibly by defining dynamic loop structures.
- Enhanced energy models for diverse PE operations beyond MACs.

#### Conclusion
Adapting ZigZag to model event-driven accelerators like Seneca involves defining a hierarchical memory structure with local register files and shared SRAM, adjusting PE energy costs for event-driven operations, and approximating sparsity through workload or mapping optimizations. The proposed YAML model provides a starting point, but further refinement and validation are needed to ensure accuracy in capturing the unique characteristics of event-driven processing.

### Key Citations
- [ZigZag HW Architecture-Mapping DSE Framework](https://github.com/KULeuven-MICAS/zigzag)
- [ZigZag v1 Fast DNN Accelerator DSE Framework](https://github.com/ZigZag-Project/zigzag-v1)
- [ZigZag-IMC HW Accelerator Mapping for In-Memory Computing](https://github.com/KULeuven-MICAS/zigzag-imc)
- [ZigZag Memory-Centric DNN Accelerator DSE Framework](https://arxiv.org/abs/2007.11360)
- [SENECA Neuromorphic Architecture Detailed Analysis](attachment:0)

Modified architecture
```yaml
# Memory pool defining different memory types
memory_pool:
  register:
    size: 64  # words, e.g., 64x16b for NPE register files
    r_bw: 128  # high bandwidth for fast access
    w_bw: 128
    r_cost: 0.1  # low energy cost, e.g., 8-12 pJ
    w_cost: 0.1
    area: 0.01  # small area per instance
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 0.5  # <1 ns
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
  shared_memory:
    size: 33554432  # 32 MB, e.g., for STT-MRAM
    r_bw: 64  # lower bandwidth
    w_bw: 64
    r_cost: 2000  # high energy cost, e.g., 2000 pJ
    w_cost: 2000
    area: 2
    r_port: 1
    w_port: 1
    rw_port: 0
    latency: 10  # higher latency, e.g., 2x SRAM

# Architecture definition
architecture:
  name: Event_Driven_Accelerator
  operational_array:
    unit_energy: 0.02  # adjusted for event-driven operations (e.g., ADD, CMP)
    unit_area: 1
    dimensions: [D1]
    sizes: [8]  # 8 NPEs
  memory_instances:
    - memory_instance: register
      bank_instances: 8  # one per PE
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
      served_dimensions: [D1]
    - memory_instance: sram
      bank_instances: 1  # shared across PEs
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
        - fh: w_port_2
          tl: r_port_2
          fl: w_port_3
          th: r_port_3
      served_dimensions: []
    - memory_instance: shared_memory
      bank_instances: 1  # shared
      operand_stored: [I, W, O]
      ports:
        - tl: r_port_1
          fl: w_port_1
      served_dimensions: []
  memory_hierarchy:
    - register
    - sram
    - shared_memory
```

#### Answer 2 from gemini deep research

An Analysis of Adapting the ZigZag Cost Model for Energy Estimation in Event-Driven Accelerators1. Introduction1.1 Context: The Rise of Event-Driven AccelerationThe relentless pursuit of energy efficiency and real-time processing capabilities in artificial intelligence (AI) has spurred significant interest in alternative computing paradigms. Among the most promising are event-driven hardware accelerators, particularly those designed for Spiking Neural Networks (SNNs).1 Inspired by the computational principles of biological brains, these systems leverage the inherent sparsity and event-based nature of SNN computations to potentially achieve orders-of-magnitude improvements in energy efficiency and latency compared to conventional Artificial Neural Network (ANN) accelerators executing dense computations.10 Key advantages stem from performing computations only when necessary (i.e., when a 'spike' or event occurs) and communicating information sparsely, thereby reducing redundant operations and data movement.13 This makes them particularly attractive for resource-constrained edge computing, robotics, real-time signal processing, and other applications demanding low power consumption and rapid response times.131.2 Problem Statement: Modeling ChallengesDespite their potential, designing and optimizing event-driven accelerators presents significant challenges. The complex interplay between SNN algorithms, hardware architectures, and algorithm-to-hardware mappings creates a vast design space. Exploring this space efficiently requires accurate and rapid performance and energy cost models, particularly during the early stages of design.3 While several sophisticated cost modeling frameworks, such as ZigZag, have been developed for conventional Deep Neural Network (DNN) accelerators 54, their direct applicability to event-driven systems is questionable due to fundamental differences in computational paradigms. DNN accelerators typically process dense tensors in a synchronous manner, whereas event-driven accelerators operate on sparse, asynchronous event streams and manage internal neuron states.131.3 Proposed Approach and Core QuestionThis report analyzes a proposed methodology that attempts to adapt the ZigZag framework for modeling event-driven accelerators. The core question addressed is: How feasible and accurate is this adaptation, particularly for energy estimation, given the inherent mismatches between ZigZag's underlying assumptions and the characteristics of event-driven, sparse, and stateful computation? The analysis critically examines the validity of the proposed adaptation's core assumptions and implementation strategies, explores necessary augmentations to ZigZag's estimates, compares the approach to specialized SNN modeling tools, and synthesizes the findings into a comprehensive critique.1.4 Report StructureThe report is structured as follows:
Section 2: Details the fundamentals of the ZigZag framework, its modeling approach, and its core assumptions regarding data density and synchronous execution, highlighting the inherent limitations for event-driven systems.
Section 3: Evaluates the proposed adaptation strategy, analyzing the impact of modeling dense computations, the potential role of workload scaling, and the effectiveness of using specific dataflows (Input Stationary) and loop orderings within ZigZag's structure.
Section 4: Investigates methods to augment ZigZag's energy estimates post-hoc, considering sparsity factors, event handling logic overhead, neuron state memory costs, and power gating savings.
Section 5: Provides a comparative analysis between the capabilities of the adapted ZigZag approach and specialized SNN modeling tools, exemplified by STEMS.
Section 6: Synthesizes the analysis into a critique of the proposed methodology, identifying fundamental limitations and suggesting potential refinements or alternative modeling strategies.
Section 7: Concludes the report with a summary of the key findings and recommendations.
2. ZigZag Framework Fundamentals and AssumptionsTo evaluate the feasibility of adapting ZigZag for event-driven accelerators, it is crucial to first understand its core functionality, modeling approach, and inherent assumptions as designed for traditional DNN accelerators.2.1 Core FunctionalityZigZag is a design space exploration (DSE) framework developed to facilitate the co-design of DNN algorithms, hardware architectures, and mappings.54 Its primary goal is to bridge the gap between high-level algorithmic choices and their resulting hardware acceleration costs by providing rapid and reasonably accurate analytical estimates for energy consumption, execution latency, and silicon area.58 It takes descriptions of the DNN workload, the target accelerator architecture (memory hierarchy, PEs), and the mapping strategy as inputs, and outputs performance/cost metrics, enabling designers to explore trade-offs without resorting to time-consuming hardware implementations or detailed cycle-accurate simulations.542.2 Modeling ApproachZigZag employs an analytical cost model built upon a structured representation of the computation and data movement involved in DNN layer execution.

Workload Representation: DNN layers, such as 2D convolution (CONV2D) or matrix multiplication (GEMM), are represented using a nested set of for-loops that define the computation.52 For instance, a typical CONV2D layer might be described by seven nested loops iterating over batch size (N), output channels (K), input channels (C), output feature map height (Y'), output feature map width (X'), filter height (R), and filter width (S). This loop-nest representation captures the parallelism and data dependencies inherent in the layer's computation. While early versions were more compute-centric, later iterations adopted an enhanced memory-centric loop format to better capture data movement.56


Hardware Representation: The target accelerator is modeled as a hierarchy of memory levels (e.g., DRAM, Global Buffer, Register Files) and a spatial array of Processing Elements (PEs).52 Each memory level is characterized by its capacity, access bandwidth, read/write energy per bit, and potentially the number of read/write ports.56 The PE array is defined by its spatial dimensions and the computational capability of each PE (e.g., performing multiply-accumulate operations).52


Mapping Representation: The mapping, or schedule, defines how the loops of the workload are executed across the hardware resources in space and time.52 This involves:

Temporal Mapping (Loop Blocking/Tiling): Assigning segments (tiles) of the loop iterations to specific memory levels. Data relevant to inner loops is kept in lower, faster memory levels, while data for outer loops resides in higher, larger levels.56
Loop Ordering: Defining the sequence of loops executed within each memory level.56
Spatial Unrolling: Parallelizing loop iterations across the spatial PE array.56
Dataflow: The combination of temporal mapping, loop order, and spatial unrolling determines the dataflow (e.g., Weight Stationary, Output Stationary, Input Stationary, Row Stationary), dictating how data operands (Weights, Inputs, Outputs) move through the memory hierarchy and PE array.69 ZigZag later introduced the concept of "uneven mapping," allowing different operands (W/I/O) to have distinct temporal mappings even at shared memory levels, increasing mapping flexibility.54


2.3 Core AssumptionsZigZag's modeling approach relies on several fundamental assumptions inherent to traditional DNN processing:

Data Density: The framework implicitly assumes that the tensors involved (weights, input activations, output activations) are dense. Its cost model calculates the number of operations (e.g., MACs) and memory accesses based on the full dimensions defined by the loop bounds.61 It does not inherently possess mechanisms to recognize or exploit zero values within tensors (sparsity). While extensions like Sparseloop explicitly model sparsity 70, the baseline ZigZag model operates under a dense computation assumption. The energy cost is computed based on the total number of accesses and operations derived from these dense loop structures.56


Synchronous Execution: The nested-for-loop representation implies a synchronous, sequential execution model.61 Computations are assumed to proceed in lockstep, dictated by the loop iterations and potentially PE array parallelism. Latency is calculated based on the number of cycles required to complete all iterations within the loop structure, considering PE utilization and memory bandwidth constraints within this synchronous framework. It does not model asynchronous event arrivals or data-driven computation triggering.

2.4 Limitations for Event-Driven ModelingThe core assumptions of data density and synchronous execution create a fundamental mismatch when attempting to model event-driven accelerators, particularly SNN accelerators:
Asynchronicity: Event-driven systems operate asynchronously, with computation triggered by the arrival of sparse events (spikes), not by fixed clock cycles iterating through dense data structures.13
Sparsity: Sparsity (both in activations/events and potentially weights) is the primary source of efficiency in SNNs, leading to significantly fewer actual computations and data movements than a dense equivalent.24
Statefulness: SNN neurons maintain internal state variables (e.g., membrane potential, threshold adaptation states) that evolve over time and influence computation.39 This state represents a significant memory and computational cost absent in standard stateless DNN layers modeled by ZigZag.
This conceptual gap suggests that directly applying ZigZag or making simple adaptations will likely face significant challenges in accurately capturing the behavior and, crucially, the energy consumption of event-driven systems.16The framework's optimization focus, derived from its dense model, centers on minimizing data movement between memory levels for weights and activations by maximizing reuse within local buffers.53 While important, this may misrepresent the dominant energy costs in SNN accelerators. SNNs introduce neuron states (e.g., membrane potential) as a critical operand.68 This state memory can be substantial, potentially requiring hundreds of megabytes for moderately sized networks, and is accessed frequently (potentially per event or timestep), often becoming a primary energy consumer.13 Additionally, the logic required for event handling—detecting spikes, decoding addresses, managing queues, and routing events through a Network-on-Chip (NoC)—also consumes non-negligible energy, especially if not carefully optimized.15 ZigZag's default cost model does not account for these SNN-specific energy sinks. Consequently, optimizing a design based solely on ZigZag's perspective of weight and activation movement could lead to suboptimal energy efficiency for SNNs, as it neglects potentially larger contributors to the total power budget.Furthermore, the introduction of "uneven mapping" in later ZigZag versions 54, while enhancing flexibility for DNNs by allowing different operands (W/I/O) to utilize the memory hierarchy differently, does not resolve the fundamental issues for SNN modeling. This feature allows, for instance, weights to reside in a larger, higher-level buffer while activations stream through smaller, lower-level buffers, catering to diverse access patterns in DNNs.56 However, it operates within the existing framework of DNN operands and synchronous loop execution. It does not introduce the necessary SNN-specific operands, most notably the neuron state, nor does it alter the underlying synchronous execution model to reflect event-driven dynamics. Therefore, uneven mapping, while beneficial for DNN flexibility, fails to bridge the core conceptual divide between the models ZigZag was designed for and the realities of event-driven SNN computation.3. Evaluating the Proposed ZigZag Adaptation for Event-Driven AcceleratorsGiven the fundamental characteristics of ZigZag, we now critically evaluate the proposed strategy for adapting it to model event-driven accelerators. This involves examining the validity of its core assumptions in this new context and the effectiveness of the suggested implementation techniques.3.1 Analysis of Core Assumptions3.1.1 Modeling a Dense Case (Ignoring Sparsity)The proposal suggests modeling the event-driven accelerator by defining an equivalent dense workload in ZigZag, effectively ignoring the inherent sparsity that is central to the efficiency of such systems. This approach has profound and detrimental impacts on the accuracy of the model, particularly for energy estimation.

Problem Definition: Event-driven accelerators, especially for SNNs, derive their primary energy advantage from sparsity.15 Activation sparsity (infrequent spikes) means that many potential computations involving zero-valued inputs are skipped.13 Weight sparsity, often achieved through pruning, further reduces computations.20 Modeling a dense equivalent forces ZigZag to calculate costs based on the full tensor dimensions defined by the loop structure (e.g., iterating over all input channels C, spatial dimensions H, W), assuming every operation is performed and every corresponding data element is accessed.61


Impact on Latency: By assuming dense computation, the model grossly overestimates the number of cycles required. It fails to account for cycle skipping opportunities that sparse accelerators employ, where cycles corresponding to zero-operand computations are bypassed.70 The actual latency in event-driven systems is often determined by the processing of non-zero events, not the total theoretical operations.


Impact on Energy: This is where the dense assumption causes the most significant distortion.

Dynamic Energy: ZigZag calculates dynamic energy based on the number of operations and memory accesses derived from the dense loop bounds.56 By ignoring sparsity, it vastly overestimates the number of active PEs, the number of MAC or Synaptic Operations (SOPs) performed, and the volume of activation data read from and written to memory. Real event-driven hardware only consumes compute and access energy for non-zero events.13 The energy cost is more closely related to the number of actual SOPs rather than the total possible MACs.17 This overestimation can be by orders of magnitude, rendering the energy prediction unreliable.
Static Energy: While less directly impacted, overestimating activity might also lead to inaccurate estimations of static power if power gating effectiveness is linked to utilization, which is misrepresented by the dense model.



Fundamental Contradiction: The core premise of event-driven computing is efficiency through sparsity exploitation.24 Modeling it as dense negates this fundamental principle. Furthermore, the optimal dataflow strategy (e.g., Input Stationary vs. Output Stationary) can depend heavily on the sparsity level; conclusions drawn from a dense model may not hold for the actual sparse execution.42


Supporting Context: The very development of specialized sparse modeling frameworks like Sparseloop 70 and others 83 arose precisely because dense analytical models like ZigZag proved inadequate for accurately evaluating sparse tensor accelerators. Studies on sparse accelerators highlight challenges like workload imbalance and underutilization at high sparsity 82, phenomena completely missed by a dense model that assumes full utilization based on loop bounds.

3.1.2 Workload ScalingAn alternative suggestion might be to retain the dense modeling structure of ZigZag but scale down the workload dimensions (e.g., reduce the loop bounds for channels C, feature map height H, or width W) to match the average computational load or data volume expected in the sparse SNN.

Mechanism: If an SNN layer is expected to have an average activation sparsity of, say, 90% (meaning only 10% of potential activations are non-zero spikes), one could potentially reduce the effective number of input channels or spatial dimensions in the ZigZag workload definition by a factor reflecting this sparsity. This would reduce the total operation and access counts calculated by ZigZag's dense cost model.


Potential Benefits: This approach might offer a very coarse, average-case approximation of energy or latency. If calibrated against known sparse behavior, it could potentially capture average bandwidth requirements or provide rough estimates for high-level architectural comparisons where absolute accuracy is not paramount.


Severe Limitations: Despite its apparent simplicity, this scaling approach suffers from critical flaws:

Loss of Temporal Dynamics: SNN activity is inherently dynamic and often bursty. Average sparsity hides the temporal characteristics of spike trains. Real systems experience peaks and troughs in activity, which significantly impact buffer occupancy requirements, instantaneous bandwidth demands, potential for pipeline stalls, and opportunities for dynamic power/clock gating.24 An average-case model cannot capture these crucial dynamic effects.
Ignoring Sparsity Patterns: Real-world sparsity is rarely uniform. It can be unstructured (random zeros) or structured (e.g., block sparsity, 2:4 sparsity).85 Different patterns have different implications for hardware efficiency and require different acceleration techniques (e.g., index-based skipping vs. structured sparsity units). A simple scaling factor treats all sparsity uniformly and misses these architectural nuances.
Inaccurate Latency Modeling: SNN latency depends on the timing and propagation of actual spikes through the network layers, often involving state dynamics and threshold crossings. It is not simply a scaled version of dense computation time. Maximum latency, critical for real-time applications, might be dictated by worst-case activity bursts, which are obscured by averaging.
Calibration Difficulty: Determining the appropriate scaling factor is non-trivial. Average sparsity can vary significantly across layers, time, and input stimuli. Relying on a single scaling factor requires prior knowledge or simulation of the SNN, partially defeating the purpose of a fast, predictive analytical model.
Missing Cost Components: Crucially, scaling the workload dimensions within ZigZag's existing framework does nothing to introduce the cost models for SNN-specific elements like neuron state memory access or event handling logic overhead. These remain unrepresented.



Supporting Context: While workload scaling is a concept used in performance modeling generally 70, its application here fails to address the fundamental mismatch in the nature of the computation (event-driven, stateful vs. synchronous, stateless loops) and the missing cost components. The need for tools that explicitly model sparsity patterns 70 and temporal dynamics 68 demonstrates that simple scaling of dense models is insufficient for accurate SNN accelerator analysis.

3.2 Analysis of Implementation StrategyBeyond the core assumptions, the proposal suggests specific implementation choices within the ZigZag framework: forcing an Input Stationary dataflow and placing input/channel loops outermost in the temporal mapping.3.2.1 Forcing Input Stationary Mapping

Rationale: The choice of Input Stationary (IS) dataflow likely stems from the intuition that input spikes trigger computations in SNNs. In a canonical IS dataflow for DNNs, input activation values are held stationary in PE registers or local buffers, while weights are streamed in to perform computations.69 This seems analogous to an incoming spike activating synapses associated with specific weights. Furthermore, for highly sparse inputs (sparsity > 50%), IS can potentially require fewer total memory accesses compared to Output Stationary (OS) dataflow, as argued in.42


Effectiveness and Limitations:

Partial Analogy: While IS captures the input-trigger aspect to some extent, it's an imperfect analogy. SNN computation is not just triggered by inputs but also produces sparse outputs (spikes) and involves reading and updating internal neuron states.
Output Sparsity Handling: IS dataflow prioritizes input reuse. However, efficiently handling the generation and writeback of sparse output spikes might be better served by an OS dataflow, where partial sums (accumulating towards the neuron membrane potential) are kept stationary.69 The trade-off is complex: OS has higher input read costs under high sparsity but potentially lower state/output write costs.42
Neuron State Access: A major deficiency is that the IS dataflow, as defined in the context of DNNs (Weights, Inputs, Outputs), does not naturally incorporate the access pattern for neuron state memory. The state (e.g., Vm) must be read, updated based on the weighted input spike, and written back, potentially for every event.39 This read-modify-write cycle is not part of the standard IS definition in tools like ZigZag.
Synchronous Constraint: Even if IS were the optimal dataflow, ZigZag models it within its synchronous, loop-based execution framework. This fails to capture the asynchronous arrival and processing of individual spike events that characterize real event-driven hardware.13
Optimality: Forcing IS is restrictive. The optimal dataflow for SNNs is an open research question and likely depends on the specific layer type (convolutional, fully connected), the degree and type of sparsity (activation, weight, structured, unstructured), the neuron model complexity, and the target hardware architecture.42 Architectures like SpinalFlow used an OS-based approach, albeit with limitations.42 Hybrid dataflows (e.g., WS-LOS 93 or hybrid WS/OS 92) are also being explored. Mandating IS within the adapted ZigZag model ignores this complexity and may lead to inaccurate comparisons or suboptimal design choices.


3.2.2 Placing Input/Channel Loops as Outer Loops

Rationale: This mapping strategy involves ordering the temporal loops in ZigZag such that loops iterating over input dimensions (e.g., batch N, input channels C, spatial dimensions H, W) are placed outermost. The intention is likely to simulate the processing of different input events or features over time before iterating through filter dimensions (K, R, S) in the inner temporal loops. This attempts to group computations related to specific inputs temporally within the synchronous loop structure.


Representation and Limitations:

Failure to Capture Asynchronicity: This loop ordering fundamentally fails to represent the asynchronous, data-driven nature of event arrival in SNNs. Spikes arrive based on the network's dynamic state and input stimuli, not according to a predetermined, fixed loop order.72 Real event-driven systems process events as they arrive, often concurrently and out of strict order.13 A fixed loop order imposes an artificial synchronicity and sequentiality that does not exist in the target system.
Ignoring Temporal Dependencies and State: SNN computation is inherently stateful and temporally dependent. A neuron's output spike depends on its internal state (Vm), which in turn depends on its previous state and the history of input spikes received over time.39 ZigZag's stateless loop iterations cannot model this evolution of neuron state across asynchronous time steps or event arrivals. The concept of discrete "timesteps" often used in SNN simulation is itself an abstraction that may not map directly to synchronous loop iterations.26
Missing Event Handling Overheads: The structured loops provide no mechanism to account for the computational and energy overhead associated with the event handling logic itself. This includes detecting an incoming spike, decoding its source address, fetching the corresponding synaptic weights and the target neuron's current state, potentially managing event queues, and scheduling the actual computation.38 These overheads can be significant, especially in highly parallel or complex systems.13
State Management Logic: The loop structure does not represent the logic for updating neuron states, including integration, leak (decay over time), threshold comparison, and reset mechanisms, which are essential parts of SNN neuron models.39



Supporting Context: The development of asynchronous hardware components like NoCs 97 and specialized asynchronous processing techniques 72 underscores the inadequacy of synchronous loop-based models for event-driven computation. Dedicated SNN modeling tools like STEMS explicitly incorporate the temporal dimension and state management beyond simple loop ordering to address these limitations.68

The proposed adaptation strategy, combining dense modeling with specific dataflow and loop ordering choices, represents a significant abstraction from the reality of event-driven computation. While potentially offering a computationally inexpensive way to get some estimate, it relies on assumptions (average-case behavior, negligible overheads, specific dataflow optimality) that are unlikely to hold true for real-world SNN accelerators operating with complex, dynamic, and sparse workloads. The predictive accuracy of such a model for performance and particularly energy consumption is expected to be low, limiting its usefulness for detailed DSE or architectural decision-making.Furthermore, the focus on manipulating ZigZag's existing inputs (workload dimensions) and mapping parameters (dataflow, loop order) sidesteps the more fundamental issue: the need to extend the cost model itself. Accurate SNN modeling requires incorporating new cost components that are absent in the standard ZigZag framework. This includes the energy associated with accessing neuron state memory (parameterized by memory technology, access frequency, and data precision) 39 and the energy consumed by the event handling logic (parameterized by event rate and hardware complexity).38 Additionally, the latency model needs to move beyond synchronous loop counts to account for asynchronous dependencies, potential queuing delays, and event propagation times.72 Without adding these SNN-specific cost dimensions and refining the performance model, simply rearranging the existing components within ZigZag's dense, synchronous framework will inevitably yield inaccurate results.4. Augmenting ZigZag Energy Estimates for Event-Driven CharacteristicsGiven the inherent limitations of applying the standard ZigZag model—even with adaptations like forcing Input Stationary dataflow or scaling workload dimensions—to event-driven accelerators, obtaining more realistic energy estimates necessitates augmenting its output. This involves post-processing the results from the dense ZigZag model or incorporating external information to account for SNN-specific characteristics like sparsity, event handling, state memory, and power gating.4.1 Applying Activity Sparsity Factors
Method: A straightforward approach is to take the compute operation counts (e.g., MACs) and memory access counts (e.g., activation reads/writes) generated by ZigZag's dense model and scale them down based on an expected or measured sparsity factor. For instance, if the average activation sparsity is 90% (10% activity), the activation read/write counts and the number of effective operations could be multiplied by 0.1 before calculating the associated dynamic energy. Weight sparsity could be handled similarly if applicable.
Pros: This method is simple to implement and provides a first-order correction acknowledging that not all potential operations occur. It can capture the average reduction in computational load and data movement due to sparsity.
Cons: This approach suffers from several inaccuracies:

Uniformity Assumption: It assumes sparsity is uniform across the tensor and throughout time, which is often not the case. Real workloads exhibit temporal variations and spatial patterns in sparsity.24
Ignoring Patterns: It doesn't differentiate between structured and unstructured sparsity, which have different hardware implications.85
Overhead Neglect: It fails to account for the overhead of managing sparsity in hardware, such as the energy cost of indexing schemes (for unstructured sparsity) or specialized logic (for structured sparsity) needed to skip zeros.70
Workload Imbalance: It doesn't capture the impact of workload imbalance caused by uneven sparsity distribution across PEs, which can lead to low utilization and increased latency/leakage energy, even if the average sparsity is high.85
Input vs. Output Sparsity: It may not correctly distinguish between the impact of input activation sparsity and output activation sparsity on different memory accesses (e.g., input reads vs. output writes).


Supporting Context: Sparsity is undeniably a defining characteristic and key efficiency driver for SNNs.24 Numerous studies quantify sparsity levels in various SNN models and benchmarks.13 However, research explicitly developing sparse accelerator models like Sparseloop highlights the inadequacy of simply scaling dense results 70, emphasizing the need to model sparse formats and execution mechanisms directly.
4.2 Estimating Event Handling Overhead
Method: This involves developing a separate energy model for the logic associated with processing events. This logic might include spike detection circuits, address event representation (AER) encoding/decoding, event queues, Network-on-Chip (NoC) routers for inter-core communication, and scheduling logic within a core.67 An estimate of the total number of events processed (e.g., from a functional SNN simulation or derived from the average sparsity factor) is multiplied by an estimated energy cost per event. This per-event cost would ideally be derived from circuit-level simulations or hardware characterization data for the specific components involved. The resulting event handling energy is then added to the augmented ZigZag estimate.
Pros: This method explicitly incorporates a significant energy cost component that is entirely absent in the standard ZigZag model. The per-event cost can be refined based on the specific architectural choices (e.g., asynchronous vs. synchronous NoC, multicast support, queue depth).
Cons:

Cost Variability: The energy cost per event can vary dramatically depending on the hardware implementation. Asynchronous logic might offer lower idle power but potentially higher switching energy per event compared to finely-gated synchronous logic.98 NoC complexity (topology, routing, flow control) significantly impacts energy.97 Multicast support can reduce total events but might increase router complexity.97
Event Count Estimation: Accurately estimating the number of events processed, especially inter-core events requiring NoC traversal, is challenging without detailed simulation, as it depends on the SNN's dynamics and mapping.
Significance: Event processing overhead is often non-trivial and can become a bottleneck, particularly if not optimized through techniques like spike grouping.13


Supporting Context: The energy consumption of event handling logic, including AER circuits and NoCs, is a recognized factor in neuromorphic chip design.38 Architectural features like multicasting 97 and optimizations like spike grouping 13 are specifically aimed at mitigating this overhead.
4.3 Estimating Neuron State Memory Costs
Method: This augmentation requires identifying the state variables associated with the SNN's neuron model (e.g., membrane potential Vm, possibly leak parameters, adaptive thresholds).1 The size and precision of these variables must be determined. An estimate of the frequency of state memory accesses (reads and writes) is needed – this could be per timestep in a time-driven simulation model or per relevant event (input spike arrival, neuron firing) in an event-driven model. Using memory energy models (similar to those used by ZigZag/Accelergy for weights/activations, but applied to the state memory characteristics), the energy per access is calculated based on the memory technology (SRAM, eNVM like RRAM, MRAM, PCM) 12, size, and the estimated number of accesses. This state memory energy cost is then added to the total.
Pros: This addresses one of the most critical omissions when applying DNN models to SNNs. It allows for modeling the impact of different neuron complexities, state precisions, and memory technologies on energy consumption.
Cons:

Access Frequency Estimation: Estimating the state access frequency accurately is difficult without simulating the SNN dynamics. It depends heavily on the neuron model, the input spike rate, the network's internal activity, and the execution model (time-stepped vs. event-driven).
Memory Size and Dominance: Neuron state memory can be very large, often significantly exceeding the parameter memory size, especially for networks with high spatial resolution or complex neuron models.13 Its energy cost can therefore dominate the overall budget.39 Inaccuracies in estimating state access energy can lead to large errors in the total energy prediction.


Supporting Context: Neuron state is fundamental to SNN operation.1 The significant memory footprint and access cost associated with neuron states are widely recognized as major challenges in neuromorphic hardware design.13 Different neuron models (e.g., LIF, Izhikevich, AdEx) impose different state storage and computation requirements.105 Specialized tools like STEMS explicitly model neuron states as first-class operands and optimize mappings considering state reuse.68 The power consumption of SNN accelerators is directly linked to the size and access rate of state memory.39
4.4 Accounting for Power/Clock Gating Savings
Method: Based on the estimated average sparsity (or activity level), calculate the fraction of time PEs or other hardware blocks are expected to be idle. Apply a reduction factor to the static/leakage power component of these blocks, assuming that effective power gating or clock gating mechanisms are implemented in the hardware. The reduction factor would depend on the assumed efficiency of the gating mechanism (i.e., how much leakage is saved when gated).
Pros: This attempts to capture potential energy savings arising from the idleness enabled by sparse, event-driven computation.
Cons:

Gating Efficiency: The actual energy savings depend heavily on the granularity (e.g., gating individual PEs vs. entire cores) and efficiency (e.g., leakage reduction percentage, energy/latency overhead of entering/exiting gated state) of the power/clock gating implementation, which varies across designs.
Idle Time Estimation: Estimating the true idle time fraction accurately is difficult. Average sparsity doesn't capture burstiness; components might be idle for short durations insufficient for effective power gating, or activity might be unevenly distributed.
Gating Logic Overhead: The energy cost of the gating control logic itself is typically ignored in this simplified approach.


Supporting Context: The event-driven nature of SNNs naturally leads to periods of inactivity, making power and clock gating viable strategies.24 Fine-grained clock/power gating is proposed in some ultra-low-power SNN designs.78 Asynchronous designs inherently avoid clock power when idle.100 SpiNNaker2 employs dynamic voltage and frequency scaling (DVFS) driven by activity, a related power management technique.124
4.5 Limitations of AugmentationWhile these augmentation methods attempt to bridge the gap between ZigZag's dense, synchronous model and the reality of event-driven computation, they are essentially patches applied to a fundamentally mismatched foundation. They improve the estimate compared to using raw ZigZag output, but they introduce their own layers of approximation and dependency on external information.A significant issue is the reliance on average values, particularly for sparsity and event rates. Event-driven systems exhibit dynamic behavior with potentially high variance in activity over time and space. Average values obscure this, potentially leading to underestimation of peak bandwidth requirements, buffer sizes needed to handle bursts, and overestimation of power gating effectiveness. The accuracy of the final augmented estimate becomes heavily dependent on the quality of the external inputs (e.g., sparsity factors, event counts derived from functional simulations) and the validity of the simplifying assumptions made in each augmentation step (e.g., uniform sparsity, average event cost). The resulting model is less an analytical prediction derived from first principles, as ZigZag aims to be for DNNs, and more a calibrated estimation framework.Furthermore, these augmentation methods typically treat each factor (sparsity, event handling, state memory, gating) in isolation, adding or subtracting their estimated costs. This fails to capture the crucial interdependencies between these factors in a real system. For example, a period of high spike activity (low sparsity) simultaneously increases the number of events to be handled by the control logic 38, raises the frequency of neuron state memory accesses 39, potentially causes NoC congestion, and reduces the opportunities for effective power gating.78 Conversely, high sparsity reduces demands on all these components. A simple additive approach misses these coupled effects, potentially leading to significant inaccuracies, especially when evaluating architectures under varying workload conditions or across different SNN models with diverse activity profiles.5. Comparative Analysis: Adapted ZigZag vs. Specialized SNN Tools (STEMS)To further understand the limitations of adapting ZigZag, it is instructive to compare its potential capabilities (even with augmentations) against tools specifically designed for modeling and exploring the design space of SNN accelerators. STEMS (Spatial-Temporal Mapping Tool For Spiking Neural Networks) serves as a pertinent example of such a specialized tool.68The following table summarizes the key differences in modeling capabilities:FeatureAdapted ZigZag (ZigZag + Augmentations)STEMS (Specialized SNN Tool)Neuron State HandlingNot a native operand; Cost estimated externally and added post-hoc.Explicitly modeled as first-class operands (e.g., Vm); Integrated into cost model & DSE.Temporal DimensionApproximated via loop ordering; State evolution across time not modeled.Explicitly incorporated in workload & mapping; Optimizes for temporal data reuse (state).Asynchronicity / Event HandlingNot modeled in core; Event logic cost added post-hoc via average rate.Framework designed for SNNs; Considers temporal scheduling (e.g., time-batched).Sparsity ModelingHandled via external scaling factors applied to dense results.Implicitly handled as SNN characteristic; Optimizes data movement for SNNs.Mapping Optimization ScopePrimarily intra-layer spatial/temporal mapping for W/I/O.Intra-layer and Inter-layer DSE across space and time, considering neuron states.Core EngineAnalytical model for synchronous, dense, stateless loops.Builds on ZigZag for intra-layer cost, adds inter-layer/temporal/stateful modeling layer.5.1 Explicit Modeling of Neuron States as Operands
Adapted ZigZag: As discussed in Section 4.3, neuron state is not a native operand type in ZigZag. Its existence and associated memory access costs must be handled through external calculations and post-hoc addition to the energy estimate derived from the W/I/O-centric model. The impact of state memory access patterns on the optimal dataflow or scheduling choices within ZigZag's core optimization logic remains unmodeled.
STEMS: In contrast, STEMS explicitly defines neuron states (e.g., the accumulator state like membrane potential in LIF neurons, and potentially other state variables for more complex models) as first-class operands within its workload description.68 This allows the framework to directly incorporate the energy and latency costs of accessing state memory into its DSE process. STEMS actively optimizes mappings to minimize data movement specifically for these neuron states, considering opportunities for state reuse both spatially across PEs and temporally across time steps or batches.68 It can explore the trade-offs associated with storing states at different memory hierarchy levels. Furthermore, STEMS can facilitate neuron state optimization, potentially identifying layers or conditions where state variables can be eliminated or reduced in precision without impacting accuracy, leading to significant memory and energy savings.68
5.2 Incorporation of the Temporal Dimension and Asynchronous Event Handling
Adapted ZigZag: The temporal aspect of SNNs is crudely approximated by manipulating the order of loops in the temporal mapping (Section 3.2.2). This fails to capture the asynchronous nature of event arrivals and the stateful evolution over time. The costs associated with event handling logic must be added via augmentation (Section 4.2). The model cannot represent dynamic, event-triggered scheduling, inter-event timing dependencies, or potential bottlenecks arising from asynchronous communication or processing.
STEMS: STEMS is explicitly designed to handle the spatial and temporal dimensions inherent in SNNs.68 Its workload representation includes time as a dimension, and its DSE explores mapping optimizations across both space and time. This allows it to reason about temporal data reuse, particularly for neuron states that persist across time steps.68 STEMS evaluates different high-level scheduling strategies that explicitly consider the temporal aspect, such as single-timestep vs. time-batched execution, and layer-by-layer vs. layer-fused processing.132 While the exact mechanism for modeling fine-grained asynchronicity versus discrete time-steps might vary, the framework's structure is built to address the temporal processing flow and state management critical to SNNs, going far beyond simple loop ordering.
5.3 Direct Modeling of Sparsity and its Impact
Adapted ZigZag: Sparsity is addressed indirectly through post-processing scaling factors applied to the results of the dense model (Section 4.1). The core analytical engine does not model sparsity. Consequently, it cannot capture the influence of specific sparsity patterns (e.g., structured vs. unstructured) on hardware efficiency, nor can it model the overheads associated with sparse data formats (e.g., coordinate lists, bitmaps) or the specific hardware logic used for skipping/gating zero computations.
STEMS: As a tool designed for SNNs, STEMS must inherently account for sparsity, as it is a defining characteristic of SNN computation and efficiency. Its primary goal is to minimize data movement considering SNN characteristics, which naturally includes sparse activations (spikes) and potentially sparse weights.68 While STEMS leverages ZigZag for intra-layer cost estimation 132, suggesting it might inherit a dense view at the lowest computational kernel level, its higher-level inter-layer and temporal optimization framework is geared towards managing the consequences of SNN behavior. This includes optimizing the movement and reuse of neuron states and intermediate features (spikes), whose patterns are dictated by the sparse, event-driven dynamics. The specific mechanisms used within STEMS to model the impact of sparsity (e.g., statistical models, event counting) require further clarification based on its detailed implementation, but its design intent is clearly focused on SNNs where sparsity is paramount.
The development and capabilities of specialized tools like STEMS serve as strong evidence that the limitations of adapting general DNN modelers like ZigZag are recognized within the research community. STEMS's architecture, which builds upon ZigZag's intra-layer modeling but adds crucial layers for inter-layer, temporal, and stateful analysis 68, effectively validates the critique that base ZigZag is insufficient for comprehensive SNN DSE. It highlights that while ZigZag might provide a useful foundation for modeling the core computation within a single layer or timestep (if abstracted as dense), it requires significant SNN-specific extensions to handle the complexities of state evolution, temporal dependencies, and data movement across layers and time.Furthermore, the optimization objectives differ. STEMS explicitly targets the minimization of off-chip data movement and energy consumption, with a keen focus on the impact of neuron states.68 This directly addresses the memory wall problem that is often exacerbated by the large state memory requirements in SNNs.13 In contrast, traditional DNN accelerator DSE often prioritizes maximizing computational throughput (measured in Tera-Operations Per Second, TOPS or Giga-Operations Per Second, GOPS) and the associated energy efficiency (TOPS/Watt).17 While throughput is important, optimizing solely for peak TOPS based on a dense model might lead to inefficient SNN accelerator designs if the primary bottlenecks lie in state memory access or event handling, rather than raw computation. The distinct focus of STEMS underscores the need for tailored optimization goals and modeling tools that accurately reflect the unique characteristics and bottlenecks of SNN accelerators.6. Synthesis and CritiqueThe analysis presented in the preceding sections provides a basis for a critical evaluation of the proposed methodology for modeling event-driven accelerators using the ZigZag framework. This section synthesizes the findings and offers a critique of the approach's feasibility and accuracy, particularly concerning energy estimation.6.1 Summary of Findings
Fundamental Mismatch: ZigZag's core modeling paradigm, based on synchronous execution of dense tensor operations within a stateless loop structure, is fundamentally misaligned with the principles of event-driven accelerators, which rely on asynchronous, sparse, and stateful computation.24
Adaptation Limitations: The proposed adaptations—modeling a dense equivalent workload, attempting workload scaling, forcing Input Stationary dataflow, and using specific loop orderings—fail to adequately capture the defining characteristics of event-driven systems. Ignoring sparsity negates the primary source of efficiency 34, workload scaling misses crucial temporal dynamics 24, and the chosen mapping strategies do not accurately reflect asynchronous event handling or state management.42
Augmentation Insufficiency: While augmenting ZigZag's outputs with estimated costs for sparsity effects, event handling, neuron state memory, and power gating savings is necessary, these methods act as patches on an inadequate foundation. They rely heavily on external data (often from more detailed simulations), introduce their own simplifying assumptions (e.g., uniform sparsity, average costs), and fail to capture the complex interdependencies between these factors.
Capability Gap: Compared to specialized SNN modeling tools like STEMS, the adapted ZigZag approach lacks the necessary mechanisms to explicitly model neuron states as operands, incorporate the temporal dimension effectively, handle asynchronous event flow, or integrate sparsity modeling directly into the optimization process.68
6.2 Critique of MethodologyBased on these findings, the proposed methodology of adapting ZigZag for event-driven accelerator modeling suffers from significant drawbacks:
Inherent Inaccuracy: The fundamental mismatch between the model's assumptions and the target system's reality makes high-fidelity prediction highly unlikely. Attempting to model asynchronous, sparse, stateful systems using a synchronous, dense, stateless framework inherently violates the core principles of event-driven computation.
Unreliable Energy Estimation: Energy estimation is particularly problematic. The approach either ignores or poorly approximates major SNN-specific energy consumers (neuron state memory access 39, event handling logic 38) and saving mechanisms (sparsity exploitation 15, power gating 78). Over-reliance on average sparsity factors hides critical dynamic behavior impacting power consumption. The complex interplay between activity levels, memory access patterns, and control logic overhead is not captured, leading to potentially large errors in energy prediction.
Misguided Design Space Exploration: Using an inaccurate cost model for DSE can lead designers toward suboptimal architectures and mappings. The model might highlight bottlenecks relevant to dense DNNs (e.g., maximizing MAC unit utilization based on dense assumptions) while failing to identify or prioritize critical SNN-specific bottlenecks, such as state memory bandwidth or event processing latency. Optimization efforts based on such a model may be misdirected, failing to achieve the potential efficiency gains of event-driven hardware.
The effort required to augment ZigZag sufficiently to achieve even moderate accuracy for SNN modeling appears substantial. It involves adding models for sparsity effects, event logic costs, state memory access patterns and energy, and potentially aspects of asynchronous timing. Each of these additions requires careful parameterization based on specific hardware details and may necessitate input from auxiliary simulations (e.g., for event counts or sparsity statistics), partially negating the speed advantage of an analytical model. Specialized tools like STEMS integrate these concepts more fundamentally.68 Therefore, the cumulative complexity of accurately patching ZigZag might approach that of developing a dedicated SNN tool, suggesting that leveraging or adapting existing specialized tools is likely a more robust and efficient path.Furthermore, the choice of modeling framework influences the types of insights designers gain. An adapted ZigZag model, rooted in dense loop analysis, will likely continue to emphasize bottlenecks related to the memory hierarchy as perceived through this dense lens (buffer sizes, bandwidth, PE utilization based on loops).56 In contrast, a tool like STEMS, designed with explicit temporal and stateful modeling, is better equipped to reveal bottlenecks unique to SNNs, such as those related to temporal dependencies, state management efficiency across time, inter-layer event propagation delays, or the impact of different scheduling strategies on state memory traffic.68 Relying solely on the adapted ZigZag approach could lead designers to overlook these critical SNN-specific optimization opportunities.6.3 Potential RefinementsIf adapting ZigZag remains the chosen path despite its limitations, several refinements beyond the initial proposal could be considered, aiming to incorporate more SNN-specific aspects directly into the model:
Integrate Statistical Sparsity Models: Instead of simple post-hoc scaling factors, incorporate statistical models of sparsity, similar to the approach used in Sparseloop.70 This would involve modeling sparsity patterns and distributions and analyzing the overhead of sparse data formats (e.g., compressed sparse row/column, coordinate lists, bitmaps) and the effectiveness of gating/skipping logic directly within the analytical framework.
Add Explicit SNN Cost Terms: Extend ZigZag's cost calculation engine to include dedicated terms for neuron state memory accesses and event handling logic. These terms should be parameterized based on technology assumptions (e.g., energy per bit for SRAM vs. RRAM state memory) and architectural choices (e.g., energy per event for a specific NoC router design). The model would need inputs for estimated state access counts and event rates.
Hybrid Simulation/Analytical Modeling: Combine the strengths of analytical modeling and simulation. Use ZigZag (potentially with integrated sparsity modeling) to estimate the cost of the core computational kernel execution. Couple this with a lightweight, abstract event-driven simulator or queuing model to capture the temporal dynamics, estimate event rates and contention more accurately, and derive realistic input parameters for the state memory and event handling cost terms. Frameworks like SANA-FE adopt a coarse-grained time-step simulation approach.3
6.4 Alternative ApproachesGiven the challenges of adapting ZigZag, pursuing alternative modeling strategies is highly recommended:
Utilize Specialized SNN Tools: Employ existing frameworks specifically designed for SNN accelerator DSE, such as STEMS 68 or SANA-FE.3 These tools inherently incorporate SNN characteristics like statefulness and temporal processing into their modeling and optimization loops. Other potential tools or simulators might be identified through surveys of neuromorphic software frameworks.3
Cycle-Accurate Simulation: If high accuracy is paramount and simulation time permits, using detailed cycle-accurate simulators specific to the target neuromorphic architecture (if available) provides the most reliable performance and energy data. However, this approach is often too slow for extensive DSE.3
Develop Novel SNN Analytical Models: Create new analytical cost models from the ground up, building in event-driven principles, state dynamics, asynchronous timing considerations, and sparsity modeling from the outset, rather than retrofitting a dense, synchronous model.
7. ConclusionThis report has critically analyzed the proposed methodology of adapting the ZigZag cost modeling framework, originally designed for dense DNN accelerators, to estimate the energy consumption of event-driven accelerators, such as those used for SNNs.The core finding is that ZigZag's fundamental assumptions—data density and synchronous, stateless computation—are fundamentally incompatible with the operational principles of event-driven systems, which are characterized by sparsity, asynchronicity, and stateful neuron dynamics. The proposed adaptations, including modeling a dense equivalent workload, applying workload scaling, forcing an Input Stationary dataflow, and using specific loop orderings, fail to adequately capture these defining characteristics. Consequently, this approach leads to significant inaccuracies, particularly in energy estimation, by neglecting or poorly approximating major cost components like neuron state memory access and event handling logic, as well as the substantial energy savings derived from sparsity.While augmenting ZigZag's outputs with post-hoc estimations for these missing factors is possible, this strategy acts as a series of patches on an inadequate foundation. It relies on external data, introduces further approximations, and fails to model the complex interdependencies between sparsity, state, and event dynamics. The resulting energy estimates are unlikely to be reliable enough for effective design space exploration or accurate architectural comparisons.Specialized SNN modeling tools, such as STEMS, explicitly incorporate neuron states, the temporal dimension, and SNN-specific optimization objectives, demonstrating a recognized need for frameworks tailored to event-driven computation. These tools offer a more robust and accurate approach compared to adapting a fundamentally mismatched DNN modeler.Therefore, the final assessment is that the proposed adaptation of ZigZag is not a feasible or reliable method for accurately modeling the energy consumption of event-driven accelerators. The inherent mismatches lead to significant limitations that cannot be adequately overcome by simple adaptations or post-hoc augmentations.For designers seeking to model and optimize event-driven accelerators, it is strongly recommended to utilize specialized SNN modeling frameworks (e.g., STEMS, SANA-FE) or to invest in developing novel analytical or hybrid models that are built from the ground up with event-driven principles, sparsity, and statefulness as core components.Future work in this area should focus on developing unified modeling frameworks capable of handling both dense ANNs and sparse SNNs within a consistent structure, advancing statistical models for characterizing sparse event-based data and asynchronous timing, and establishing standardized benchmarks and metrics specifically for evaluating and comparing the performance and efficiency of diverse event-driven and neuromorphic computing systems.4 Such efforts are crucial for accelerating progress in the design of next-generation, energy-efficient AI hardware.

#### Question 3:
Along with the Architecture design the cost model also needs to be evolved/modified to be able to effectively calculate the Latency, energy and EDP. How this cost model needs to be modified especially for the latency and energy calculations to achieve correct estimated values for Event driven models

Here is the cost model:
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


#### What are the difference b/w the seneca and the frame based accelerator in terms of memory for being modelled on to the zigzag?

Now, A final research can be done where the following context from the previous chat is considered along with the actual system requirements to answer the question of what final changes need to be made to the zigzag cost model to develop an event driven accelerator exploration tool.



Here is the context that needs to be extracted from the previous chats

1. The event driven accelerator model exactly like seneca and its difference from the frame based accelerators

2. The Working of the mapping exploration tool Zigzag and how it functions



Here are my sytem design choices

1. I am going to run conventional ANN on my event driven accelerator rather than SNN

2. I am first going to model the Dense case exploration even though the latency and energy values are more as I am intrested in the flow developement rather than the sparstity for now. Once the flow is ready I will include sparsity.



Now, I would want you to take the STEMS tool that works for the SNN exploration, and help me understand what exact changes it made and where it made in the tools of Zigzag and stream to achieve the desired functionality

As in my case too I need to model the Event driven accelerator behaviour on the top of Zigzag and Stream.

Both are same, Only difference is that STEMS focus on modeling a specific form of Network and I am focusing on the specific type of accelerator.



I need you to do deep research on

1. Identifying the specific changes STEMS made to Zigzag and Stream and what blocks are changed in this standard tool

2. From the previous discussion extract the specific event driven accelerator behaviour that needs to be modelled into Zigzag and Stream for my specific system design

3. Finally show a block diagram which indicates these changes are made to the zigzag and stream tools to achieve the event driven architecture behaviour by utilising the already exsiting tools and adapting their features for our need.


## The new model developed is as follows

Based on the seneca design and the Zigzag way of modeling the architecture. I modelled the Seneca architectue in zigzag framwork for exploration

Can you verify this modeling that is being done of the event-driven architecture in the zigzag tool?

The architecture: 
"name: Event_Driven_Accelerator

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
    served_dimensions: [D1]

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
    served_dimensions: []

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
    served_dimensions: []

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
    served_dimensions: []
"

Here the modeling of the NOC is done by the "shared_memory_inputs_outputs", which stores the input and passes them to the NPEs, and the Final output are written here for transmission and this cost is used for the modeling of the NOC. The rest of everything is as it is the seneca architecture.

Please verify my design of the seneca that is being modelled in zigzag.


Below is the table that I followed to model Seneca
"Yes, incorporating the corrections we discussed, here is the final merged table summarising the storage and access of inputs, weights, and outputs across the memory levels in SENECA, drawing on information from the sources:

| Component | Memory Level                                      | Storage Location(s)                                                                                                                                   | Access Mechanism(s)                                                                                                                                                                                                                                                                                                                         |
| :-------- | :------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Inputs** | **Register File (within NPEs)**                  | Potentially transient data associated with input events during processing [Implied from context in multiple sources].                                    | Input events **arrive at the SENECA core via the Network on Chip (NoC)**. The **RISC-V controller** is interrupted upon arrival and performs pre-processing, which may involve retrieving **local information from the Data Memory (SRAM)**. Pre-processed information or immediate data associated with the input event might then be loaded into and used within the **NPE's register file during computation**.                                                                                                                                                                                                                                                                                                                                                                                         |
|           | **Local Data Memory (SRAM) per core**           | Addresses of relevant parameters related to input events may be stored here.                                                                  | The **RISC-V controller** might retrieve information from the Data Memory based on the incoming event. The **NPEs access data from the Data Memory** during computations.                                                                                                                                                                |
|           | **Shared Memory (MRAM/HBM) per cluster (optional)** | If input-related parameters are very large, they might reside here.                                                                        | Accessed by **NPEs** via the **Shared Memory Pre-fetch Unit (SMPU)** [Implied from context related to parameter storage in shared memory]. It's important to note that **shared memory primarily stores parameters (like weights)** when local SRAM is insufficient, rather than being the primary path for receiving initial input events.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Weights** | **Register File (within NPEs)**                  | Weights can be loaded into the **register file within each NPE** and reused for multiple computations to improve energy efficiency.         | **NPEs** can directly access weights from their internal register file.                                                                                                                                                                                                                                                                 |
|           | **Local Data Memory (SRAM) per core**           | **Synaptic weights are primarily stored in the Data Memory (SRAM) of each SENECA core**.                                                 | The **NPEs access these weights directly from the Data Memory** to perform neural computations.                                                                                                                                                                                                                                 |
|           | **Shared Memory (MRAM/HBM) per cluster (optional)** | When synaptic weights exceed the capacity of the local Data Memory, they can be stored in **shared memory (MRAM or HBM)**. | Accessed by **NPEs** via the **Shared Memory Pre-fetch Unit (SMPU)**, which can pre-fetch necessary parameters to mitigate latency. SENECA has the flexibility to dynamically allocate weights to different memory blocks.                                                                                                         |
| **Outputs** | **Register File (within NPEs)**                  | Intermediate neuron states leading to output spikes are held here during integration and processing [Implied from 79, 87, 92].                       | **NPEs** perform computations that update neuron states in their register files. These states are then used to determine if a neuron fires.                                                                                                                                                                                             |
|           | **Local Data Memory (SRAM) per core**           | **Neuron states** (membrane potential) are stored in the local Data Memory.                                                 | **NPEs** read neuron states from and write updated states back to the Data Memory.                                                                                                                                                                                                                                              |
|           | **Via NoC from the core**                        | Output spikes, after processing by the event generator, are formatted into **Address Event Representation (AER)** packets and sent via the **NoC**. | The **event generator unit** within the core converts the generated spikes into AER format. The **RISC-V controller** may perform post-processing before transmission via the **NoC**. Within the same core, outputs can be directly passed to subsequent layers without involving the NoC.                                                      |

This table now more accurately reflects the initial arrival of input events via the NoC and the subsequent processing steps involving the RISC-V controller, local SRAM, and the NPE register file. It also clarifies the primary role of shared memory for parameter storage."

First verify this information with the paper, then verify my zigzag design with the table and finally if there are any corrections required let me know