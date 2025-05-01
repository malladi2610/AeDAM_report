# Mapping Constraints for Deep Learning Accelerators

## Key Points
- Spatial mapping likely involves splitting workload dimensions across processing elements for parallel execution, as you described.
- Temporal mapping seems to define the execution order within each processing element, aligning with your dataflow strategies.
- Your pseudo-code for output, weight, and input stationary dataflows appears simplified but may need adjustment for completeness.
- The evidence leans toward a table that clearly maps each variable to spatial or temporal roles for intuitive understanding.

## Spatial Mapping
Spatial mapping distributes parts of a neural network's workload, like output channels (K) or spatial dimensions (OX, OY), across multiple processing elements (NPEs) in your architecture. This allows parallel computation, boosting speed. For example, splitting K across 8 NPEs means each NPE handles a subset of output channels simultaneously.

## Temporal Mapping
Temporal mapping sets the order of operations within each NPE. Different strategies—output stationary, weight stationary, and input stationary—prioritize keeping certain data (outputs, weights, or inputs) fixed while looping over others. Your pseudo-code shows how loop orders change, like prioritizing output dimensions (OX, OY) in output stationary.

## Verification and Corrections
Your explanations are mostly accurate:
- **Spatial Mapping**: Correctly described as vectorization for parallel execution across NPEs.
- **Temporal Mapping**: Accurately captures execution order, but your pseudo-code omits loops for K and C, which are typically included in convolution operations.

**Corrections**:
- The pseudo-code should include loops for all dimensions (K, C, FX, FY, OX, OY) to fully represent a convolution.
- For input stationary, looping over IX, IY as outer loops is unconventional; typically, inputs are spatially mapped, and outputs are computed in inner loops.

## Mapping Table
Below is an intuitive table showing how each variable is typically mapped in each dataflow strategy, based on common practices and adjusted to align with your architecture's 8 NPEs.

| Dataflow          | K  | C  | FX | FY | IX | IY | OX | OY |
|-------------------|----|----|----|----|----|----|----|----|
| Output Stationary | S  | I  | I  | I  | I  | I  | O  | O  |
| Weight Stationary | O  | O  | S  | S  | I  | I  | I  | I  |
| Input Stationary  | I  | S  | I  | I  | S  | S  | O  | O  |

**Legend**:
- **S**: Spatially mapped across NPEs (parallelized).
- **O**: Outer loop in temporal mapping (sequential, higher-level).
- **I**: Inner loop in temporal mapping (sequential, lower-level).

This table assumes typical mappings where, for example, output stationary parallelizes K across NPEs, while OX and OY are outer temporal loops, and C, FX, FY, IX, IY are inner loops due to their role in convolution calculations.

---

# Comprehensive Analysis of Mapping Constraints in Deep Learning Accelerators

This report provides a detailed examination of spatial and temporal mapping constraints for deep learning accelerators, focusing on their application within mapping exploration tools like ZigZag. It verifies and refines the provided explanations, corrects inaccuracies, and presents an intuitive table to summarize the mapping of workload variables across different dataflow strategies. The goal is to create a clear, slide-friendly resource for understanding how neural network computations are distributed and scheduled on hardware accelerators.

## Background and Context
Deep learning accelerators, such as those with multiple processing elements (NPEs), are designed to efficiently execute neural network workloads, particularly convolutional neural networks (CNNs). A key challenge is mapping the workload—comprising dimensions like output channels (K), input channels (C), filter sizes (FX, FY), input spatial dimensions (IX, IY), and output spatial dimensions (OX, OY)—onto the hardware. This mapping is split into two components:
- **Spatial Mapping**: Distributing computations across NPEs for parallel execution.
- **Temporal Mapping**: Scheduling the order of operations within each NPE.

Tools like ZigZag ([ZigZag GitHub](https://github.com/KULeuven-MICAS/zigzag)) automate this process by exploring different mapping strategies to optimize performance metrics like latency and energy efficiency. The provided explanations describe spatial mapping as vectorization for parallel execution and temporal mapping as the execution order, with specific dataflow strategies (output stationary, weight stationary, input stationary) illustrated via pseudo-code.

## Verification of Provided Information
The provided explanations and pseudo-code are evaluated for accuracy and completeness.

### Spatial Mapping
**Provided Description**: "Vectorisation of the selected workload for parallel execution."
- **Verification**: This is accurate. Spatial mapping involves assigning different parts of the workload to different NPEs, enabling simultaneous computation. The block diagram splitting K, C, OX, OY, IX, IY, FX, FY across 8 NPEs aligns with this concept, indicating parallelization of these dimensions.
- **Strengths**: Clearly conveys the idea of distributing workload for parallelism.
- **Weaknesses**: Lacks detail on how dimensions are chosen for splitting (e.g., prioritizing K for output stationary).

### Temporal Mapping
**Provided Description**: "The order of execution of the workload within the accelerator."
- **Verification**: Correct, as temporal mapping defines the sequence of operations within each NPE, influenced by dataflow strategies.
- **Pseudo-Code Analysis**:
  - **Output Stationary**:
    ```plaintext
    //outerloop
    for ox in range of OX
        for oy in range of OY
    //innerloops
        for fx in range of FX
            for fy in range of FY
                ...
    ```
    - **Strengths**: Highlights OX, OY as outer loops, consistent with prioritizing output computation.
    - **Weaknesses**: Omits loops for K and C, which are critical for a complete convolution. Typically, output stationary includes K in outer loops or spatial mapping.
  - **Weight Stationary**:
    ```plaintext
    //outerloop
    for fx in range of FX
        for fy in range of FY
    //innerloops
        for ox in range of OX
            for oy in range of OY
                ...
    ```
    - **Strengths**: Shows FX, FY as outer loops, aligning with keeping weights stationary.
    - **Weaknesses**: FX, FY are often spatially mapped in weight stationary, not temporally looped, suggesting a possible misunderstanding.
  - **Input Stationary**:
    ```plaintext
    //outerloop
    for ix in range of IX
        for iy in range of IY
    //innerloops
        for ox in range of OX
            for oy in range of OY
                ...
    ```
    - **Strengths**: Indicates focus on input dimensions.
    - **Weaknesses**: Looping over IX, IY as outer loops is unconventional; input stationary typically maps C, IX, IY spatially, with OX, OY computed in inner loops.

**Corrections Needed**:
- **Completeness**: Pseudo-code should include all dimensions (K, C, FX, FY, OX, OY) to represent a full convolution loop nest, e.g.:
  ```plaintext
  for k in K
      for c in C
          for oy in OY
              for ox in OX
                  for fy in FY
                      for fx in FX
                          y[k, oy, ox] += w[k, c, fy, fx] * x[c, oy + fy, ox + fx]
  ```
- **Input Stationary**: The pseudo-code suggests temporal loops over IX, IY, which contradicts standard input stationary dataflows where IX, IY are spatially mapped, and outputs are computed based on filter positions.

## Spatial Mapping: Detailed Explanation
Spatial mapping determines how workload dimensions are parallelized across NPEs. In an architecture with 8 NPEs, dimensions like K, C, OX, OY, IX, IY, FX, or FY can be split, depending on the dataflow strategy. For example:
- **Output Stationary**: Often maps K spatially, assigning different output channels to each NPE, maximizing parallelism in output computation.
- **Weight Stationary**: May map FX, FY or K, C, with each NPE holding specific weights, reducing weight movement.
- **Input Stationary**: Typically maps C, IX, IY, allowing NPEs to process different input channels or spatial positions in parallel.

The choice of dimensions depends on the number of NPEs and hardware constraints, such as memory bandwidth and interconnects. ZigZag and similar tools ([MAESTRO](https://maestro.ece.gatech.edu/)) use directives like `SpatialMap` to specify which dimensions are distributed across PEs, optimizing data reuse and minimizing off-chip memory accesses.

## Temporal Mapping: Detailed Explanation
Temporal mapping schedules operations within each NPE, defined by the loop nest order. Different dataflow strategies prioritize keeping certain data stationary to reduce data movement:
- **Output Stationary**: Keeps output activations (y[k, oy, ox]) fixed in NPEs, looping over inputs and weights to accumulate results. Outer loops are typically over K, OX, OY, with inner loops over C, FX, FY.
- **Weight Stationary**: Keeps weights (w[k, c, fy, fx]) fixed, streaming inputs and accumulating outputs. Outer loops may be over K, C, with inner loops over OX, OY.
- **Input Stationary**: Keeps input activations (x[c, iy, ix]) fixed, applying weights to compute outputs. Outer loops are often over K, with inner loops over FX, FY, computing OX, OY based on input positions.

The provided pseudo-code simplifies these strategies but omits key dimensions, affecting clarity. Standard loop nests, as shown above, provide a complete picture.

## Mapping Constraints in ZigZag
ZigZag is a design space exploration framework that integrates algorithmic decisions with hardware costs ([ZigZag Paper](https://arxiv.org/abs/2007.11360)). It uses a memory-centric approach, defining mappings as:
- **Spatial Mapping**: Assigning loop indices to NPEs, e.g., mapping K across 8 NPEs means each NPE computes a subset of output channels.
- **Temporal Mapping**: Specifying loop orders within NPEs, supporting even or uneven scheduling to optimize data reuse.

**Input Constraints**:
- **Workload Description**: The CNN layer dimensions (K, C, FX, FY, IX, IY, OX, OY), typically provided via an ONNX model or similar format.
- **Hardware Architecture**: Details of the 8 NPEs, including memory hierarchy, interconnects, and compute capabilities.
- **Mapping Directives**: Specifications for spatial mapping (e.g., `SpatialMap(K, size=8)` to split K across NPEs) and temporal mapping (e.g., loop order like `for oy, for ox, for c, for fy, for fx`).
- **Optimization Goals**: Metrics like latency, energy, or throughput to guide the exploration.

ZigZag processes these inputs to generate optimal mappings, balancing parallelism and data locality.

## Intuitive Mapping Table
To make the mapping constraints intuitive for slides, the table below categorizes each variable (K, C, FX, FY, IX, IY, OX, OY) as:
- **S**: Spatially mapped across NPEs.
- **O**: Outer loop in temporal mapping (sequential, higher-level).
- **I**: Inner loop in temporal mapping (sequential, lower-level).

| Dataflow          | K  | C  | FX | FY | IX | IY | OX | OY |
|-------------------|----|----|----|----|----|----|----|----|
| Output Stationary | S  | I  | I  | I  | I  | I  | O  | O  |
| Weight Stationary | O  | O  | S  | S  | I  | I  | I  | I  |
| Input Stationary  | I  | S  | I  | I  | S  | S  | O  | O  |

**Explanations**:
- **Output Stationary**:
  - **K**: Spatially mapped (S) to parallelize output channels across NPEs.
  - **OX, OY**: Outer temporal loops (O), processing output positions sequentially.
  - **C, FX, FY, IX, IY**: Inner temporal loops (I), as they are used in convolution calculations within each NPE.
- **Weight Stationary**:
  - **FX, FY**: Spatially mapped (S), with NPEs holding different filter positions.
  - **K, C**: Outer temporal loops (O), processing channels sequentially.
  - **OX, OY, IX, IY**: Inner temporal loops (I), computing outputs within each NPE.
- **Input Stationary**:
  - **C, IX, IY**: Spatially mapped (S), parallelizing input channels and positions.
  - **OX, OY**: Outer temporal loops (O), computing output positions.
  - **K, FX, FY**: Inner temporal loops (I), applying filters within each NPE.

This table assumes typical mappings but adjusts for your architecture’s 8 NPEs, ensuring clarity at a glance.

## Alignment with Pseudo-Code
The provided pseudo-code partially aligns with standard dataflows:
- **Output Stationary**: Matches with OX, OY as outer loops and FX, FY as inner loops, but misses K, C.
- **Weight Stationary**: Shows FX, FY as outer loops, which conflicts with typical spatial mapping of FX, FY; corrected in the table to reflect standard practice.
- **Input Stationary**: Lists IX, IY as outer loops, which is adjusted in the table to spatial mapping, with OX, OY as outer loops for output computation.

## Practical Considerations
- **ZigZag Flexibility**: Supports uneven mappings, allowing operands to use different memory levels, enhancing efficiency ([ZigZag IEEE](https://ieeexplore.ieee.org/document/9360462/)).
- **Hardware Constraints**: With only 8 NPEs, spatial mapping is limited, so choices like mapping K or FX, FY must balance parallelism and memory usage.
- **Data Reuse**: Dataflows optimize reuse (e.g., output stationary maximizes convolutional reuse), impacting energy efficiency ([MAESTRO Paper](https://ieeexplore.ieee.org/document/9076333)).

## Conclusion
The provided explanations are a solid foundation but benefit from including all dimensions in pseudo-code and adjusting input stationary mappings. The table above offers an intuitive summary, ideal for slides, showing how each variable is typically mapped in output, weight, and input stationary dataflows. This structure aligns with tools like ZigZag, ensuring clarity for both technical and non-technical audiences.

## Key Citations
- [ZigZag: HW Architecture-Mapping Design Space Exploration Framework](https://github.com/KULeuven-MICAS/zigzag)
- [ZigZag: A Memory-Centric Rapid DNN Accelerator Design Space Exploration](https://arxiv.org/abs/2007.11360)
- [ZigZag: Enlarging Joint Architecture-Mapping Design Space Exploration](https://ieeexplore.ieee.org/document/9360462/)
- [MAESTRO: A Data-Centric Approach to Understand Reuse, Performance](https://ieeexplore.ieee.org/document/9076333)
- [MAESTRO: An Open-source Infrastructure for Modeling Dataflows](https://deepai.org/publication/maestro-an-open-source-infrastructure-for-modeling-dataflows-within-deep-learning-accelerators)
- [Efficient Hardware Architectures for Accelerating Deep Neural Networks](https://www.mdpi.com/1999-5903/12/7/113)