# Mid-term presentation

# High level over-view of an exploration

#### Question 1: 
I want an high level block-diagram of over view of the Hardware - Software co-optimisation frame work which are bascially exploration tool that find the best mapping for a given DNN on the Architecture provided.

Here is the description I want you to improve further, Streamline it and make it coherent. So, I can create a block diagram from it.

Description which is taken directly from the timeloop and zigzag

"An exploration tool as stated by in Zigzag and Timeloop.

As stated in the Timeloop paper it provides an infrastructure for evaluating and exploring the architecture design space of the DNN accelerators. it evaluate the described those models by emulating the hardware by generating an accurate projection of the performance and energy efficiency for a a DNN workload through a mapper that find s the best way to schedule operations and  stage data on the specified architecture. This enables a fair comparirios across different architectures and makes DNN accelerators design more systematic. Timeloop makes the various case studies possible because of this tool when these done manually would be really difficult and time consuming tasks.

As stated in the Zigzag paper,  Each accelerator is designed with a different architecture and different choice of dataflow/Mapping, due to this it is always difficult to justify if the topology selected for the accelerator is the best one, given the vast design space available. Therefore it is essential to have a framework that can rapidly explore the available design space and guide the designers to find the perto-optimal architectures with optimal mappings while taking hardware constraints, technology dependent cost and algorithmic workload into account. Ans there are many tools in the market that can make this possible.

Both these tools have common underlying building blocks:
1. Hardware design space which is given as the input along with Modelling of the workload and the constraints
2. Mapspace creation based on the given workload mapped on the hardware with all the valid combination according to the constraints provided
3. To create a mapspace there is a mapping search engine to make it possible
4. Hardware analytical model which is used for the estimation of the costs of the various mapping in the map space and then provide the best mapping available in them based on the optimisation criteria provided.
"

Task: Validate and create more general information from this above writing.

Which could be ustilised to create a more general high level block diagram


#### Answer 1:

Design Space Exploration (DSE) frameworks like Timeloop and ZigZag are instrumental in optimizing the deployment of Deep Neural Networks (DNNs) on specialized hardware accelerators. These tools systematically analyze and identify the most efficient mappings of DNN workloads onto given hardware architectures, ensuring optimal performance and energy efficiency.

**Key Components of DSE Frameworks:**

1. **Input Specifications:**
   - **Workload Model:** Defines the DNN layer's computational requirements, including tensor dimensions and operations.
   - **Hardware Architecture Description:** Details the organization of computational units, memory hierarchies, and interconnections within the accelerator.
   - **Design Constraints:** Encompasses hardware limitations, technology-specific considerations, and performance targets.

2. **Mapping Space Generation:**
   - Based on the provided workload and hardware specifications, the framework generates a comprehensive set of possible mappings, representing various ways the DNN operations can be scheduled and data can be staged across the hardware resources.

3. **Mapping Search Engine:**
   - Explores the mapping space to identify optimal mappings that meet the defined constraints and objectives.
   - Utilizes strategies ranging from exhaustive searches to heuristic methods to efficiently navigate the vast design space.

4. **Analytical Cost Model:**
   - Evaluates each potential mapping by estimating key metrics such as performance (e.g., latency, throughput), energy consumption, and resource utilization.
   - Facilitates comparisons across different mappings and architectures to guide design decisions.

**Workflow Overview:**

- **Input Phase:**
  - The user provides the DNN workload specifications, hardware architecture details, and any pertinent design constraints.

- **Mapping Exploration Phase:**
  - The framework constructs the mapping space and employs the search engine to traverse this space, seeking mappings that optimize the desired criteria.

- **Evaluation Phase:**
  - Each candidate mapping undergoes assessment through the analytical cost model to predict performance and energy efficiency.

- **Output Phase:**
  - The framework outputs the optimal mapping(s) along with their associated metrics, providing insights into the most efficient deployment strategies for the DNN on the specified hardware.

By integrating these components, DSE frameworks like Timeloop and ZigZag enable systematic and efficient exploration of the complex design spaces inherent in DNN accelerator development, facilitating informed decisions that balance performance, energy consumption, and resource constraints. 



#### Question 2:
The high level block dragram with a detailed template as the flow progress

#### Answer 2:

Below is a high-level flow diagram of a mapping exploration tool for DNN accelerators, along with a detailed template for the inputs required at each stage. You can use these templates as a starting point to explain or further detail each block when creating your diagram.

---

### High-Level Flow Diagram

```plaintext
+--------------------------+
|       Input Phase        |
| (Workload, Architecture, |
|   & Design Constraints)  |
+------------+-------------+
             |
             v
+--------------------------+
| Mapping Space Generation |
|       Phase              |
| (Define mapping options, |
|   tiling, loop order,    |
|   and data reuse)        |
+------------+-------------+
             |
             v
+--------------------------+
|   Mapping Search Engine  |
|       Phase              |
| (Explore mapping space,  |
|   apply search strategies|
|   and constraints)       |
+------------+-------------+
             |
             v
+--------------------------+
| Analytical Cost Model    |
|     Evaluation Phase     |
| (Estimate performance,   |
|   energy, and resource   |
|   utilization for each   |
|   candidate mapping)     |
+------------+-------------+
             |
             v
+--------------------------+
|       Output Phase       |
| (Report optimal mapping, |
|   metrics, and insights) |
+--------------------------+
```

---

### Detailed Template for Inputs at Each Block

#### 1. **Input Phase**

**Purpose:**  
Gather all necessary high-level information about the DNN workload, target hardware architecture, and design constraints.

**Template:**

- **Workload Specification:**
  - **DNN Model Name:** e.g., *ResNet-50, VGG-16*
  - **Layer Details:** List of layers (e.g., convolution, pooling, fully connected)
  - **Tensor Dimensions:** Input/output sizes, filter sizes, etc.
  - **Operation Complexity:** Number of MACs, FLOPs per layer

- **Hardware Architecture Description:**
  - **Compute Units:** Number, type, and capabilities (e.g., SIMD units, MAC arrays)
  - **Memory Hierarchy:** Details on caches, scratchpad memories, global memory sizes, and bandwidths
  - **Interconnect Topology:** Bus architecture, network-on-chip details
  - **Specialized Components:** e.g., accelerators for convolution, dedicated dataflow engines

- **Design Constraints:**
  - **Performance Targets:** Desired latency, throughput, or cycles per operation
  - **Energy Efficiency Targets:** Energy per operation, total energy budget
  - **Resource Limits:** Area constraints, power budget, technology nodes
  - **Additional Considerations:** Thermal limits, reliability, cost factors

---

#### 2. **Mapping Space Generation Phase**

**Purpose:**  
Generate all valid mapping configurations by combining the workload and hardware specifications while considering the design constraints.

**Template:**

- **Mapping Space Parameters:**
  - **Tiling Factors:** Options for partitioning input/output tensors (e.g., tile sizes for height, width, channel)
  - **Loop Order Options:** Permutations of loop nests for scheduling operations
  - **Data Reuse Patterns:** Strategies for caching and reusing data across loops
  - **Memory Mapping Strategies:** Allocation of data across different memory levels
  - **Constraints Validation:** Rules that ensure each mapping adheres to the hardware and design limits

---

#### 3. **Mapping Search Engine Phase**

**Purpose:**  
Explore the mapping space to identify candidate configurations that meet the optimization objectives and constraints.

**Template:**

- **Search Engine Configuration:**
  - **Search Strategy:** Specify the algorithm (e.g., exhaustive search, genetic algorithm, simulated annealing)
  - **Optimization Objectives:** Define criteria (e.g., minimize latency, energy consumption, maximize throughput)
  - **Search Boundaries:** Set limits for parameters (e.g., range of tiling factors, acceptable loop order permutations)
  - **Constraint Checks:** Input the design constraints that each candidate mapping must satisfy

---

#### 4. **Analytical Cost Model Evaluation Phase**

**Purpose:**  
Assess each candidate mapping using an analytical model to estimate performance metrics, energy consumption, and resource utilization.

**Template:**

- **Cost Model Inputs:**
  - **Mapping Configuration:** Details of the candidate mapping under evaluation
  - **Hardware Performance Parameters:** Latency per operation, energy per MAC, memory access costs
  - **Workload Execution Profile:** Computation and data transfer patterns based on the DNN workload
  - **Evaluation Metrics:** Define metrics such as total latency, energy consumption, throughput, and area utilization
  - **Calibration Factors:** Adjustments based on empirical data or technology-specific parameters

---

#### 5. **Output Phase**

**Purpose:**  
Select and present the optimal mapping along with its associated metrics and insights to guide design decisions.

**Template:**

- **Optimal Mapping Output:**
  - **Selected Mapping Configuration:** The best mapping parameters (tiling factors, loop order, dataflow strategy)
  - **Performance Metrics:** Predicted latency, throughput, energy consumption, etc.
  - **Evaluation Summary:** Comparative analysis of candidate mappings and rationale for the selection
  - **Design Recommendations:** Insights and suggestions for hardware or software modifications to further optimize performance

---

### Summary

This high-level flow and corresponding detailed templates provide a systematic way to set up a mapping exploration tool. Each block in the flow diagram is associated with specific input templates that help in gathering and processing the required data, generating mapping options, exploring these options through a search engine, evaluating them using an analytical cost model, and finally presenting the optimal solution.

This structure not only streamlines the process of mapping exploration but also provides a clear framework that can be easily expanded and explained further in documentation or presentations.



# High level flow of the project

Assumptions

We are using Zigzag and Stream are used for the calculations and they by default cann't do the right exploarion of the event driven architecture exploration and would need to be provided externally with diffent spatial and temporal mapping configurationsas input to find the best mapping out of them.


The high level flow that I proposed which works around zigzag/stream is as follows

For the dense case implementation for using zigzag stream the high level flow of the tool is as follows

Inputs
1. Hardware configuration
2. Python script (This generates all the possible Spatial and temporal ordering possible by fixing the Input variables constant)
3. Workload configuration 

Exploration block
By zigzag where for every configuration the block is run and the output specifications are noted down - For single core
By Stream where for every configuration the block is run and the ouput specifications are noted down - For multi core

Output block
Final output specification 
Best Mapping configuration


#### Question 2: 
Can you polish this high level flow futher which is used for modeling the event driven architectures using the zigzag/stream exploration tool as you gave the general exploration tool flow, but this time it's going to be more detailed towards this flow and the two tools I mentioned


"Assumptions

We are using Zigzag and Stream are used for the calculations and they by default cann't do the right exploarion of the event driven architecture exploration and would need to be provided externally with diffent spatial and temporal mapping configurationsas input to find the best mapping out of them.


The high level flow that I proposed which works around zigzag/stream is as follows

For the dense case implementation for using zigzag stream the high level flow of the tool is as follows

Inputs
1. Hardware configuration
2. Python script (This generates all the possible Spatial and temporal ordering possible by fixing the Input variables constant)
3. Workload configuration 

Exploration block
By zigzag where for every configuration the block is run and the output specifications are noted down - For single core
By Stream where for every configuration the block is run and the ouput specifications are noted down - For multi core

Output block
Final output specification 
Best Mapping configuration"


#### Answer 2:
Designing event-driven architectures for deep neural network (DNN) accelerators requires a specialized approach to design space exploration (DSE). Tools like ZigZag and Stream offer robust frameworks for such explorations but necessitate external configurations to effectively navigate the unique challenges posed by event-driven designs. The following refined high-level flow outlines the integration of these tools, detailing the inputs required at each stage to facilitate a comprehensive exploration process.

---

**Assumptions:**

- **Tool Capabilities:** ZigZag and Stream are employed for DSE but require externally provided spatial and temporal mapping configurations to effectively explore event-driven architectures.

---

**High-Level Flow for Event-Driven Architecture Exploration:**

1. **Input Phase:**

   - **Hardware Configuration:**
     - **Description:** Detailed specifications of the target hardware architecture, including compute units, memory hierarchy, interconnects, and any specialized components relevant to event-driven operations.
     - **Purpose:** Provides the foundational parameters upon which mappings will be evaluated.

   - **Spatial and Temporal Mapping Configurations:**
     - **Description:** A comprehensive set of predefined spatial (data distribution across hardware resources) and temporal (operation scheduling over time) mapping configurations.
     - **Purpose:** Supplies the necessary mappings for ZigZag and Stream to assess, compensating for their limited native support in generating event-driven mappings.
     - **Generation Method:** Typically produced by an external Python script that systematically explores the mapping space, considering the fixed input variables and design constraints.

   - **Workload Configuration:**
     - **Description:** Specifications of the DNN workload, detailing layer types, tensor dimensions, computational dependencies, and dataflow characteristics.
     - **Purpose:** Ensures that the mappings are evaluated against realistic and representative DNN tasks.

2. **Exploration Phase:**

   - **Single-Core Analysis with ZigZag:**
     - **Process:** For each provided mapping configuration, ZigZag evaluates the performance and energy efficiency on a single-core implementation.
     - **Output:** Metrics such as latency, throughput, energy consumption, and resource utilization for each mapping.
     - **Purpose:** Identifies optimal mappings for single-core architectures and serves as a baseline for multi-core comparisons.

   - **Multi-Core Analysis with Stream:**
     - **Process:** Similarly, Stream assesses each mapping configuration in the context of a multi-core architecture, considering inter-core communication and synchronization.
     - **Output:** Detailed performance and efficiency metrics tailored to multi-core implementations.
     - **Purpose:** Determines how mappings scale across multiple cores and identifies configurations that leverage parallelism effectively.

3. **Output Phase:**

   - **Compilation of Results:**
     - **Description:** Aggregates the evaluation metrics from both ZigZag and Stream analyses, providing a comprehensive overview of each mapping's performance across different architectural scenarios.
     - **Purpose:** Facilitates direct comparison between single-core and multi-core implementations, highlighting trade-offs and potential bottlenecks.

   - **Selection of Optimal Mapping Configuration:**
     - **Process:** Identifies the mapping configuration that best meets the predefined optimization criteria (e.g., lowest energy consumption, highest throughput) while adhering to design constraints.
     - **Purpose:** Guides the final design decision, ensuring that the selected mapping aligns with the overarching goals of the event-driven architecture.

---

**Summary:**

This refined flow integrates external generation of spatial and temporal mappings with the analytical capabilities of ZigZag and Stream to facilitate a thorough exploration of event-driven DNN accelerator architectures. By systematically evaluating each configuration in both single-core and multi-core contexts, this approach ensures that the final design is optimized for performance and efficiency, tailored to the unique demands of event-driven processing. 