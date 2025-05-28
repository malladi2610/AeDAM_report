# Structure to perform DSE for Event driven accelerators.

1. Exisiting literature on the DSE performed for the Event driven accelerators and the results obtained.

2. Based on the memory layout paper provided, exisitng literature available that showcased the effect of the memory layout

3. Similar to the memory layout paper are there any literature on the PE arrangements effect for the event driven accelerators

4. Combining the Reseach 2 and 3 can a reseach scope can be devised which is novel and address the limitations of the exisitng research and differentiates the new plan with the previous research

5. Structure the final plan to what DSE will be performed and the what is the novelty of this exploration backed by the previous research.

------------------------------------------------------------------------------------------------------------------------------------------------------

Use of the AI tools 

1. Grok Deeper research (To improve the research question further and structure them more clearly with previous research)
2. Gemini Deep research (To build on the top of grok research and give a detailed explanation)
2. Chatgpt Deep research (To build on the top of grok research and give a detailed explanation)
3. A consolidatied plan that combines the understanding of the Gemini and Chatgpt for the final answer of this chat

======================================================================================================================================================


# Question 1.

Now, with the discussion we had previously about the basics of the the DSE and the parameters that can be modified based on the exploration tool being utilised.

Here is the next question I have which needs to be answered by taking the awareness of the previously discussed topics.

This time the seach will be narrowed down to a specific type of accelerators known as event drievn accelerators


1. Exisiting literature on the DSE performed for the Event driven accelerators and the results obtained.

2. Based on the memory layout paper provided, exisitng literature available that showcased the effect of the memory layout

3. Similar to the memory layout paper are there any literature on the PE arrangements effect for the event driven accelerators

4. Combining the Reseach 2 and 3 can a reseach scope can be devised which is novel and address the limitations of the exisitng research and differentiates the new plan with the previous research

5. Structure the final plan to what DSE will be performed and the what is the novelty of this exploration backed by the previous research.


Do the following:
1. Structure the entire research questions correctly
2. Do an intial search and find relevant research that could briefly answer the question in the research.
3. Give a final output in markdown, with a more detailed and structured reseach question and supporting citation from relevant papers that explain what is being planned to be achieved with these research question.


## Answer 1:

### Key Points
- Research suggests that Design Space Exploration (DSE) for event-driven accelerators, particularly neuromorphic hardware, focuses on optimizing energy and latency, but comprehensive studies are limited.
- Memory layouts, like SRAM versus MRAM, significantly impact energy and area in event-driven architectures, with MRAM offering up to 40% area savings.
- Processing element (PE) arrangements in event-driven accelerators affect parallelism and efficiency, though specific studies on their impact are scarce.
- A novel research scope could involve jointly optimizing memory and PE configurations using tools like ZigZag, tailored for single-event processing, addressing gaps in holistic DSE approaches.
- The proposed DSE plan leverages existing tools and research to explore unique event-driven behaviors, potentially improving efficiency for edge computing applications.

### Understanding DSE for Event-Driven Accelerators
Event-driven accelerators, often used in neuromorphic computing, process data only when specific events occur, making them energy-efficient for tasks like edge inference. Design Space Exploration (DSE) helps find the best hardware setups by testing different configurations. Research indicates that DSE for these accelerators is still developing, with studies like those on spiking neural networks (SNNs) showing promise in optimizing energy and speed.

### Memory Layout Impacts
The choice of memory, such as SRAM or MRAM, affects how much power and space an accelerator uses. For example, one study found that MRAM can save about 40% in area compared to SRAM, which is crucial for compact edge devices. This aligns with findings that different memory setups can change energy use and processing speed in event-driven systems.

### Processing Element Arrangements
The way processing elements (like neurons in neuromorphic chips) are organized influences how efficiently an accelerator handles events. While some research explores dataflow architectures for event-driven tasks, there’s less focus on how different PE layouts directly impact performance, suggesting an area for further study.

### Novel Research Scope
Combining memory and PE optimization in a single DSE framework could be a new approach. Most studies look at these separately, missing how they work together. Using tools like ZigZag, modified to handle single-event processing (where each event is processed once with all computations at once), could lead to better designs for specific applications like real-time edge computing.

### Structured DSE Plan
The plan involves defining a range of memory types and PE layouts, using ZigZag to model event-driven behavior, and testing how these setups affect energy, speed, and size. This approach is unique because it focuses on single-event processing and combines memory and compute optimization, building on existing research to create more efficient accelerators.

---



# Literature Survey on Design Space Exploration for Event-Driven Accelerators

This survey addresses a research plan focused on Design Space Exploration (DSE) for event-driven accelerators, particularly those designed for single-event processing with architectures similar to multiply-and-fire models like SENECA. The plan is structured around five key research questions, rephrased for clarity and supported by relevant literature. Each question explores existing research, the impact of memory layouts and processing element (PE) arrangements, a novel research scope, and a structured DSE plan, with citations to academic papers providing insights into methodologies, parameter impacts, and optimization techniques.

## 1. Existing Literature on DSE for Event-Driven Accelerators

### Overview of DSE in Event-Driven Contexts
Event-driven accelerators, often associated with neuromorphic computing and spiking neural networks (SNNs), process computations triggered by discrete events, offering energy efficiency for applications like edge inference. DSE in this context involves exploring hardware configurations to optimize metrics such as latency, energy, and area. Research suggests that while DSE frameworks for neuromorphic hardware exist, comprehensive studies tailored to event-driven accelerators with single-event processing (where a single event triggers parallelized computations and does not recur) are limited.

### Key Studies and Findings
- **Neuromorphic Hardware DSE**: A study on hardware spiking neurons proposes a framework for DSE that defines architectures based on application-specific constraints, focusing on SNNs. It explores neural coding methods and hardware configurations, achieving energy-efficient designs for embedded AI [Design Space Exploration of Hardware Spiking Neurons](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041). Results indicate significant energy savings when optimizing neuron models and interconnects.
- **Rapid DSE for Neuromorphic Processors**: An SBIR proposal outlines a tool for rapid DSE, evaluating deep learning accelerators, spiking neuron accelerators, memristive systems, and photonic accelerators. It supports mapping algorithms to heterogeneous neuromorphic systems, showing improved performance for complex workloads [Rapid Design Space Exploration for Neuromorphic Processors](https://www.sbir.gov/node/1925299).
- **ASCENT Neuromorphic DSE**: A presentation discusses rapid DSE for neuromorphic hardware using emerging devices, exploring mappings, architectures, and devices. It highlights trade-offs in energy, area, and delay, with sparse SNN characteristics impacting dataflow designs [ASCENT Theme 4 Neuromorphic DSE](https://ascent.nd.edu/events/2021/09/16/ascent-theme-4-liaison-meeting-rapid-design-space-exploration-of-neuromorphic-hardware-using-emerging-devices/).
- **Event-Driven Execution Models**: A dissertation on hardware acceleration for irregular applications like Parallel Discrete Events Simulation (PDES) designs dataflow architectures with event-driven execution, improving parallelism and memory bandwidth utilization [Hardware Acceleration Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd).

### Results and Implications
These studies demonstrate that DSE for event-driven accelerators can yield significant improvements in energy efficiency (up to 10x in some neuromorphic designs) and performance, particularly when tailored to specific workloads. However, they often focus on general neuromorphic systems or specific applications, with less emphasis on single-event processing models, suggesting a gap in targeted DSE methodologies.

## 2. Impact of Memory Layouts in Event-Driven Accelerators

### Insights from the Provided Paper
The paper "Scaling NVMs in Event-Driven Architectures for Edge Inference" examines memory layouts in a multicore event-driven architecture for edge AI [Scaling NVMs in Event-Driven Architectures](attachment:0). Key findings include:
- **SRAM vs. MRAM**: SRAM offers high speed but high static power and lower density, while MRAM (e.g., STT-MRAM, VGSOT-MRAM) provides non-volatility, higher density, and lower static power, reducing area by ~40% across technology nodes (22nm to 14A).
- **Energy and Latency Trade-offs**: MRAM reduces total energy by ~25% and leakage energy by ~40% in 22nm and 12nm nodes but increases energy and latency at 14A due to slower write speeds.
- **Distributed Memory**: Near-memory processing with smaller, distributed memory blocks minimizes data movement, enhancing energy efficiency and reducing latency.
- **Depth-First Mapping**: Buffering only incomplete neural states reduces memory requirements, lowering latency and energy use.

### Additional Literature
- **Synaptic Memory Devices**: A study on Mn-doped HfO2-based synaptic memory devices compares them to VLSI SRAM synapses, showing a ~10x area reduction and >106 times lower power consumption per learning cycle [Novel Synaptic Memory Device](https://www.nature.com/articles/srep05333). This highlights the potential of emerging memories for neuromorphic applications.
- **Multi-State MRAM Cells**: Research on serially connected magnetic tunnel junctions (MTJs) forming multi-state MRAM cells for neuromorphic computing demonstrates their use in quantized weight storage, improving energy efficiency in neural networks [Multi-State MRAM Cells](https://www.nature.com/articles/s41598-022-11199-4).
- **Efficient Synapse Memory**: A study on synapse memory structures for digital neuromorphic hardware compares energy efficiency and throughput with systems like TrueNorth and Loihi, showing optimized memory layouts can enhance performance [Efficient Synapse Memory Structure](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00829/full).

### Table: Memory Layout Impacts
| **Memory Type** | **Advantages** | **Disadvantages** | **Impact on Metrics** | **Source** |
|-----------------|----------------|-------------------|-----------------------|------------|
| SRAM | High speed, low latency | High static power, low density | Higher energy, larger area | [Scaling NVMs in Event-Driven Architectures](attachment:0) |
| MRAM | Non-volatile, high density, low static power | Slower write speeds, higher write energy | ~40% area reduction, ~25% energy savings | [Scaling NVMs in Event-Driven Architectures](attachment:0) |
| Mn-doped HfO2 | Analog conductance, small size | Complex fabrication | ~10x area reduction, >106x power savings | [Novel Synaptic Memory Device](https://www.nature.com/articles/srep05333) |
| Multi-State MRAM | Quantized weights, energy-efficient | Limited scalability | Improved neural network efficiency | [Multi-State MRAM Cells](https://www.nature.com/articles/s41598-022-11199-4) |

### Implications
Memory layout significantly affects event-driven accelerators, with emerging technologies like MRAM offering substantial area and energy benefits. However, trade-offs in latency and write energy at advanced nodes require further optimization, particularly for single-event processing where data access patterns may differ.

## 3. Impact of PE Arrangements in Event-Driven Accelerators

### Existing Research
PE arrangements in event-driven accelerators, analogous to neuron or compute unit configurations in neuromorphic hardware, influence parallelism, throughput, and energy efficiency. Literature on this topic is less abundant but includes:
- **Event-Driven Dataflow Architectures**: The dissertation on irregular applications designs dataflow architectures for event-driven execution, showing increased parallelism and memory bandwidth utilization for PDES and graph processing [Hardware Acceleration Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd). Different PE configurations (e.g., interconnected cores) impact performance.
- **Neuromorphic Hardware DSE**: The study on hardware spiking neurons explores various neuron models and interconnects, effectively varying PE arrangements, to optimize SNN performance [Design Space Exploration of Hardware Spiking Neurons](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041). Results suggest that grid or hierarchical PE layouts affect latency and energy.
- **Neuromorphic Architectures**: The Neuromorphic Hardware Guide describes chips like SpiNNaker2 (mesh of ARM cores) and DynapCNN (event-driven neuron arrays), where PE arrangements influence computational efficiency [Neuromorphic Hardware Guide](https://open-neuromorphic.org/neuromorphic-computing/hardware/).

### Specific Findings
- **Parallelism and Efficiency**: Dataflow architectures with event-driven PEs can achieve higher throughput by optimizing interconnects, but specific impacts depend on workload characteristics.
- **SNN Performance**: In SNNs, PE arrangements like grid-based (TrueNorth) or mesh-based (SpiNNaker) affect spike propagation and energy use, with sparse workloads benefiting from optimized topologies.
- **Lack of Direct Studies**: Few studies explicitly quantify how different PE arrangements (e.g., number, topology) impact event-driven accelerators, particularly for single-event processing.

### Table: PE Arrangement Impacts
| **PE Arrangement** | **Description** | **Impact on Performance** | **Source** |
|--------------------|-----------------|---------------------------|------------|
| Dataflow Cores | Interconnected cores for event-driven tasks | Increased parallelism, higher throughput | [Hardware Acceleration Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd) |
| Grid-Based Neurons | 2D grid of spiking neurons (e.g., TrueNorth) | Efficient for sparse SNNs, lower latency | [Design Space Exploration of Hardware Spiking Neurons](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041) |
| Mesh of Cores | ARM cores in a mesh (e.g., SpiNNaker2) | High flexibility, energy-efficient for SNNs | [Neuromorphic Hardware Guide](https://open-neuromorphic.org/neuromorphic-computing/hardware/) |

### Implications
While PE arrangements are critical, the literature lacks detailed studies on their specific impact in event-driven accelerators, especially for single-event processing. This gap suggests an opportunity for research focusing on how PE topologies affect performance in such contexts.

## 4. Novel Research Scope

### Limitations in Existing Research
- **Separate Optimization**: Most studies optimize memory layouts or PE arrangements independently, missing their interplay in event-driven accelerators.
- **Generalized Workloads**: Existing DSE frameworks often target general neuromorphic or SNN workloads, not specific single-event processing models where events are processed once with parallelized computations.
- **Tool Adaptation**: Tools like ZigZag are designed for DNN accelerators, and their use for event-driven behaviors via restricted mappings is underexplored.
- **Performance Trade-offs**: Limited exploration of how memory and PE configurations together affect latency, energy, and area in event-driven systems.

### Proposed Research Scope
The novel research scope is to develop a **Joint Design Space Exploration Framework for Memory Hierarchies and PE Arrangements in Event-Driven Accelerators for Single-Event Processing**. This involves:
- **Integrated Optimization**: Simultaneously exploring memory configurations (e.g., SRAM, MRAM, hierarchy levels) and PE arrangements (e.g., number, interconnect topologies) to identify optimal designs.
- **Tool Adaptation**: Modifying tools like ZigZag to model single-event processing by restricting loop structures, ensuring accurate representation of event-driven behavior.
- **Workload Specificity**: Focusing on workloads where a single event triggers parallel computations that complete before the next event, relevant for edge applications like real-time sensor processing.
- **Performance Analysis**: Evaluating the impact of configurations on latency, energy, and area, providing insights into trade-offs and optimal designs.

### Differentiation from Previous Research
- **Holistic Approach**: Unlike studies focusing on memory or PEs separately, this research integrates both, addressing their combined effect.
- **Single-Event Focus**: Targets a unique computational model, differing from batch or stream processing, with distinct data access and compute patterns.
- **Tool Innovation**: Adapts existing DSE tools for event-driven contexts, potentially developing new mapping strategies to enhance accuracy and efficiency.

### Table: Comparison with Existing Research
| **Aspect** | **Existing Research** | **Proposed Research** |
|------------|-----------------------|-----------------------|
| Optimization Focus | Memory or PEs separately | Joint memory and PE optimization |
| Workload Type | General SNNs, DNNs | Single-event processing |
| Tool Usage | Standard ZigZag, Timeloop | Modified ZigZag with event-driven mappings |
| Performance Metrics | Energy, latency | Energy, latency, area with trade-off analysis |

## 5. Structured Plan for DSE

### DSE Methodology
The DSE plan for event-driven accelerators with single-event processing includes:
1. **Define the Design Space**:
   - **Memory Configurations**: Types (SRAM, MRAM), sizes (e.g., 64 Byte to 2 MByte), hierarchy levels, and data placement strategies.
   - **PE Arrangements**: Number of PEs, spatial topologies (e.g., grid, mesh, hierarchical), and interconnection patterns.
2. **Model Event-Driven Behavior**:
   - Adapt ZigZag to restrict loop structures, modeling single-event processing where inputs arrive sequentially, and computations are parallelized per event.
   - Develop or modify mapping strategies to ensure accurate representation of event-driven workloads.
3. **Evaluate Performance Metrics**:
   - Use simulations or analytical models to assess latency, energy consumption, and area for various configurations.
   - Employ tools like NeuroFlow (from the provided paper) or ZigZag’s Hardware Cost Estimator for accurate performance estimation.
4. **Optimize and Analyze Trade-offs**:
   - Apply heuristic search or machine learning to identify Pareto-optimal configurations.
   - Analyze trade-offs to provide design guidelines for event-driven accelerators.
5. **Validate Results**:
   - Compare DSE outcomes with hardware measurements or detailed simulations, targeting errors below 10% (similar to ZigZag’s 5-7.5% error rates).

### Novelty of the Exploration
- **Integrated DSE**: Combines memory and PE optimization, addressing a gap in holistic approaches.
- **Single-Event Processing**: Targets a specific computational model, relevant for edge applications with sporadic events.
- **Tool Adaptation**: Enhances tools like ZigZag for event-driven contexts, potentially introducing new mapping techniques.
- **Comprehensive Metrics**: Evaluates latency, energy, and area, providing a complete picture of design trade-offs.

### Backing from Previous Research
- **DSE Frameworks**: Builds on frameworks from neuromorphic DSE studies, adapting them for single-event processing [Design Space Exploration of Hardware Spiking Neurons](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041).
- **Memory Insights**: Leverages findings on SRAM vs. MRAM trade-offs to inform memory configuration choices [Scaling NVMs in Event-Driven Architectures](attachment:0).
- **PE Configurations**: Draws on dataflow and neuromorphic architecture studies to guide PE arrangement exploration [Hardware Acceleration Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd).

### Table: DSE Plan Components
| **Component** | **Description** | **Tools/Methods** | **Expected Outcome** |
|---------------|-----------------|-------------------|----------------------|
| Design Space | Memory types, PE topologies | ZigZag, custom definitions | Comprehensive parameter set |
| Event-Driven Model | Restricted loop structures | Modified ZigZag mappings | Accurate workload representation |
| Performance Evaluation | Latency, energy, area | NeuroFlow, Hardware Cost Estimator | Quantitative metrics |
| Optimization | Heuristic search, ML | Pareto analysis | Optimal configurations |
| Validation | Hardware comparisons | Simulation benchmarks | <10% error in estimates |

## Conclusion
This survey outlines a comprehensive research plan for DSE in event-driven accelerators, focusing on single-event processing. It reviews existing literature, highlights the impact of memory layouts and PE arrangements, proposes a novel joint optimization framework, and structures a detailed DSE plan. The approach builds on prior research while addressing gaps in holistic optimization and workload-specific DSE, offering potential advancements in energy-efficient accelerator design for edge computing.

## Key Citations
- [Design Space Exploration of Hardware Spiking Neurons for Embedded Artificial Intelligence](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041)
- [Rapid Design Space Exploration for Mapping Applications to Neuromorphic Processors](https://www.sbir.gov/node/1925299)
- [Novel Synaptic Memory Device for Neuromorphic Computing](https://www.nature.com/articles/srep05333)
- [Multi-State MRAM Cells for Hardware Neuromorphic Computing](https://www.nature.com/articles/s41598-022-11199-4)
- [Efficient Synapse Memory Structure for Reconfigurable Digital Neuromorphic Hardware](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00829/full)
- [Hardware Acceleration of Irregular Applications Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd)
- [ASCENT Theme 4 Rapid Design Space Exploration of Neuromorphic Hardware](https://ascent.nd.edu/events/2021/09/16/ascent-theme-4-liaison-meeting-rapid-design-space-exploration-of-neuromorphic-hardware-using-emerging-devices/)
- [Neuromorphic Hardware Guide by Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/hardware/)

# Question 2:

Here is the next question that I have about the Design space exploration specifically for the the Event driven accelerators. This research is going to be build on top of the previous research findings about the general Design space exploration, where in this research question we narrow it doen to only for a specific Event driven accelerators and how the plan that I propose can be novel supported by the exsiting research.


Here are the question I wanted to have answer for :
1. Existing literature on the DSE performed for the event-driven accelerators and the results obtained.

2. Based on the memory layout paper provided, existing literature is available that showcases the effect of the memory layout

3. Similar to the memory layout paper, is there any literature on the PE arrangements effect for event-driven accelerators

4. Combining Research 2 and 3 can a research scope can be devised which is novel and addresses the limitations of the existing research and differentiates the new plan from the previous research

5. Structure the final plan to what DSE will be performed and what is the novelty of this exploration backed by the previous research.

The assumed event-driven behaviour is as follows: a single event arrives, all the computations corresponding to it are parallelized, and that event never arrives again. The architecture is similar to SENECA, multiply, and fire architecture.

Using this definition of event-driven, can you do the deep research

The event-driven behaviour from the Tool like Zigzag is modelled by restricting the Loop structure (Mapping constraints) where the inputs are forced to come one by one, and as the system is being modelled for the dense case, this type of modelling is acceptable.


Here is the preliminary research that I have done on the above questions:


### Key Points
- Research suggests that Design Space Exploration (DSE) for event-driven accelerators, particularly neuromorphic hardware, focuses on optimizing energy and latency, but comprehensive studies are limited.
- Memory layouts, like SRAM versus MRAM, significantly impact energy and area in event-driven architectures, with MRAM offering up to 40% area savings.
- Processing element (PE) arrangements in event-driven accelerators affect parallelism and efficiency, though specific studies on their impact are scarce.
- A novel research scope could involve jointly optimizing memory and PE configurations using tools like ZigZag, tailored for single-event processing, addressing gaps in holistic DSE approaches.
- The proposed DSE plan leverages existing tools and research to explore unique event-driven behaviors, potentially improving efficiency for edge computing applications.

### Understanding DSE for Event-Driven Accelerators
Event-driven accelerators, often used in neuromorphic computing, process data only when specific events occur, making them energy-efficient for tasks like edge inference. Design Space Exploration (DSE) helps find the best hardware setups by testing different configurations. Research indicates that DSE for these accelerators is still developing, with studies like those on spiking neural networks (SNNs) showing promise in optimizing energy and speed.

### Memory Layout Impacts
The choice of memory, such as SRAM or MRAM, affects how much power and space an accelerator uses. For example, one study found that MRAM can save about 40% in area compared to SRAM, which is crucial for compact edge devices. This aligns with findings that different memory setups can change energy use and processing speed in event-driven systems.

### Processing Element Arrangements
The way processing elements (like neurons in neuromorphic chips) are organized influences how efficiently an accelerator handles events. While some research explores dataflow architectures for event-driven tasks, there’s less focus on how different PE layouts directly impact performance, suggesting an area for further study.

### Novel Research Scope
Combining memory and PE optimization in a single DSE framework could be a new approach. Most studies look at these separately, missing how they work together. Using tools like ZigZag, modified to handle single-event processing (where each event is processed once with all computations at once), could lead to better designs for specific applications like real-time edge computing.

### Structured DSE Plan
The plan involves defining a range of memory types and PE layouts, using ZigZag to model event-driven behavior, and testing how these setups affect energy, speed, and size. This approach is unique because it focuses on single-event processing and combines memory and compute optimization, building on existing research to create more efficient accelerators.

---



# Literature Survey on Design Space Exploration for Event-Driven Accelerators

This survey addresses a research plan focused on Design Space Exploration (DSE) for event-driven accelerators, particularly those designed for single-event processing with architectures similar to multiply-and-fire models like SENECA. The plan is structured around five key research questions, rephrased for clarity and supported by relevant literature. Each question explores existing research, the impact of memory layouts and processing element (PE) arrangements, a novel research scope, and a structured DSE plan, with citations to academic papers providing insights into methodologies, parameter impacts, and optimization techniques.

## 1. Existing Literature on DSE for Event-Driven Accelerators

### Overview of DSE in Event-Driven Contexts
Event-driven accelerators, often associated with neuromorphic computing and spiking neural networks (SNNs), process computations triggered by discrete events, offering energy efficiency for applications like edge inference. DSE in this context involves exploring hardware configurations to optimize metrics such as latency, energy, and area. Research suggests that while DSE frameworks for neuromorphic hardware exist, comprehensive studies tailored to event-driven accelerators with single-event processing (where a single event triggers parallelized computations and does not recur) are limited.

### Key Studies and Findings
- **Neuromorphic Hardware DSE**: A study on hardware spiking neurons proposes a framework for DSE that defines architectures based on application-specific constraints, focusing on SNNs. It explores neural coding methods and hardware configurations, achieving energy-efficient designs for embedded AI [Design Space Exploration of Hardware Spiking Neurons](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041). Results indicate significant energy savings when optimizing neuron models and interconnects.
- **Rapid DSE for Neuromorphic Processors**: An SBIR proposal outlines a tool for rapid DSE, evaluating deep learning accelerators, spiking neuron accelerators, memristive systems, and photonic accelerators. It supports mapping algorithms to heterogeneous neuromorphic systems, showing improved performance for complex workloads [Rapid Design Space Exploration for Neuromorphic Processors](https://www.sbir.gov/node/1925299).
- **ASCENT Neuromorphic DSE**: A presentation discusses rapid DSE for neuromorphic hardware using emerging devices, exploring mappings, architectures, and devices. It highlights trade-offs in energy, area, and delay, with sparse SNN characteristics impacting dataflow designs [ASCENT Theme 4 Neuromorphic DSE](https://ascent.nd.edu/events/2021/09/16/ascent-theme-4-liaison-meeting-rapid-design-space-exploration-of-neuromorphic-hardware-using-emerging-devices/).
- **Event-Driven Execution Models**: A dissertation on hardware acceleration for irregular applications like Parallel Discrete Events Simulation (PDES) designs dataflow architectures with event-driven execution, improving parallelism and memory bandwidth utilization [Hardware Acceleration Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd).

### Results and Implications
These studies demonstrate that DSE for event-driven accelerators can yield significant improvements in energy efficiency (up to 10x in some neuromorphic designs) and performance, particularly when tailored to specific workloads. However, they often focus on general neuromorphic systems or specific applications, with less emphasis on single-event processing models, suggesting a gap in targeted DSE methodologies.

## 2. Impact of Memory Layouts in Event-Driven Accelerators

### Insights from the Provided Paper
The paper "Scaling NVMs in Event-Driven Architectures for Edge Inference" examines memory layouts in a multicore event-driven architecture for edge AI [Scaling NVMs in Event-Driven Architectures](attachment:0). Key findings include:
- **SRAM vs. MRAM**: SRAM offers high speed but high static power and lower density, while MRAM (e.g., STT-MRAM, VGSOT-MRAM) provides non-volatility, higher density, and lower static power, reducing area by ~40% across technology nodes (22nm to 14A).
- **Energy and Latency Trade-offs**: MRAM reduces total energy by ~25% and leakage energy by ~40% in 22nm and 12nm nodes but increases energy and latency at 14A due to slower write speeds.
- **Distributed Memory**: Near-memory processing with smaller, distributed memory blocks minimizes data movement, enhancing energy efficiency and reducing latency.
- **Depth-First Mapping**: Buffering only incomplete neural states reduces memory requirements, lowering latency and energy use.

### Additional Literature
- **Synaptic Memory Devices**: A study on Mn-doped HfO2-based synaptic memory devices compares them to VLSI SRAM synapses, showing a ~10x area reduction and >106 times lower power consumption per learning cycle [Novel Synaptic Memory Device](https://www.nature.com/articles/srep05333). This highlights the potential of emerging memories for neuromorphic applications.
- **Multi-State MRAM Cells**: Research on serially connected magnetic tunnel junctions (MTJs) forming multi-state MRAM cells for neuromorphic computing demonstrates their use in quantized weight storage, improving energy efficiency in neural networks [Multi-State MRAM Cells](https://www.nature.com/articles/s41598-022-11199-4).
- **Efficient Synapse Memory**: A study on synapse memory structures for digital neuromorphic hardware compares energy efficiency and throughput with systems like TrueNorth and Loihi, showing optimized memory layouts can enhance performance [Efficient Synapse Memory Structure](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00829/full).

### Table: Memory Layout Impacts
| **Memory Type** | **Advantages** | **Disadvantages** | **Impact on Metrics** | **Source** |
|-----------------|----------------|-------------------|-----------------------|------------|
| SRAM | High speed, low latency | High static power, low density | Higher energy, larger area | [Scaling NVMs in Event-Driven Architectures](attachment:0) |
| MRAM | Non-volatile, high density, low static power | Slower write speeds, higher write energy | ~40% area reduction, ~25% energy savings | [Scaling NVMs in Event-Driven Architectures](attachment:0) |
| Mn-doped HfO2 | Analog conductance, small size | Complex fabrication | ~10x area reduction, >106x power savings | [Novel Synaptic Memory Device](https://www.nature.com/articles/srep05333) |
| Multi-State MRAM | Quantized weights, energy-efficient | Limited scalability | Improved neural network efficiency | [Multi-State MRAM Cells](https://www.nature.com/articles/s41598-022-11199-4) |

### Implications
Memory layout significantly affects event-driven accelerators, with emerging technologies like MRAM offering substantial area and energy benefits. However, trade-offs in latency and write energy at advanced nodes require further optimization, particularly for single-event processing where data access patterns may differ.

## 3. Impact of PE Arrangements in Event-Driven Accelerators

### Existing Research
PE arrangements in event-driven accelerators, analogous to neuron or compute unit configurations in neuromorphic hardware, influence parallelism, throughput, and energy efficiency. Literature on this topic is less abundant but includes:
- **Event-Driven Dataflow Architectures**: The dissertation on irregular applications designs dataflow architectures for event-driven execution, showing increased parallelism and memory bandwidth utilization for PDES and graph processing [Hardware Acceleration Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd). Different PE configurations (e.g., interconnected cores) impact performance.
- **Neuromorphic Hardware DSE**: The study on hardware spiking neurons explores various neuron models and interconnects, effectively varying PE arrangements, to optimize SNN performance [Design Space Exploration of Hardware Spiking Neurons](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041). Results suggest that grid or hierarchical PE layouts affect latency and energy.
- **Neuromorphic Architectures**: The Neuromorphic Hardware Guide describes chips like SpiNNaker2 (mesh of ARM cores) and DynapCNN (event-driven neuron arrays), where PE arrangements influence computational efficiency [Neuromorphic Hardware Guide](https://open-neuromorphic.org/neuromorphic-computing/hardware/).

### Specific Findings
- **Parallelism and Efficiency**: Dataflow architectures with event-driven PEs can achieve higher throughput by optimizing interconnects, but specific impacts depend on workload characteristics.
- **SNN Performance**: In SNNs, PE arrangements like grid-based (TrueNorth) or mesh-based (SpiNNaker) affect spike propagation and energy use, with sparse workloads benefiting from optimized topologies.
- **Lack of Direct Studies**: Few studies explicitly quantify how different PE arrangements (e.g., number, topology) impact event-driven accelerators, particularly for single-event processing.

### Table: PE Arrangement Impacts
| **PE Arrangement** | **Description** | **Impact on Performance** | **Source** |
|--------------------|-----------------|---------------------------|------------|
| Dataflow Cores | Interconnected cores for event-driven tasks | Increased parallelism, higher throughput | [Hardware Acceleration Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd) |
| Grid-Based Neurons | 2D grid of spiking neurons (e.g., TrueNorth) | Efficient for sparse SNNs, lower latency | [Design Space Exploration of Hardware Spiking Neurons](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041) |
| Mesh of Cores | ARM cores in a mesh (e.g., SpiNNaker2) | High flexibility, energy-efficient for SNNs | [Neuromorphic Hardware Guide](https://open-neuromorphic.org/neuromorphic-computing/hardware/) |

### Implications
While PE arrangements are critical, the literature lacks detailed studies on their specific impact in event-driven accelerators, especially for single-event processing. This gap suggests an opportunity for research focusing on how PE topologies affect performance in such contexts.

## 4. Novel Research Scope

### Limitations in Existing Research
- **Separate Optimization**: Most studies optimize memory layouts or PE arrangements independently, missing their interplay in event-driven accelerators.
- **Generalized Workloads**: Existing DSE frameworks often target general neuromorphic or SNN workloads, not specific single-event processing models where events are processed once with parallelized computations.
- **Tool Adaptation**: Tools like ZigZag are designed for DNN accelerators, and their use for event-driven behaviors via restricted mappings is underexplored.
- **Performance Trade-offs**: Limited exploration of how memory and PE configurations together affect latency, energy, and area in event-driven systems.

### Proposed Research Scope
The novel research scope is to develop a **Joint Design Space Exploration Framework for Memory Hierarchies and PE Arrangements in Event-Driven Accelerators for Single-Event Processing**. This involves:
- **Integrated Optimization**: Simultaneously exploring memory configurations (e.g., SRAM, MRAM, hierarchy levels) and PE arrangements (e.g., number, interconnect topologies) to identify optimal designs.
- **Tool Adaptation**: Modifying tools like ZigZag to model single-event processing by restricting loop structures, ensuring accurate representation of event-driven behavior.
- **Workload Specificity**: Focusing on workloads where a single event triggers parallel computations that complete before the next event, relevant for edge applications like real-time sensor processing.
- **Performance Analysis**: Evaluating the impact of configurations on latency, energy, and area, providing insights into trade-offs and optimal designs.

### Differentiation from Previous Research
- **Holistic Approach**: Unlike studies focusing on memory or PEs separately, this research integrates both, addressing their combined effect.
- **Single-Event Focus**: Targets a unique computational model, differing from batch or stream processing, with distinct data access and compute patterns.
- **Tool Innovation**: Adapts existing DSE tools for event-driven contexts, potentially developing new mapping strategies to enhance accuracy and efficiency.

### Table: Comparison with Existing Research
| **Aspect** | **Existing Research** | **Proposed Research** |
|------------|-----------------------|-----------------------|
| Optimization Focus | Memory or PEs separately | Joint memory and PE optimization |
| Workload Type | General SNNs, DNNs | Single-event processing |
| Tool Usage | Standard ZigZag, Timeloop | Modified ZigZag with event-driven mappings |
| Performance Metrics | Energy, latency | Energy, latency, area with trade-off analysis |

## 5. Structured Plan for DSE

### DSE Methodology
The DSE plan for event-driven accelerators with single-event processing includes:
1. **Define the Design Space**:
   - **Memory Configurations**: Types (SRAM, MRAM), sizes (e.g., 64 Byte to 2 MByte), hierarchy levels, and data placement strategies.
   - **PE Arrangements**: Number of PEs, spatial topologies (e.g., grid, mesh, hierarchical), and interconnection patterns.
2. **Model Event-Driven Behavior**:
   - Adapt ZigZag to restrict loop structures, modeling single-event processing where inputs arrive sequentially, and computations are parallelized per event.
   - Develop or modify mapping strategies to ensure accurate representation of event-driven workloads.
3. **Evaluate Performance Metrics**:
   - Use simulations or analytical models to assess latency, energy consumption, and area for various configurations.
   - Employ tools like NeuroFlow (from the provided paper) or ZigZag’s Hardware Cost Estimator for accurate performance estimation.
4. **Optimize and Analyze Trade-offs**:
   - Apply heuristic search or machine learning to identify Pareto-optimal configurations.
   - Analyze trade-offs to provide design guidelines for event-driven accelerators.
5. **Validate Results**:
   - Compare DSE outcomes with hardware measurements or detailed simulations, targeting errors below 10% (similar to ZigZag’s 5-7.5% error rates).

### Novelty of the Exploration
- **Integrated DSE**: Combines memory and PE optimization, addressing a gap in holistic approaches.
- **Single-Event Processing**: Targets a specific computational model, relevant for edge applications with sporadic events.
- **Tool Adaptation**: Enhances tools like ZigZag for event-driven contexts, potentially introducing new mapping techniques.
- **Comprehensive Metrics**: Evaluates latency, energy, and area, providing a complete picture of design trade-offs.

### Backing from Previous Research
- **DSE Frameworks**: Builds on frameworks from neuromorphic DSE studies, adapting them for single-event processing [Design Space Exploration of Hardware Spiking Neurons](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041).
- **Memory Insights**: Leverages findings on SRAM vs. MRAM trade-offs to inform memory configuration choices [Scaling NVMs in Event-Driven Architectures](attachment:0).
- **PE Configurations**: Draws on dataflow and neuromorphic architecture studies to guide PE arrangement exploration [Hardware Acceleration Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd).

### Table: DSE Plan Components
| **Component** | **Description** | **Tools/Methods** | **Expected Outcome** |
|---------------|-----------------|-------------------|----------------------|
| Design Space | Memory types, PE topologies | ZigZag, custom definitions | Comprehensive parameter set |
| Event-Driven Model | Restricted loop structures | Modified ZigZag mappings | Accurate workload representation |
| Performance Evaluation | Latency, energy, area | NeuroFlow, Hardware Cost Estimator | Quantitative metrics |
| Optimization | Heuristic search, ML | Pareto analysis | Optimal configurations |
| Validation | Hardware comparisons | Simulation benchmarks | <10% error in estimates |

## Conclusion
This survey outlines a comprehensive research plan for DSE in event-driven accelerators, focusing on single-event processing. It reviews existing literature, highlights the impact of memory layouts and PE arrangements, proposes a novel joint optimization framework, and structures a detailed DSE plan. The approach builds on prior research while addressing gaps in holistic optimization and workload-specific DSE, offering potential advancements in energy-efficient accelerator design for edge computing.

## Key Citations
- [Design Space Exploration of Hardware Spiking Neurons for Embedded Artificial Intelligence](https://www.sciencedirect.com/science/article/abs/pii/S0893608019303041)
- [Rapid Design Space Exploration for Mapping Applications to Neuromorphic Processors](https://www.sbir.gov/node/1925299)
- [Novel Synaptic Memory Device for Neuromorphic Computing](https://www.nature.com/articles/srep05333)
- [Multi-State MRAM Cells for Hardware Neuromorphic Computing](https://www.nature.com/articles/s41598-022-11199-4)
- [Efficient Synapse Memory Structure for Reconfigurable Digital Neuromorphic Hardware](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00829/full)
- [Hardware Acceleration of Irregular Applications Using Event-Driven Execution](https://escholarship.org/uc/item/3tj261gd)
- [ASCENT Theme 4 Rapid Design Space Exploration of Neuromorphic Hardware](https://ascent.nd.edu/events/2021/09/16/ascent-theme-4-liaison-meeting-rapid-design-space-exploration-of-neuromorphic-hardware-using-emerging-devices/)
- [Neuromorphic Hardware Guide by Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/hardware/)


In my priliminary reserch about the DSE for the Event driven accelerators, I wasn't fully able to find a contrasting novelty that suggests that this study is missing and could be considered as a contribution to work on. I need strong evidence that working on DSE for the Event driven accelerators using my own flow build on top of zigzag is a novel one. Right now I have two ideas for the DSE for event drven accelrators
1. Observing the effect of changing the SRAM parameters and keeping the PE constant for a workload like the Mobile Net v2
2. Observing the effect of changing the PE and keeping the SRAM parameters constant for the same workload
3. Replacing the SRAM with the MRAM and visualising it's effects for the same workload.

I would in the end want to achieve a pareto optimal curve with the best configuration of the architecture for Energy, latency and EDP (energy delay product) or the Other metrics like ED^2P (Energy delay square product).

Now can you perform a deep research and from the litrature find/show what I am doing is novel or what else can be added to the exiting plan to make it novel.

In the end of the research you will be answering my research question and also providing support to my research I am performing.

## Answer 2:

Here is the answer from the Gemini and Chatgpt: https://docs.google.com/document/d/1vC0WFAnhy9yVUh4YbdpWcfONLXasZ35qCyc_xVidwUU/edit?usp=sharing

=======================================================================================================================================================

# Exploring the effects of the SRAM and MRAM with constant PEs

## Steps:

1. Development of the Excel sheet with various configuration for exploring the effects

### Question 1:

The DSE of an event driven accelerator needs to be performed to observer the effect of the varying the SRAM parameters as shown below

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

  NOC_inputs_outputs:
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

"

Here I want you to take the workload as into consideration, which is VGGnet the configuration is as shown below



# ID Name Type Input Size Output Size Kernel/Stride Loop Sizes Operand Source

# --0 conv1_1 Conv 224x224x3 224x224x64 3x3, s=1 [1, 64, 3, 224, 224, 3, 3] I:0, W:0

# --1 conv1_2 Conv 224x224x64 224x224x64 3x3, s=1 [1, 64, 64, 224, 224, 3, 3] I:0, W:1

# --2 maxpool1 Pooling 224x224x64 112x112x64 2x2, s=2 [1, 64, 112, 112, 2, 2] I:1, W:2

# --3 conv2_1 Conv 112x112x64 112x112x128 3x3, s=1 [1, 128, 64, 112, 112, 3, 3] I:2, W:3

# --4 conv2_2 Conv 112x112x128 112x112x128 3x3, s=1 [1, 128, 128, 112, 112, 3, 3] I:3, W:4

# --5 maxpool2 Pooling 112x112x128 56x56x128 2x2, s=2 [1, 128, 56, 56, 2, 2] I:4, W:5

# --6 conv3_1 Conv 56x56x128 56x56x256 3x3, s=1 [1, 256, 128, 56, 56, 3, 3] I:5, W:6

# --7 conv3_2 Conv 56x56x256 56x56x256 3x3, s=1 [1, 256, 256, 56, 56, 3, 3] I:6, W:7

# --8 conv3_3 Conv 56x56x256 56x56x256 3x3, s=1 [1, 256, 256, 56, 56, 3, 3] I:7, W:8

# --9 maxpool3 Pooling 56x56x256 28x28x256 2x2, s=2 [1, 256, 28, 28, 2, 2] I:8, W:9

# --10 conv4_1 Conv 28x28x256 28x28x512 3x3, s=1 [1, 512, 256, 28, 28, 3, 3] I:9, W:10

# --11 conv4_2 Conv 28x28x512 28x28x512 3x3, s=1 [1, 512, 512, 28, 28, 3, 3] I:10, W:11

# --12 conv4_3 Conv 28x28x512 28x28x512 3x3, s=1 [1, 512, 512, 28, 28, 3, 3] I:11, W:12

# --13 maxpool4 Pooling 28x28x512 14x14x512 2x2, s=2 [1, 512, 14, 14, 2, 2] I:12, W:13

# --14 conv5_1 Conv 14x14x512 14x14x512 3x3, s=1 [1, 512, 512, 14, 14, 3, 3] I:13, W:14

# -15 conv5_2 Conv 14x14x512 14x14x512 3x3, s=1 [1, 512, 512, 14, 14, 3, 3] I:14, W:15

# -16 conv5_3 Conv 14x14x512 14x14x512 3x3, s=1 [1, 512, 512, 14, 14, 3, 3] I:15, W:16

# -17 maxpool5 Pooling 14x14x512 7x7x512 2x2, s=2 [1, 512, 7, 7, 2, 2] I:16, W:17

# --18 fc1 Conv 7x7x512 1x1x4096 7x7, global [1, 4096, 512, 1, 1, 7, 7] I:17, W:18

# --19 fc2 Conv 1x1x4096 1x1x4096 1x1, s=1 [1, 4096, 4096, 1, 1, 1, 1] I:18, W:19

# --20 fc3 Conv 1x1x4096 1x1x1000 1x1, s=1 [1, 1000, 4096, 1, 1, 1, 1] I:19, W:20


Above is the architecture and the VGGnet model which are being used for the DSE of the given architecture. The observation is specifically being made on what is the effect on the latency, energy and the EDP on varying the SRAM parameters. 

I need your help to vary these parameters in a specific manner than just randomly changing the number one by one. I want the configuration that I test have a specific reason and are worth considering to perform the exploration adn finally having a pareto optimal curve. 

Can you help me in developing a sequence of configuration of the SRAM that I can test on to see it's impact on the performance of the VGGnet model which is being modelled.


Here is the google docs explanaing the novely: https://docs.google.com/document/d/1vC0WFAnhy9yVUh4YbdpWcfONLXasZ35qCyc_xVidwUU/edit?usp=sharing

The end conclusion is that, combined study of the Memory and PEs is not that novel, but with proper validation and justification it will be an intresting research.

--------------------------------------------------------------------------------------------------------------------------------------------------

# Steps followed in the DSE of the SRAM

The Architecture, Workload and the mapping constraints are as follows

### Architecture

``` yaml
name: Event_Driven_Accelerator

operational_array:
  unit_energy: 0.04  # for ANN MAC operations
  unit_area: 1
  dimensions: [D1]
  sizes: [8]  # 8 NPEs

memories:
  register:
    size: 64  #size of each register e.g., 64x16b for NPE register files
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
    size: 3221225472  #3GB.., 24Gb Data memory # 256 KB, e.g., 2 Mb Data Memory   1048576  # 1 MB, e.g., 8 Mb Data Memory
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

  NOC_inputs_outputs:
    size: 134217728  # 128 MB, e.g., for STT-MRAM
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
```

### Workload

``` yaml

# ID	Name	Type	Input Size	Output Size	Kernel/Stride	Loop Sizes	Operand Source
# --0	conv1_1	Conv	224x224x3	224x224x64	3x3, s=1	[1, 64, 3, 224, 224, 3, 3]	I:0, W:0
# --1	conv1_2	Conv	224x224x64	224x224x64	3x3, s=1	[1, 64, 64, 224, 224, 3, 3]	I:0, W:1
# --2	maxpool1	Pooling	224x224x64	112x112x64	2x2, s=2	[1, 64, 112, 112, 2, 2]	I:1, W:2
# --3	conv2_1	Conv	112x112x64	112x112x128	3x3, s=1	[1, 128, 64, 112, 112, 3, 3]	I:2, W:3
# --4	conv2_2	Conv	112x112x128	112x112x128	3x3, s=1	[1, 128, 128, 112, 112, 3, 3]	I:3, W:4
# --5	maxpool2	Pooling	112x112x128	56x56x128	2x2, s=2	[1, 128, 56, 56, 2, 2]	I:4, W:5
# --6	conv3_1	Conv	56x56x128	56x56x256	3x3, s=1	[1, 256, 128, 56, 56, 3, 3]	I:5, W:6
# --7	conv3_2	Conv	56x56x256	56x56x256	3x3, s=1	[1, 256, 256, 56, 56, 3, 3]	I:6, W:7
# --8	conv3_3	Conv	56x56x256	56x56x256	3x3, s=1	[1, 256, 256, 56, 56, 3, 3]	I:7, W:8
# --9	maxpool3	Pooling	56x56x256	28x28x256	2x2, s=2	[1, 256, 28, 28, 2, 2]	I:8, W:9
# --10	conv4_1	Conv	28x28x256	28x28x512	3x3, s=1	[1, 512, 256, 28, 28, 3, 3]	I:9, W:10
# --11	conv4_2	Conv	28x28x512	28x28x512	3x3, s=1	[1, 512, 512, 28, 28, 3, 3]	I:10, W:11
# --12	conv4_3	Conv	28x28x512	28x28x512	3x3, s=1	[1, 512, 512, 28, 28, 3, 3]	I:11, W:12
# --13	maxpool4	Pooling	28x28x512	14x14x512	2x2, s=2	[1, 512, 14, 14, 2, 2]	I:12, W:13
# --14	conv5_1	Conv	14x14x512	14x14x512	3x3, s=1	[1, 512, 512, 14, 14, 3, 3]	I:13, W:14
# -15	conv5_2	Conv	14x14x512	14x14x512	3x3, s=1	[1, 512, 512, 14, 14, 3, 3]	I:14, W:15
# -16	conv5_3	Conv	14x14x512	14x14x512	3x3, s=1	[1, 512, 512, 14, 14, 3, 3]	I:15, W:16
# -17	maxpool5	Pooling	14x14x512	7x7x512	2x2, s=2	[1, 512, 7, 7, 2, 2]	I:16, W:17
# --18	fc1	Conv	7x7x512	1x1x4096	7x7, global	[1, 4096, 512, 1, 1, 7, 7]	I:17, W:18
# --19	fc2	Conv	1x1x4096	1x1x4096	1x1, s=1	[1, 4096, 4096, 1, 1, 1, 1]	I:18, W:19
# --20	fc3	Conv	1x1x4096	1x1x1000	1x1, s=1	[1, 1000, 4096, 1, 1, 1, 1]	I:19, W:20

- id: 0 # Conv1 Stride 1
  name: layer_0
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 64, 3, 224, 224, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 0


- id: 1 # Conv1 Stride 1
  name: layer_1
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 64, 64, 224, 224, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 0
    W: 1

# - id: 2 # Pool1 #Stride 2
#   name: pooling_layer_1
#   operator_type: Pooling
#   equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
#   dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
#   loop_dims: [B, G, IY, IX, FY, FX]
#   loop_sizes: [1, 64, 224, 224, 2, 2]
#   operand_precision:
#     W: 16
#     I: 16
#     O: 16
#     O_final: 16
#   operand_source:
#     I: 1
#     W: 2

- id: 2 # Pool1 #Stride 2
  name: pooling_layer_1
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, G, OY, OX, FY, FX]
  loop_sizes: [1, 64, 112, 112, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 1
    W: 2


- id: 3 # Conv1 Stride 1
  name: layer_3
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 128, 64, 112, 112, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 2
    W: 3

- id: 4 # Conv1 Stride 1
  name: layer_4
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 128, 128, 112, 112, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 3
    W: 4

# - id: 5 # Pool1 #Stride 2
#   name: pooling_layer_2
#   operator_type: Pooling
#   equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
#   dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
#   loop_dims: [B, G, IY, IX, FY, FX]
#   loop_sizes: [1, 128, 112, 112, 2, 2]
#   operand_precision:
#     W: 16
#     I: 16
#     O: 16
#     O_final: 16
#   operand_source:
#     I: 4
#     W: 5


- id: 5 # Pool1 #Stride 2
  name: pooling_layer_2
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, G, OY, OX, FY, FX]
  loop_sizes: [1, 128, 56, 56, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 4
    W: 5

- id: 6 # Conv1 Stride 1
  name: layer_6
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 256, 128, 56, 56, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 5
    W: 6

- id: 7 # Conv1 Stride 1
  name: layer_7
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 256, 256, 56, 56, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 6
    W: 7

- id: 8 # Conv1 Stride 1
  name: layer_8
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 256, 256, 56, 56, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 7
    W: 8

# - id: 9 # Pool1 #Stride 2
#   name: pooling_layer_3
#   operator_type: Pooling
#   equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
#   dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
#   loop_dims: [B, G, IY, IX, FY, FX]
#   loop_sizes: [1, 256, 56, 56, 2, 2]
#   operand_precision:
#     W: 16
#     I: 16
#     O: 16
#     O_final: 16
#   operand_source:
#     I: 8
#     W: 9


- id: 9 # Pool1 #Stride 2
  name: pooling_layer_3
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, G, OY, OX, FY, FX]
  loop_sizes: [1, 256, 28, 28, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 8
    W: 9

- id: 10 # Conv1 Stride 1
  name: layer_10
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 512, 256, 28, 28, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 9
    W: 10

- id: 11 # Conv1 Stride 1
  name: layer_11
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 512, 256, 28, 28, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 10
    W: 11

- id: 12 # Conv1 Stride 1
  name: layer_12
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 512, 512, 28, 28, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 11
    W: 12

# - id: 13 # Pool1 #Stride 2
#   name: pooling_layer_4
#   operator_type: Pooling
#   equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
#   dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
#   loop_dims: [B, G, IY, IX, FY, FX]
#   loop_sizes: [1, 512, 28, 28, 2, 2]
#   operand_precision:
#     W: 16
#     I: 16
#     O: 16
#     O_final: 16
#   operand_source:
#     I: 12
#     W: 13

- id: 13 # Pool1 #Stride 2
  name: pooling_layer_4
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, G, OY, OX, FY, FX]
  loop_sizes: [1, 512, 14, 14, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 12
    W: 13

- id: 14 # Conv1 Stride 1
  name: layer_14
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 512, 512, 14, 14, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 13
    W: 14

- id: 15 # Conv1 Stride 1
  name: layer_15
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 512, 512, 14, 14, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 14
    W: 15

- id: 16 # Conv1 Stride 1
  name: layer_16
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 512, 512, 14, 14, 3, 3]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 15
    W: 16

# - id: 17 # Pool1 #Stride 2
#   name: pooling_layer_5
#   operator_type: Pooling
#   equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
#   dimension_relations: [ox=2*ix+1*fx, oy=2*iy+1*fy]
#   loop_dims: [B, G, IY, IX, FY, FX]
#   loop_sizes: [1, 512, 14, 14, 2, 2]
#   operand_precision:
#     W: 16
#     I: 16
#     O: 16
#     O_final: 16
#   operand_source:
#     I: 16
#     W: 17

- id: 17 # Pool1 #Stride 2
  name: pooling_layer_5
  operator_type: Pooling
  equation: O[b][g][oy][ox]+=W[fy][fx]*I[b][g][iy][ix]
  dimension_relations: [ix=2*ox+1*fx, iy=2*oy+1*fy]
  loop_dims: [B, G, OY, OX, FY, FX]
  loop_sizes: [1, 512, 7, 7, 2, 2]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 16
    W: 17


- id: 18 #Flatenning
  name: layer_18
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 4096, 512, 7, 7, 7, 7]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 17
    W: 18
  
- id: 19 #FC
  name: layer_19
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1, 4096, 4096, 1, 1, 1, 1]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 18
    W: 19

- id: 20 #FC
  name: layer_20
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ox=1*ix+1*fx, oy=1*iy+1*fy]
  loop_dims: [B, K, C, IY, IX, FY, FX]
  loop_sizes: [1,1000,4096, 1, 1, 1, 1]
  operand_precision:
    W: 16
    I: 16
    O: 16
    O_final: 16
  operand_source:
    I: 19
    W: 20

```

#### Mapping constraints

As the intrest is to perform the exploraiton for the event driven case then spatial unrolling across FX, FY and K

Now, the first question is to idently the computationally demanding layer in the entire VGG net


