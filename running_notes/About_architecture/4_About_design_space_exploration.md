# Structure of the Literature Review and the plan

1. What is Design space exploration (DSE) in detail and what are the steps followed to perform it in a structured manner ?

2. Using an exploration tool like Zigzag or Timeloop when DSE is performed the input metrics in control are the Memory arrangements and PE elements and their arrangements and interconnections? How can the steps found in the above research be linked to this question ?

3. Based on the variable that can be changed in hand because of the using the tools like zigzag and timeloop what is the supporting literature availbe that show case that by changing the what parameters (Memory or PE arrangements), the effects can be seen in which metric Latency, Energy and the EDP any correlation that was already found which can be used for the validation on getting the results on performing the exploration ?

4. Finally how to set a bound on the design space exploration on until what extent this exploration can be done, Is there any research that could help to set this limit and explore within that bound ?

------------------------------------------------------------------------------------------------------------------------------------------------------

Use of the AI tools 

1. Grok Deeper research (To improve the research question further and structure them more clearly with previous research)
2. Gemini Deep research (To build on the top of grok research and give a detailed explanation)
2. Chatgpt Deep research (To build on the top of grok research and give a detailed explanation)
3. A consolidatied plan that combines the understanding of the Gemini and Chatgpt for the final answer of this chat

======================================================================================================================================================

# Question 1 (Grok)

Can you by taking this plan shown below that I have about the literature survey on the DSE and how it is performed and gathering the supporting literature. 

"My research plan in the form of questions:
1. What is Design space exploration (DSE) in detail and what are the steps followed to perform it in a structured manner ?

2. Using an exploration tool like Zigzag or Timeloop when DSE is performed the input metrics in control are the Memory arrangements and PE elements and their arrangements and interconnections? How can the steps found in the above research be linked to this question ?

3. Based on the variable that can be changed in hand because of the using the tools like zigzag and timeloop what is the supporting literature availbe that show case that by changing the what parameters (Memory or PE arrangements), the effects can be seen in which metric Latency, Energy and the EDP any correlation that was already found which can be used for the validation on getting the results on performing the exploration ?

4. Finally how to set a bound on the design space exploration on until what extent this exploration can be done, Is there any research that could help to set this limit and explore within that bound ?"


Do the following:
1. Structure the entire research questions correctly
2. Do an intial search and find relevant research that could briefly answer the question in the research.
3. Give a final output in markdown, with a more detailed and structured reseach question and supporting citation from relevant papers that explain what is being planned to be achieved with these research question.


## Answer 1:
Literature Survey on Design Space Exploration for Hardware Design
This survey addresses a research plan focused on Design Space Exploration (DSE) in the context of hardware design, particularly for deep learning accelerators using tools like ZigZag and Timeloop. The plan is structured around four key research questions, which are rephrased for clarity and supported by relevant literature. Each question is explored in detail, with citations to academic papers and resources that provide insights into DSE methodologies, parameter impacts, and bounding techniques.
1. Detailed Explanation of Design Space Exploration and Structured Steps
Definition of DSE
Design Space Exploration (DSE) is a systematic process used in system design, particularly for electronic and embedded systems, to evaluate and select optimal design alternatives before implementation. It involves exploring a vast set of possible configurations—termed the design space—to optimize metrics such as performance, power consumption, area, or cost. The complexity arises from the numerous design choices, including component selection, their configurations, interconnections, and algorithmic implementations, which can result in millions or billions of possibilities Design Space Exploration - Wikipedia.
Structured Steps in DSE
While DSE methodologies vary depending on the application and tools, a general structured approach includes the following steps:

Define the Design Space: Identify all variable parameters, such as architectural components (e.g., memory types, processing elements), their configurations, and interconnections.
Specify Objectives and Constraints: Define the metrics to optimize (e.g., latency, energy, energy-delay product) and constraints (e.g., area, cost) that designs must satisfy.
Generate Design Points: Create or sample possible designs from the design space, either exhaustively for small spaces or selectively using algorithms for larger ones.
Evaluate Designs: Assess each design using simulations, analytical models, or cost estimators to estimate performance against objectives and check constraint compliance.
Analyze and Select: Perform trade-off analysis to identify the best designs, often using multi-objective optimization techniques.
Refine and Iterate: Adjust the design space or exploration strategy based on results and iterate to improve outcomes.

Supporting Literature

Kang et al. (2010) describe DSE as a process to maximize performance parameters like time and power density while minimizing cost and area, emphasizing the need for systematic exploration due to the large design space An Approach for Effective Design Space Exploration.
The Wikipedia entry on DSE highlights its role in electronic system design, noting that trade-off analysis based on parameters like power and performance forms the basis of DSE Design Space Exploration - Wikipedia.
Shathanaa and Ramasubramanian provide a survey on DSE for architectural synthesis, describing it as an iterative process reliant on the architect’s decisions to navigate vast design choices Design Space Exploration for Architectural Synthesis—A Survey.

Application in Context
In the context of deep learning accelerators, tools like ZigZag implement DSE by defining a design space that includes memory hierarchies and processing element (PE) arrangements, then systematically evaluating these to optimize energy and latency ZigZag: A Memory-Centric Rapid DNN Accelerator.
2. Key Parameters in ZigZag and Timeloop and Their Relation to DSE Steps
Parameters Controlled in ZigZag and Timeloop
Tools like ZigZag and Timeloop are specialized for DSE of deep neural network (DNN) accelerators, focusing on hardware architecture and algorithm-to-hardware mappings. The key parameters that can be controlled include:

Memory Hierarchy: Configurations such as memory levels, sizes (e.g., 64 Byte to 2 MByte), bandwidths, and data placement strategies.
Processing Elements (PEs): Number of PEs, their spatial arrangements (e.g., multi-dimensional MAC arrays), and interconnection patterns.
Mappings: Temporal and spatial scheduling of computations, including even and uneven mapping strategies that determine how data and operations are assigned to hardware resources.

These parameters define the design space explored by these tools. For instance, ZigZag’s Architecture Generator creates various memory hierarchies, while its Temporal Mapping Generator explores different scheduling options, both of which are evaluated using a Hardware Cost Estimator ZigZag: A Memory-Centric Rapid DNN Accelerator.
Linking to DSE Steps
The parameters relate directly to the general DSE steps:

Define the Design Space: Memory hierarchies and PE arrangements form the core of the design space in ZigZag and Timeloop.
Specify Objectives and Constraints: Objectives like minimizing energy or latency are set, with constraints like area limits.
Generate Design Points: ZigZag’s Architecture and Mapping Generators produce various configurations.
Evaluate Designs: The Hardware Cost Estimator assesses energy, latency, and throughput.
Analyze and Select: Results are compared to select optimal designs.
Refine and Iterate: Search strategies (e.g., heuristic search) refine the exploration.

Supporting Literature

Mei et al. (2020) detail ZigZag’s approach, emphasizing its memory-centric nested-for-loop format to integrate algorithm, accelerator, and mapping, allowing exploration of uneven mappings ZigZag: A Memory-Centric Rapid DNN Accelerator.
Parashar et al. (2019) describe Timeloop, which evaluates DNN accelerators by exploring mappings and architectures, focusing on similar parameters Timeloop: A Systematic Approach.
The GitHub repository for ZigZag provides practical insights into how these parameters are configured in input files ZigZag GitHub.

Table: Parameters in ZigZag and Timeloop



Parameter
Description
Tool Example



Memory Hierarchy
Levels, sizes, bandwidths, data placement
ZigZag: 64 Byte to 2 MByte memories


Processing Elements
Number, arrangement, interconnections
ZigZag: Multi-dimensional MAC arrays


Mappings
Temporal/spatial scheduling of computations
ZigZag: Even/uneven scheduling


3. Impact of Parameter Changes on Performance Metrics
Effects of Parameter Variations
Research using ZigZag and Timeloop demonstrates that varying memory and PE arrangements significantly impacts performance metrics like latency, energy, and energy-delay product (EDP):

Memory Hierarchy: Different memory sizes and levels affect data access times and energy costs. Flexible hierarchies can reduce energy if overheads are managed ZigZag: A Memory-Centric Rapid DNN Accelerator.
PE Arrangements: The number and layout of PEs influence computation parallelism and throughput. Spatial unrolling can lead to up to 5× energy differences in non-shared memory setups.
Mappings: Scheduling strategies (e.g., even vs. uneven mappings) cause significant variations. ZigZag’s case studies show up to 4.7× energy variance and 8× throughput variance for AlexNet CONV2 on an Eyeriss-like architecture.

Validation and Correlations
To validate DSE results, tools compare estimates with hardware measurements or detailed simulations:

ZigZag reports a 5% error for Eyeriss and 7.5% for ENVISION accelerators, with 6% energy and 9% PE utilization errors against in-house RTL simulations.
These correlations allow researchers to trust the tools’ predictions and use them to guide design decisions.

Supporting Literature

Mei et al. (2020) provide case studies showing that ZigZag’s uneven mapping can yield up to 33% more energy-efficient solutions compared to tools like Timeloop+Accelergy ZigZag: A Memory-Centric Rapid DNN Accelerator.
Mei et al. (2021) further explore ZigZag’s capabilities, demonstrating improved design points through joint architecture-mapping exploration ZigZag: Enlarging Joint Architecture-Mapping.
Timeloop’s evaluation framework likely provides similar insights, though specific case studies are less detailed in the provided references Timeloop: A Systematic Approach.

Table: Impact of Parameters on Metrics



Parameter Changed
Metric Affected
Observed Impact
Source



Mapping (Scheduling)
Energy, Throughput
4.7× energy, 8× throughput variance
ZigZag: A Memory-Centric Rapid DNN Accelerator


Memory Hierarchy
Energy
30% energy cost variation
ZigZag: A Memory-Centric Rapid DNN Accelerator


Spatial Unrolling
Energy
Up to 5× energy difference
ZigZag: A Memory-Centric Rapid DNN Accelerator


4. Setting Bounds on Design Space Exploration
Techniques for Bounding DSE
The large size of design spaces—often millions or billions of possibilities—makes exhaustive exploration infeasible. Several techniques are used to bound or reduce the design space:

Heuristic Search: Algorithms like simulated annealing, genetic algorithms, or iterative searches focus on promising regions, reducing evaluation time. ZigZag’s heuristic search achieves a 2.5× speedup with no energy loss ZigZag: A Memory-Centric Rapid DNN Accelerator.
Machine Learning: Predictive models estimate design quality, guiding exploration to high-potential areas Design-Space Exploration - ScienceDirect.
Equivalence Classes: Grouping equivalent designs to avoid redundant evaluations, as proposed by Kang et al. using symmetry breaking predicates An Approach for Effective Design Space Exploration.
Constraints: Applying prior knowledge or requirements (e.g., area limits) to restrict the design space.

Research on Bounding Techniques

Kang et al. (2010) introduce a method where user-defined equivalence notions generate symmetry breaking predicates, ensuring only a subset of the design space is explored An Approach for Effective Design Space Exploration.
Mei et al. (2020) demonstrate ZigZag’s use of iterative search (7.5× speedup, 1.6% energy penalty) and spatial utilization thresholds (e.g., >0.75 MAC array utilization) to limit exploration ZigZag: A Memory-Centric Rapid DNN Accelerator.
Siemens Software discusses parametric exploration and topology optimization to focus on optimal shapes within constraints Design Space Exploration - Siemens.

Practical Application
In ZigZag, bounding is achieved by setting thresholds (e.g., spatial utilization) and using efficient search strategies. For example, evaluating 7,168 final schedules instead of 4,887,334 saves significant time while maintaining near-optimal results ZigZag GitHub.
Table: Bounding Techniques in DSE



Technique
Description
Example Application
Source



Heuristic Search
Uses algorithms to explore promising designs
ZigZag: 2.5× speedup, no energy loss
ZigZag: A Memory-Centric Rapid DNN Accelerator


Machine Learning
Predicts design quality to guide exploration
General DSE frameworks
Design-Space Exploration - ScienceDirect


Equivalence Classes
Groups equivalent designs to avoid redundancy
Symmetry breaking predicates
An Approach for Effective Design Space Exploration


Constraints
Limits design space with prior requirements
Area constraints in ZigZag
ZigZag GitHub


Conclusion
This survey provides a comprehensive overview of DSE in the context of hardware design, focusing on tools like ZigZag and Timeloop. It defines DSE and its structured steps, identifies key parameters and their impacts, and discusses methods to bound exploration. The literature highlights the importance of systematic approaches and efficient techniques to navigate large design spaces, ensuring optimal designs for deep learning accelerators.
Key Citations

An Approach for Effective Design Space Exploration
Design Space Exploration - Wikipedia
Design Space Exploration for Architectural Synthesis—A Survey
ZigZag: A Memory-Centric Rapid DNN Accelerator Design Space Exploration Framework
Timeloop: A Systematic Approach to DNN Accelerator Evaluation
ZigZag: Enlarging Joint Architecture-Mapping Design Space Exploration for DNN Accelerators
Design-Space Exploration - an overview | ScienceDirect Topics
ZigZag GitHub - HW Architecture-Mapping Design Space Exploration Framework
Design Space Exploration and Optimization | Siemens Software

# Question 2: (Gemini and Chatgpt)

I have done a priliminary research about these research questions as follows

Research question: 

1. What is Design space exploration (DSE) in detail and what are the steps followed to perform it in a structured manner ?

2. Using an exploration tool like Zigzag or Timeloop when DSE is performed the input metrics in control are the Memory arrangements and PE elements and their arrangements and interconnections? How can the steps found in the above research be linked to this question ?

3. Based on the variable that can be changed in hand because of the using the tools like zigzag and timeloop what is the supporting literature availbe that show case that by changing the what parameters (Memory or PE arrangements), the effects can be seen in which metric Latency, Energy and the EDP any correlation that was already found which can be used for the validation on getting the results on performing the exploration ?

4. Finally how to set a bound on the design space exploration on until what extent this exploration can be done, Is there any research that could help to set this limit and explore within that bound ?

Priliminary Reseach by refering to the basic literature:

Literature Survey on Design Space Exploration for Hardware Design
This survey addresses a research plan focused on Design Space Exploration (DSE) in the context of hardware design, particularly for deep learning accelerators using tools like ZigZag and Timeloop. The plan is structured around four key research questions, which are rephrased for clarity and supported by relevant literature. Each question is explored in detail, with citations to academic papers and resources that provide insights into DSE methodologies, parameter impacts, and bounding techniques.
1. Detailed Explanation of Design Space Exploration and Structured Steps
Definition of DSE
Design Space Exploration (DSE) is a systematic process used in system design, particularly for electronic and embedded systems, to evaluate and select optimal design alternatives before implementation. It involves exploring a vast set of possible configurations—termed the design space—to optimize metrics such as performance, power consumption, area, or cost. The complexity arises from the numerous design choices, including component selection, their configurations, interconnections, and algorithmic implementations, which can result in millions or billions of possibilities Design Space Exploration - Wikipedia.
Structured Steps in DSE
While DSE methodologies vary depending on the application and tools, a general structured approach includes the following steps:

Define the Design Space: Identify all variable parameters, such as architectural components (e.g., memory types, processing elements), their configurations, and interconnections.
Specify Objectives and Constraints: Define the metrics to optimize (e.g., latency, energy, energy-delay product) and constraints (e.g., area, cost) that designs must satisfy.
Generate Design Points: Create or sample possible designs from the design space, either exhaustively for small spaces or selectively using algorithms for larger ones.
Evaluate Designs: Assess each design using simulations, analytical models, or cost estimators to estimate performance against objectives and check constraint compliance.
Analyze and Select: Perform trade-off analysis to identify the best designs, often using multi-objective optimization techniques.
Refine and Iterate: Adjust the design space or exploration strategy based on results and iterate to improve outcomes.

Supporting Literature

Kang et al. (2010) describe DSE as a process to maximize performance parameters like time and power density while minimizing cost and area, emphasizing the need for systematic exploration due to the large design space An Approach for Effective Design Space Exploration.
The Wikipedia entry on DSE highlights its role in electronic system design, noting that trade-off analysis based on parameters like power and performance forms the basis of DSE Design Space Exploration - Wikipedia.
Shathanaa and Ramasubramanian provide a survey on DSE for architectural synthesis, describing it as an iterative process reliant on the architect’s decisions to navigate vast design choices Design Space Exploration for Architectural Synthesis—A Survey.

Application in Context
In the context of deep learning accelerators, tools like ZigZag implement DSE by defining a design space that includes memory hierarchies and processing element (PE) arrangements, then systematically evaluating these to optimize energy and latency ZigZag: A Memory-Centric Rapid DNN Accelerator.
2. Key Parameters in ZigZag and Timeloop and Their Relation to DSE Steps
Parameters Controlled in ZigZag and Timeloop
Tools like ZigZag and Timeloop are specialized for DSE of deep neural network (DNN) accelerators, focusing on hardware architecture and algorithm-to-hardware mappings. The key parameters that can be controlled include:

Memory Hierarchy: Configurations such as memory levels, sizes (e.g., 64 Byte to 2 MByte), bandwidths, and data placement strategies.
Processing Elements (PEs): Number of PEs, their spatial arrangements (e.g., multi-dimensional MAC arrays), and interconnection patterns.
Mappings: Temporal and spatial scheduling of computations, including even and uneven mapping strategies that determine how data and operations are assigned to hardware resources.

These parameters define the design space explored by these tools. For instance, ZigZag’s Architecture Generator creates various memory hierarchies, while its Temporal Mapping Generator explores different scheduling options, both of which are evaluated using a Hardware Cost Estimator ZigZag: A Memory-Centric Rapid DNN Accelerator.
Linking to DSE Steps
The parameters relate directly to the general DSE steps:

Define the Design Space: Memory hierarchies and PE arrangements form the core of the design space in ZigZag and Timeloop.
Specify Objectives and Constraints: Objectives like minimizing energy or latency are set, with constraints like area limits.
Generate Design Points: ZigZag’s Architecture and Mapping Generators produce various configurations.
Evaluate Designs: The Hardware Cost Estimator assesses energy, latency, and throughput.
Analyze and Select: Results are compared to select optimal designs.
Refine and Iterate: Search strategies (e.g., heuristic search) refine the exploration.

Supporting Literature

Mei et al. (2020) detail ZigZag’s approach, emphasizing its memory-centric nested-for-loop format to integrate algorithm, accelerator, and mapping, allowing exploration of uneven mappings ZigZag: A Memory-Centric Rapid DNN Accelerator.
Parashar et al. (2019) describe Timeloop, which evaluates DNN accelerators by exploring mappings and architectures, focusing on similar parameters Timeloop: A Systematic Approach.
The GitHub repository for ZigZag provides practical insights into how these parameters are configured in input files ZigZag GitHub.

Table: Parameters in ZigZag and Timeloop



Parameter
Description
Tool Example



Memory Hierarchy
Levels, sizes, bandwidths, data placement
ZigZag: 64 Byte to 2 MByte memories


Processing Elements
Number, arrangement, interconnections
ZigZag: Multi-dimensional MAC arrays


Mappings
Temporal/spatial scheduling of computations
ZigZag: Even/uneven scheduling


3. Impact of Parameter Changes on Performance Metrics
Effects of Parameter Variations
Research using ZigZag and Timeloop demonstrates that varying memory and PE arrangements significantly impacts performance metrics like latency, energy, and energy-delay product (EDP):

Memory Hierarchy: Different memory sizes and levels affect data access times and energy costs. Flexible hierarchies can reduce energy if overheads are managed ZigZag: A Memory-Centric Rapid DNN Accelerator.
PE Arrangements: The number and layout of PEs influence computation parallelism and throughput. Spatial unrolling can lead to up to 5× energy differences in non-shared memory setups.
Mappings: Scheduling strategies (e.g., even vs. uneven mappings) cause significant variations. ZigZag’s case studies show up to 4.7× energy variance and 8× throughput variance for AlexNet CONV2 on an Eyeriss-like architecture.

Validation and Correlations
To validate DSE results, tools compare estimates with hardware measurements or detailed simulations:

ZigZag reports a 5% error for Eyeriss and 7.5% for ENVISION accelerators, with 6% energy and 9% PE utilization errors against in-house RTL simulations.
These correlations allow researchers to trust the tools’ predictions and use them to guide design decisions.

Supporting Literature

Mei et al. (2020) provide case studies showing that ZigZag’s uneven mapping can yield up to 33% more energy-efficient solutions compared to tools like Timeloop+Accelergy ZigZag: A Memory-Centric Rapid DNN Accelerator.
Mei et al. (2021) further explore ZigZag’s capabilities, demonstrating improved design points through joint architecture-mapping exploration ZigZag: Enlarging Joint Architecture-Mapping.
Timeloop’s evaluation framework likely provides similar insights, though specific case studies are less detailed in the provided references Timeloop: A Systematic Approach.

Table: Impact of Parameters on Metrics



Parameter Changed
Metric Affected
Observed Impact
Source



Mapping (Scheduling)
Energy, Throughput
4.7× energy, 8× throughput variance
ZigZag: A Memory-Centric Rapid DNN Accelerator


Memory Hierarchy
Energy
30% energy cost variation
ZigZag: A Memory-Centric Rapid DNN Accelerator


Spatial Unrolling
Energy
Up to 5× energy difference
ZigZag: A Memory-Centric Rapid DNN Accelerator


4. Setting Bounds on Design Space Exploration
Techniques for Bounding DSE
The large size of design spaces—often millions or billions of possibilities—makes exhaustive exploration infeasible. Several techniques are used to bound or reduce the design space:

Heuristic Search: Algorithms like simulated annealing, genetic algorithms, or iterative searches focus on promising regions, reducing evaluation time. ZigZag’s heuristic search achieves a 2.5× speedup with no energy loss ZigZag: A Memory-Centric Rapid DNN Accelerator.
Machine Learning: Predictive models estimate design quality, guiding exploration to high-potential areas Design-Space Exploration - ScienceDirect.
Equivalence Classes: Grouping equivalent designs to avoid redundant evaluations, as proposed by Kang et al. using symmetry breaking predicates An Approach for Effective Design Space Exploration.
Constraints: Applying prior knowledge or requirements (e.g., area limits) to restrict the design space.

Research on Bounding Techniques

Kang et al. (2010) introduce a method where user-defined equivalence notions generate symmetry breaking predicates, ensuring only a subset of the design space is explored An Approach for Effective Design Space Exploration.
Mei et al. (2020) demonstrate ZigZag’s use of iterative search (7.5× speedup, 1.6% energy penalty) and spatial utilization thresholds (e.g., >0.75 MAC array utilization) to limit exploration ZigZag: A Memory-Centric Rapid DNN Accelerator.
Siemens Software discusses parametric exploration and topology optimization to focus on optimal shapes within constraints Design Space Exploration - Siemens.

Practical Application
In ZigZag, bounding is achieved by setting thresholds (e.g., spatial utilization) and using efficient search strategies. For example, evaluating 7,168 final schedules instead of 4,887,334 saves significant time while maintaining near-optimal results ZigZag GitHub.
Table: Bounding Techniques in DSE



Technique
Description
Example Application
Source



Heuristic Search
Uses algorithms to explore promising designs
ZigZag: 2.5× speedup, no energy loss
ZigZag: A Memory-Centric Rapid DNN Accelerator


Machine Learning
Predicts design quality to guide exploration
General DSE frameworks
Design-Space Exploration - ScienceDirect


Equivalence Classes
Groups equivalent designs to avoid redundancy
Symmetry breaking predicates
An Approach for Effective Design Space Exploration


Constraints
Limits design space with prior requirements
Area constraints in ZigZag
ZigZag GitHub


Conclusion
This survey provides a comprehensive overview of DSE in the context of hardware design, focusing on tools like ZigZag and Timeloop. It defines DSE and its structured steps, identifies key parameters and their impacts, and discusses methods to bound exploration. The literature highlights the importance of systematic approaches and efficient techniques to navigate large design spaces, ensuring optimal designs for deep learning accelerators.
Key Citations

An Approach for Effective Design Space Exploration
Design Space Exploration - Wikipedia
Design Space Exploration for Architectural Synthesis—A Survey
ZigZag: A Memory-Centric Rapid DNN Accelerator Design Space Exploration Framework
Timeloop: A Systematic Approach to DNN Accelerator Evaluation
ZigZag: Enlarging Joint Architecture-Mapping Design Space Exploration for DNN Accelerators
Design-Space Exploration - an overview | ScienceDirect Topics
ZigZag GitHub - HW Architecture-Mapping Design Space Exploration Framework
Design Space Exploration and Optimization | Siemens Software

Using this prilimiary research can you do a through research on each of the question and provide relevant citations for each papers referred to find the answer for each question and in the end summarise all the research done the explains what is the DSE, How tool like Zigzag, Timeloop etc.., make it possible, what are the parameters to be varied while performing the DSE and also where the effect can be seen on varing the parameters and finally any relevant bounds that can be set during the research.


## Answer 2:
Detailed response from Chatgpt and gemini - https://docs.google.com/document/d/1iZXi5Gd7QY453JYm8FoyeOZINfB2vVGWogsXykP3OBs/edit?usp=sharing



