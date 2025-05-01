#### Question 1:
Explaning the mapping constraint summary table and verify its points with the research from the papers

#### Answers 1:
### Key Points
- Research suggests the provided table accurately captures standard mappings for Output Stationary, Weight Stationary, and Input Stationary dataflows, with minor clarification needed for the Weight Stationary row.
- It seems likely that the Event Driven dataflow aligns with an input-stationary approach, where inputs trigger parallel computations across processing elements (PEs).
- The evidence leans toward the table being valid, supported by studies like Eyeriss and SCNN, though the Event Driven dataflow’s exact implementation may vary by architecture.
- There’s some controversy around the flexibility of mappings, as hardware constraints can alter optimal configurations.

### Verification and Corrections
The table summarizing mapping constraints for dataflows in CNN accelerators appears largely accurate, aligning with standard practices for Output Stationary, Weight Stationary, and Input Stationary dataflows. A minor correction is needed for the Weight Stationary row, where the notation “S/T(C)” for FX is likely a typo and should be “S/T(O)”. The Event Driven dataflow matches an input-stationary approach, consistent with research like the SCNN accelerator.

### Event-Driven Dataflow Explanation
In an event-driven dataflow, inputs (like a specific input channel and position) are broadcast to all PEs, which compute contributions to outputs in parallel. For example, when an input arrives, each PE might calculate results for different output channels or filter positions, completing all computations for that input before moving to the next. This approach, similar to input-stationary dataflows, optimizes for scenarios where inputs trigger immediate processing, as seen in the SCNN architecture ([SCNN Paper](https://arxiv.org/abs/1708.04485)).

### Supporting Research
- **Eyeriss** confirms Output Stationary mappings, with output channels and dimensions as outer loops or spatially mapped ([Eyeriss Paper](https://arxiv.org/abs/1606.01283)).
- **TPU** supports Weight Stationary, with weights kept stationary and inputs streamed ([TPU Paper](https://arxiv.org/abs/1704.04760)).
- **SCNN** validates the Event Driven and Input Stationary mappings, using an input-stationary dataflow where inputs are processed sequentially, and outputs are computed in parallel ([SCNN Paper](https://arxiv.org/abs/1708.04485)).
- **MAESTRO** provides a framework for modeling various dataflows, supporting the table’s flexibility ([MAESTRO Paper](https://arxiv.org/abs/1805.02566)).

---


# Mapping Constraint Summary for CNN Dataflows

This table summarizes how dimensions in convolutional neural network (CNN) accelerators are mapped to achieve specific dataflows: Output Stationary, Weight Stationary, Input Stationary, and Event Driven. Each dimension is assigned as Spatial (S), Temporal Outer loop (T(O)), or Temporal Inner loop (T(I)).

| Dataflow          | K       | C       | FX      | FY      | IX      | IY      | OX      | OY      |
|-------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| Output Stationary | S/T(O)  | T(I)    | T(I)    | T(I)    | T(I)    | T(I)    | S/T(O)  | S/T(O)  |
| Weight Stationary | S/T(O)  | S/T(O)  | S/T(O)  | S/T(O)  | T(I)    | T(I)    | T(I)    | T(I)    |
| Input Stationary  | S/T(I)  | S/T(O)  | T(I)    | T(I)    | S/T(O)  | S/T(O)  | T(I)    | T(I)    |
| Event Driven      | S/T(I)  | T(O)    | S/T(I)  | S/T(I)  | T(O)    | T(O)    | T(I)    | T(I)    |

## Legend
- **S**: Spatial ordering (parallelized across processing elements).
- **T(O)**: Temporal Outer loop (sequential, higher-level).
- **T(I)**: Temporal Inner loop (sequential, lower-level).

## Notes
- The table assumes a typical CNN accelerator with multiple processing elements (e.g., 8 PEs).
- The Event Driven dataflow is akin to Input Stationary, where inputs trigger parallel computations across PEs.
- For Weight Stationary, the original FX entry “S/T(C)” is corrected to “S/T(O)” for consistency with standard mappings.


### Comprehensive Analysis of Mapping Constraints in CNN Accelerators

This report provides a detailed examination of the provided mapping constraint table for convolutional neural network (CNN) accelerators, verifying its accuracy, correcting any errors with research-backed evidence, and explaining the event-driven dataflow concept with supporting citations. The analysis is structured as a professional survey note, ensuring depth and clarity for technical audiences, suitable for slide presentations where brevity and precision are key.

#### Background and Context
CNN accelerators optimize the execution of convolutional layers by mapping workload dimensions—output channels (K), input channels (C), filter dimensions (FX, FY), input dimensions (IX, IY), and output dimensions (OX, OY)—across spatial and temporal dimensions. Dataflow strategies like Output Stationary (OS), Weight Stationary (WS), Input Stationary (IS), and Event Driven determine how data (activations, weights, outputs) is reused to minimize memory access and maximize parallelism. The provided table summarizes these mappings, using notations like S (spatial), T(O) (temporal outer loop), and T(I) (temporal inner loop), to guide design space exploration in tools like ZigZag.

The table is intended to explain how dimensions are split to achieve specific dataflows, with a focus on the Event Driven dataflow, where inputs trigger immediate computations across processing elements (PEs). The verification process checks the table’s alignment with standard practices, and the event-driven explanation leverages research to validate its implementation.

#### Verification of the Table
The table is evaluated for accuracy against standard dataflow definitions and research literature. Each dataflow is analyzed, with corrections proposed where necessary.

##### Output Stationary
- **Table Mapping**: K: S/T(O), C: T(I), FX: T(I), FY: T(I), IX: T(I), IY: T(I), OX: S/T(O), OY: S/T(O).
- **Analysis**: In OS, output activations (y[k, oy, ox]) are kept stationary in PEs, accumulating partial sums over input channels and filter dimensions. K (output channels) and OX, OY (output spatial dimensions) are typically spatially mapped across PEs or processed as outer temporal loops, while C, FX, FY, IX, IY are inner loops for convolution calculations. This aligns with Eyeriss’s row stationary dataflow, a variant of OS, where output channels are parallelized across PEs ([Eyeriss Paper](https://arxiv.org/abs/1606.01283)).
- **Verdict**: Accurate, consistent with standard OS practices.

##### Weight Stationary
- **Table Mapping**: K: S/T(O), C: S/T(O), FX: S/T(C), FY: S/T(O), IX: T(I), IY: T(I), OX: T(I), OY: T(I).
- **Analysis**: In WS, weights (w[k, c, fy, fx]) are kept stationary, with K, C, FX, FY typically mapped spatially or as outer loops to minimize weight movement. IX, IY, OX, OY are inner loops, as inputs are streamed, and outputs are computed. The notation “S/T(C)” for FX is unclear and likely a typo, as it should be “S/T(O)” to indicate spatial mapping or outer loop, consistent with K, C, FY. This aligns with TPU’s WS dataflow, where weights are pre-loaded, and inputs are streamed ([TPU Paper](https://arxiv.org/abs/1704.04760)).
- **Correction**: Change FX from “S/T(C)” to “S/T(O)” for consistency with standard WS mappings.
- **Verdict**: Accurate with the correction of FX to “S/T(O)”.

##### Input Stationary
- **Table Mapping**: K: S/T(I), C: S/T(O), FX: T(I), FY: T(I), IX: S/T(O), IY: S/T(O), OX: T(I), OY: T(I).
- **Analysis**: In IS, input activations (x[c, iy, ix]) are kept stationary, with C, IX, IY mapped spatially or as outer loops to maximize input reuse. K, FX, FY are inner loops, as weights are applied to compute outputs, and OX, OY are derived from input and filter positions. This matches the SCNN accelerator’s input-stationary dataflow, where input activations are reused against multiple weights ([SCNN Paper](https://arxiv.org/abs/1708.04485)).
- **Verdict**: Accurate, consistent with IS practices.

##### Event Driven
- **Table Mapping**: K: S/T(I), C: T(O), FX: S/T(I), FY: S/T(I), IX: T(O), IY: T(O), OX: T(I), OY: T(I).
- **Analysis**: The Event Driven dataflow, as described by the user, involves each PE receiving the same input (e.g., x[c, iy, ix]), performing all computations for that input at once, and moving to the next input. This resembles an input-stationary dataflow, where inputs are processed sequentially (C, IX, IY as T(O)), and computations for output channels (K) and filter positions (FX, FY) are parallelized across PEs (S/T(I)). OX, OY as T(I) indicate that output positions are computed within inner loops, derived from input and filter positions (e.g., oy = iy - fy). This aligns with SCNN’s PT-IS-CP-sparse dataflow, where inputs are delivered to PEs for parallel computation, though SCNN focuses on sparse networks ([SCNN Paper](https://arxiv.org/abs/1708.04485)). MAESTRO’s flexible dataflow modeling also supports such configurations ([MAESTRO Paper](https://arxiv.org/abs/1805.02566)).
- **Verdict**: Accurate, resembling an input-stationary dataflow tailored for event-driven processing.

##### Correction Summary
- **Weight Stationary**: Change FX from “S/T(C)” to “S/T(O)” to align with standard WS mappings, as supported by TPU’s dataflow ([TPU Paper](https://arxiv.org/abs/1704.04760)).
- **Other Rows**: No corrections needed, as mappings align with research.

#### Event-Driven Dataflow Explanation
The Event Driven dataflow, as specified in the table, is a strategy where computations are triggered by the arrival of input data, with each input processed fully before moving to the next. In the context of CNN accelerators, this means:

- **Input Trigger**: An input value (x[c, iy, ix]) is broadcast to all PEs, which compute contributions to output activations (y[k, oy, ox]) for various output channels (k) and filter positions (fy, fx), where oy = iy - fy and ox = ix - fx (assuming no padding for simplicity).
- **Parallel Computation**: PEs are assigned to different K, FX, or FY, computing partial sums in parallel (S/T(I)), as indicated by the table’s K: S/T(I), FX: S/T(I), FY: S/T(I).
- **Sequential Input Processing**: Inputs are processed one at a time, with outer loops over C, IX, IY (T(O)), ensuring each input triggers a complete computation cycle.
- **Output Computation**: Output positions (OX, OY) are computed within inner loops (T(I)), derived from the input and filter positions, aligning with the convolution operation’s structure.

This approach is similar to input-stationary dataflows, where inputs are kept stationary or processed sequentially, and PEs compute outputs in parallel. The SCNN accelerator exemplifies this, using a PlanarTiled-InputStationary-CartesianProduct-sparse (PT-IS-CP-sparse) dataflow, where input activations are delivered to PEs, which compute partial sums for outputs using a multiplier array ([SCNN Paper](https://arxiv.org/abs/1708.04485)). While SCNN focuses on sparse networks, the core concept of input-triggered parallel computation applies to dense CNNs as well.

The MAESTRO framework further supports this by allowing flexible dataflow modeling, including input-stationary configurations, enabling exploration of such mappings for performance and energy efficiency ([MAESTRO Paper](https://arxiv.org/abs/1805.02566)). The COAC framework also discusses spatial unrolling across input channels, which could be adapted for event-driven setups, reinforcing the feasibility of parallelizing computations triggered by inputs ([COAC Paper](https://arxiv.org/abs/2406.13752)).

#### Supporting Research and Citations
The table’s mappings and the event-driven dataflow concept are validated by several authoritative sources:

- **Eyeriss ([Eyeriss Paper](https://arxiv.org/abs/1606.01283))**:
  - Describes row stationary, a variant of OS, where output channels (K) are spatially mapped, and output dimensions (OX, OY) are processed as outer loops, matching the table’s OS row.
  - Supports the flexibility of spatial mapping for K and OX, OY, applicable to event-driven setups if adapted.
- **TPU ([TPU Paper](https://arxiv.org/abs/1704.04760))**:
  - Uses a WS dataflow, with weights (K, C, FX, FY) kept stationary, aligning with the corrected WS row.
  - Demonstrates input streaming, which can be adapted for broadcasting in event-driven contexts.
- **SCNN ([SCNN Paper](https://arxiv.org/abs/1708.04485))**:
  - Employs an input-stationary dataflow (PT-IS-CP-sparse), where inputs are delivered to PEs for parallel computation, directly supporting the Event Driven and IS rows.
  - Achieves 2.7× performance and 2.3× energy improvements, validating the efficiency of input-triggered parallel processing.
- **MAESTRO ([MAESTRO Paper](https://arxiv.org/abs/1805.02566))**:
  - Provides a framework for modeling arbitrary dataflows, including IS, supporting the table’s flexibility and the event-driven concept.
  - Enables rapid design space exploration, confirming the validity of various mappings.
- **COAC ([COAC Paper](https://arxiv.org/abs/2406.13752))**:
  - Discusses spatial unrolling across input channels (C), which can be adapted for event-driven dataflows, supporting the parallel computation aspect of the Event Driven row.
  - Achieves up to 38% energy-delay-product savings, highlighting the benefits of flexible mappings.

#### Practical Considerations
- **Hardware Constraints**: With a limited number of PEs (e.g., 8), spatial mapping (S) for dimensions like K, FX, FY in the Event Driven dataflow requires efficient interconnects for partial sum accumulation, as noted in COAC’s discussion of reshuffling buffers.
- **Data Reuse**: The Event Driven dataflow maximizes input reuse by processing each input fully, but may increase weight access if K, FX, FY are spatially mapped, a trade-off discussed in MAESTRO.
- **Scalability**: For layers with large C, IX, IY, sequential processing (T(O)) may increase latency, but parallelizing K, FX, FY mitigates this, as supported by SCNN’s performance gains.

#### Comparison with Standard Dataflows
The Event Driven dataflow shares similarities with IS, as both prioritize input processing, but differs in its event-triggered nature, where inputs broadcast to PEs trigger immediate parallel computations. The table below compares the dataflows:

| Dataflow          | Stationary Data | Outer Loops (T(O)) | Spatial/Inner Loops (S/T(I)) | Research Support |
|-------------------|-----------------|---------------------|-----------------------------|------------------|
| Output Stationary | Outputs (K, OX, OY) | K, OX, OY | C, FX, FY, IX, IY | Eyeriss |
| Weight Stationary | Weights (K, C, FX, FY) | K, C, FX, FY | IX, IY, OX, OY | TPU |
| Input Stationary  | Inputs (C, IX, IY) | C, IX, IY | K, FX, FY, OX, OY | SCNN |
| Event Driven      | Inputs (C, IX, IY) | C, IX, IY | K, FX, FY, OX, OY | SCNN, MAESTRO |

This table highlights the Event Driven dataflow’s alignment with IS, with research backing its implementation.

#### Conclusion
The provided mapping constraint table is accurate for OS, IS, and Event Driven dataflows, with a minor correction needed for WS (FX: S/T(C) to S/T(O)). The Event Driven dataflow, resembling an input-stationary approach, involves inputs triggering parallel computations across PEs for output channels and filter positions, supported by SCNN’s dataflow and MAESTRO’s modeling capabilities. The table effectively summarizes how dimensions are split to achieve specific dataflows, making it a valuable tool for explaining mapping constraints in CNN accelerators, particularly for event-driven architectures.

### Key Citations
- [Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks](https://arxiv.org/abs/1606.01283)
- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)
- [SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks](https://arxiv.org/abs/1708.04485)
- [MAESTRO: An Open-source Infrastructure for Modeling Dataflows within Deep Learning Accelerators](https://arxiv.org/abs/1805.02566)
- [COAC: Cross-layer Optimization of Accelerator Configurability for Efficient CNN Processing](https://arxiv.org/abs/2406.13752)