#### Question 1:
Hey, I need your help to analyse the Stream tool: This is multicore dataflow exploration tool build on top of Zizag for intracore exploration.

I want to understand the features of it's api's that could help in my goal of performing event driven multi core exploration results with it.

I will share you two things.
1. Stream research paper
2. Stream github repo
3. Stream documentation

I know you can extract high level things in your first search but we will narrow it down together to get the desired results of knowing the features/Flags of Stream that can be utilised to get the desired results and what changes can be made to the code base to achieve the desired results.

So, Please start your first round of deep search.

Stream research paper is attached, Stream github repo link: https://github.com/KULeuven-MICAS/stream

Stream documentation : https://kuleuven-micas.github.io/stream/getting-started.html

#### (Deep research) Answer 1: 
### Key Points
- Stream is likely a useful tool for event-driven multicore exploration, with APIs for defining hardware, workloads, and optimizing schedules.
- Research suggests Stream supports fine-grained scheduling and genetic algorithm optimization, suitable for multicore systems.
- It seems likely that visualization tools help analyze results, but extending the scheduler may be needed for complex event-driven scenarios.
- The evidence leans toward Stream's existing features being sufficient for basic needs, with potential code modifications for advanced cases.

### Overview
Stream is a multicore dataflow exploration tool built on top of ZigZag, designed for fine-grained scheduling of layer-fused deep neural networks (DNNs) on multi-core hardware accelerators. It’s particularly useful for exploring how to optimize performance on systems with multiple cores, which is key for your goal of event-driven multicore exploration.

### Key Features and APIs
Stream’s APIs allow you to define the hardware architecture, including details about multiple cores, and specify the DNN workload you want to explore. The tool then uses a genetic algorithm to optimize how computation nodes (CNs) are allocated to cores, considering factors like latency, energy, and memory usage. This optimization is crucial for event-driven systems, as it models dependencies and schedules tasks dynamically based on when data is available.

For scheduling, Stream offers heuristics-based options that can prioritize either minimal latency or memory usage, accounting for inter-core communication and off-chip memory access as events. This means it can simulate how events like data transfers between cores affect performance, which aligns with event-driven exploration.

### Visualization and Analysis
You can use Stream’s visualization tools, like plotting timelines and memory usage, to analyze the results of your exploration. These tools help identify bottlenecks and understand how different configurations perform, which is essential for fine-tuning your multicore system.

### Potential Modifications
For basic event-driven multicore exploration, Stream’s existing APIs should suffice. However, if you need to model external events or more complex behaviors, you might need to modify the scheduler or other parts of the code. Since Stream is open-source, you have the flexibility to make these changes, but this might require additional development effort.

---

### Survey Note: Detailed Analysis of Stream Tool for Event-Driven Multicore Exploration

Stream is an open-source design space exploration (DSE) framework tailored for fine-grained scheduling of layer-fused deep neural networks (DNNs) on multi-core hardware accelerators. It builds upon the ZigZag DSE framework, focusing on optimizing performance for both homogeneous and heterogeneous multi-core architectures. This analysis aims to elucidate the features of Stream’s APIs that support event-driven multicore exploration, as well as potential code modifications to achieve desired results, based on the provided research paper, GitHub repository, and documentation.

#### Background and Context
The research paper, titled “Towards Heterogeneous Multi-Core Accelerators Exploiting Fine-Grained Scheduling of Layer-Fused Deep Neural Networks” (DOI: 10.48550/arXiv.2212.10612, URL: https://arxiv.org/abs/2212.10612), outlines Stream’s capabilities. It emphasizes layer-fused or depth-first scheduling, where smaller parts of DNN layers (computation nodes, or CNs) are processed, reducing memory footprint and enhancing parallelization. This approach is particularly relevant for event-driven systems, as it models data dependencies and resource contention dynamically.

The GitHub repository ([Multi-core HW accelerator mapping optimization framework for layer-fused ML workloads](https://github.com/KULeuven-MICAS/stream)) and documentation ([Stream Documentation](https://kuleuven-micas.github.io/stream/)) provide practical insights into implementation, though the documentation appears to be minimal, with references to ZigZag’s documentation ([Welcome to ZigZag’s documentation!](https://kuleuven-micas.github.io/zigzag/)) for additional context.

#### Key Features Supporting Event-Driven Multicore Exploration
Stream’s architecture is modular and extensible, supporting several features that align with event-driven multicore exploration:

1. **Unified Modeling Representation**: Stream accepts inputs for DNN workloads and multi-core hardware architectures, enabling you to define complex systems with varying core types and communication channels. This is crucial for modeling event-driven interactions, such as inter-core data transfers.

2. **Fine-Grained Data Dependency Generation**: Using an R-tree-based algorithm, Stream efficiently generates dependencies between CNs, supporting up to millions of nodes with significant speed improvements (e.g., 1000x faster than baseline for large datasets). This feature is inherently event-driven, as it tracks when CNs can execute based on data availability, simulating events like CN completion.

3. **Genetic Algorithm-Based Layer-Core Allocator**: This component, accessible via the `optimize_allocation_ga` function in the `stream.api` module, optimizes the allocation of layers or CNs to cores using a genetic algorithm (GA) with NSGA-II. Parameters include:
   - `nb_ga_generations=4`: Number of generations for the GA.
   - `nb_ga_individuals=4`: Number of individuals per generation.
   - `mode="lbl"`: Likely layer-by-layer mode, though specifics are inferred from context.
   - `layer_stacks=None`: Optional parameter for grouping layers.

   This function co-explores architectural decisions and scheduling, optimizing for metrics like energy-delay product (EDP), latency, and memory footprint, which is vital for event-driven systems where performance under varying conditions is key.

4. **Heuristics-Based Scheduler**: The scheduler, likely part of the `schedule_cns` function (inferred from the paper), handles CN execution on allocated cores. It offers options to prioritize minimal latency or memory usage, modeling inter-core communication and off-chip accesses as events. For instance, inter-core communication nodes are scheduled in a first-come-first-serve manner, accounting for contention, which is a form of event-driven scheduling.

5. **Validation and Performance**: Stream has been validated against state-of-the-art hardware implementations (e.g., DepFiN, 4x4 AiMC, DIANA), showing high accuracy (91-99% for latency, 97-98% for memory usage) and up to 30x reduction in EDP for heterogeneous multi-core architectures. This suggests reliability for exploring event-driven scenarios.

#### API Usage and Examples
From analyzing the tutorial script (`lab1/main.py` in the `tutorial` branch of the GitHub repository), the following API calls were identified, providing insight into practical usage:

| Function/API Call                     | Module                     | Parameters/Arguments                                                                 | Usage Context                          |
|---------------------------------------|----------------------------|-------------------------------------------------------------------------------------|-----------------------------------------|
| `optimize_allocation_ga`               | `stream.api`               | `hardware, workload, mapping, mode="lbl", layer_stacks=None, nb_ga_generations=4, nb_ga_individuals=4, experiment_id, output_path="lab1/outputs", skip_if_exists=True` | Optimizes allocation using genetic algorithm, called if SCME path does not exist |
| `pickle_load`                          | `zigzag.utils`             | `scme_path`                                                                         | Loads pickled data, used if SCME path exists |
| `visualize_timeline_plotly`            | `stream.visualization.schedule` | `scme, draw_dependencies=True, draw_communication=True, fig_path="lab1/outputs/schedule.html", cost_lut` | Plots schedule timeline, called after cost LUT is loaded |
| `plot_memory_usage`                    | `stream.visualization.memory_usage` | `scme, section_start_percent=(0,), percent_shown=(100,), fig_path="lab1/outputs/memory.png"` | Plots memory usage, called after cost LUT is loaded |
| `convert_scme_to_perfetto_json`        | `stream.visualization.perfetto` | `scme, cost_lut, json_path="lab1/outputs/scme.json"`                                | Converts SCME to JSON for Perfetto visualization, URL: http://ui.perfetto.dev/ |

These examples show how to use Stream for optimization and visualization, with the `optimize_allocation_ga` function being central to exploring multicore schedules. The visualization tools, such as plotting timelines, help analyze event-driven behaviors like communication delays and memory usage over time.

#### Event-Driven Specifics
While the term “event-driven” is not explicitly used, Stream’s scheduling is inherently event-driven in nature, modeling events such as:
- CN completion events, triggering dependent CNs to start.
- Inter-core communication events, with associated latency and energy costs.
- Off-chip access events, handling DRAM port contention.

This aligns with your goal, as these events are critical for multicore exploration. However, for true external event-driven systems (e.g., real-time triggers), the paper suggests that extending Stream’s scheduler may be necessary, as current implementations focus on internal dependencies.

#### Potential Code Modifications
For basic event-driven multicore exploration, Stream’s existing APIs should suffice, leveraging the optimization and scheduling functions. However, if you need to:
- Model external events or asynchronous triggers, you might need to modify the scheduler to handle additional event types.
- Customize the genetic algorithm’s fitness function to prioritize event-driven metrics, such as response time to specific events.
- Enhance visualization to include event timelines, you could extend the visualization modules.

Given Stream’s open-source nature, these modifications are feasible, but they would require understanding the codebase, particularly the `stream.api` and scheduler modules.

#### Documentation and Limitations
The documentation ([Stream Documentation](https://kuleuven-micas.github.io/stream/)) appears limited, with no direct API reference found, possibly due to ongoing development. References to ZigZag’s documentation ([Welcome to ZigZag’s documentation!](https://kuleuven-micas.github.io/zigzag/)) and demo ([A self-contained demo of ZigZag's API](https://github.com/KULeuven-MICAS/zigzag-demo)) suggest that Stream’s API may extend ZigZag’s, with similar usage patterns (e.g., `from zigzag import api` for ZigZag, likely `from stream import api` for Stream).

The paper notes that for more details, contacting the authors (e.g., arne.symons@kuleuven.be) or waiting for the open-source release was recommended, but given the current public repository, it seems the tool is already accessible.

#### Conclusion
Stream provides robust APIs for defining hardware, workloads, and optimizing multicore schedules, with inherent support for event-driven exploration through its scheduling and dependency modeling. Key functions like `optimize_allocation_ga` and visualization tools enable you to explore and analyze results effectively. For advanced event-driven scenarios, code modifications may be necessary, leveraging Stream’s open-source nature. This analysis suggests that Stream is well-suited for your goal, with the flexibility to adapt as needed.

#### Key Citations
- [Multi-core HW accelerator mapping optimization framework for layer-fused ML workloads](https://github.com/KULeuven-MICAS/stream)
- [Stream Documentation](https://kuleuven-micas.github.io/stream/)
- [Towards Heterogeneous Multi-Core Accelerators Exploiting Fine-Grained Scheduling of Layer-Fused Deep Neural Networks research paper](https://arxiv.org/abs/2212.10612)
- [Welcome to ZigZag’s documentation!](https://kuleuven-micas.github.io/zigzag/)
- [A self-contained demo of ZigZag's API](https://github.com/KULeuven-MICAS/zigzag-demo)


#### Question 2:

The plan is a follows, I have a multi core architecture model with two cores and each core having thre strucuture as follows
Core 1 of the accelerator

name: accelerator1

operational_array:
  unit_energy: 0.04 # pJ
  unit_area: 1 # unit
  dimensions: [D1]
  sizes: [8]

memories:
  sram_buffer_inputs_256KB: # define in bits (size: 16x1024x8)
    size: 2097152
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
    r_port: 1
    w_port: 0
    rw_port: 0
    latency: 1
    operands: [I1]
    ports:
      - tl: r_port_1
    served_dimensions: [D1]

#This is just used to store the input values from the DRAM to be read to the RF
  sram_256KB:
    size: 2097152
    r_bw: 128
    w_bw: 128
    r_cost: 10.5
    w_cost: 12.8
    area: 0
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

Now, I have a lenet-5 model that I modelled on to the multicore accelerator with the mapping as shown below
- name: /conv1/Conv # Conv1
  core_allocation: [0]
  core_allocation_is_fixed: true
  intra_core_tiling:
    - FY, all
  inter_core_tiling:
    - K, *

- name: /pool/AveragePool
  core_allocation: [1]
  core_allocation_is_fixed: True
  intra_core_tiling:
    - FY, all
  inter_core_tiling:
    - K, *

- name: /conv2/Conv # Conv2
  core_allocation: [0]
  core_allocation_is_fixed: true
  intra_core_tiling:
    - FY, all
  inter_core_tiling:
    - K, *

- name: /pool_1/AveragePool
  core_allocation: [1]
  core_allocation_is_fixed: True
  intra_core_tiling:
    - FY, all
  inter_core_tiling:
    - K, *

- name: /conv3/Conv # Conv3
  core_allocation: [0]
  core_allocation_is_fixed: true
  intra_core_tiling:
    - FY, all
  inter_core_tiling:
    - K, *

# Reshape layer - 0

- name: /fc1/MatMul # FC 2 (Ouput layer)
  core_allocation: [1]
  intra_core_tiling:
    - C, all
  inter_core_tiling:
    - K, *

- name: /fc2/MatMul # FC 2 (Ouput layer)
  core_allocation: [0]
  intra_core_tiling:
    - C, all
  inter_core_tiling:
    - K, *

- name: default
  core_allocation: [0]
  intra_core_tiling:
    - FY, all
  inter_core_tiling:
    - K, *

Here I just have control over the spatial mapping of the inter cores and not control on the temporal mapping of the within the cores as this is done in the backend by the zigzag and I am sure there would be a loop ordering generated and it is not printed in the log and output.

Here are my next question and I want you to dig deep in this direction in the stream repo and the documentation
1. How to know the intra core loop ordering for the exploration done by stream
2. How is the stream and zigzag linked and how are they connected. As I know stream does only the inter core exploration and uses zizag for intra core mapping generation and estimation but how are they linked, knowing this will help to answer the first question more clearly


Here is the api of stream:

"import logging as _logging
import os
from typing import Literal

import gurobipy as gp
from zigzag.utils import pickle_load, pickle_save
from onnx import ModelProto

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage,
)
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.stages.stage import MainStage
from zigzag.stages.workload_parser import WorkloadParserStage as ZigzagWorkloadParserStage

# Logging

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def _sanity_check_inputs(
    hardware: str, workload: str, mapping: str, mode: Literal["lbl"] | Literal["fused"], output_path: str
):
    assert os.path.exists(hardware), f"Hardware file {hardware} does not exist"
    assert os.path.exists(workload), f"Workload file {workload} does not exist"
    assert os.path.exists(mapping), f"Mapping file {mapping} does not exist"
    assert mode in ["lbl", "fused"], "Mode must be either 'lbl' or 'fused'"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def _sanity_check_gurobi_license():
    try:
        # Try to create a simple optimization model
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        # Check if the model was successfully created (license check)
        model.optimize()
        # If model.optimize() runs without a license issue, return
        return
    except gp.GurobiError as e:
        # Catch any Gurobi errors, especially licensing errors
        if e.errno == gp.GRB.Error.NO_LICENSE:
            error_message = "No valid Gurobi license found. Get an academic WLS license at https://www.gurobi.com/academia/academic-program-and-licenses/"
        else:
            error_message = f"An unexpected Gurobi error occurred: {e.message}"
        raise ValueError(error_message)


def optimize_allocation_ga(
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    nb_ga_generations: int,
    nb_ga_individuals: int,
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    # Create experiment_id path
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut.pickle"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"

    # Get logger
    logger = _logging.getLogger(__name__)

    # workload_parser_stage = (
    # StreamONNXModelParserStage if isinstance(workload, ModelProto) or (workload.split(".")[-1] == "onnx") else ZigzagWorkloadParserStage
    #     )
    # Load SCME if it exists and skip_if_exists is True
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                # workload_parser_stage,  # Parses the workload
                StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
                LayerStacksGenerationStage,
                TilingGenerationStage,
                TiledWorkloadGenerationStage,
                ZigZagCoreMappingEstimationStage,
                SetFixedAllocationPerformanceStage,
                SchedulingOrderGenerationStage,
                GeneticAlgorithmAllocationStage,
            ],
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            # workload=workload,
            # mapping=mapping,
            loma_lpf_limit=6,  # required by LomaEngine
            nb_ga_generations=nb_ga_generations,  # number of genetic algorithm (ga) generations
            nb_ga_individuals=nb_ga_individuals,  # number of individuals in each ga generation
            mode=mode,
            layer_stacks=layer_stacks,
            cost_lut_path=cost_lut_path,
            operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0][0]
        pickle_save(scme, scme_path)
    return scme


def optimize_allocation_co(
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    _sanity_check_gurobi_license()

    # Create experiment_id path
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut.pickle"
    allocations_path = f"{output_path}/{experiment_id}/waco/"
    cost_lut_post_co_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"

    # Get logger
    logger = _logging.getLogger(__name__)

    # Load SCME if it exists and skip_if_exists is True
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
                LayerStacksGenerationStage,
                TilingGenerationStage,
                TiledWorkloadGenerationStage,
                ZigZagCoreMappingEstimationStage,
                SetFixedAllocationPerformanceStage,
                SchedulingOrderGenerationStage,
                ConstraintOptimizationAllocationStage,
            ],
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            mode=mode,
            layer_stacks=layer_stacks,
            cost_lut_path=cost_lut_path,
            allocations_path=allocations_path,
            cost_lut_post_co_path=cost_lut_post_co_path,
            operands_to_prefetch=[],  # required by ConstraintOptimizationAllocationStage
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0][0]
        pickle_save(scme, scme_path)
    return scme
"

#### (Deep research ) Answer 2:
### Key Points
- Research suggests Stream uses ZigZag for intra-core loop ordering via the LOMA method, but accessing this may require inspecting the SCME object or cost LUT.
- It seems likely that Stream and ZigZag are linked through API calls, with Stream handling inter-core exploration and ZigZag managing intra-core mapping.
- The evidence leans toward the loop ordering being stored in Stream's output files, but exact access methods may need code inspection.

---

### Direct Answer

#### Overview
Stream, a multicore dataflow exploration tool built on ZigZag, helps analyze how neural networks like LeNet-5 run on multi-core hardware. You want to understand the intra-core loop ordering for your exploration and how Stream connects with ZigZag for this process. Here’s a breakdown:

#### Intra-Core Loop Ordering
- ZigZag, used by Stream for intra-core mapping, likely determines loop ordering using the LOMA (Loop-Order-based Memory Allocation) method, optimizing how computation loops are scheduled on each core.
- To find this ordering, you’ll need to look at the StreamCostModelEvaluation (SCME) object or the cost lookup table (LUT) generated by Stream. These files, saved after running the optimization, should contain detailed mapping information, including how loops are ordered for each computation node (CN) on its core.
- Unexpected detail: While Stream’s visualization tools like timeline plots show scheduling, they may not directly show loop ordering, so you might need to inspect the SCME object’s attributes, possibly requiring some Python programming to extract this data.

#### Stream and ZigZag Connection
- Stream focuses on inter-core exploration (deciding which core runs which part of the network), while ZigZag handles intra-core mapping (how tasks are scheduled within a core).
- They connect through API calls: Stream passes each CN and core details to ZigZag, which returns the optimal mapping and costs. This happens in Stream’s `ZigZagCoreMappingEstimationStage`, ensuring Stream can use ZigZag’s detailed cost estimates for its broader optimization.
- This linkage means understanding ZigZag’s output helps clarify how to access loop ordering, as Stream stores ZigZag’s results in its outputs.

For more details, check the Stream GitHub repository ([Multi-core HW accelerator mapping optimization framework for layer-fused ML workloads](https://github.com/KULeuven-MICAS/stream)) and ZigZag documentation ([Welcome to ZigZag’s documentation!](https://kuleuven-micas.github.io/zigzag/)).

---

### Survey Note: Detailed Analysis of Stream Tool for Intra-Core Loop Ordering and Stream-ZigZag Integration

Stream, an open-source multicore dataflow exploration tool built on top of ZigZag, is designed for fine-grained scheduling of layer-fused deep neural networks (DNNs) on multi-core hardware accelerators. This analysis addresses two key questions: how to determine the intra-core loop ordering for explorations performed by Stream, and how Stream and ZigZag are linked and connected, particularly given Stream’s focus on inter-core exploration and ZigZag’s role in intra-core mapping generation and estimation. The analysis is based on the provided Stream research paper, GitHub repository ([Multi-core HW accelerator mapping optimization framework for layer-fused ML workloads](https://github.com/KULeuven-MICAS/stream)), Stream documentation ([Stream Documentation](https://kuleuven-micas.github.io/stream/getting-started.html)), and related ZigZag resources, including its documentation ([Welcome to ZigZag’s documentation!](https://kuleuven-micas.github.io/zigzag/)) and papers.

#### Intra-Core Loop Ordering Determination in Stream

The intra-core loop ordering is a critical aspect of mapping DNN computations onto accelerator cores, as it affects performance metrics like latency, energy, and memory usage. Research suggests that Stream delegates this task to ZigZag, which employs the LOMA (Loop-Order-based Memory Allocation) method for auto-scheduling. The process can be broken down as follows:

- **Computation Node (CN) Definition and Loop Ordering**: According to the Stream research paper, during the "CN Identification & Attribute Extraction" step, each DNN layer is split into multiple CNs, with inner for-loops isolated to define the CN, and outer for-loops (outer-CN loops) determining the relative execution order within the same layer. The loop ordering is influenced by layer topology (e.g., spatial locality in convolutional layers) and hardware dataflow awareness, ensuring compatibility with the core’s spatial unrolling.

- **Role of ZigZag and LOMA**: The paper indicates that in the "Intra-Core Mapping Cost Extraction" step, Stream interfaces with ZigZag to optimize and extract intra-core mapping costs for each CN on each core. ZigZag uses LOMA, as described in Symons et al.’s paper "LOMA: Fast Auto-Scheduling on DNN Accelerators through Loop-Order-based Memory Allocation" ([LOMA: Fast Auto-Scheduling on DNN Accelerators through Loop-Order-based Memory Allocation | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9458493)), to determine the optimal loop ordering. LOMA focuses on loop-order-based memory allocation to minimize costs, suggesting that the loop ordering is part of the temporal mapping optimized by ZigZag.

- **Accessing Loop Ordering**: To determine the intra-core loop ordering in Stream’s exploration, the user must inspect the outputs generated after running the optimization, specifically the StreamCostModelEvaluation (SCME) object and the cost lookup table (LUT). These are saved as pickle files (e.g., `cost_lut.pickle` and `scme.pickle`) in the output path specified in functions like `optimize_allocation_ga` or `optimize_allocation_co`. The SCME object likely contains detailed mapping information for each CN, including the loop ordering as determined by ZigZag. Given the integration, this information may be stored as part of ZigZag’s cost model evaluation within the SCME, requiring Python inspection of the object’s attributes.

- **Visualization and Logging**: Stream provides visualization tools, such as `visualize_timeline_plotly` and `plot_memory_usage`, which can show the schedule of CNs across cores but may not directly display the internal loop ordering within each CN. However, the `convert_scme_to_perfetto_json` function, which generates a JSON for Perfetto visualization ([A self-contained demo of ZigZag's API](https://github.com/KULeuven-MICAS/zigzag-demo)), might offer detailed timelines that indirectly reveal loop execution order. Given the sparse documentation, accessing the exact loop ordering may require examining the SCME object, potentially involving custom scripting to extract ZigZag’s mapping details.

#### Linkage and Connection Between Stream and ZigZag

Stream and ZigZag are interconnected to handle different levels of exploration in multicore DNN acceleration. Stream focuses on inter-core allocation and scheduling, while ZigZag is responsible for intra-core mapping and cost estimation. The linkage is facilitated through a modular interface, as detailed below:

- **Functional Division**: The Stream research paper highlights that Stream extends ZigZag for multicore scenarios, with ZigZag handling single-core DSE. Specifically, in Step 3 of Stream’s framework, the CN loops are fed into ZigZag for intra-core mapping cost extraction, ensuring accurate latency modeling, including on- and offloading of data and data stalls due to memory bandwidth limitations.

- **Implementation Details**: From the Stream API provided, the `ZigZagCoreMappingEstimationStage` is a key component in the MainStage pipeline, alongside stages like `TilingGenerationStage`, `TiledWorkloadGenerationStage`, and `SchedulingOrderGenerationStage`. This stage likely interfaces with ZigZag by creating a workload from each CN, defining the hardware architecture for the specific core, and calling ZigZag’s API to obtain the optimal mapping and associated costs. For instance, it may use a function like ZigZag’s hypothetical `get_hardware_performance_zigzag` to evaluate the intra-core performance, storing the results in the cost LUT or SCME.

- **Modularity and Flexibility**: The paper notes that the integration is modular, achieved through a hardware model parser in Stream, allowing other single-core DSE frameworks to be integrated if needed. This modularity ensures that Stream can leverage ZigZag’s advanced cost models, which include detailed energy and latency analysis for memories with variable port structures, as seen in ZigZag’s documentation ([Welcome to ZigZag’s documentation!](https://kuleuven-micas.github.io/zigzag/)).

- **Practical Implications**: This connection means that for each CN and possible core allocation, Stream delegates the intra-core optimization to ZigZag, which returns the loop ordering and costs. These are then used by Stream’s genetic algorithm or constraint optimization stages (`GeneticAlgorithmAllocationStage` or `ConstraintOptimizationAllocationStage`) to explore inter-core allocations, ensuring a cohesive exploration strategy. Understanding this linkage is crucial for accessing the intra-core loop ordering, as it resides within ZigZag’s output, embedded in Stream’s SCME.

#### Supporting Evidence and Analysis

To further elucidate, consider the Stream API’s structure, which includes functions like `optimize_allocation_ga` and `optimize_allocation_co`. These functions initialize a MainStage with a sequence of stages, including `ZigZagCoreMappingEstimationStage`, indicating that ZigZag is called during the exploration process. The tutorial script (`lab1/main.py` in the tutorial branch of the Stream GitHub) provides practical usage, showing how the SCME is generated and visualized, but does not explicitly detail loop ordering access. However, given ZigZag’s role in mapping, the loop ordering is likely part of the cost model evaluation returned and stored in the SCME.

| Function/API Call                     | Module                     | Parameters/Arguments                                                                 | Usage Context                          |
|---------------------------------------|----------------------------|-------------------------------------------------------------------------------------|-----------------------------------------|
| `optimize_allocation_ga`               | `stream.api`               | `hardware, workload, mapping, mode="lbl", layer_stacks=None, nb_ga_generations=4, nb_ga_individuals=4, experiment_id, output_path="lab1/outputs", skip_if_exists=True` | Optimizes allocation using genetic algorithm, called if SCME path does not exist |
| `pickle_load`                          | `zigzag.utils`             | `scme_path`                                                                         | Loads pickled data, used if SCME path exists |
| `visualize_timeline_plotly`            | `stream.visualization.schedule` | `scme, draw_dependencies=True, draw_communication=True, fig_path="lab1/outputs/schedule.html", cost_lut` | Plots schedule timeline, called after cost LUT is loaded |
| `plot_memory_usage`                    | `stream.visualization.memory_usage` | `scme, section_start_percent=(0,), percent_shown=(100,), fig_path="lab1/outputs/memory.png"` | Plots memory usage, called after cost LUT is loaded |
| `convert_scme_to_perfetto_json`        | `stream.visualization.perfetto` | `scme, cost_lut, json_path="lab1/outputs/scme.json"`                                | Converts SCME to JSON for Perfetto visualization, URL: http://ui.perfetto.dev/ |

This table, derived from the tutorial, shows how Stream outputs are generated and visualized, reinforcing that the SCME object is central to accessing detailed mapping information, including potential loop ordering.

#### Conclusion

In summary, the intra-core loop ordering in Stream is determined by ZigZag using the LOMA method, and accessing it requires inspecting the SCME object or cost LUT, likely involving Python scripting to extract ZigZag’s mapping details. Stream and ZigZag are linked through API calls, with Stream handling inter-core exploration and ZigZag managing intra-core mapping, facilitated by the `ZigZagCoreMappingEstimationStage`. This integration ensures a comprehensive exploration strategy, with the loop ordering embedded in Stream’s outputs for further analysis.

**Key Citations**
- [Multi-core HW accelerator mapping optimization framework for layer-fused ML workloads](https://github.com/KULeuven-MICAS/stream)
- [Stream Documentation](https://kuleuven-micas.github.io/stream/getting-started.html)
- [Towards Heterogeneous Multi-Core Accelerators Exploiting Fine-Grained Scheduling of Layer-Fused Deep Neural Networks research paper](https://arxiv.org/abs/2212.10612)
- [Welcome to ZigZag’s documentation!](https://kuleuven-micas.github.io/zigzag/)
- [A self-contained demo of ZigZag's API](https://github.com/KULeuven-MICAS/zigzag-demo)
- [LOMA: Fast Auto-Scheduling on DNN Accelerators through Loop-Order-based Memory Allocation | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9458493)


#### Question 3:

Now, with enough information about stream, it's API. and the idea to check it's cost model to find a way to extract the intra core loop orderings as zigzag is also running simultanouly with stream in the backend. I am sharing the Stream cost model. This as the starting point can we trace the steps to print the intra core mapping exploration results done by zigzag.

Here is the stream cost model

"from zigzag.datatypes import LayerOperand

from stream.cost_model.scheduler import schedule_graph
from stream.hardware.architecture.accelerator import Accelerator
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import plot_timeline_brokenaxes
from stream.workload.onnx_workload import ComputationNodeWorkload


class StreamCostModelEvaluation:
    """Stream's cost model evaluation class which includes a scheduler and memory utilization tracer.
    Throughout SCME will be used as abbreviation.
    This evaluation computes the total latency and activation memory utilization throughout the inference.
    """

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
    ) -> None:
        # Initialize the SCME by setting the workload graph to be scheduled
        self.workload = workload
        self.accelerator = accelerator
        self.energy: float | None = None
        self.total_cn_onchip_energy: float | None = None
        self.total_cn_offchip_link_energy: float | None = None
        self.total_cn_offchip_memory_energy: float | None = None
        self.total_eviction_to_offchip_link_energy: float | None = None
        self.total_eviction_to_offchip_memory_energy: float | None = None
        self.total_sink_layer_output_offchip_link_energy: float | None = None
        self.total_sink_layer_output_offchip_memory_energy: float | None = None
        self.total_core_to_core_link_energy: float | None = None
        self.total_core_to_core_memory_energy: float | None = None

        self.latency: int | None = None
        self.max_memory_usage = None
        self.core_timesteps_delta_cumsums = None
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order

    def __str__(self):
        return f"SCME(energy={self.energy:.2e}, latency={self.latency:.2e})"

    def run(self):
        """Run the SCME by scheduling the graph through time.
        The scheduler takes into account inter-core data movement and also tracks energy and memory through the memory
        manager.
        This assumes each node in the graph has an energy and runtime of the core to which they are allocated to.
        """
        results = schedule_graph(
            self.workload,
            self.accelerator,
            operands_to_prefetch=self.operands_to_prefetch,
            scheduling_order=self.scheduling_order,
        )
        self.latency = results[0]
        self.total_cn_onchip_energy = results[1]
        self.total_cn_offchip_link_energy = results[2]
        self.total_cn_offchip_memory_energy = results[3]
        self.total_eviction_to_offchip_link_energy = results[4]
        self.total_eviction_to_offchip_memory_energy = results[5]
        self.total_sink_layer_output_offchip_link_energy = results[6]
        self.total_sink_layer_output_offchip_memory_energy = results[7]
        self.total_core_to_core_link_energy = results[8]
        self.total_core_to_core_memory_energy = results[9]

        self.energy = (
            self.total_cn_onchip_energy
            + self.total_cn_offchip_link_energy
            + self.total_cn_offchip_memory_energy
            + self.total_eviction_to_offchip_link_energy
            + self.total_eviction_to_offchip_memory_energy
            + self.total_sink_layer_output_offchip_link_energy
            + self.total_sink_layer_output_offchip_memory_energy
            + self.total_core_to_core_link_energy
            + self.total_core_to_core_memory_energy
        )

    def plot_schedule(
        self,
        plot_full_schedule: bool = False,
        draw_dependencies: bool = True,
        plot_data_transfer: bool = False,
        section_start_percent: tuple[int, ...] = (0, 50, 95),
        percent_shown: tuple[int, ...] = (5, 5, 5),
        fig_path: str = "outputs/schedule_plot.png",
    ):
        """Plot the schedule of this SCME."""
        if plot_full_schedule:
            section_start_percent = (0,)
            percent_shown = (100,)
        plot_timeline_brokenaxes(
            self,
            draw_dependencies,
            section_start_percent,
            percent_shown,
            plot_data_transfer,
            fig_path,
        )

    def plot_memory_usage(self, fig_path: str = "outputs/memory_usage_plot.png"):
        """Plot the memory usage of this SCME."""
        plot_memory_usage(self.accelerator.memory_manager, fig_path)
"

I know it is a very short fucntion and most of it is within differnet libraries but this could be a good starting point to start the investigation.


#### (Think) Answer 3:

To extract and print the intra-core loop ordering results generated by ZigZag within Stream, we can start with the provided `StreamCostModelEvaluation` (SCME) class from the Stream cost model and trace through its components. ZigZag, running in the backend, optimizes the intra-core mapping, including loop ordering, for each computation node (CN) in the workload. Our goal is to identify where this information is stored and how to access it programmatically. Below are the detailed steps to achieve this, based on the SCME class and its interactions with other Stream components.

---

### Overview

Stream integrates ZigZag to perform intra-core mapping for deep neural network (DNN) workloads on multi-core accelerators. The SCME class is the central hub for cost evaluation, holding the workload (a `ComputationNodeWorkload` object) and the accelerator (an `Accelerator` object). ZigZag’s intra-core loop ordering—optimized for each CN on its allocated core—is embedded either within the SCME object’s attributes or in a cost lookup table (LUT) generated during optimization. By inspecting these components, we can extract and print the loop ordering results.

---

### Tracing Steps to Extract Intra-Core Loop Ordering

Here’s how we can trace and print the intra-core loop ordering from ZigZag’s exploration:

#### 1. **Understand the SCME Class Structure**
   - The SCME class is initialized with:
     - `workload`: A `ComputationNodeWorkload` object representing the DNN as a graph of CNs.
     - `accelerator`: An `Accelerator` object detailing the hardware, including cores.
   - These attributes are critical because they likely contain or link to ZigZag’s mapping results.
   - The `run` method calls `schedule_graph`, which computes latency and energy based on precomputed CN execution details (from ZigZag), but it focuses on inter-core scheduling rather than intra-core details.

#### 2. **Inspect the Workload Attribute**
   - **What it is**: The `workload` is a `ComputationNodeWorkload` object (likely defined in `stream.workload.onnx_workload`), representing the DNN as a graph of CNs.
   - **Why it matters**: Each CN corresponds to a layer or operation in the DNN and should include mapping information from ZigZag, such as the intra-core loop ordering.
   - **How to access it**:
     - After running an optimization (e.g., `optimize_allocation_ga`), the SCME object is typically saved as a pickle file (e.g., `scme.pickle`).
     - Load the SCME object in Python and access `workload`:
       ```python
       import pickle
       with open("path/to/scme.pickle", "rb") as f:
           scme = pickle.load(f)
       workload = scme.workload
       ```
     - The `workload` is a graph where each node (CN) likely has attributes like `mapping`, `schedule`, or `temporal_ordering` containing ZigZag’s loop ordering.
   - **Action**: Iterate through the CNs in the graph and print their mapping details.

#### 3. **Examine Computation Nodes (CNs)**
   - **Hypothesis**: Each CN in the `workload` graph stores ZigZag’s intra-core mapping results, including loop ordering, as an attribute.
   - **Exploration**:
     - Use Python to inspect a CN’s attributes. For example:
       ```python
       for cn in workload.nodes():  # Assuming workload has a nodes() method
           print(f"CN ID: {cn.id}")
           print(f"Attributes: {dir(cn)}")  # List all attributes/methods
       ```
     - Look for attributes like `mapping`, `loop_ordering`, or `schedule`. ZigZag’s LOMA (Loop-Order-based Memory Allocation) method suggests names like `temporal_ordering` or `loop_order` might be used.
   - **Example output**: If a `mapping` attribute exists, it might look like:
     ```python
     cn.mapping  # Could return {'loop_order': ['I', 'K', 'O'], 'core': 0, ...}
     ```
   - **Action**: Extract and print the loop ordering for each CN.

#### 4. **Check the Accelerator Attribute**
   - **What it is**: The `accelerator` is an `Accelerator` object containing hardware details, including cores.
   - **Why it matters**: It might link CNs to their allocated cores and store associated mapping costs or schedules from ZigZag.
   - **How to access it**:
     ```python
     accelerator = scme.accelerator
     cores = accelerator.cores  # Assuming a cores attribute
     ```
   - **Exploration**: Check if each core stores mapping data for its assigned CNs:
     ```python
     for core in cores:
         print(f"Core ID: {core.id}")
         print(f"Attributes: {dir(core)}")
     ```
   - **Action**: If mapping details are here, correlate them with CNs from the workload.

#### 5. **Investigate the Cost Lookup Table (LUT)**
   - **What it is**: A precomputed table (`cost_lut.pickle`) generated during optimization, containing costs for CN-core combinations.
   - **Why it matters**: It likely includes ZigZag’s optimal loop ordering as part of the cost evaluation.
   - **How to access it**:
     ```python
     with open("path/to/cost_lut.pickle", "rb") as f:
         cost_lut = pickle.load(f)
     print(cost_lut)  # Inspect structure
     ```
   - **Exploration**: The LUT might be a dictionary mapping CN-core pairs to cost details:
     ```python
     # Hypothetical structure
     # cost_lut = {(cn_id, core_id): {'cost': ..., 'loop_order': ['I', 'K', 'O'], ...}, ...}
     for (cn_id, core_id), details in cost_lut.items():
         print(f"CN {cn_id} on Core {core_id}: Loop Order = {details.get('loop_order')}")
     ```
   - **Action**: Extract and print loop orderings from the LUT.

#### 6. **Run the SCME and Access Results**
   - **Process**: If you’re running Stream from scratch:
     1. Initialize SCME with a workload, accelerator, and other parameters.
     2. Call `scme.run()` to schedule the graph.
     3. Access `workload` or the LUT post-execution.
   - **Code example**:
     ```python
     from stream.cost_model.cost_model import StreamCostModelEvaluation
     from stream.workload.onnx_workload import ComputationNodeWorkload
     from stream.hardware.architecture.accelerator import Accelerator

     # Placeholder initialization (adjust based on actual API)
     workload = ComputationNodeWorkload(...)  # Load your ONNX model
     accelerator = Accelerator(...)  # Define your hardware
     scme = StreamCostModelEvaluation(workload, accelerator, [], [])
     scme.run()

     # Extract loop ordering from workload
     for cn in scme.workload.nodes():
         loop_order = getattr(cn, 'mapping', {}).get('loop_order', 'Not found')
         print(f"CN {cn.id}: Loop Order = {loop_order}")
     ```

---

### Complete Code Example

Here’s a self-contained script to load an SCME object and print the intra-core loop ordering:

```python
import pickle
from stream.cost_model.cost_model import StreamCostModelEvaluation

# Load SCME from pickle file (after optimization)
with open("path/to/scme.pickle", "rb") as f:
    scme = pickle.load(f)

# Access workload and print loop ordering
print("Extracting Intra-Core Loop Ordering from Workload:")
for cn in scme.workload.nodes():  # Adjust based on actual graph API
    try:
        # Attempt common attribute names
        mapping = getattr(cn, 'mapping', None) or getattr(cn, 'schedule', None) or getattr(cn, 'temporal_ordering', None)
        loop_order = mapping.get('loop_order') if mapping else "Not found"
        core_id = mapping.get('core') if mapping else "Unknown"
        print(f"CN {cn.id} (Core {core_id}): Loop Order = {loop_order}")
    except AttributeError:
        print(f"CN {cn.id}: No mapping attribute found")

# Optionally, check the cost LUT
try:
    with open("path/to/cost_lut.pickle", "rb") as f:
        cost_lut = pickle.load(f)
    print("\nExtracting Intra-Core Loop Ordering from Cost LUT:")
    for (cn_id, core_id), details in cost_lut.items():
        loop_order = details.get('loop_order', 'Not found')
        print(f"CN {cn_id} (Core {core_id}): Loop Order = {loop_order}")
except FileNotFoundError:
    print("Cost LUT not found.")
```

---

### Notes
- **Assumptions**: The script assumes `workload.nodes()` provides CNs and that mapping attributes exist (e.g., `mapping`, `loop_order`). Adjust based on actual class definitions in `stream.workload.onnx_workload`.
- **Further Investigation**: If attributes are not immediately obvious, use `dir()` or consult Stream’s GitHub ([Multi-core HW accelerator mapping optimization framework](https://github.com/KULeuven-MICAS/stream)) and ZigZag’s documentation ([ZigZag Documentation](https://kuleuven-micas.github.io/zigzag/)).
- **Fallback**: If loop ordering isn’t in `workload` or the LUT, it might be computed on-the-fly by ZigZag during `schedule_graph`. Check `stream.cost_model.scheduler` for calls to ZigZag APIs.

---

### Conclusion

By starting with the SCME class, we can trace ZigZag’s intra-core loop ordering through the `workload` attribute (via CNs) or the cost LUT. The steps outlined—loading the SCME object, inspecting CN attributes, and optionally checking the LUT—provide a practical approach to extract and print these results. This method leverages Stream’s integration with ZigZag, ensuring access to the optimized intra-core mappings.




#### Question 4:

There is this zigzag_core_mapping_estimtion.py

'import logging
import os
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.hardware.architecture.memory_port import DataDirection, PortAllocation
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.main import MainStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.results.reduce_stages import MinimalLatencyStage
from zigzag.utils import pickle_deepcopy

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.stages.stage import Stage, StageCallable
from stream.utils import CostModelEvaluationLUT, get_unique_nodes
from stream.visualization.cost_model_evaluation_lut import (
    visualize_cost_lut_pickle,
)
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class ZigZagCoreMappingEstimationStage(Stage):
    """
    Class that saves the optimal CME for each valid node-core allocation to the node.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        loma_lpf_limit: int,
        cost_lut_path: str,
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the stage by:
        - extracting all the unique nodes that will have to be evaluated
        - initializing the valid node-core allocations (which are used later by the InterCoreMappingStage)
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.loma_lpf_limit = loma_lpf_limit
        self.cost_lut_path = cost_lut_path
        self.visualize_cost_lut_path = os.path.splitext(self.cost_lut_path)[0] + ".png"
        self.loma_show_progress_bar: bool = kwargs.get("loma_show_progress_bar", False)

        # Extract all unique nodes that will have to be evaluated
        self.unique_nodes = get_unique_nodes(self.workload)

        # Initialize the valid node-core allocations.
        self.valid_allocations: dict[ComputationNode, list[int]] = {}
        for node in self.unique_nodes:
            assert isinstance(
                node, ComputationNode
            ), f"ZigZagCoreMappingEstimationStage received node {node} of type {type(node)}."
            assert isinstance(node.possible_core_allocation, list), f"Core allocation is not a list for node {node}."
            self.valid_allocations[node] = node.possible_core_allocation

        # Initialize CostModelEvaluationLUT
        self.cost_lut = CostModelEvaluationLUT(self.cost_lut_path)

    def run(self):
        logger.info("Start ZigZagCoreMappingEstimationStage.")
        for node in self.unique_nodes:
            # TODO This should never evaluate to true: enforce core_allocation as list everywhere
            if isinstance(self.valid_allocations[node], tuple):
                raise ValueError

            core_ids = self.valid_allocations[node]
            for core_id in core_ids:
                core = self.accelerator.get_core(core_id)
                # Offchip memory core doesn't have operational units
                if core.operational_array.total_unit_count == 0:
                    continue
                # If the (node, core) combination has already been optimized, we skip it
                if self.cost_lut.has_cme(node, core):
                    continue
                # If an equal performance has already been computed, we take it
                equal_node = self.cost_lut.get_equal_node(node)
                equal_core = self.cost_lut.get_equal_core(equal_node, core) if equal_node else None
                if equal_node and equal_core:
                    cme = pickle_deepcopy(self.cost_lut.get_cme(equal_node, equal_core))
                    # Update the CME attributes for this node-core combination
                    cme.layer.core_allocation = [core_id]
                    cme.core_id = core_id
                    self.cost_lut.add_cme(node, core, cme, allow_overwrite=False)
                    continue
                else:
                    node_duplicate = pickle_deepcopy(node)
                    # Remove duplicate cores with same id in case the core definition has changed
                    self.cost_lut.remove_cores_with_same_id(node, core)
                    # We need to compute the optimal performance for this node-core combination
                    # It's possible this node might not fully fit within the core's top level memories.
                    #  If so, we update the core
                    too_large_operands_for_cme = self.check_core_capacity_for_node(core, node_duplicate)
                    node_duplicate.set_chosen_core_allocation(core_id)

                    # Attempt to override the node's spatial mapping based on the core's dataflow
                    if core.dataflows:
                        node_duplicate.spatial_mapping = core.dataflows

                    # Initialize the flow that will be followed to extract the optimal HW performance of every
                    #  unique node-core allocation
                    main_stage = self.get_intra_core_mapping_flow(
                        node=node_duplicate,
                        too_large_operands=too_large_operands_for_cme,
                        core_id=core_id,
                    )
                    answers = main_stage.run()
                    assert len(answers) == 1, "ZigZagCoreMappingEstimationStage's subflow returned more than one CME"
                    cme: CostModelEvaluation = answers[0][0]  # type: ignore
                    node_duplicate.set_chosen_core_allocation(None)  # Reset the node's chosen core allocation
                    self.cost_lut.add_cme(node, core, cme, allow_overwrite=False)
            self.cost_lut.save()

        self.visualize_cost_lut()
        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["accelerator"] = self.accelerator
        kwargs["cost_lut"] = self.cost_lut

        logger.info("Finished ZigZagCoreMappingEstimationStage.")
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def visualize_cost_lut(self):
        # Get the scale factors
        scale_factors = {
            n: len([cn for cn in self.workload.node_list if cn.has_same_performance(n)])
            for n in self.cost_lut.get_nodes()
        }
        # Run the visualization
        visualize_cost_lut_pickle(self.cost_lut, scale_factors, self.visualize_cost_lut_path)

    def get_intra_core_mapping_flow(self, node: ComputationNode, too_large_operands: list[MemoryOperand], core_id: int):
        logger.info(f"Launching intra-core mapping optimization for {node} -> core {core_id} ...")

        core = self.accelerator.get_core(core_id)

        if too_large_operands:
            core = self.add_offchip_to_core(core, too_large_operands, node.id)

        main_stage = MainStage(
            [  # Initializes the MainStage as entry point
                MinimalLatencyStage,
                SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
                MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
                TemporalMappingGeneratorStage,  # Generates multiple temporal mappings (TM)
                CostModelStage,  # Evaluates generated SM and TM through cost model
            ],
            layer=node,
            accelerator=core,  # Accelerator in zigzag corresponds to Core in stream
            loma_lpf_limit=self.loma_lpf_limit,  # required by LomaEngine
            loma_show_progress_bar=self.loma_show_progress_bar,
        )
        return main_stage

    def check_core_capacity_for_node(self, core: Core, node: ComputationNode) -> list[MemoryOperand]:
        """Check if we need to add a DRAM memory to the given core for the given node.
        The DRAM can span one or more operands, based on the total size of available on-chip memory
        and the stored operands inside each memory.

        Args:
            core (Core): The core onto which we want to map the node
            node (ComputationNode): The node we want to map onto the core

        Returns:
            list: A list of memory operands for which the capacity on the core is insufficient.
        """
        too_large_operands_for_cme: list[MemoryOperand] = []

        # Step 1: get all the unique top level memories of the core
        memory_hierarchy_dict = core.mem_hierarchy_dict
        top_memories = [memory[-1] for (_, memory) in memory_hierarchy_dict.items()]
        unique_top_memories = set(top_memories)

        # Step 2: for each top level memory, for each operand this memory holds, calculate the required capacity
        # (in bit) for holding them
        memory_operand_link = node.memory_operand_links
        constant_operands = node.constant_operands
        output_operand = node.output_operand
        for top_memory in unique_top_memories:
            top_level_capacity = top_memory.memory_instance.size
            memory_operands = list(top_memory.mem_level_of_operands.keys())
            layer_operands = [memory_operand_link.mem_to_layer_op(mem_operand) for mem_operand in memory_operands]
            bits_to_be_stored_in_top_level: dict[MemoryOperand, int] = {}
            for layer_operand, memory_operand in zip(layer_operands, memory_operands):
                # Case 1: constant operand (e.g. 'W' and the first layer's 'I') or output operand
                if layer_operand in constant_operands + [output_operand]:
                    nb_bits = node.operand_size_bit[layer_operand]
                # Case 2: variable operand -> sum up the the total data amount on the in edges
                # (can be larger than the ideal required data size)
                elif len(self.workload.in_edges(node)) > 0:
                    in_edges_data = [data for (_, _, data) in self.workload.in_edges(node, data=True)]
                    nb_bits = sum(
                        [
                            data["bits"]
                            for data in in_edges_data
                            if "operand" in data and data["operand"] == layer_operand
                        ]
                    )
                # Case 3: not constant, but no edges found
                else:
                    nb_bits = node.operand_size_bit[layer_operand]

                # Patchwork for edge cases where the node has input data that is not present in the edges (e.g.
                # in case of KV-cache). The data on the edges should always be > operand_size_bit, except in this case
                nb_bits = max(nb_bits, node.operand_size_bit[layer_operand])

                bits_to_be_stored_in_top_level[memory_operand] = nb_bits
            total_required_capacity = sum(bits_to_be_stored_in_top_level.values())

            # Step 3: compare the total required capacity with the top level memory capacity
            if top_level_capacity < total_required_capacity:
                # when the memory capacity is smaller than the requirement,
                # sort the required capacity of each operand that shares this memory based on the operand's required
                #  size, from small to large
                # fit the operands to the memory from small to large
                bits_to_be_stored_in_top_level = {
                    k: v for k, v in sorted(bits_to_be_stored_in_top_level.items(), key=lambda item: item[1])
                }
                nb_operands_in_top_level = len(bits_to_be_stored_in_top_level)
                while top_level_capacity < sum(
                    list(bits_to_be_stored_in_top_level.values())[:nb_operands_in_top_level]
                ):
                    nb_operands_in_top_level -= 1
                    if nb_operands_in_top_level == 0:
                        break
                operands_stored_in_top_level = list(bits_to_be_stored_in_top_level.keys())[:nb_operands_in_top_level]
                operands_stored_in_offchip = list(bits_to_be_stored_in_top_level.keys())[nb_operands_in_top_level:]

                # Step 4: Check when some operand(s) fit in the top level core memory, and some cannot fit
                # (too_large_operands), the top level core memory has enough space for supporting the SU of not-fitted
                #  operands
                if not operands_stored_in_top_level or not operands_stored_in_offchip:
                    pass
                else:
                    rest_capacity = self.get_top_level_memory_rest_capacity(
                        operands_stored_in_top_level,
                        bits_to_be_stored_in_top_level,
                        top_level_capacity,
                    )
                    required_capacity = self.get_too_large_operands_minimal_required_capacity_in_top_level_memory(
                        operands_stored_in_offchip, core
                    )
                    while rest_capacity < required_capacity:
                        # put_the_largest operands_stored_in_top_level to operands_stored_in_offchip
                        nb_operands_in_top_level -= 1
                        operands_stored_in_top_level = list(bits_to_be_stored_in_top_level.keys())[
                            :nb_operands_in_top_level
                        ]
                        operands_stored_in_offchip = list(bits_to_be_stored_in_top_level.keys())[
                            nb_operands_in_top_level:
                        ]
                        if not operands_stored_in_top_level:
                            break
                        rest_capacity = self.get_top_level_memory_rest_capacity(
                            operands_stored_in_top_level,
                            bits_to_be_stored_in_top_level,
                            top_level_capacity,
                        )
                        required_capacity = self.get_too_large_operands_minimal_required_capacity_in_top_level_memory(
                            operands_stored_in_offchip, core
                        )

                too_large_operands_for_cme += operands_stored_in_offchip
        return too_large_operands_for_cme

    @staticmethod
    def get_top_level_memory_rest_capacity(
        operands_stored_in_top_level: list[MemoryOperand],
        bits_to_be_stored_in_top_level: dict[MemoryOperand, int],
        top_level_capacity_bits: int,
    ) -> int:
        """Calculate the remaining capacity in the top level core memory after storing the operands_stored_in_top_level

        Args:
            operands_stored_in_top_level (list): list of operands that can fit in the top memory level of the core
            bits_to_be_stored_in_top_level (dict): the data size in bit for each variable operands
            top_level_capacity_bits (int): the total capacity of the top level core memory

        Returns:
            int: the memory capacity left after storing the operands_stored_in_top_level
        """
        rest_capacity = top_level_capacity_bits
        for mem_operand in operands_stored_in_top_level:
            rest_capacity -= bits_to_be_stored_in_top_level[mem_operand]
        return rest_capacity

    def get_too_large_operands_minimal_required_capacity_in_top_level_memory(
        self,
        operands_stored_in_offchip: list[MemoryOperand],
        core: Core,
    ) -> int:
        """Calculate the required capacity in the top level core memory for operands_stored_in_offchip due to spatial
        unrolling

        Args:
            operands_stored_in_offchip (list): list of operands that cannot fit in the top memory level of the core
            dataflows (list of dict): the dataflows (spatial mappings) that current core supports
            node (ComputationNode): The computational node we want to map onto the core

        Returns:
            int: the required memory capacity in the top memory of the core for operands_stored_in_offchip
        """

        def get_lowest_level_unrolled_memory_capacity(memory_operand: MemoryOperand):
            memory_level = core.memory_hierarchy.get_memory_levels(memory_operand)[0]
            return memory_level.memory_instance.size * memory_level.unroll_count

        unroll_dict: dict[MemoryOperand, int] = {}
        for mem_operand in operands_stored_in_offchip:
            capacity = get_lowest_level_unrolled_memory_capacity(mem_operand)
            unroll_dict[mem_operand] = capacity
        return round(sum(unroll_dict.values()))

    def add_offchip_to_core(self, core: Core, too_large_operands: list[MemoryOperand], layer_idx: int):
        """Add the offchip memory as the top level memory of the core with core_id in a copy of the accelerator

        Args:
            core_id (int): The id of the core to which we want to add the off-chip memory for cost evaluation.
            too_large_operands (list): The memory operands the off-chip memory should store.
            layer_idx (int): workload layer index.
        """
        assert self.accelerator.offchip_core_id is not None
        logger.warning(f"Adding offchip memory for {core}, layer={layer_idx}, memory_operands={too_large_operands}.")
        offchip_core: Core = pickle_deepcopy(self.accelerator.get_core(self.accelerator.offchip_core_id))

        # Sanity checks
        # Make sure that there is only one offchip memory
        offchip_memory_levels = offchip_core.memory_hierarchy.mem_level_list
        assert (
            len(offchip_memory_levels) == 1
        ), "There is more than one offchip memory, unsure which one to take for intra core mapping"
        offchip_memory_level: MemoryLevel = pickle_deepcopy(offchip_memory_levels[0])
        offchip_memory_instance = offchip_memory_level.memory_instance
        offchip_memory_operands = too_large_operands
        # Recreate the port allocation
        offchip_port_alloc_raw: dict[MemoryOperand, dict[DataDirection, str]] = {}
        for memory_operand in offchip_memory_operands:
            offchip_port_alloc_raw[memory_operand] = offchip_memory_level.port_alloc_raw.get_alloc_for_mem_op(
                memory_operand
            )

        offchip_port_alloc = PortAllocation(offchip_port_alloc_raw)
        offchip_served_dimensions = offchip_memory_level.served_dimensions

        # Create new core with updated memory hierarchy
        updated_core: Core = pickle_deepcopy(core)
        updated_core.memory_hierarchy.add_memory(
            offchip_memory_instance,
            offchip_memory_operands,
            offchip_port_alloc,
            offchip_served_dimensions,
        )
        updated_core.recalculate_memory_hierarchy_information()  # Recalculates some internals of the Core object

        return updated_core
"

There is stream utils.py

import os
import pickle
import pprint
from typing import TYPE_CHECKING, Any, TypeAlias

from numpy.typing import NDArray
from onnx import ModelProto, NodeProto
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving
from zigzag.parser.onnx.utils import get_onnx_tensor_type

from stream.hardware.architecture.core import Core
from stream.workload.mapping import TILING_T

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.computation.computation_node import ComputationNode
    from stream.workload.onnx_workload import ComputationNodeWorkload

ARRAY_T: TypeAlias = NDArray[Any]


def get_onnx_input_shapes(node: NodeProto, onnx_model: ModelProto) -> tuple[list[int], list[int]]:
    if len(node.input) != 2:
        raise ValueError(f"Node {node.name} does not have two inputs")
    input_name1 = node.input[0]
    input_name2 = node.input[1]
    input_shape1 = get_onnx_tensor_type(input_name1, onnx_model).shape
    input_shape2 = get_onnx_tensor_type(input_name2, onnx_model).shape
    return input_shape1, input_shape2


def has_asymmetric_input_data(node: NodeProto, onnx_model: ModelProto):
    """Return true iff the node has two inputs and the input nodes have a different shape"""
    if len(node.input) != 2:
        return False

    input_shape1, input_shape2 = get_onnx_input_shapes(node, onnx_model)
    return input_shape1 != input_shape2


def get_too_large_operands(cme: CostModelEvaluation, accelerator: "Accelerator", core_id: int) -> list[MemoryOperand]:
    """Create a list of memory operands for which an extra memory level (i.e. offchip) was added.

    Args:
        cme (CostModelEvaluation): The CostModelEvaluation containing information wrt the memory utilization.
        accelerator (Accelerator): The accelerator object containing the different cores.
        core_id (int): The id of the core of which we wish to get the too large operands.
    """
    too_large_operands: list[MemoryOperand] = []
    core = accelerator.get_core(core_id)
    core_nb_memory_levels = core.memory_hierarchy.nb_levels
    for layer_operand, lvl in cme.mapping.data_elem_per_level.items():
        memory_operand = cme.layer.memory_operand_links.layer_to_mem_op(layer_operand)
        if len(lvl) > core_nb_memory_levels[memory_operand] + 1:  # +1 because of spatial level
            too_large_operands.append(memory_operand)
    return too_large_operands


def save_core_allocation(
    workload: "ComputationNodeWorkload", path: str, type: str = "fixed", format: str = "py"
) -> dict:
    """Saves the core allocations of a workload to a python or pickle file.
    In fixed mode: if a layer has been split into multiple groups, the allocation of each group is saved to a tuple.
    In flexible mode: for each layer, the possible allocations are saved to a list.
    # TODO: Update this function to work with new mapping definition

    Args:
        workload (DiGraph): The graph of CNs
        path (str): The filepath to save the dict to.
        type (str, optional): The type of core allocation: fixed or flexible.

    Returns:
        allocations (dict): The dictionary containing core allocations for each node name
    """
    node_allocations = {}
    node_allocations_grouped = {}
    for n in workload.node_list:
        if n.name not in node_allocations:
            node_allocations[n.name] = {"core_allocation": [n.chosen_core_allocation]}
            node_allocations_grouped[n.name] = {n.group: n.chosen_core_allocation}
        else:
            node_allocations[n.name]["core_allocation"].append(n.chosen_core_allocation)
            if n.group not in node_allocations_grouped[n.name]:
                node_allocations_grouped[n.name][n.group] = n.chosen_core_allocation
    if type == "fixed":
        mapping = {
            k: {"core_allocation": tuple(list(zip(*sorted(v.items())))[1])} for k, v in node_allocations_grouped.items()
        }
    else:
        mapping = {k: {"core_allocation": sorted(set(v["core_allocation"]))} for k, v in node_allocations.items()}
    # Create folder structure if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The dict is saved with variable name 'mapping' as this is expected for running
    if format in ["python", "py"]:
        assert path.split(".")[-1] == "py", "format is python but file path doesn't end in .py"
        with open(path, "w") as handle:
            handle.write("mapping = ")
            handle.write(pprint.pformat(mapping))
    elif format in ["pickle", "pkl"]:
        with open(path, "wb") as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Invalid format: {format}.")
    return mapping


def get_unique_nodes(workload: "ComputationNodeWorkload") -> list["ComputationNode"]:
    """! Get the unique nodes from a workload."""
    unique_nodes: list[ComputationNode] = []
    for node in workload.node_list:
        equal_nodes = list(
            (
                unique_node
                for unique_node in unique_nodes
                if node.has_same_performance(unique_node) and node.group == unique_node.group
            )
        )
        if not equal_nodes:
            unique_nodes.append(node)
    return unique_nodes


def get_required_offchip_bandwidth(
    cme: CostModelEvaluation, too_large_operands: list[MemoryOperand]
) -> FourWayDataMoving:
    if not too_large_operands:
        return FourWayDataMoving(0, 0, 0, 0)
    # If there was offchip memory added for some operands, get the offchip bandwidth required
    offchip_level = cme.accelerator.get_memory_level(too_large_operands[0], -1)
    req_offchip_bw = cme.get_total_inst_bandwidth(offchip_level)
    return req_offchip_bw


def contains_wildcard(tiling: TILING_T):
    """Returns wether the given tiling contains a wildcard number `*`. The wildcard must later be replaced by the
    constraint optimization into the optimal number of tiles"""
    return any(tiling == "*" for _, tiling in tiling)


class CostModelEvaluationLUT:
    """A class to store the cost model evaluations in a look-up table.
    The look-up table is a dictionary with the following structure:
    {
        node0: {
            core0: CostModelEvaluation,
            ...
        },
        ...
    }
    """

    def __init__(self, cache_path: str | None, load: bool = True):
        self.lut: dict["ComputationNode", dict[Core, CostModelEvaluation]] = {}
        self.cache_path = cache_path
        if load and self.cache_path and os.path.exists(self.cache_path):
            self.load()

    def load(self):
        if not self.cache_path:
            raise ValueError("No cache_path provided.")
        try:
            with open(self.cache_path, "rb") as fp:
                self.lut = pickle.load(fp)
        except Exception as e:
            raise ValueError(
                f"Could not load look-up table from {self.cache_path}. Try removing the file if it exists."
            ) from e

    def save(self):
        if not self.cache_path:
            raise ValueError("No cache_path provided.")
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.lut, fp)

    def add_cme(self, node: "ComputationNode", core: Core, cme: CostModelEvaluation, allow_overwrite: bool = True):
        """Add a CostModelEvaluation to the look-up table for a given node and core.
        If a node with equal performance already exists in the look-up table,
        the CostModelEvaluation is added to that node."""
        if not allow_overwrite and self.has_cme(node, core):
            raise ValueError(f"CostModelEvaluation for node {node} and core {core} already exists.")
        if node not in self.lut:
            self.lut[node] = {}
        self.lut[node][core] = cme

    def has_cme(self, node: "ComputationNode", core: Core):
        """Check if a CostModelEvaluation exists for a given node and core."""
        return self.get_equal_node(node) is not None and node in self.get_nodes() and core in self.lut[node]

    def get_cme(self, node: "ComputationNode", core: Core):
        """Retrieve the CostModelEvaluation for a given node and core."""
        if not self.has_cme(node, core):
            raise ValueError(f"No CostModelEvaluation found for node {node} and core {core}.")
        return self.lut[node][core]

    def get_equal_node(self, node: "ComputationNode"):
        """Retrieve the node in the look-up table that is equal to the given node."""
        if any((n.has_same_performance(node) for n in self.lut)):
            return next(n for n in self.lut if n.has_same_performance(node))
        else:
            return None

    def get_equal_core(self, node: "ComputationNode", core: Core):
        """Retrieve the core in the look-up table that is equal to the given core."""
        try:
            return next(c for c in self.lut[node] if c.has_same_performance(core))
        except StopIteration or KeyError:
            return None

    def get_nodes(self):
        return list(self.lut.keys())

    def get_cores(self, node: "ComputationNode"):
        return list(self.lut.get(node, {}).keys())

    def remove_cores_with_same_id(self, node: "ComputationNode", core: Core):
        """! Removes cores with the same id as core for node from the look-up table."""
        if node in self.lut:
            self.lut[node] = {c: v for c, v in self.lut[node].items() if c.id != core.id}



Here is the scripts I believe could help us with the priting of the mapping.

Ans also it says about the data storing in the pickle file. How to ready a pickle file?