# Question 1:

In Zigzag when the workload is passed in the yaml file there is an if condition  as shown below that chooses the parser based on the type of the format in the API

    workload_parser_stage = (
        ONNXModelParserStage
        if isinstance(workload, ModelProto) or (workload.split(".")[-1] == "onnx")
        else WorkloadParserStage
    )

But in the Stream api, I couldn't find it and it just parses the ONNX format.

Below is the stream api.py program
``` python
import logging as _logging
import os
from typing import Literal

import gurobipy as gp
from zigzag.utils import pickle_load, pickle_save

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
                GeneticAlgorithmAllocationStage,
            ],
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
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
```

Now, As I have to modify the workload in the input stationary format I need to have the format in the yaml file and stream doen't have the parser for that case.

So, My idea is to adapt the yaml parser from the zigzag but it just need to adapt the functionality as the data that stream and zigzag parser is sa bit different and I need to match them to ensure I don't face any challenges in the later part of exploration.

So, Here is the ONNX parser of the Stream which gives an idea of what data is being parsed and sent to other stages

ONNX parser of stream
"import logging
from typing import Any

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class ONNXModelParserStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload_path: str,
        mapping_path: str,
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload_path = workload_path
        self.accelerator = accelerator
        self.mapping_parser = MappingParser(mapping_path)

    def run(self):
        all_mappings = self.mapping_parser.run()
        onnx_model_parser = ONNXModelParser(self.workload_path, all_mappings, self.accelerator)
        onnx_model_parser.run()
        onnx_model = onnx_model_parser.onnx_model
        workload = onnx_model_parser.workload

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["all_mappings"] = all_mappings
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            onnx_model=onnx_model,
            workload=workload,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
"

Here is the workload parser from the zigzag which I need to adapt for the stream to parse an yaml file
"import logging
from typing import Any

from zigzag.parser.mapping_validator import MappingValidator
from zigzag.parser.workload_factory import WorkloadFactory
from zigzag.parser.workload_validator import WorkloadValidator
from zigzag.stages.stage import Stage, StageCallable
from zigzag.utils import open_yaml
from zigzag.workload.dnn_workload import DNNWorkload

logger = logging.getLogger(__name__)


class WorkloadParserStage(Stage):
    """! Parses a user-provided workload from a yaml file."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: str,
        mapping: str,
        **kwargs: Any,
    ):
        assert mapping.endswith(".yaml"), "Mapping is not a yaml file path"
        assert workload.endswith(".yaml"), "Workload is not a yaml file path"
        super().__init__(list_of_callables, **kwargs)
        self.workload_yaml_path = workload
        self.mapping_yaml_path = mapping

    def run(self):
        workload = self.parse_workload()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def parse_workload(self) -> DNNWorkload:
        workload_data = self._parse_workload_data()
        mapping_data = self._parse_mapping_data()
        factory = WorkloadFactory(workload_data, mapping_data)
        return factory.create()

    def _parse_workload_data(self) -> list[dict[str, Any]]:
        """! Parse, validate and normalize workload"""
        workload_data = open_yaml(self.workload_yaml_path)
        workload_validator = WorkloadValidator(workload_data)
        workload_data = workload_validator.normalized_data
        workload_validate_succes = workload_validator.validate()
        if not workload_validate_succes:
            raise ValueError("Failed to validate user provided workload.")
        return workload_data

    def _parse_mapping_data(self) -> list[dict[str, Any]]:
        return self.parse_mapping_data(self.mapping_yaml_path)

    @staticmethod
    def parse_mapping_data(mapping_yaml_path: str) -> list[dict[str, Any]]:
        """Parse, validate and normalize workload mapping from a given yaml file path"""
        mapping_data = open_yaml(mapping_yaml_path)
        mapping_validator = MappingValidator(mapping_data)
        mapping_data = mapping_validator.normalized_data
        mapping_validate_success = mapping_validator.validate()
        if not mapping_validate_success:
            raise ValueError("Failed to validate user provided mapping.")
        return mapping_data
"

Now, Can you help me with the workload_parser for the stream that I can use to parse the yaml files like I did for the zigzag.


#### Answer 1
The grok generate the code but there are some assumptions made which needs to be corrected


#### Question 2
Here are the assumptions made to generate the previous code

Since I don’t have access to the full Stream codebase (e.g., ONNXModelParser, Layer class, or the exact workload type), I’ll make these assumptions based on the provided code and typical DNN workload representations:

Workload Format: Stream’s workload is a networkx.DiGraph with nodes as Layer objects (from stream.workload.layer) and edges representing dependencies, similar to ZigZag’s DNNWorkload.
Layer Class: Stream has a Layer class that takes parameters like id, type, and mapping.
Mapping: all_mappings from MappingParser is a dictionary mapping layer types or IDs to hardware mappings.
YAML Structure: The YAML workload file contains a list of layer dictionaries with keys like id, type, and predecessors (for dependencies), similar to ZigZag.
If these assumptions differ from Stream’s actual implementation, you’ll need to adjust the create_workload method accordingly.

Now, to clarify that
1. Stream uses the concept of ComputationNode
Here is the code of it

```python
"from copy import deepcopy
from math import prod
from typing import TypeAlias

from zigzag.datatypes import Constants, LayerDim, LayerOperand, MemoryOperand
from zigzag.utils import hash_sha512
from zigzag.visualization.results.plot_cme import shorten_onnx_layer_name
from zigzag.workload.layer_attributes import (
    LayerPadding,
)
from zigzag.workload.layer_node import LayerNode, LayerNodeAttributes

from stream.node_tensor import NodeTensor
from stream.workload.mapping import INTRA_CORE_MAPPING_DEFAULT, InterCoreMappingAttributes
from stream.workload.node import Node
from stream.workload.tensor import Tensor

OperandTensorReshape: TypeAlias = dict[LayerOperand, tuple[int, ...]]
LoopRanges: TypeAlias = dict[LayerDim, tuple[int, int]]


class ComputationNode(LayerNode, Node):
    """Extension of ZigZag's concept of a "LayerNode" into a more general concept
    called "ComputationNode", which is not necessarily an entire layer,
    but can represent a smaller chunk of a layer.
    This object also inherits from the "Node" class, which is an abstract baseclass to represent
    different types of onnx nodes needed to accurately schedule the fine-grained graph.
    On top of that, some new information is added for correct dependency generation
    for the finer graph that is built when a layer is split into one and is a
    producer/consumer of another layer.
    """

    too_large_operands: list[MemoryOperand]

    # Map the node's op_type to the corresponding layer dimension to split on for fusion
    FUSION_DIM_MAPPING: dict[str, list[LayerDim]] = {
        "conv": [LayerDim("OY")],
        "matmul": [LayerDim("D")],
        "gemm": [LayerDim("D")],
        "pooling": [LayerDim("OY")],
        "add": [LayerDim("D")],
        "mul": [LayerDim("D")],
        "softmax": [LayerDim("K")],
        "max": [LayerDim("K")],
        "div": [LayerDim("K")],
        "exp": [LayerDim("K")],
        "sum": [LayerDim("K")],
        "relu": [LayerDim("K")],
        "gelu": [LayerDim("K")],
        "silu": [LayerDim("K")],
    }  # TODO default to "K" ?

    def __init__(
        self,
        node_id: int,
        node_name: str,
        node_attr: LayerNodeAttributes,
        mapping_attr: InterCoreMappingAttributes,
        op_type: str = "computation",
        operand_tensor_reshape: OperandTensorReshape | None = None,
        produces_final_output: bool = False,
        group_id: int = 0,
        sub_id: int = -1,  # To distinguish alternative versions of this node
    ):
        op_type = op_type.lower()

        LayerNode.__init__(
            self, layer_id=node_id, node_name=node_name, node_attr=node_attr, mapping_attr=INTRA_CORE_MAPPING_DEFAULT
        )
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type=op_type,
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=mapping_attr.core_allocation,
        )

        # Overwrite default spatial mapping with given one
        self.spatial_mapping = mapping_attr.spatial_mapping
        # Unpack other mapping attributes
        self.core_allocation = mapping_attr.core_allocation
        self.core_allocation_is_fixed = mapping_attr.core_allocation_is_fixed
        self.intra_core_tiling = mapping_attr.intra_core_tiling
        self.inter_core_tiling = mapping_attr.inter_core_tiling

        self.sub_id = sub_id
        self.group = group_id
        self.operand_tensor_reshape = (
            operand_tensor_reshape if operand_tensor_reshape is not None else self.get_operand_tensor_reshape_default()
        )
        self.produces_final_output = produces_final_output
        self.loop_ranges: LoopRanges = {  # type: ignore
            layer_dim: (0, size) for layer_dim, size in self.layer_dim_sizes.items()
        }
        self.operand_dimensionality_order: dict[LayerOperand, list[LayerDim]] = {
            layer_op: self.equation.get_r_layer_dims(layer_op) for layer_op in self.equation.get_contained_operands()
        }

        # adds pr dimensions loop ranges to self.loop_ranges
        self.calculate_pr_loop_ranges()
        # Rename function
        self.get_node_operand = self.memory_operand_links.mem_to_layer_op
        self.extract_node_info = self.extract_layer_info

        # Number of real predecessors is saved to deal with edge cases where some nodes of the same layer have differing predecessors
        # This is used to hash the node and to get accurate knowledge of the number of unique nodes.
        # This should be set after the node is created and the number of predecessors is known.
        self.nb_real_predecessors = None
        self._static_hash_value = self.__compute_static_hash()

        try:
            self.fusion_partition_dims = ComputationNode.FUSION_DIM_MAPPING[op_type]
        except KeyError:
            raise NotImplementedError(f"Fusion partitioning dimensions not defined for {op_type}")

        # Each ComputationNode will save a tensor for all its defined operands.
        # For example, a conv layer will have an I tensor, W tensor and O tensor.
        self.operand_tensors: dict[LayerOperand, Tensor] = {}
        self.set_operand_tensors()

    def set_operand_tensors(self):
        for op in self.layer_operands:
            if op == Constants.OUTPUT_LAYER_OP:
                precision = self.operand_precision.final_output_precision
            else:
                precision = self.operand_precision[op]

            op_dimensionality_order = self.operand_dimensionality_order[op]
            ranges = tuple([self.loop_ranges[dim] for dim in op_dimensionality_order])
            size = prod([upper_bound - lower_bound for (lower_bound, upper_bound) in ranges]) * precision
            self.operand_tensors[op] = Tensor(
                size=size,
                origin=self,
                layer_operand=op,
                loop_dimensions=op_dimensionality_order,
                loop_ranges=ranges,
            )

    def get_operand_tensor_reshape_default(self) -> OperandTensorReshape | None:
        try:
            size_B = self.layer_dim_sizes[LayerDim("B")]
            size_OX = self.layer_dim_sizes[LayerDim("OX")]
            size_OY = self.layer_dim_sizes[LayerDim("OY")]
            size_IX = self.pr_layer_dim_sizes[LayerDim("IX")]
            size_IY = self.pr_layer_dim_sizes[LayerDim("IY")]
            return {
                LayerOperand("I"): (size_B, -1, size_IX, size_IY),
                LayerOperand("O"): (size_B, -1, size_OX, size_OY),
            }
        except KeyError:
            return None

    @property
    def short_name(self) -> str:
        return shorten_onnx_layer_name(self.name)

    def __compute_static_hash(self):
        """Return a value that can be used to identify unique nodes in sets, dicts and equality. It is pre-computed at
        initialization time to speed up dict lookup and instance equality"""
        return hash_sha512(
            (
                self.layer_dim_sizes,
                frozenset(self.dimension_relations),
                self.operand_precision,
                self.memory_operand_links,
                self.id,
                self.sub_id,
                self.nb_real_predecessors,
            )
        )

    def __str__(self):
        return f"ComputationNode{self.id}_{self.sub_id}"

    def __hash__(self) -> int:
        """The hash operator of a node.

        Returns:
            the pre-computed hash
        """
        return self._static_hash_value

    def __eq__(self, other: object):
        """Fast equality comparison between two nodes"""
        # Optimization: this method is used many times to compare with `0`, to count empty tensor elements
        if not other:
            return False
        return isinstance(other, ComputationNode) and self._static_hash_value == other._static_hash_value

    def has_same_performance(self, other: object) -> bool:
        """Compare the equality between two nodes.
        Two nodes are considered equal if they have equal hardware performance, which happens following attributes are
        equal:
        - loop_dim_size: The size of the loops.
        - dimension_relations: The partial relevancy between a dimension and two others.
        - operand_precision: The precision at which the operands are stored, which means the operand identifiers should
          be equal.
        - memory_operand_links: The link between memory operand (paths in mem hierarchy) and this node's operands
          accurate knowledge of the number of unique nodes.
        - nb_real_predecessors: The number of predecessors of the node. This impacts the required memory size.

        Args:
            other (Node): The other node to compare this node with

        Returns:
            bool: Whether the nodes are equal or not
        """
        return (
            isinstance(other, ComputationNode)
            and self.layer_dim_sizes == other.layer_dim_sizes
            and self.dimension_relations == other.dimension_relations
            and self.operand_precision == other.operand_precision
            and self.memory_operand_links == other.memory_operand_links
            and self.id == other.id
            and self.nb_real_predecessors == other.nb_real_predecessors
            # NOTE: don't include sub_id
        )

    def __lt__(self, other: "ComputationNode"):
        """Compare two ComputationNodes for the 'less than (<)' operator.

        Args:
            other (ComputationNode): The other ComputationNode.

        Returns:
            bool: self < other
        """
        return (self.id, self.sub_id) < (other.id, other.sub_id)

    def get_operand_for_dim(self, dim: LayerDim) -> LayerOperand:
        """Return the first operand in the operand_list that has this dim as one of is dimensions

        Args:
            dim (str): The dimension for which to find the operand

        Returns:
            str: The operand that has dim as one of its dimensions
        """
        for op in self.layer_operands:
            if dim in self.operand_dimensionality_order[op]:
                return op
        raise ValueError(f"The given dim {dim} doesn't appear in any operand's dimensionality order")

    def calculate_pr_loop_ranges(self):
        """Add the loop ranges of the partially revelant dimensions for this node to self.loop_ranges"""
        for pr_dim, related_dims_and_scalings in self.pr_scaling_factors.items():
            dim_padding = self.padding[pr_dim] if pr_dim in self.padding else LayerPadding.DEFAULT
            padding_begin = dim_padding[0]
            # Assume that there is always 2 dimensions involved in the calculation of a pr dimension
            pr_dim_val_min = -padding_begin
            pr_dim_val_max = -padding_begin
            for related_dimension, scaling_factor in related_dims_and_scalings:
                pr_dim_val_min += scaling_factor * self.loop_ranges[related_dimension][0]
                # convert to inclusive upper limit
                pr_dim_val_max += scaling_factor * (self.loop_ranges[related_dimension][1] - 1)
            pr_dim_val_max += 1  # convert to exclusive upper range
            self.loop_ranges[pr_dim] = (pr_dim_val_min, pr_dim_val_max)

    def reshape_operand_tensor(self, tensor: NodeTensor, operand: LayerOperand):
        """Reshape the tensor back to the representation needed for producer/consumer."""
        if self.operand_tensor_reshape is None or operand not in self.operand_tensor_reshape:
            return tensor
        else:
            new_shape = self.operand_tensor_reshape[operand]
            return tensor.reshape(new_shape)

    def set_too_large_operands(self, too_large_operands: list[MemoryOperand]):
        self.too_large_operands = too_large_operands

    def update_loop_ranges(self, new_ranges: LoopRanges):
        """Override the loop ranges with a new value for each of the given LayerDims. Keep the old range for the
        LayerDims not defined in `new_ranges`"""
        for layer_dim in new_ranges:
            self.loop_ranges[layer_dim] = new_ranges[layer_dim]

    def extract_inter_core_mapping_attr(self):
        mapping_attr = InterCoreMappingAttributes(
            op_type=self.type,
            spatial_mapping=self.spatial_mapping,
            core_allocation=self.core_allocation,
            core_allocation_is_fixed=self.core_allocation_is_fixed,
            intra_core_tiling=self.intra_core_tiling,
            inter_core_tiling=self.inter_core_tiling,
        )
        return deepcopy(mapping_attr)

    @property
    def nb_real_predecessors(self):
        return self.__nb_real_predecessors

    @nb_real_predecessors.setter
    def nb_real_predecessors(self, nb_real_predecessors: int | None):
        self.__nb_real_predecessors = nb_real_predecessors
        self._static_hash_value = self.__compute_static_hash()
"
```

And the subsequent stages using the ONNXparser as follows
LayerStacksGenerationStage : it uses the  ComputationNodeWorkload, from the "stream.workload.onnx_workload import ComputationNodeWorkload"



#### Question 3:

In Stream documentation it says this "Besides these stages, the implemented stages from the ZigZag framework can be used as well." It means I can used the stage workload_parser directly and use it to let the stream do it's exploration as "WorkloadParserStage: Parse the input workload residing in workload_path. Used when workload is defined manually by the user." is an implemented stage in zigzag.

Now with this what do you think as you have access to the ONNX_parser of the stream and the workloadparserstage of the zigzag, Can I just plug and play with this stage in the stream?


#### Answer

The changes were made as shown below

```python
import logging as _logging
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

    workload_parser_stage = (
    StreamONNXModelParserStage if isinstance(workload, ModelProto) or (workload.split(".")[-1] == "onnx") else ZigzagWorkloadParserStage
        )
    # Load SCME if it exists and skip_if_exists is True
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                workload_parser_stage,  # Parses the workload
                # StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
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
```

#### Question 5: The above changes made to read the yaml file, has given the follwing error can you help in solving them

The final api.py code

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

    workload_parser_stage = (
    StreamONNXModelParserStage if isinstance(workload, ModelProto) or (workload.split(".")[-1] == "onnx") else ZigzagWorkloadParserStage
        )
    # Load SCME if it exists and skip_if_exists is True
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                workload_parser_stage,  # Parses the workload
                # StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
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

The new errors are as follows

Traceback (most recent call last):
  File "/home/subhash/Thesis/stream/stream/March_experiments/results-23-03.1/main.py", line 61, in <module>
    scme = optimize_allocation_ga(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/subhash/Thesis/stream/stream/stream/api.py", line 118, in optimize_allocation_ga
    answers = mainstage.run()
              ^^^^^^^^^^^^^^^
  File "/home/subhash/Thesis/stream/stream/stream/stages/stage.py", line 62, in run
    for cme, extra_info in self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs).run():
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/subhash/Thesis/stream/stream/stream/stages/parsing/accelerator_parser.py", line 28, in run
    sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: WorkloadParserStage.__init__() missing 2 required keyword-only arguments: 'workload' and 'mapping'

