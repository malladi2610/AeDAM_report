# The in depth analysis of the zigzag cost model is as follows

## Zigzag cost model is as follows
```python
## Class that stores inputs and runs them through the zigzag cost model.
#
# Initialize the cost model evaluation with the following inputs:
# - accelerator: the accelerator that includes the core on which to run the layer
# - layer: the layer to run
# - spatial_mapping: the spatial mapping
# - temporal_mapping: the temporal mapping
#
# From these parameters, the following attributes are computed:
# * core: The core on which the layer is ran. This should be specified in the LayerNode attributes.
# * mapping: The combined spatial and temporal mapping object where access patterns are computed.
#
# The following cost model attributes are also initialized:
# - mem_energy_breakdown: The energy breakdown for all operands
# - energy: The total energy
#
# After initialization, the cost model evaluation is run.
class CostModelEvaluation:

    ## The class constructor
    # After initialization, the cost model evaluation is run
    # @param accelerator the accelerator that includes the core on which to run the
    # @param layer the layer to run
    # @param spatial_mapping the spatial mapping
    # @param temporal_mapping the temporal mapping
    # @param access_same_data_considered_as_no_access (optional)
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        spatial_mapping_int,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.spatial_mapping_int = (
            spatial_mapping_int  # the original spatial mapping without decimal
        )
        self.temporal_mapping = temporal_mapping
        self.access_same_data_considered_as_no_access = (
            access_same_data_considered_as_no_access
        )

        self.core_id = layer.core_allocation
        self.mem_level_list = (
            accelerator.get_core(self.core_id).get_memory_hierarchy().mem_level_list
        )
        self.mem_hierarchy_dict = accelerator.get_core(
            self.core_id
        ).get_memory_hierarchy_dict()
        self.mem_size_dict = accelerator.get_core(self.core_id).get_memory_size_dict()
        self.mem_r_bw_dict, self.mem_w_bw_dict = accelerator.get_core(
            self.core_id
        ).get_memory_bw_dict()
        self.mem_r_bw_min_dict, self.mem_w_bw_min_dict = accelerator.get_core(
            self.core_id
        ).get_memory_bw_min_dict()
        self.mem_sharing_list = accelerator.get_core(
            self.core_id
        ).get_memory_sharing_list()
        self.layer_op_to_mem_op = layer.memory_operand_links
        self.mem_op_to_layer_op = dict(
            [(value, key) for key, value in self.layer_op_to_mem_op.items()]
        )

        """ generate the integer spatial mapping from fractional spatial mapping (due to greedy mapping support).
        Later the fractional one is used for calculating energy, and the integer one is used for calculating latency"""
        # self.spatial_mapping_dict_int = spatial_mapping_fractional_to_int(
        #     self.spatial_mapping.mapping_dict_origin
        # )
        self.spatial_mapping_dict_int = self.spatial_mapping_int.mapping_dict_origin

        # For constructing Mapping object,  the last parameter "self.access_same_data_considered_as_no_access" is optional
        self.mapping = Mapping(
            self.accelerator,
            self.spatial_mapping,
            self.temporal_mapping,
            self.layer,
            self.access_same_data_considered_as_no_access,
        )
        self.mapping_int = Mapping(
            self.accelerator,
            self.spatial_mapping_dict_int,
            self.temporal_mapping,
            self.layer,
            self.access_same_data_considered_as_no_access,
        )

        self.active_mem_level = self.mapping.mem_level

        # Run the cost model evaluation
        self.run()

    def __str__(self):
        return f"CostModelEvaluation(layer={self.layer}, core={self.core_id})"

    def __repr__(self):
        return str(self)

    # JSON representation used for saving this object to a json file.
    def __jsonrepr__(self):
        return {
            "outputs": {
                "memory": {
                    "utilization": self.mem_utili_shared
                    if hasattr(self, "mem_utili_shared")
                    else None,
                    "word_accesses": self.memory_word_access,
                },
                "energy": {
                    "energy_total": self.energy_total,
                    "operational_energy": self.MAC_energy,
                    "memory_energy": self.mem_energy,
                    "memory_energy_breakdown_per_level": self.mem_energy_breakdown,
                    "memory_energy_breakdown_per_level_per_operand": self.mem_energy_breakdown_further,
                },
                "latency": {
                    "data_onloading": self.latency_total1 - self.latency_total0,
                    "computation": self.latency_total0,
                    "data_offloading": self.latency_total2 - self.latency_total1,
                },
                "spatial": {
                    "mac_utilization": {
                        "ideal": self.MAC_spatial_utilization,
                        "stalls": self.MAC_utilization0,
                        "stalls_onloading": self.MAC_utilization1,
                        "stalls_onloading_offloading": self.MAC_utilization2,
                    }
                },
            },
            "inputs": {
                "accelerator": self.accelerator,
                "layer": self.layer,
                "spatial_mapping": self.spatial_mapping_int
                if hasattr(self, "spatial_mapping_int")
                else None,
                "temporal_mapping": self.temporal_mapping
                if hasattr(self, "temporal_mapping")
                else None,
            },
        }

    ## Simple JSON representation used for saving this object to a simple json file.
    def __simplejsonrepr__(self):
        return {"energy": self.energy_total, "latency": self.latency_total2}

    ## Run the cost model evaluation.
    def run(self):
        self.calc_memory_utilization()
        self.calc_memory_word_access()
        self.calc_energy()
        self.calc_latency()

    ## Calculate occupancy for each physical memory based on the mapping.
    def calc_memory_utilization(self):
        # mem_utili_individual: the memory utilization of each operand individually.
        # mem_utili_shared: the memory utilization taking operand memory sharing into consideration.
        mem_utili_individual = {}
        effective_mem_utili_individual = {}
        for layer_op in self.layer.operand_list:
            mem_utili_individual[layer_op] = []
            effective_mem_utili_individual[layer_op] = []
            for mem_lv in range(self.active_mem_level[layer_op]):
                mem_utilization = (
                    self.mapping.data_bit_per_level_unrolled[layer_op][mem_lv + 1]
                    / self.mem_size_dict[self.layer_op_to_mem_op[layer_op]][mem_lv]
                )
                assert mem_utilization <= 1, (
                    f"Operand {layer_op} memory level {mem_lv}'s individual memory utilization is "
                    f"{mem_utilization}, which is larger than 1 "
                    f"(memory level starts from 0)"
                )
                mem_utili_individual[layer_op].append(mem_utilization)

                # if we do not count copied data in parallel memories as effective, what is the utilization then? =>
                effective_mem_utilization = (
                    self.mapping.effective_data_bit[layer_op][mem_lv + 1]
                    / self.mem_size_dict[self.layer_op_to_mem_op[layer_op]][mem_lv]
                )
                effective_mem_utili_individual[layer_op].append(
                    effective_mem_utilization
                )

        mem_utili_shared = pickle_deepcopy(mem_utili_individual)
        effective_mem_utili_shared = pickle_deepcopy(effective_mem_utili_individual)
        for mem_share_dict in self.mem_sharing_list:
            mem_utilization = 0
            effective_mem_utilization = 0
            for mem_op, mem_lv in mem_share_dict.items():
                try:
                    layer_op = self.mem_op_to_layer_op[mem_op]
                except:  # mem to layer op might not contain this mem op (e.g. pooling layer)
                    continue
                mem_utilization += mem_utili_individual[layer_op][mem_lv]
                effective_mem_utilization += effective_mem_utili_individual[layer_op][
                    mem_lv
                ]
            assert mem_utilization <= 1, (
                f"Memory shared by {mem_share_dict} (memory operand, memory level) has shared utilization of "
                f"{mem_utilization}, which is > 1 "
                f"(memory level starts from 0)."
            )
            for mem_op, mem_lv in mem_share_dict.items():
                try:
                    layer_op = self.mem_op_to_layer_op[mem_op]
                except:  # mem to layer op might not contain this mem op (e.g. pooling layer)
                    continue
                mem_utili_shared[layer_op][mem_lv] = mem_utilization
                effective_mem_utili_shared[layer_op][mem_lv] = effective_mem_utilization

        self.mem_utili_individual = mem_utili_individual
        self.mem_utili_shared = mem_utili_shared
        self.effective_mem_utili_individual = effective_mem_utili_individual
        self.effective_mem_utili_shared = effective_mem_utili_shared

    ## Calculates the memory word access based on unit memory's data element move count and the physical memory bw.
    def calc_memory_word_access(self):
        memory_word_access = {}
        for layer_op in self.layer.operand_list:
            memory_word_access[layer_op] = []
            for mem_lv in range(self.mapping.mem_level[layer_op]):
                """wr_in_by_low"""
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.wr_in_by_low
                data_precision = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_low
                if data_elem_move_per_period == 0 or data_precision == 0:
                    wr_in_by_low = 0
                else:
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.wr_in_by_low
                    max_bw = self.mem_w_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_w_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    if mem_lv > 0:
                        another_side_bw = self.mem_r_bw_dict[
                            self.layer_op_to_mem_op[layer_op]
                        ][mem_lv - 1] * (
                            self.spatial_mapping.unit_unique[layer_op][mem_lv]
                            / self.spatial_mapping.unit_unique[layer_op][mem_lv + 1]
                        )
                        data_elem_move_per_cycle_in_a_period = min(
                            (another_side_bw / data_precision),
                            (max_bw / data_precision),
                            data_elem_move_per_period,
                        )
                        cycle_in_a_period = ceil(
                            data_elem_move_per_period
                            / data_elem_move_per_cycle_in_a_period
                        )
                    else:
                        data_elem_move_per_cycle_in_a_period = data_elem_move_per_period
                        cycle_in_a_period = 1
                    # wr_in_by_low = (
                    #    ceil(
                    #        (data_elem_move_per_cycle_in_a_period * data_precision)
                    #        / min_bw
                    #    )
                    #    * (min_bw / max_bw)
                    #    * total_period_count
                    #    * cycle_in_a_period
                    #    * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    # )

                    # 2023/06/30, solve the memory access granuarity issue - Jiacong Sun, Linyan Mei
                    # Originally we used the cycle_in_a_period to compute the memory word access.
                    # This neglected the finer-grained memory access possibility (the min_bw, the minimal memory access granuarity, like half-word access).
                    # Now we changed to calculation based on min_bw.
                    wr_in_by_low = (
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ rd_out_to_low """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.rd_out_to_low
                data_precision = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_low
                if data_elem_move_per_period == 0 or data_precision == 0:
                    rd_out_to_low = 0
                else:
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.rd_out_to_low
                    max_bw = self.mem_r_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_r_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    if mem_lv > 0:
                        another_side_bw = self.mem_w_bw_dict[
                            self.layer_op_to_mem_op[layer_op]
                        ][mem_lv - 1] * (
                            self.spatial_mapping.unit_unique[layer_op][mem_lv]
                            / self.spatial_mapping.unit_unique[layer_op][mem_lv + 1]
                        )
                        data_elem_move_per_cycle_in_a_period = min(
                            (another_side_bw / data_precision),
                            (max_bw / data_precision),
                            data_elem_move_per_period,
                        )
                        cycle_in_a_period = ceil(
                            data_elem_move_per_period
                            / data_elem_move_per_cycle_in_a_period
                        )
                        # rd_out_to_low = (
                        #    ceil(
                        #        (data_elem_move_per_cycle_in_a_period * data_precision)
                        #        / min_bw
                        #    )
                        #    * (min_bw / max_bw)
                        #    * total_period_count
                        #    * cycle_in_a_period
                        #    * self.mapping.spatial_mapping.unit_count[layer_op][
                        #        mem_lv + 1
                        #    ]
                        # )
                    # else:

                    # 2023/06/30, solve the memory access granuarity issue - Jiacong Sun, Linyan Mei
                    # Originally we used the cycle_in_a_period to compute the memory word access.
                    # This neglected the finer-grained memory access possibility (the min_bw, the minimal memory access granuarity, like half-word access).
                    # Now we changed to calculation based on min_bw.
                    rd_out_to_low = (
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ rd_out_to_high """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.rd_out_to_high
                if data_elem_move_per_period == 0:
                    rd_out_to_high = 0
                else:
                    data_precision = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_precision.rd_out_to_high
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.rd_out_to_high
                    max_bw = self.mem_r_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_r_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    rd_out_to_high = (
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ wr_in_by_high """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.wr_in_by_high
                if data_elem_move_per_period == 0:
                    wr_in_by_high = 0
                else:
                    data_precision = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_precision.wr_in_by_high
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.wr_in_by_high
                    max_bw = self.mem_w_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_w_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    wr_in_by_high = (
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ All """
                memory_word_access_single = FourWayDataMoving(
                    rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high
                )
                memory_word_access[layer_op].append(memory_word_access_single)

        self.memory_word_access = memory_word_access

    ## Calculates the energy cost of this cost model evaluation by calculating the memory reading/writing energy.
    def calc_energy(self):
        # - TODO: Interconnection energy
        self.calc_MAC_energy_cost()
        self.calc_memory_energy_cost()

    ## Calculate the dynamic MAC energy
    def calc_MAC_energy_cost(self):
        core = self.accelerator.get_core(self.core_id)
        single_MAC_energy = core.operational_array.unit.cost
        self.MAC_energy = single_MAC_energy * self.layer.total_MAC_count

    ## Computes the memories reading/writing energy by converting the access patterns in self.mapping to
    # energy breakdown using the memory hierarchy of the core on which the layer is mapped.
    #
    # The energy breakdown is saved in self.mem_energy_breakdown.
    #
    # The energy total consumption is saved in self.energy_total.
    def calc_memory_energy_cost(self):
        core = self.accelerator.get_core(self.core_id)
        mem_hierarchy = core.memory_hierarchy

        mem_energy_breakdown = {}
        mem_energy_breakdown_further = {}
        energy_total = 0
        for (layer_op, mem_access_list_per_op) in self.memory_word_access.items():
            """Retrieve the memory levels in the hierarchy for this memory operand"""
            mem_op = self.layer_op_to_mem_op[layer_op]
            memory_levels = mem_hierarchy.get_memory_levels(mem_op=mem_op)

            breakdown = (
                []
            )  # Stores the energy breakdown of a single layer operand (W, I, ...)
            breakdown_further = []  # Stores
            for (access_count, memory_level) in zip(
                mem_access_list_per_op, memory_levels
            ):
                energy_cost_per_read_out = memory_level.read_energy
                energy_cost_per_write_in = memory_level.write_energy
                read_out_energy_to_above = access_count.get_total_read_outs_to_above(
                    scaling=energy_cost_per_read_out
                )
                write_in_energy_from_above = (
                    access_count.get_total_write_ins_from_above(
                        scaling=energy_cost_per_write_in
                    )
                )
                read_out_energy_to_below = access_count.get_total_read_outs_to_below(
                    scaling=energy_cost_per_read_out
                )
                write_in_energy_from_below = (
                    access_count.get_total_write_ins_from_below(
                        scaling=energy_cost_per_write_in
                    )
                )
                total_read_out_energy = (
                    read_out_energy_to_above + read_out_energy_to_below
                )
                total_write_in_energy = (
                    write_in_energy_from_above + write_in_energy_from_below
                )
                total_energy_cost_memory = total_read_out_energy + total_write_in_energy
                breakdown.append(
                    total_energy_cost_memory
                )  # Here the breakdown only saves the total energy cost per memory level
                breakdown_further.append(
                    FourWayDataMoving(
                        read_out_energy_to_below,
                        write_in_energy_from_below,
                        read_out_energy_to_above,
                        write_in_energy_from_above,
                    )
                )  # here it contains the full split
                energy_total += total_energy_cost_memory
            mem_energy_breakdown[layer_op] = breakdown
            mem_energy_breakdown_further[layer_op] = breakdown_further
        self.mem_energy_breakdown = mem_energy_breakdown
        self.mem_energy_breakdown_further = mem_energy_breakdown_further
        self.mem_energy = energy_total
        self.energy_total = self.mem_energy + self.MAC_energy
        logger.debug(f"Ran {self}. Total energy = {self.energy_total}")

    ##  Calculate latency in 4 steps
    #
    # 1) As we already calculated the ideal data transfer rate in combined_mapping.py (in the Mapping class),
    # here we start with calculating the required (or allowed) memory updating window by comparing the effective
    # data size with the physical memory size at each level. If the effective data size is smaller than 50%
    # of the physical memory size, then we take the whole period as the allowed memory updating window (double buffer effect);
    # otherwise we take the the period divided by the top_ir_loop as the allowed memory updating window.
    #
    # 2) Then, we compute the real data transfer rate given the actual memory bw per functional port pair,
    # assuming we have enough memory ports.
    #
    # 3) In reality, there is no infinite memory port to use. So, as the second step, we combine the real
    # data transfer attributes per physical memory port.
    #
    # 4) Finally, we combine the stall/slack of each memory port to get the final latency.
    def calc_latency(self):
        self.calc_double_buffer_flag()
        self.calc_allowed_and_real_data_transfer_cycle_per_DTL()
        self.combine_data_transfer_rate_per_physical_port()
        self.calc_data_loading_offloading_latency()
        self.calc_overall_latency()

    ## This function checks the double-buffer possibility for each operand at each memory level
    # (minimal memory BW requirement case) by comparing the physical memory size with the effective
    # data size, taking into account the memory sharing between operands.
    def calc_double_buffer_flag(self):
        double_buffer_true = {}
        for layer_op in self.layer.operand_list:
            mem_op = self.layer_op_to_mem_op[layer_op]
            """ start with False for each operand at the lowest arch level (MAC array level) """
            double_buffer_true[layer_op] = [False]
            for mem_lv in range(0, self.mapping_int.mem_level[layer_op]):
                if self.effective_mem_utili_shared[layer_op][mem_lv] <= 0.5:
                    double_buffer_true[layer_op].append(True)
                elif (
                    self.effective_mem_utili_individual[layer_op][mem_lv]
                    <= 1 - self.effective_mem_utili_shared[layer_op][mem_lv]
                ):
                    double_buffer_true[layer_op].append(True)
                    shared_mem_list = get_shared_mem_list(
                        mem_op, mem_lv, self.mem_sharing_list
                    )
                    """ When one of the operand in the shared memory get the "double-buffer" chance, 
                    all operands of that shared memory level need to update the memory utilization 
                    for later memory free space evaluation """
                    for shared_mem_op, shared_mem_lv in shared_mem_list:
                        try:
                            shared_layer_op = self.mem_op_to_layer_op[shared_mem_op]
                        except:  # mem op to layer op might not have this mem op (e.g. pooling layer)
                            continue
                        self.effective_mem_utili_shared[shared_layer_op][
                            shared_mem_lv
                        ] += self.effective_mem_utili_individual[layer_op][mem_lv]
                else:
                    double_buffer_true[layer_op].append(False)

        self.double_buffer_true = double_buffer_true

    ## Construct a 4-way data transfer pattern for each unit mem, calculate
    # {allowed_mem_updating_cycle, real_data_trans_cycle, DTL_SS_cycle} per period
    def calc_allowed_and_real_data_transfer_cycle_per_DTL(self):
        allowed_mem_updat_cycle = {}
        real_data_trans_cycle = {}
        """ stall (+) or slack (-) cycle within each period per virtual data transfer link (DTL) """
        DTL_SS_cycle = {}

        for layer_op in self.layer.operand_list:
            allowed_mem_updat_cycle[layer_op] = []
            real_data_trans_cycle[layer_op] = []
            DTL_SS_cycle[layer_op] = []
            mem_op = self.layer_op_to_mem_op[layer_op]
            for mem_lv in range(self.mapping_int.mem_level[layer_op]):
                """======================================allowed_mem_updating_cycle(below)====================================="""
                """ wr_in_by_low & rd_out_to_low"""
                if self.double_buffer_true[layer_op][mem_lv]:
                    wr_in_by_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.wr_in_by_low
                    rd_out_to_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.rd_out_to_low
                else:
                    wr_in_by_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.wr_in_by_low
                    rd_out_to_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.rd_out_to_low

                """ wr_in_by_high & rd_out_to_high """
                if self.double_buffer_true[layer_op][mem_lv + 1]:
                    wr_in_by_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.wr_in_by_high
                    rd_out_to_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.rd_out_to_high
                else:
                    wr_in_by_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.wr_in_by_high
                    rd_out_to_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.rd_out_to_high

                """ All """
                updating_window = FourWayDataMoving(
                    rd_out_to_low_allowed,
                    wr_in_by_low_allowed,
                    rd_out_to_high_allowed,
                    wr_in_by_high_allowed,
                )
                allowed_mem_updat_cycle[layer_op].append(updating_window)
                """ ======================================allowed_mem_updating_cycle(above)===================================== """

                """ =========================================real_data_trans_cycle(below)======================================== """
                """ wr_in_by_low """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_low
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_low
                mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                wr_in_by_low_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ rd_out_to_low """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_low
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_low
                mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                rd_out_to_low_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ rd_out_to_high """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_high
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_high
                mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                rd_out_to_high_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ wr_in_by_high """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_high
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_high
                mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                wr_in_by_high_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ All """
                real_data_trans = FourWayDataMoving(
                    rd_out_to_low_real,
                    wr_in_by_low_real,
                    rd_out_to_high_real,
                    wr_in_by_high_real,
                )
                real_data_trans_cycle[layer_op].append(real_data_trans)
                """ =========================================real_data_trans_cycle(above)======================================= """

        self.allowed_mem_updat_cycle = allowed_mem_updat_cycle
        self.real_data_trans_cycle = real_data_trans_cycle

    ## Consider memory sharing and port sharing, combine the data transfer activity
    # Step 1: collect port activity per memory instance per physical memory port
    # Step 2: calculate SS combine and MUW union parameters per physical memory port
    def combine_data_transfer_rate_per_physical_port(self):
        # Step 1: collect port activity per memory instance per physical memory port
        port_activity_collect = []
        for mem_instance in self.mem_level_list:
            port_activity_single = {}
            port_list = mem_instance.port_list
            for port in port_list:
                port_activity_single[str(port)] = []
                for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
                    try:
                        layer_op = self.mem_op_to_layer_op[mem_op]
                    except:  # mem op to layer might not have this mem op (e.g. pooling layer)
                        continue
                    period_count = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period_count,
                        mov_dir,
                    )
                    if period_count == 0:
                        # skip the inactive data movement activities because they won't impact SS
                        continue
                    period = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period,
                        mov_dir,
                    )
                    real_cycle = getattr(
                        self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                    )
                    allowed_cycle = getattr(
                        self.allowed_mem_updat_cycle[layer_op][mem_lv], mov_dir
                    )
                    port_activity = PortActivity(
                        real_cycle,
                        allowed_cycle,
                        period,
                        period_count,
                        layer_op,
                        mem_lv,
                        mov_dir,
                    )
                    port_activity_single[str(port)].append(port_activity)
            port_activity_collect.append(port_activity_single)
        self.port_activity_collect = port_activity_collect

        # Step 2: calculate SS combine and MUW union parameters per physical memory port
        SS_comb_collect = [
            {port: None for port in mem_ports} for mem_ports in port_activity_collect
        ]
        SS_comb_list = [0]
        # intermediate parameters saved for debugging purpose
        MUW_union_collect = [
            {port: None for port in mem_ports} for mem_ports in port_activity_collect
        ]

        for idx, mem_ports in enumerate(port_activity_collect):
            for port_name, port_activity in mem_ports.items():
                if len(port_activity) == 1:
                    MUW_union_collect[idx][port_name] = port_activity[0].allowed_cycle
                    SS_comb_collect[idx][port_name] = port_activity[0].SS
                    SS_comb_list.append(port_activity[0].SS)
                elif len(port_activity) != 0:
                    MUW_union_collect[idx][port_name] = calc_MUW_union(port_activity)
                    SS_positive_sum = 0
                    SS_negative_sum = 0
                    MUW_sum = 0
                    for port_d in port_activity:
                        if port_d.SS > 0:
                            SS_positive_sum += port_d.SS
                        else:
                            SS_negative_sum += port_d.SS
                        MUW_sum += port_d.MUW
                    SS_comb = SS_positive_sum + max(
                        0, SS_negative_sum + MUW_sum - MUW_union_collect[idx][port_name]
                    )
                    SS_comb_collect[idx][port_name] = SS_comb
                    SS_comb_list.append(SS_comb)

        self.MUW_union_collect = MUW_union_collect
        self.SS_comb_collect = SS_comb_collect
        # Assuming all the memory ports can work in parallel
        self.SS_comb = max(SS_comb_list)

    ## Calculate the initial/final data loading/off-loading cycle by separating out
    # the first-time input operands' / the last-time output operand's data movement
    # on corresponding ports.
    def calc_data_loading_offloading_latency(self):
        # Collect ports' initial data-loading and final data-offloading activities
        data_loading_per_mem_inst = []
        data_loading_cc_per_op = {op: {} for op in self.layer.input_operands}
        data_offloading_per_mem_inst = []
        data_offloading_cc_per_op = {}
        for mem_inst_idx, mem_instance in enumerate(self.mem_level_list):
            data_loading_single = {}
            data_offloading_single = {}
            port_list = mem_instance.port_list
            for port in port_list:
                data_loading_single[str(port)] = []
                data_offloading_single[str(port)] = []
                served_operands = set(
                    s[0] for s in port.served_op_lv_dir if s[0] in ["I1", "I2"]
                )
                port_is_shared_by_two_input_operands = len(served_operands) > 1
                for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
                    try:
                        layer_op = self.mem_op_to_layer_op[mem_op]
                    except:  # mem op to layer op might not have this mem op (e.g. pooling layer)
                        continue
                    period_count = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period_count,
                        mov_dir,
                    )
                    if period_count == 0:
                        # skip for the inactive data movement
                        continue
                    if mem_op in ["I1", "I2"]:
                        real_cycle = getattr(
                            self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                        )
                        data_in_charge = getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_trans_amount_per_period,
                            mov_dir,
                        ) * getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_precision,
                            mov_dir,
                        )
                        if mov_dir[:2] == "rd":
                            mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                        else:
                            mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                        port_activity = PortBeginOrEndActivity(
                            real_cycle,
                            data_in_charge,
                            mem_bw,
                            layer_op,
                            mem_lv,
                            mov_dir,
                        )
                        data_loading_single[str(port)].append(port_activity)
                        data_loading_cc_per_op[layer_op][
                            layer_op + str(mem_lv) + "_" + mov_dir
                        ] = (real_cycle, port_is_shared_by_two_input_operands)
                    else:
                        if mov_dir in ["rd_out_to_low", "wr_in_by_high"]:
                            # don't consider partial sum flowing in the final data off-loading stage
                            continue
                        real_cycle = getattr(
                            self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                        )
                        data_in_charge = getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_trans_amount_per_period,
                            mov_dir,
                        ) * getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_precision,
                            mov_dir,
                        )
                        if mov_dir[:2] == "rd":
                            mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                        else:
                            mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                        port_activity = PortBeginOrEndActivity(
                            real_cycle,
                            data_in_charge,
                            mem_bw,
                            layer_op,
                            mem_lv,
                            mov_dir,
                        )
                        data_offloading_single[str(port)].append(port_activity)
                        data_offloading_cc_per_op[
                            layer_op + str(mem_lv) + "_" + mov_dir
                        ] = real_cycle

            data_loading_per_mem_inst.append(data_loading_single)
            data_offloading_per_mem_inst.append(data_offloading_single)
        self.data_loading_per_mem_inst = data_loading_per_mem_inst
        self.data_loading_cc_per_op = data_loading_cc_per_op
        self.data_offloading_per_mem_inst = data_offloading_per_mem_inst
        self.data_offloading_per_op = data_offloading_cc_per_op

        # Combine ports' initial data-loading activities to get the data loading cycle amount
        data_loading_cc_pair_combined_per_op = {
            op: [] for op in self.layer.input_operands
        }
        data_loading_individual_part = {op: 0 for op in self.layer.input_operands}
        data_loading_half_shared_part = {op: 0 for op in self.layer.input_operands}
        data_loading_shared_part = {op: 0 for op in self.layer.input_operands}
        for layer_op in self.layer.input_operands:
            for mem_lv in range(self.active_mem_level[layer_op] - 1):
                elem1 = data_loading_cc_per_op[layer_op][
                    layer_op + str(mem_lv) + "_" + "wr_in_by_high"
                ]
                elem2 = data_loading_cc_per_op[layer_op][
                    layer_op + str(mem_lv + 1) + "_" + "rd_out_to_low"
                ]
                completely_shared = elem1[1] and elem2[1]
                completely_separate = not (elem1[1]) and not (elem2[1])
                longest_loading_cc = max(elem1[0], elem2[0])
                # for the ports that serve the same data movement purpose, take the longest data loading cycle
                data_loading_cc_pair_combined = longest_loading_cc
                data_loading_cc_pair_combined_per_op[layer_op].append(
                    data_loading_cc_pair_combined
                )
                if completely_separate:
                    data_loading_individual_part[layer_op] += longest_loading_cc
                elif completely_shared:
                    data_loading_shared_part[layer_op] += longest_loading_cc
                else:
                    # the data transfer link between two memory levels is half-shared,
                    # i.e. on one memory side, the port is shared, while on another memory side,
                    # there are different memories with separate ports
                    data_loading_half_shared_part[layer_op] = longest_loading_cc

        if len(self.layer.input_operands) == 1:
            data_loading_cycle = data_loading_individual_part[
                self.layer.input_operands[0]
            ]
        else:
            op1 = self.layer.input_operands[0]
            op2 = self.layer.input_operands[1]
            possible1 = data_loading_shared_part[op1] + max(
                data_loading_shared_part[op2]
                + data_loading_half_shared_part[op2]
                + data_loading_individual_part[op2],
                data_loading_half_shared_part[op1] + data_loading_individual_part[op1],
            )
            possible2 = data_loading_shared_part[op2] + max(
                data_loading_shared_part[op1]
                + data_loading_half_shared_part[op1]
                + data_loading_individual_part[op1],
                data_loading_half_shared_part[op2] + data_loading_individual_part[op2],
            )
            data_loading_cycle = min(possible1, possible2)

        self.data_loading_cc_pair_combined_per_op = data_loading_cc_pair_combined_per_op
        self.data_loading_individual_part = data_loading_individual_part
        self.data_loading_half_shared_part = data_loading_half_shared_part
        self.data_loading_shared_part = data_loading_shared_part
        self.data_loading_cycle = data_loading_cycle

        # Combine ports' final data-offloading activities to get the data offloading cycle amount
        # TODO Only considered the worst case for now
        #  (assumed that all the ports are working in series during the final data off-loading phase)
        data_offloading_cc_pair_combined = []
        layer_op = self.layer.output_operand
        for mem_lv in range(self.active_mem_level[layer_op] - 1):
            elem1 = data_offloading_cc_per_op[
                layer_op + str(mem_lv) + "_" + "rd_out_to_high"
            ]
            elem2 = data_offloading_cc_per_op[
                layer_op + str(mem_lv + 1) + "_" + "wr_in_by_low"
            ]
            longest_offloading_cc = max(elem1, elem2)
            # for the ports that serve the same data movement purpose, take the longest data loading cycle
            data_offloading_cc_pair_combined.append(longest_offloading_cc)
        data_offloading_cycle = sum(data_offloading_cc_pair_combined)

        self.data_offloading_cc_pair_combined = data_offloading_cc_pair_combined
        self.data_offloading_cycle = data_offloading_cycle

    ## This function integrates the previous calculated SScomb, data loading and off-loading cycle to get the overall latency
    def calc_overall_latency(self, cycles_per_mac=1):
        # @param cycles_per_mac: cycle counts per mac operand (>1 for bit-serial computation)
        # the ideal cycle count assuming the MAC array is 100% utilized
        ideal_cycle = ceil(
            self.layer.total_MAC_count
            / self.accelerator.get_core(self.core_id).operational_array.total_unit_count
        ) * cycles_per_mac

        # the ideal temporal cycle count given the spatial mapping (the spatial mapping can be non-ideal)
        ideal_temporal_cycle = self.mapping_int.temporal_mapping.total_cycle * cycles_per_mac
        MAC_spatial_utilization = ideal_cycle / ideal_temporal_cycle

        # Total latency without the initial data loading and the final data off-loading
        latency_total0 = ideal_temporal_cycle + self.SS_comb
        MAC_utilization0 = ideal_cycle / latency_total0

        # Total latency with the initial data loading, but without the final data off-loading
        latency_total1 = ideal_temporal_cycle + self.SS_comb + self.data_loading_cycle
        MAC_utilization1 = ideal_cycle / latency_total1

        # Total latency with both the initial data loading and the final data off-loading
        latency_total2 = (
            ideal_temporal_cycle
            + self.SS_comb
            + self.data_loading_cycle
            + self.data_offloading_cycle
        )
        MAC_utilization2 = ideal_cycle / latency_total2

        self.ideal_cycle = ideal_cycle
        self.ideal_temporal_cycle = ideal_temporal_cycle
        self.MAC_spatial_utilization = MAC_spatial_utilization
        self.latency_total0 = latency_total0
        self.latency_total1 = latency_total1
        self.latency_total2 = latency_total2
        self.MAC_utilization0 = MAC_utilization0
        self.MAC_utilization1 = MAC_utilization1
        self.MAC_utilization2 = MAC_utilization2

    def __add__(self, other):
        sum = pickle_deepcopy(self)

        ## Energy
        sum.MAC_energy += other.MAC_energy
        sum.mem_energy += other.mem_energy
        for op in sum.mem_energy_breakdown.keys():
            if op in other.mem_energy_breakdown.keys():
                l = []
                for i in range(
                    min(len(self.mem_energy_breakdown[op]), len(other.mem_energy_breakdown[op]))
                ):
                    l.append(
                        self.mem_energy_breakdown[op][i] + other.mem_energy_breakdown[op][i]
                    )
                i = min(len(self.mem_energy_breakdown[op]), len(other.mem_energy_breakdown[op]))
                l += self.mem_energy_breakdown[op][i:]
                l += other.mem_energy_breakdown[op][i:]
                sum.mem_energy_breakdown[op] = l

        for op in sum.mem_energy_breakdown_further.keys():
            if op in other.mem_energy_breakdown_further.keys():
                l = []
                for i in range(
                    min(
                        len(self.mem_energy_breakdown_further[op]),
                        len(other.mem_energy_breakdown_further[op]),
                    )
                ):
                    l.append(
                        self.mem_energy_breakdown_further[op][i]
                        + other.mem_energy_breakdown_further[op][i]
                    )
                i = min(
                    len(self.mem_energy_breakdown_further[op]),
                    len(other.mem_energy_breakdown_further[op]),
                )
                l += self.mem_energy_breakdown_further[op][i:]
                l += other.mem_energy_breakdown_further[op][i:]
                sum.mem_energy_breakdown_further[op] = l

        # Get all the operands from other that are not in self and add them to the energy breakdown as well
        op_diff = set(other.mem_energy_breakdown.keys()) - set(self.mem_energy_breakdown.keys())
        for op in op_diff:
            sum.mem_energy_breakdown[op] = other.mem_energy_breakdown[op]
            sum.mem_energy_breakdown_further[op] = other.mem_energy_breakdown_further[op]

        sum.energy_total += other.energy_total

        ## Memory access
        for op in sum.memory_word_access.keys():
            if op in other.memory_word_access.keys():
                l = []
                for i in range(
                    min(
                        len(self.memory_word_access[op]),
                        len(other.memory_word_access[op]),
                    )
                ):
                    l.append(
                        self.memory_word_access[op][i] + other.memory_word_access[op][i]
                    )
                i = min(
                    len(self.memory_word_access[op]), len(other.memory_word_access[op])
                )
                l += self.memory_word_access[op][i:]
                l += other.memory_word_access[op][i:]
                sum.memory_word_access[op] = l
        for op in op_diff:
            sum.memory_word_access[op] = other.memory_word_access[op]

        ## Latency
        sum.data_loading_cycle += other.data_loading_cycle
        sum.data_offloading_cycle += other.data_offloading_cycle
        sum.ideal_cycle += other.ideal_cycle
        sum.ideal_temporal_cycle += other.ideal_temporal_cycle  # ideal computation cycles without stalling
        sum.latency_total0 += other.latency_total0
        sum.latency_total1 += other.latency_total1
        sum.latency_total2 += other.latency_total2

        ## MAC utilization
        sum.MAC_spatial_utilization = sum.ideal_cycle / sum.ideal_temporal_cycle
        sum.MAC_utilization0 = sum.ideal_cycle / sum.latency_total0
        sum.MAC_utilization1 = sum.ideal_cycle / sum.latency_total1
        sum.MAC_utilization2 = sum.ideal_cycle / sum.latency_total2

        ## layer
        if type(sum.layer) != list:
            sum.layer = [sum.layer.id]
        if type(other.layer) != list:
            other_layer = [other.layer.id]
        sum.layer += other_layer

        ## core_id
        if type(sum.core_id) != list:
            sum.core_id = [sum.core_id]
        if type(other.layer) != list:
            other_core_id = [other.core_id]
        sum.core_id += other_core_id

        ## Not addable
        func = [
            "calc_allowed_and_real_data_transfer_cycle_per_DTL",
            "calc_data_loading_offloading_latency",
            "calc_double_buffer_flag",
            "calc_overall_latency",
            "calc_MAC_energy_cost",
            "calc_energy",
            "calc_latency",
            "calc_memory_energy_cost",
            "calc_memory_utilization",
            "calc_memory_word_access",
            "combine_data_transfer_rate_per_physical_port",
            "run",
        ]
        add_attr = [
            "MAC_energy",
            "mem_energy",
            "mem_energy_breakdown",
            "mem_energy_breakdown_further",
            "energy_total",
            "memory_word_access",
            "data_loading_cycle",
            "data_offloading_cycle",
            "ideal_cycle",
            "ideal_temporal_cycle",
            "latency_total0",
            "latency_total1",
            "latency_total2",
            "MAC_spatial_utilization",
            "MAC_utilization0",
            "MAC_utilization1",
            "MAC_utilization2",
            "layer",
            "core_id",
        ]

        if hasattr(self, "accelerator") and hasattr(other, "accelerator"):
            if self.accelerator.name.startswith(other.accelerator.name):
                sum.accelerator = other.accelerator
                add_attr.append("accelerator")
            elif other.accelerator.name.startswith(self.accelerator.name):
                add_attr.append("accelerator")
        else:
            pass

        for attr in dir(sum):
            if attr not in (func + add_attr) and attr[0] != "_":
                delattr(sum, attr)

        return sum

    def __mul__(self, number):
        mul = pickle_deepcopy(self)

        # Energy
        mul.MAC_energy *= number
        mul.mem_energy *= number
        mul.mem_energy_breakdown = {
            op: [
                mul.mem_energy_breakdown[op][i] * number
                for i in range(len(mul.mem_energy_breakdown[op]))
            ]
            for op in mul.mem_energy_breakdown.keys()
        }
        mul.mem_energy_breakdown_further = {
            op: [
                mul.mem_energy_breakdown_further[op][i] * number
                for i in range(len(mul.mem_energy_breakdown_further[op]))
            ]
            for op in mul.mem_energy_breakdown_further.keys()
        }
        mul.energy_total *= number

        # Memory access
        mul.memory_word_access = {
            op: [
                mul.memory_word_access[op][i] * number
                for i in range(len(mul.memory_word_access[op]))
            ]
            for op in mul.memory_word_access.keys()
        }

        # Latency
        mul.data_loading_cycle *= number
        mul.data_offloading_cycle *= number
        mul.ideal_cycle *= number
        mul.ideal_temporal_cycle *= number
        mul.latency_total0 *= number
        mul.latency_total1 *= number
        mul.latency_total2 *= number

        # MAC utilization
        mul.MAC_spatial_utilization = mul.ideal_cycle / mul.ideal_temporal_cycle
        mul.MAC_utilization0 = mul.ideal_cycle / mul.latency_total0
        mul.MAC_utilization1 = mul.ideal_cycle / mul.latency_total1
        mul.MAC_utilization2 = mul.ideal_cycle / mul.latency_total2

        # Not addable
        func = [
            "calc_allowed_and_real_data_transfer_cycle_per_DTL",
            "calc_data_loading_offloading_latency",
            "calc_double_buffer_flag",
            "calc_overall_latency",
            "calc_MAC_energy_cost",
            "calc_energy",
            "calc_latency",
            "calc_memory_energy_cost",
            "calc_memory_utilization",
            "calc_memory_word_access",
            "combine_data_transfer_rate_per_physical_port",
            "run",
        ]
        mul_attr = [
            "MAC_energy",
            "mem_energy",
            "mem_energy_breakdown",
            "mem_energy_breakdown_further",
            "energy_total",
            "memory_word_access",
            "data_loading_cycle",
            "data_offloading_cycle",
            "ideal_cycle",
            "ideal_temporal_cycle",
            "latency_total0",
            "latency_total1",
            "latency_total2",
            "MAC_spatial_utilization",
            "MAC_utilization0",
            "MAC_utilization1",
            "MAC_utilization2",
            "layer",
            "accelerator",
        ]

        for attr in dir(mul):
            if attr not in (func + mul_attr) and attr[0] != "_":
                delattr(mul, attr)

        return mul
```

## Grok analysis of the cost model is as follows

Below, I provide detailed formulas for calculating **latency**, **word access**, **memory utilization**, and **spatial utilization** based on the ZigZag cost model, as implemented in the provided code and described in the associated paper. These formulas are derived from the `CostModelEvaluation` class and are intended for use in an Excel sheet to verify whether the exploration results of a workload are event-driven or not. The calculations are agnostic to whether the accelerator is event-driven or traditional; only the modeling approach differs, but the underlying mathematical formulas remain consistent.

---

## 1. Latency
Latency in the ZigZag cost model represents the total time (in cycles) to complete the computation of a layer, including computation time, stalls due to memory bandwidth limitations, and data loading/offloading overheads. The final latency is stored as `latency_total2`, which includes all components. The calculation is performed in the `calc_overall_latency` method, with intermediate steps from other methods.

### Formula
\[
\text{Latency} = \text{ideal_temporal_cycle} + \text{SS_comb} + \text{data_loading_cycle} + \text{data_offloading_cycle}
\]

Where:
- **ideal_temporal_cycle**: The ideal number of cycles for computation based on the temporal mapping, assuming no stalls.
- **SS_comb**: Combined stall cycles due to memory bandwidth constraints.
- **data_loading_cycle**: Cycles required to load initial input data.
- **data_offloading_cycle**: Cycles required to offload final output data.

### Detailed Breakdown
#### a. Ideal Temporal Cycle
\[
\text{ideal_temporal_cycle} = \text{temporal_mapping.total_cycle} \times \text{cycles_per_mac}
\]
- **temporal_mapping.total_cycle**: Total cycles derived from the temporal mapping, representing the number of cycles needed to execute all MAC operations across the temporal schedule.
- **cycles_per_mac**: Number of cycles per MAC operation (default = 1, but >1 for bit-serial computation).

Additionally, an **ideal_cycle** is computed for reference:
\[
\text{ideal_cycle} = \lceil \frac{\text{total_MAC_count}}{\text{total_unit_count}} \rceil \times \text{cycles_per_mac}
\]
- **total_MAC_count**: Total number of MAC operations in the layer.
- **total_unit_count**: Total number of MAC units in the accelerator’s operational array.

#### b. Stall Cycles (SS_comb)
`SS_comb` is the maximum stall cycle across all memory ports, calculated in `combine_data_transfer_rate_per_physical_port`. For each physical memory port:
\[
\text{SS_comb_port} = \text{SS_positive_sum} + \max(0, \text{SS_negative_sum} + \text{MUW_sum} - \text{MUW_union})
\]
- **SS_positive_sum**: Sum of positive stall cycles (SS > 0) for all data transfer activities on the port.
- **SS_negative_sum**: Sum of negative stall cycles (slack, SS < 0).
- **MUW_sum**: Sum of memory updating windows (MUW) for all activities.
- **MUW_union**: The union of memory updating windows, calculated as the effective overlap (see `calc_MUW_union` in the code).

Then:
\[
\text{SS_comb} = \max(\text{SS_comb_list})
\]
- **SS_comb_list**: List of `SS_comb_port` values across all ports, including a default 0 for no activity.

For each data transfer activity (e.g., `wr_in_by_low`, `rd_out_to_high`):
- **SS** = **real_data_trans_cycle** - **allowed_mem_updat_cycle**
  - **real_data_trans_cycle**: Cycles needed to transfer data based on bandwidth:
    \[
    \text{real_data_trans_cycle} = \lceil \frac{\text{data_trans_amount_per_period} \times \text{data_precision}}{\text{mem_bw}} \rceil
    \]
    - **data_trans_amount_per_period**: Data elements moved per period.
    - **data_precision**: Bits per data element.
    - **mem_bw**: Memory bandwidth (read or write, depending on direction).
  - **allowed_mem_updat_cycle**: Allowed cycles for data transfer, determined by double buffering:
    - If double buffering is true (effective memory utilization ≤ 0.5 or sufficient free space):
      \[
      \text{allowed_mem_updat_cycle} = \text{data_trans_period}
      \]
    - Otherwise:
      \[
      \text{allowed_mem_updat_cycle} = \text{inst_data_trans_window}
      \]
    - **data_trans_period**: Total period length.
    - **inst_data_trans_window**: Instantaneous transfer window without double buffering.

#### c. Data Loading Cycle
Calculated in `calc_data_loading_offloading_latency` for input operands (e.g., "I1", "I2"):
- For each memory level pair (mem_lv, mem_lv+1):
  \[
  \text{data_loading_cc_pair} = \max(\text{real_cycle}_{\text{wr_in_by_high, mem_lv}}, \text{real_cycle}_{\text{rd_out_to_low, mem_lv+1}})
  \]
- Combine based on port sharing:
  - If one input operand:
    \[
    \text{data_loading_cycle} = \text{data_loading_individual_part}
    \]
  - If two input operands (op1, op2):
    \[
    \text{data_loading_cycle} = \min(\text{possible1}, \text{possible2})
    \]
    Where:
    - \(\text{possible1} = \text{data_loading_shared_part[op1]} + \max(\text{shared[op2]} + \text{half_shared[op2]} + \text{individual[op2]}, \text{half_shared[op1]} + \text{individual[op1]})\)
    - \(\text{possible2} = \text{data_loading_shared_part[op2]} + \max(\text{shared[op1]} + \text{half_shared[op1]} + \text{individual[op1]}, \text{half_shared[op2]} + \text{individual[op2]})\)

#### d. Data Offloading Cycle
For the output operand:
\[
\text{data_offloading_cycle} = \sum_{\text{mem_lv}} \max(\text{real_cycle}_{\text{rd_out_to_high, mem_lv}}, \text{real_cycle}_{\text{wr_in_by_low, mem_lv+1}})
\]

### Excel Implementation
- **Inputs**: `total_MAC_count`, `total_unit_count`, `temporal_mapping.total_cycle`, `cycles_per_mac`, memory bandwidths, data transfer amounts, periods, and precisions per operand and level.
- **Steps**: Compute `ideal_temporal_cycle`, then `SS_comb` per port, take the maximum, add `data_loading_cycle` and `data_offloading_cycle`.

---

## 2. Word Access
Word access represents the number of memory words read or written per operand at each memory level, calculated in `calc_memory_word_access`. It accounts for four directions: `wr_in_by_low`, `rd_out_to_low`, `rd_out_to_high`, `wr_in_by_high`.

### Formula (per direction, per operand, per memory level)
\[
\text{word_access} = \lceil \frac{\text{data_elem_move_per_period} \times \text{data_precision}}{\text{min_bw}} \rceil \times \left( \frac{\text{min_bw}}{\text{max_bw}} \right) \times \text{total_period_count} \times \text{unit_count}
\]

Where:
- **data_elem_move_per_period**: Number of data elements moved per period.
- **data_precision**: Bits per data element.
- **min_bw**: Minimum memory bandwidth (granularity of access).
- **max_bw**: Maximum memory bandwidth.
- **total_period_count**: Number of periods for the data transfer.
- **unit_count**: Number of memory units at the level above (mem_lv + 1).

### Notes
- If `data_elem_move_per_period` or `data_precision` is 0, `word_access` = 0.
- The formula adjusts for memory access granularity by using `min_bw` and scales by the bandwidth ratio and parallelism (`unit_count`).

### Excel Implementation
- **Inputs**: Per operand and memory level: `data_elem_move_per_period`, `data_precision`, `min_bw`, `max_bw`, `total_period_count`, `unit_count`.
- **Output**: Four values per operand per level (one per direction).

---

## 3. Memory Utilization
Memory utilization measures how much of the physical memory is used per operand at each level, calculated in `calc_memory_utilization`. Two versions exist: individual and shared (considering memory sharing).

### Formula (Individual Utilization)
\[
\text{mem_utilization} = \frac{\text{data_bit_per_level_unrolled}}{\text{mem_size}}
\]
- **data_bit_per_level_unrolled**: Total bits required at the memory level, including unrolling.
- **mem_size**: Physical memory size in bits.

### Effective Utilization
\[
\text{effective_mem_utilization} = \frac{\text{effective_data_bit}}{\text{mem_size}}
\]
- **effective_data_bit**: Bits of unique data (excluding copies in parallel memories).

### Shared Utilization
For shared memory levels:
\[
\text{mem_utilization_shared} = \sum_{\text{shared_ops}} \text{mem_utili_individual[op]}
\]
- Summed over operands sharing the memory, ensuring ≤ 1.

### Excel Implementation
- **Inputs**: `data_bit_per_level_unrolled`, `effective_data_bit`, `mem_size` per operand per level, and memory sharing list.
- **Output**: Utilization per operand per level, adjusted for sharing.

---

## 4. Spatial Utilization
Spatial utilization, specifically MAC spatial utilization, measures how effectively the MAC array is used spatially, calculated in `calc_overall_latency`.

### Formula
\[
\text{MAC_spatial_utilization} = \frac{\text{ideal_cycle}}{\text{ideal_temporal_cycle}}
\]
- **ideal_cycle**: Minimum cycles if all MACs are fully utilized.
- **ideal_temporal_cycle**: Cycles based on the temporal mapping, reflecting spatial mapping efficiency.

### Additional Utilization Metrics
- **MAC_utilization0**: \(\frac{\text{ideal_cycle}}{\text{latency_total0}}\) (without loading/offloading).
- **MAC_utilization1**: \(\frac{\text{ideal_cycle}}{\text{latency_total1}}\) (with loading).
- **MAC_utilization2**: \(\frac{\text{ideal_cycle}}{\text{latency_total2}}\) (with loading and offloading).

### Excel Implementation
- **Inputs**: `ideal_cycle`, `ideal_temporal_cycle`, `latency_total0`, `latency_total1`, `latency_total2`.
- **Output**: Utilization percentages.

---

## Verification in Excel
To verify if the exploration is event-driven:
1. **Inputs**: Extract layer parameters (e.g., `total_MAC_count`), accelerator specs (e.g., `total_unit_count`, memory bandwidths), and mapping details (e.g., `temporal_mapping.total_cycle`, data transfer amounts).
2. **Calculate**: Use the formulas above to compute each metric.
3. **Compare**: Match results against ZigZag outputs. Event-driven exploration may show different stall patterns (SS_comb) or data transfer timings due to dynamic scheduling, but the formulas hold.

These formulas encapsulate the ZigZag cost model’s logic, enabling manual verification in Excel regardless of the exploration type.


#### Question 1:

With the parameters mentioned below:
"    #Workload configuration
    B = int(input("Enter the value for B: "))
    C = int(input("Enter the value for C: "))
    IX = int(input("Enter the value for IX: "))
    IY = int(input("Enter the value for IY: "))
    K = int(input("Enter the value for K: "))
    FX = int(input("Enter the value for FX: "))
    FY = int(input("Enter the value for FY: "))
    P = int(input("Enter the value for P: "))
    S = int(input("Enter the value for S: "))
    
    output_height = (IY + 2 * P - FY) // S + 1
    output_width = (IX + 2 * P - FX) // S + 1
    
    #Hardware configuration
    num_levels = int(input("Enter number of memory levels: "))
    memory_levels = []
    for i in range(num_levels):
        print(f"Enter configuration for memory level {i+1}:")
        mem_config = {
            "size": int(input("  size: ")),
            "r_bw": int(input("  r_bw: ")),
            "w_bw": int(input("  w_bw: ")),
            "r_cost": float(input("  r_cost: ")),
            "w_cost": float(input("  w_cost: ")),
            "area": int(input("  area: ")),
            "r_port": int(input("  r_port: ")),
            "w_port": int(input("  w_port: ")),
            "rw_port": int(input("  rw_port: ")),
            "latency": int(input("  latency: "))
        }
        memory_levels.append(mem_config)
    
    #Worload specification and print them
    input_config = {'Batch': B, 'Input height': IY, 'Input width': IX, 'Input channels': C}
    weight_config = {'Input channels': C, 'Weight height': FY, 'Weight width': FX, 'Output channels': K}
    output_config = {'Batch': B, 'Output height': output_height, 'Output width': output_width, 'Output channels': K}
        
    # Print the workload configuration
    print("Test case configuration : \n")
    print("Input Matrix:\n", input_config)
    print("Weight Matrix:\n",weight_config)
    print("output_size:\n", output_config)
    
    # Print the memory configuration
    print("\nMemory configuration:")
    for i, mem_config in enumerate(memory_levels, start=1):
        print(f"\nMemory Level {i}:")
        for key, value in mem_config.items():
            print(f"  {key:<8}: {value}")
            "

Can you help me calculate the parameters shown below for the latency as it as how zigzag calculates?
Here is how zigzag calcualted the latency 
"## 1. Latency
Latency in the ZigZag cost model represents the total time (in cycles) to complete the computation of a layer, including computation time, stalls due to memory bandwidth limitations, and data loading/offloading overheads. The final latency is stored as `latency_total2`, which includes all components. The calculation is performed in the `calc_overall_latency` method, with intermediate steps from other methods.

### Formula
\[
\text{Latency} = \text{ideal_temporal_cycle} + \text{SS_comb} + \text{data_loading_cycle} + \text{data_offloading_cycle}
\]

Where:
- **ideal_temporal_cycle**: The ideal number of cycles for computation based on the temporal mapping, assuming no stalls.
- **SS_comb**: Combined stall cycles due to memory bandwidth constraints.
- **data_loading_cycle**: Cycles required to load initial input data.
- **data_offloading_cycle**: Cycles required to offload final output data.

### Detailed Breakdown
#### a. Ideal Temporal Cycle
\[
\text{ideal_temporal_cycle} = \text{temporal_mapping.total_cycle} \times \text{cycles_per_mac}
\]
- **temporal_mapping.total_cycle**: Total cycles derived from the temporal mapping, representing the number of cycles needed to execute all MAC operations across the temporal schedule.
- **cycles_per_mac**: Number of cycles per MAC operation (default = 1, but >1 for bit-serial computation).

Additionally, an **ideal_cycle** is computed for reference:
\[
\text{ideal_cycle} = \lceil \frac{\text{total_MAC_count}}{\text{total_unit_count}} \rceil \times \text{cycles_per_mac}
\]
- **total_MAC_count**: Total number of MAC operations in the layer.
- **total_unit_count**: Total number of MAC units in the accelerator’s operational array.

#### b. Stall Cycles (SS_comb)
`SS_comb` is the maximum stall cycle across all memory ports, calculated in `combine_data_transfer_rate_per_physical_port`. For each physical memory port:
\[
\text{SS_comb_port} = \text{SS_positive_sum} + \max(0, \text{SS_negative_sum} + \text{MUW_sum} - \text{MUW_union})
\]
- **SS_positive_sum**: Sum of positive stall cycles (SS > 0) for all data transfer activities on the port.
- **SS_negative_sum**: Sum of negative stall cycles (slack, SS < 0).
- **MUW_sum**: Sum of memory updating windows (MUW) for all activities.
- **MUW_union**: The union of memory updating windows, calculated as the effective overlap (see `calc_MUW_union` in the code).

Then:
\[
\text{SS_comb} = \max(\text{SS_comb_list})
\]
- **SS_comb_list**: List of `SS_comb_port` values across all ports, including a default 0 for no activity.

For each data transfer activity (e.g., `wr_in_by_low`, `rd_out_to_high`):
- **SS** = **real_data_trans_cycle** - **allowed_mem_updat_cycle**
  - **real_data_trans_cycle**: Cycles needed to transfer data based on bandwidth:
    \[
    \text{real_data_trans_cycle} = \lceil \frac{\text{data_trans_amount_per_period} \times \text{data_precision}}{\text{mem_bw}} \rceil
    \]
    - **data_trans_amount_per_period**: Data elements moved per period.
    - **data_precision**: Bits per data element.
    - **mem_bw**: Memory bandwidth (read or write, depending on direction).
  - **allowed_mem_updat_cycle**: Allowed cycles for data transfer, determined by double buffering:
    - If double buffering is true (effective memory utilization ≤ 0.5 or sufficient free space):
      \[
      \text{allowed_mem_updat_cycle} = \text{data_trans_period}
      \]
    - Otherwise:
      \[
      \text{allowed_mem_updat_cycle} = \text{inst_data_trans_window}
      \]
    - **data_trans_period**: Total period length.
    - **inst_data_trans_window**: Instantaneous transfer window without double buffering.

#### c. Data Loading Cycle
Calculated in `calc_data_loading_offloading_latency` for input operands (e.g., "I1", "I2"):
- For each memory level pair (mem_lv, mem_lv+1):
  \[
  \text{data_loading_cc_pair} = \max(\text{real_cycle}_{\text{wr_in_by_high, mem_lv}}, \text{real_cycle}_{\text{rd_out_to_low, mem_lv+1}})
  \]
- Combine based on port sharing:
  - If one input operand:
    \[
    \text{data_loading_cycle} = \text{data_loading_individual_part}
    \]
  - If two input operands (op1, op2):
    \[
    \text{data_loading_cycle} = \min(\text{possible1}, \text{possible2})
    \]
    Where:
    - \(\text{possible1} = \text{data_loading_shared_part[op1]} + \max(\text{shared[op2]} + \text{half_shared[op2]} + \text{individual[op2]}, \text{half_shared[op1]} + \text{individual[op1]})\)
    - \(\text{possible2} = \text{data_loading_shared_part[op2]} + \max(\text{shared[op1]} + \text{half_shared[op1]} + \text{individual[op1]}, \text{half_shared[op2]} + \text{individual[op2]})\)

#### d. Data Offloading Cycle
For the output operand:
\[
\text{data_offloading_cycle} = \sum_{\text{mem_lv}} \max(\text{real_cycle}_{\text{rd_out_to_high, mem_lv}}, \text{real_cycle}_{\text{wr_in_by_low, mem_lv+1}})
\]

### Excel Implementation
- **Inputs**: `total_MAC_count`, `total_unit_count`, `temporal_mapping.total_cycle`, `cycles_per_mac`, memory bandwidths, data transfer amounts, periods, and precisions per operand and level.
- **Steps**: Compute `ideal_temporal_cycle`, then `SS_comb` per port, take the maximum, add `data_loading_cycle` and `data_offloading_cycle`.
"

#### Answer: 
Let’s create a Python function to calculate the **latency** based on the ZigZag cost model, using the inputs you provided (workload configuration and hardware memory hierarchy) and the detailed formulas we derived earlier. The function will compute the total latency (`latency_total2`) as per ZigZag’s methodology, incorporating `ideal_temporal_cycle`, `SS_comb` (stall cycles), `data_loading_cycle`, and `data_offloading_cycle`. Below, I’ll explain how to derive each component from your inputs and implement the function.

---

### Step 1: Understanding the Inputs and Mapping to ZigZag’s Formula

From your code snippet:
- **Workload Configuration**:
  - `B`: Batch size.
  - `C`: Input channels.
  - `IX`, `IY`: Input width and height.
  - `K`: Output channels.
  - `FX`, `FY`: Filter width and height.
  - `P`: Padding.
  - `S`: Stride.
  - Derived: `output_height = (IY + 2 * P - FY) // S + 1`, `output_width = (IX + 2 * P - FX) // S + 1`.

- **Hardware Configuration**:
  - `memory_levels`: List of dictionaries with `size`, `r_bw` (read bandwidth), `w_bw` (write bandwidth), etc., for each memory level.
  - Missing: Number of MAC units (`total_unit_count`), which we’ll assume or prompt for.

To compute latency:
- **total_MAC_count**: \( B \times K \times C \times \text{output_height} \times \text{output_width} \times FY \times FX \).
- **ideal_temporal_cycle**: Requires `temporal_mapping.total_cycle`, which isn’t directly provided. We’ll estimate it or assume a full temporal unroll initially.
- **SS_comb**, **data_loading_cycle**, **data_offloading_cycle**: Depend on memory bandwidth, data sizes, and transfer schedules, derived from workload and memory config.

---

### Step 2: Assumptions and Simplifications

Since your inputs don’t include the spatial and temporal mappings explicitly (unlike the earlier ZigZag example), we’ll make some assumptions:
- **Temporal Mapping**: Assume all loops (B, K, C, OY, OX, FY, FX) are fully unrolled temporally unless specified, so `temporal_mapping.total_cycle` is the product of loop sizes.
- **Spatial Mapping**: Assume no spatial unrolling (single MAC unit, `total_unit_count = 1`) unless you add this input.
- **Data Precision**: Assume 8-bit precision for inputs (I), weights (W), and outputs (O) unless specified.
- **Double Buffering**: Check memory utilization to determine if it’s ≤ 0.5 for each level.

---

### Step 3: Python Implementation

Here’s a function that calculates latency based on your inputs, incorporating ZigZag’s formulas:

```python
import math

def compute_zigzag_latency():
    # --- Manual Input Collection ---
    # Workload Configuration
    B = int(input("Enter the value for B (Batch size): "))
    C = int(input("Enter the value for C (Input channels): "))
    IX = int(input("Enter the value for IX (Input width): "))
    IY = int(input("Enter the value for IY (Input height): "))
    K = int(input("Enter the value for K (Output channels): "))
    FX = int(input("Enter the value for FX (Filter width): "))
    FY = int(input("Enter the value for FY (Filter height): "))
    P = int(input("Enter the value for P (Padding): "))
    S = int(input("Enter the value for S (Stride): "))
    
    # Derived Output Dimensions
    output_height = (IY + 2 * P - FY) // S + 1
    output_width = (IX + 2 * P - FX) // S + 1

    # Hardware Configuration
    total_unit_count = int(input("Enter the number of MAC units (total_unit_count): "))
    cycles_per_mac = int(input("Enter cycles per MAC operation (default=1): ") or 1)
    num_levels = int(input("Enter number of memory levels: "))
    memory_levels = []
    for i in range(num_levels):
        print(f"Enter configuration for memory level {i+1}:")
        mem_config = {
            "size": int(input("  size (bits): ")),
            "r_bw": int(input("  r_bw (bits/cycle): ")),
            "w_bw": int(input("  w_bw (bits/cycle): ")),
            "r_port": int(input("  r_port: ")),
            "w_port": int(input("  w_port: ")),
            "rw_port": int(input("  rw_port: "))
        }
        memory_levels.append(mem_config)

    # Assume operand precision (you can add inputs for these if needed)
    precision_I = 8  # Input precision (bits)
    precision_W = 8  # Weight precision (bits)
    precision_O = 8  # Output precision (bits)

    # --- Compute Intermediate Values ---
    # Total MAC Count
    total_MAC_count = B * K * C * output_height * output_width * FY * FX

    # Ideal Cycle (minimum cycles assuming full MAC utilization)
    ideal_cycle = math.ceil(total_MAC_count / total_unit_count) * cycles_per_mac

    # Temporal Mapping Total Cycle (assume full unroll if no mapping provided)
    temporal_mapping_total_cycle = B * K * C * output_height * output_width * FY * FX
    ideal_temporal_cycle = temporal_mapping_total_cycle * cycles_per_mac

    # Data Sizes
    input_size_bits = B * C * IY * IX * precision_I  # Input data size
    weight_size_bits = K * C * FY * FX * precision_W  # Weight data size
    output_size_bits = B * K * output_height * output_width * precision_O  # Output data size

    # --- Stall Cycles (SS_comb) ---
    SS_comb = 0  # Default to 0, refine with bandwidth constraints
    for level_idx, mem in enumerate(memory_levels):
        # Assume lowest level serves MACs, higher levels feed lower levels
        mem_bw = min(mem["r_bw"], mem["w_bw"])  # Use minimum of read/write bandwidth
        if level_idx == 0:  # Lowest level (e.g., feeding MACs)
            data_trans_amount = total_MAC_count * precision_I  # Input data to MACs
            real_data_trans_cycle = math.ceil(data_trans_amount / mem_bw)
            allowed_mem_updat_cycle = ideal_temporal_cycle  # Assume full cycle if double buffering
            # Check memory utilization for double buffering
            util = max(input_size_bits, weight_size_bits) / mem["size"]
            if util > 0.5:
                allowed_mem_updat_cycle = ideal_temporal_cycle // (FY * FX)  # Simplified window
            SS = max(0, real_data_trans_cycle - allowed_mem_updat_cycle)
            SS_comb = max(SS_comb, SS)

    # --- Data Loading Cycle ---
    data_loading_cycle = 0
    for level_idx in range(1, num_levels):  # Between memory levels
        lower_level = memory_levels[level_idx - 1]
        upper_level = memory_levels[level_idx]
        data_size = max(input_size_bits, weight_size_bits)  # Assume max of inputs/weights
        real_cycle_lower = math.ceil(data_size / lower_level["w_bw"])
        real_cycle_upper = math.ceil(data_size / upper_level["r_bw"])
        data_loading_cc_pair = max(real_cycle_lower, real_cycle_upper)
        data_loading_cycle += data_loading_cc_pair  # Simplified: assume serial transfer

    # --- Data Offloading Cycle ---
    data_offloading_cycle = 0
    for level_idx in range(1, num_levels):
        lower_level = memory_levels[level_idx - 1]
        upper_level = memory_levels[level_idx]
        real_cycle_lower = math.ceil(output_size_bits / lower_level["r_bw"])
        real_cycle_upper = math.ceil(output_size_bits / upper_level["w_bw"])
        data_offloading_cc_pair = max(real_cycle_lower, real_cycle_upper)
        data_offloading_cycle += data_offloading_cc_pair

    # --- Compute Total Latency ---
    latency_total = ideal_temporal_cycle + SS_comb + data_loading_cycle + data_offloading_cycle

    # --- Print Results ---
    print("\nComputed Latency Components:")
    print(f"Ideal Temporal Cycle: {ideal_temporal_cycle} cycles")
    print(f"Stall Cycles (SS_comb): {SS_comb} cycles")
    print(f"Data Loading Cycle: {data_loading_cycle} cycles")
    print(f"Data Offloading Cycle: {data_offloading_cycle} cycles")
    print(f"Total Latency: {latency_total} cycles")

    return latency_total

# Run the function
if __name__ == "__main__":
    latency = compute_zigzag_latency()
```

---

### Step 4: Explanation of the Implementation

#### Inputs and Derivations
- **Workload Inputs**:
  - Directly collected from user input as per your code snippet.
  - `total_MAC_count` is calculated based on the convolutional layer’s dimensions.

- **Hardware Inputs**:
  - Memory hierarchy details are collected level-by-level.
  - `total_unit_count` is added as an extra input since it’s not in your snippet but is critical for ZigZag’s latency.

- **Temporal Mapping**:
  - Since no mapping is provided, we assume a full temporal unroll (all loops executed sequentially). If you have a specific mapping, adjust `temporal_mapping_total_cycle` accordingly (e.g., product of mapped loop sizes).

#### Formula Components
1. **Ideal Temporal Cycle**:
   - Computed as `temporal_mapping_total_cycle * cycles_per_mac`.
   - Here, `temporal_mapping_total_cycle` is the full unroll of all loops, which may overestimate if partial mapping is used.

2. **Stall Cycles (SS_comb)**:
   - Estimated for the lowest memory level feeding MACs.
   - `real_data_trans_cycle` uses input data size and bandwidth.
   - `allowed_mem_updat_cycle` adjusts based on double buffering (utilization ≤ 0.5).
   - Simplified to take the max stall across levels; refine with port-specific data if available.

3. **Data Loading Cycle**:
   - Computed between memory levels (e.g., DRAM to SRAM).
   - Takes the maximum of write cycles to lower level and read cycles from upper level.
   - Assumes serial transfer; adjust for parallel ports if specified.

4. **Data Offloading Cycle**:
   - Similar to loading but for output data moving up the hierarchy.
   - Uses output size and bandwidth constraints.

#### Output
- Prints all components and the total latency for verification against ZigZag’s output.

---

### Step 5: How to Use This Function

1. **Run the Script**:
   - Execute it in a Python environment.
   - Enter values as prompted (e.g., from your workload and architecture).

2. **Example Input**:
   - Workload: `B=1`, `C=1`, `IX=5`, `IY=5`, `K=8`, `FX=3`, `FY=3`, `P=0`, `S=1`.
     - Output: `output_height = (5 - 3) + 1 = 3`, `output_width = 3`.
     - `total_MAC_count = 1 * 8 * 1 * 3 * 3 * 3 * 3 = 648`.
   - Hardware: `total_unit_count=1`, `cycles_per_mac=1`.
   - Memory Levels: 
     - Level 1: `size=2097152`, `r_bw=128`, `w_bw=128`, `r_port=1`, `w_port=1`, `rw_port=0`.
     - Level 2: `size=8388608`, `r_bw=16`, `w_bw=16`, `r_port=1`, `w_port=1`, `rw_port=0`.

3. **Verify**:
   - Compare the printed `Total Latency` with ZigZag’s `latency_total2` (e.g., from `"computation" + "data_onloading" + "data_offloading"`).

---

### Step 6: Refinements and Notes

- **Temporal Mapping**: If you have a specific mapping (e.g., loop ordering), input `temporal_mapping_total_cycle` manually instead of computing it as a full unroll.
- **Stalls**: The stall calculation is simplified. For accuracy, you’d need data transfer schedules (e.g., `data_trans_amount_per_period`) from the mapping, which ZigZag derives internally.
- **Ports**: This assumes single-port transfers; adjust for multiple ports or shared operands if your architecture specifies them.
- **Precision**: Hardcoded at 8 bits; add inputs for `precision_I`, `precision_W`, `precision_O` if they vary.

Let me know if you want to refine this further with specific mapping data or test it with your example inputs!

#### Question 4

Can the below fucntion be modified
"```python
import math

def compute_zigzag_latency():
    # --- Manual Input Collection ---
    # Workload Configuration
    B = int(input("Enter the value for B (Batch size): "))
    C = int(input("Enter the value for C (Input channels): "))
    IX = int(input("Enter the value for IX (Input width): "))
    IY = int(input("Enter the value for IY (Input height): "))
    K = int(input("Enter the value for K (Output channels): "))
    FX = int(input("Enter the value for FX (Filter width): "))
    FY = int(input("Enter the value for FY (Filter height): "))
    P = int(input("Enter the value for P (Padding): "))
    S = int(input("Enter the value for S (Stride): "))
    
    # Derived Output Dimensions
    output_height = (IY + 2 * P - FY) // S + 1
    output_width = (IX + 2 * P - FX) // S + 1

    # Hardware Configuration
    total_unit_count = int(input("Enter the number of MAC units (total_unit_count): "))
    cycles_per_mac = int(input("Enter cycles per MAC operation (default=1): ") or 1)
    num_levels = int(input("Enter number of memory levels: "))
    memory_levels = []
    for i in range(num_levels):
        print(f"Enter configuration for memory level {i+1}:")
        mem_config = {
            "size": int(input("  size (bits): ")),
            "r_bw": int(input("  r_bw (bits/cycle): ")),
            "w_bw": int(input("  w_bw (bits/cycle): ")),
            "r_port": int(input("  r_port: ")),
            "w_port": int(input("  w_port: ")),
            "rw_port": int(input("  rw_port: "))
        }
        memory_levels.append(mem_config)

    # Assume operand precision (you can add inputs for these if needed)
    precision_I = 8  # Input precision (bits)
    precision_W = 8  # Weight precision (bits)
    precision_O = 8  # Output precision (bits)

    # --- Compute Intermediate Values ---
    # Total MAC Count
    total_MAC_count = B * K * C * output_height * output_width * FY * FX

    # Ideal Cycle (minimum cycles assuming full MAC utilization)
    ideal_cycle = math.ceil(total_MAC_count / total_unit_count) * cycles_per_mac

    # Temporal Mapping Total Cycle (assume full unroll if no mapping provided)
    temporal_mapping_total_cycle = B * K * C * output_height * output_width * FY * FX
    ideal_temporal_cycle = temporal_mapping_total_cycle * cycles_per_mac

    # Data Sizes
    input_size_bits = B * C * IY * IX * precision_I  # Input data size
    weight_size_bits = K * C * FY * FX * precision_W  # Weight data size
    output_size_bits = B * K * output_height * output_width * precision_O  # Output data size

    # --- Stall Cycles (SS_comb) ---
    SS_comb = 0  # Default to 0, refine with bandwidth constraints
    for level_idx, mem in enumerate(memory_levels):
        # Assume lowest level serves MACs, higher levels feed lower levels
        mem_bw = min(mem["r_bw"], mem["w_bw"])  # Use minimum of read/write bandwidth
        if level_idx == 0:  # Lowest level (e.g., feeding MACs)
            data_trans_amount = total_MAC_count * precision_I  # Input data to MACs
            real_data_trans_cycle = math.ceil(data_trans_amount / mem_bw)
            allowed_mem_updat_cycle = ideal_temporal_cycle  # Assume full cycle if double buffering
            # Check memory utilization for double buffering
            util = max(input_size_bits, weight_size_bits) / mem["size"]
            if util > 0.5:
                allowed_mem_updat_cycle = ideal_temporal_cycle // (FY * FX)  # Simplified window
            SS = max(0, real_data_trans_cycle - allowed_mem_updat_cycle)
            SS_comb = max(SS_comb, SS)

    # --- Data Loading Cycle ---
    data_loading_cycle = 0
    for level_idx in range(1, num_levels):  # Between memory levels
        lower_level = memory_levels[level_idx - 1]
        upper_level = memory_levels[level_idx]
        data_size = max(input_size_bits, weight_size_bits)  # Assume max of inputs/weights
        real_cycle_lower = math.ceil(data_size / lower_level["w_bw"])
        real_cycle_upper = math.ceil(data_size / upper_level["r_bw"])
        data_loading_cc_pair = max(real_cycle_lower, real_cycle_upper)
        data_loading_cycle += data_loading_cc_pair  # Simplified: assume serial transfer

    # --- Data Offloading Cycle ---
    data_offloading_cycle = 0
    for level_idx in range(1, num_levels):
        lower_level = memory_levels[level_idx - 1]
        upper_level = memory_levels[level_idx]
        real_cycle_lower = math.ceil(output_size_bits / lower_level["r_bw"])
        real_cycle_upper = math.ceil(output_size_bits / upper_level["w_bw"])
        data_offloading_cc_pair = max(real_cycle_lower, real_cycle_upper)
        data_offloading_cycle += data_offloading_cc_pair

    # --- Compute Total Latency ---
    latency_total = ideal_temporal_cycle + SS_comb + data_loading_cycle + data_offloading_cycle

    # --- Print Results ---
    print("\nComputed Latency Components:")
    print(f"Ideal Temporal Cycle: {ideal_temporal_cycle} cycles")
    print(f"Stall Cycles (SS_comb): {SS_comb} cycles")
    print(f"Data Loading Cycle: {data_loading_cycle} cycles")
    print(f"Data Offloading Cycle: {data_offloading_cycle} cycles")
    print(f"Total Latency: {latency_total} cycles")

    return latency_total

# Run the function
if __name__ == "__main__":
    latency = compute_zigzag_latency()
```"

Accorning to these input main fucntion shown below:

```python
def parse_zigzag_mapping(mapping_str):
    lines = mapping_str.strip().split('\n')
    temporal_loops = []
    spatial_loops = []
    current_section = None

    for line in lines:
        line = line.strip()
        if 'Temporal Loops' in line:
            current_section = 'temporal'
            continue
        elif 'Spatial Loops' in line:
            current_section = 'spatial'
            continue
        elif not line or '====' in line or '----' in line:
            continue  # Skip empty lines or separators

        # Process loop lines
        if ('for' in line or 'parfor' in line) and 'in' in line and '[' in line:
            # Determine loop type
            loop_type = 'for' if 'for' in line and 'parfor' not in line else 'parfor'
            
            # Extract variable name and range
            parts = line.split('in', 1)  # Split only on first 'in'
            if len(parts) != 2:
                continue
            var_name = parts[0].replace(loop_type, '').strip()
            range_part = parts[1].split(')')[0].strip()  # Get up to ')'
            range_str = range_part.replace('[', '').replace(')', '')
            try:
                start, end = map(int, range_str.split(','))
            except ValueError:
                continue  # Skip if range parsing fails
            
            loop = {'var': var_name, 'start': start, 'end': end}

            # Assign to appropriate loop list
            if current_section == 'temporal' and loop_type == 'for':
                temporal_loops.append(loop)
            elif current_section == 'spatial' and loop_type == 'parfor':
                spatial_loops.append(loop)

    return {
        'temporal_loops': temporal_loops,
        'spatial_loops': spatial_loops,
        'memory_access' : {
        'O': 'sram_2Mb_I2_O',
        'W': 'sram_2Mb_I2_O',
        'I': 'dram_8Mb_I1'
    }
    }

#Inputs
def main():
    #Workload configuration
    B = int(input("Enter the value for B: "))
    C = int(input("Enter the value for C: "))
    IX = int(input("Enter the value for IX: "))
    IY = int(input("Enter the value for IY: "))
    K = int(input("Enter the value for K: "))
    FX = int(input("Enter the value for FX: "))
    FY = int(input("Enter the value for FY: "))
    P = int(input("Enter the value for P: "))
    S = int(input("Enter the value for S: "))
    
    output_height = (IY + 2 * P - FY) // S + 1
    output_width = (IX + 2 * P - FX) // S + 1
    
    #Hardware configuration
    num_levels = int(input("Enter number of memory levels: "))
    memory_levels = []
    for i in range(num_levels):
        print(f"Enter configuration for memory level {i+1}:")
        mem_config = {
            "size": int(input("  size: ")),
            "r_bw": int(input("  r_bw: ")),
            "w_bw": int(input("  w_bw: ")),
            "r_cost": float(input("  r_cost: ")),
            "w_cost": float(input("  w_cost: ")),
            "area": int(input("  area: ")),
            "r_port": int(input("  r_port: ")),
            "w_port": int(input("  w_port: ")),
            "rw_port": int(input("  rw_port: ")),
            "latency": int(input("  latency: "))
        }
        memory_levels.append(mem_config)
        
    #Mapping configuration
    zigzag_mapping = """
    Loop ordering for default_analysis_by_zigzag
    =====================================================================================
    Temporal Loops                 O                 W                 I                 
    =====================================================================================
    for FX in [0, 3):              sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
      for FY in [0, 3):            sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
        for OX in [0, 3):          sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
          for OY in [0, 3):        sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
    =====================================================================================
    Spatial Loops                                                                        
    =====================================================================================
            parfor K in [0, 8):                                                          
    -------------------------------------------------------------------------------------
    """
    
    
    #Worload specification and print them
    input_config = {'Batch': B, 'Input height': IY, 'Input width': IX, 'Input channels': C}
    weight_config = {'Input channels': C, 'Weight height': FY, 'Weight width': FX, 'Output channels': K}
    output_config = {'Batch': B, 'Output height': output_height, 'Output width': output_width, 'Output channels': K}
        
    # Print the workload configuration
    print("Test case configuration : \n")
    print("Input Matrix:\n", input_config)
    print("Weight Matrix:\n",weight_config)
    print("output_size:\n", output_config)
    
    # Print the memory configuration
    print("\nMemory configuration:")
    for i, mem_config in enumerate(memory_levels, start=1):
        print(f"\nMemory Level {i}:")
        for key, value in mem_config.items():
            print(f"  {key:<8}: {value}")
            
    # Print the mapping configuration
    parsed_mapping = parse_zigzag_mapping(zigzag_mapping)
    print("\nMapping configuration:")
    print(parsed_mapping)
```

Can you generate a new script the takes the inputs from the main function runs a cal_latency fucntion which calculates all the different types of latency and give the final latency according to the zigzag cost fucntion which are alredy modelled above.


#### Answer 2:
To address your query, I'll modify the provided `compute_zigzag_latency` function to work seamlessly with the inputs collected in your `main` function. The goal is to create a new script that takes inputs from `main`, passes them to a `calc_latency` function (renamed for clarity), calculates various latency components based on the ZigZag cost model, and returns the total latency. Below is the complete solution.

### Solution Overview
- **Input Collection**: The `main` function gathers workload, hardware, and mapping configurations.
- **Latency Calculation**: The `calc_latency` function computes ideal temporal cycles, stall cycles, data loading/offloading cycles, and total latency using the ZigZag cost model.
- **Integration**: The script integrates parsing of the ZigZag mapping and ensures all parameters are passed correctly.

Here’s the new script:

```python
import math

def parse_zigzag_mapping(mapping_str):
    """Parse the ZigZag mapping string into temporal and spatial loops."""
    lines = mapping_str.strip().split('\n')
    temporal_loops = []
    spatial_loops = []
    current_section = None

    for line in lines:
        line = line.strip()
        if 'Temporal Loops' in line:
            current_section = 'temporal'
            continue
        elif 'Spatial Loops' in line:
            current_section = 'spatial'
            continue
        elif not line or '====' in line or '----' in line:
            continue

        if ('for' in line or 'parfor' in line) and 'in' in line and '[' in line:
            loop_type = 'for' if 'for' in line and 'parfor' not in line else 'parfor'
            parts = line.split('in', 1)
            if len(parts) != 2:
                continue
            var_name = parts[0].replace(loop_type, '').strip()
            range_part = parts[1].split(')')[0].strip()
            range_str = range_part.replace('[', '').replace(')', '')
            try:
                start, end = map(int, range_str.split(','))
            except ValueError:
                continue
            loop = {'var': var_name, 'start': start, 'end': end}
            if current_section == 'temporal' and loop_type == 'for':
                temporal_loops.append(loop)
            elif current_section == 'spatial' and loop_type == 'parfor':
                spatial_loops.append(loop)

    return {
        'temporal_loops': temporal_loops,
        'spatial_loops': spatial_loops,
        'memory_access': {'O': 'sram_2Mb_I2_O', 'W': 'sram_2Mb_I2_O', 'I': 'dram_8Mb_I1'}
    }

def calc_latency(B, C, IX, IY, K, FX, FY, P, S, total_unit_count, cycles_per_mac, memory_levels, parsed_mapping):
    """Calculate latency components and total latency based on the ZigZag cost model."""
    # Derived Output Dimensions
    output_height = (IY + 2 * P - FY) // S + 1
    output_width = (IX + 2 * P - FX) // S + 1

    # Operand Precision (fixed for simplicity; can be parameterized if needed)
    precision_I = 8  # Input precision (bits)
    precision_W = 8  # Weight precision (bits)
    precision_O = 8  # Output precision (bits)

    # ### Compute Intermediate Values ###
    # Total MAC Count
    total_MAC_count = B * K * C * output_height * output_width * FY * FX

    # Ideal Cycle (minimum cycles with full MAC utilization)
    ideal_cycle = math.ceil(total_MAC_count / total_unit_count) * cycles_per_mac

    # Temporal Mapping Total Cycle
    temporal_loops = parsed_mapping['temporal_loops']
    temporal_mapping_total_cycle = 1
    for loop in temporal_loops:
        temporal_mapping_total_cycle *= (loop['end'] - loop['start'])
    ideal_temporal_cycle = temporal_mapping_total_cycle * cycles_per_mac

    # Data Sizes
    input_size_bits = B * C * IY * IX * precision_I
    weight_size_bits = K * C * FY * FX * precision_W
    output_size_bits = B * K * output_height * output_width * precision_O

    # ### Stall Cycles (SS_comb) ###
    SS_comb = 0
    for level_idx, mem in enumerate(memory_levels):
        mem_bw = min(mem["r_bw"], mem["w_bw"])
        if level_idx == 0:  # Lowest level feeds MACs
            data_trans_amount = total_MAC_count * precision_I
            real_data_trans_cycle = math.ceil(data_trans_amount / mem_bw)
            allowed_mem_updat_cycle = ideal_temporal_cycle  # Default with double buffering
            util = max(input_size_bits, weight_size_bits) / mem["size"]
            if util > 0.5:
                allowed_mem_updat_cycle = ideal_temporal_cycle // (FY * FX)  # Simplified window
            SS = max(0, real_data_trans_cycle - allowed_mem_updat_cycle)
            SS_comb = max(SS_comb, SS)

    # ### Data Loading Cycle ###
    data_loading_cycle = 0
    for level_idx in range(1, len(memory_levels)):
        lower_level = memory_levels[level_idx - 1]
        upper_level = memory_levels[level_idx]
        data_size = max(input_size_bits, weight_size_bits)
        real_cycle_lower = math.ceil(data_size / lower_level["w_bw"])
        real_cycle_upper = math.ceil(data_size / upper_level["r_bw"])
        data_loading_cc_pair = max(real_cycle_lower, real_cycle_upper)
        data_loading_cycle += data_loading_cc_pair

    # ### Data Offloading Cycle ###
    data_offloading_cycle = 0
    for level_idx in range(1, len(memory_levels)):
        lower_level = memory_levels[level_idx - 1]
        upper_level = memory_levels[level_idx]
        real_cycle_lower = math.ceil(output_size_bits / lower_level["r_bw"])
        real_cycle_upper = math.ceil(output_size_bits / upper_level["w_bw"])
        data_offloading_cc_pair = max(real_cycle_lower, real_cycle_upper)
        data_offloading_cycle += data_offloading_cc_pair

    # ### Total Latency ###
    latency_total = ideal_temporal_cycle + SS_comb + data_loading_cycle + data_offloading_cycle

    # ### Print Results ###
    print("\nComputed Latency Components:")
    print(f"Ideal Temporal Cycle: {ideal_temporal_cycle} cycles")
    print(f"Stall Cycles (SS_comb): {SS_comb} cycles")
    print(f"Data Loading Cycle: {data_loading_cycle} cycles")
    print(f"Data Offloading Cycle: {data_offloading_cycle} cycles")
    print(f"Total Latency: {latency_total} cycles")

    return latency_total

def main():
    """Collect inputs and compute latency using the ZigZag cost model."""
    # ### Workload Configuration ###
    B = int(input("Enter the value for B: "))
    C = int(input("Enter the value for C: "))
    IX = int(input("Enter the value for IX: "))
    IY = int(input("Enter the value for IY: "))
    K = int(input("Enter the value for K: "))
    FX = int(input("Enter the value for FX: "))
    FY = int(input("Enter the value for FY: "))
    P = int(input("Enter the value for P: "))
    S = int(input("Enter the value for S: "))
    output_height = (IY + 2 * P - FY) // S + 1
    output_width = (IX + 2 * P - FX) // S + 1

    # ### Hardware Configuration ###
    total_unit_count = int(input("Enter the number of MAC units (total_unit_count): "))
    cycles_per_mac = int(input("Enter cycles per MAC operation (default=1): ") or 1)
    num_levels = int(input("Enter number of memory levels: "))
    memory_levels = []
    for i in range(num_levels):
        print(f"Enter configuration for memory level {i+1}:")
        mem_config = {
            "size": int(input("  size: ")),
            "r_bw": int(input("  r_bw: ")),
            "w_bw": int(input("  w_bw: ")),
            "r_cost": float(input("  r_cost: ")),
            "w_cost": float(input("  w_cost: ")),
            "area": int(input("  area: ")),
            "r_port": int(input("  r_port: ")),
            "w_port": int(input("  w_port: ")),
            "rw_port": int(input("  rw_port: ")),
            "latency": int(input("  latency: "))
        }
        memory_levels.append(mem_config)

    # ### Mapping Configuration ###
    zigzag_mapping = """
    Loop ordering for default_analysis_by_zigzag
    =====================================================================================
    Temporal Loops                 O                 W                 I                 
    =====================================================================================
    for FX in [0, 3):              sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
      for FY in [0, 3):            sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
        for OX in [0, 3):          sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
          for OY in [0, 3):        sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
    =====================================================================================
    Spatial Loops                                                                        
    =====================================================================================
            parfor K in [0, 8):                                                          
    -------------------------------------------------------------------------------------
    """
    parsed_mapping = parse_zigzag_mapping(zigzag_mapping)

    # ### Display Configurations ###
    input_config = {'Batch': B, 'Input height': IY, 'Input width': IX, 'Input channels': C}
    weight_config = {'Input channels': C, 'Weight height': FY, 'Weight width': FX, 'Output channels': K}
    output_config = {'Batch': B, 'Output height': output_height, 'Output width': output_width, 'Output channels': K}

    print("Test case configuration:\n")
    print("Input Matrix:\n", input_config)
    print("Weight Matrix:\n", weight_config)
    print("Output Size:\n", output_config)

    print("\nMemory configuration:")
    for i, mem_config in enumerate(memory_levels, start=1):
        print(f"\nMemory Level {i}:")
        for key, value in mem_config.items():
            print(f"  {key:<8}: {value}")

    print("\nMapping configuration:")
    print(parsed_mapping)

    # ### Compute Latency ###
    latency = calc_latency(B, C, IX, IY, K, FX, FY, P, S, total_unit_count, cycles_per_mac, memory_levels, parsed_mapping)

if __name__ == "__main__":
    main()
```

### Key Modifications
1. **Function Renaming**:
   - Renamed `compute_zigzag_latency` to `calc_latency` for clarity and to reflect its role in the new script.

2. **Parameter Passing**:
   - Removed manual input collection from `calc_latency` and modified it to accept parameters (`B`, `C`, `IX`, etc.) directly from `main`.

3. **Integration with Mapping**:
   - Incorporated the `parsed_mapping` from `parse_zigzag_mapping` to compute the `ideal_temporal_cycle` based on temporal loops, replacing the previous assumption of full unrolling.

4. **Latency Components**:
   - Kept the core latency calculations (ideal temporal cycle, stall cycles, data loading/offloading cycles) consistent with the ZigZag cost model from your original function.

5. **Main Function Adjustments**:
   - Added `total_unit_count` and `cycles_per_mac` inputs to `main` to ensure all necessary hardware parameters are collected and passed to `calc_latency`.

### How It Works
- **Execution Flow**:
  1. `main()` collects workload parameters (e.g., `B`, `C`, `IX`), hardware details (e.g., `memory_levels`), and defines the ZigZag mapping string.
  2. The mapping is parsed into temporal and spatial loops using `parse_zigzag_mapping`.
  3. All inputs are displayed for verification.
  4. `calc_latency` computes and prints the latency components and total latency.

- **Latency Components**:
  - **Ideal Temporal Cycle**: Based on the product of temporal loop ranges.
  - **Stall Cycles (SS_comb)**: Estimated from memory bandwidth constraints at the lowest level.
  - **Data Loading Cycle**: Calculated for data transfers between memory levels.
  - **Data Offloading Cycle**: Calculated for output data movement.
  - **Total Latency**: Sum of all components.

### Running the Script
1. Run the script in a Python environment.
2. Enter the prompted values for workload, hardware, and memory configurations.
3. Review the printed configurations and latency results.

This script fulfills your request by integrating the `main` function’s inputs with a modified latency calculation function, leveraging the ZigZag cost model. Let me know if you need further adjustments!



#### Question 3:

Here is the updated main()
def main():
    """Collect inputs and compute latency using the ZigZag cost model."""
    # ### Workload Configuration ###
    B = int(input("Enter the value for B: "))
    C = int(input("Enter the value for C: "))
    IX = int(input("Enter the value for IX: "))
    IY = int(input("Enter the value for IY: "))
    K = int(input("Enter the value for K: "))
    FX = int(input("Enter the value for FX: "))
    FY = int(input("Enter the value for FY: "))
    P = int(input("Enter the value for P: "))
    S = int(input("Enter the value for S: "))
    output_height = (IY + 2 * P - FY) // S + 1
    output_width = (IX + 2 * P - FX) // S + 1

    # ### Hardware Configuration ###
    total_unit_count = int(input("Enter the number of MAC units (total_unit_count): "))
    cycles_per_mac = int(input("Enter cycles per MAC operation (default=1): ") or 1)
    
    num_levels = int(input("Enter number of memory levels: "))
    memory_levels = []
    for i in range(num_levels):
        print(f"Enter configuration for memory level {i+1}:")
        mem_config = {
            "size": int(input("  size: ")),
            "r_bw": int(input("  r_bw: ")),
            "w_bw": int(input("  w_bw: ")),
            "r_cost": float(input("  r_cost: ")),
            "w_cost": float(input("  w_cost: ")),
            "area": int(input("  area: ")),
            "r_port": int(input("  r_port: ")),
            "w_port": int(input("  w_port: ")),
            "rw_port": int(input("  rw_port: ")),
            "latency": int(input("  latency: "))
        }
        
        # Additional configuration
        operands = input("  Enter operands (comma separated, e.g., I2,O): ").strip()
        mem_config["operands"] = [op.strip() for op in operands.split(',')]

        auto_cost = input("  Enable auto cost extraction (True/False): ").strip().lower()
        mem_config["auto_cost_extraction"] = True if auto_cost in ("true", "1", "yes") else False

        double_buffer = input("  Enable double buffering support (True/False): ").strip().lower()
        mem_config["double_buffering_support"] = True if double_buffer in ("true", "1", "yes") else False

        # Collect port information as a list of dictionaries
        num_ports = int(input("  Enter number of port entries: "))
        ports = []
        for j in range(num_ports):
            print(f"    Enter details for port entry {j+1}:")
            # If you want a fixed set of keys (for example: fl and tl), you can do:
            port_entry = {}
            port_entry["tl"] = input("      Enter tl value: ").strip()
            # If the entry has an optional 'fl' key, you can prompt for it:
            fl_val = input("      Enter fl value (or press Enter to skip): ").strip()
            if fl_val:
                port_entry["fl"] = fl_val
            fh_val = input("      Enter fl value (or press Enter to skip): ").strip()
            if fh_val:
                port_entry["fh"] = fh_val
            th_val = input("      Enter fl value (or press Enter to skip): ").strip()
            if th_val:
                port_entry["th"] = th_val
            ports.append(port_entry)
        mem_config["ports"] = ports

        served_dims = input("  Enter served dimensions (comma separated, e.g., D1): ").strip()
        mem_config["served_dimensions"] = [dim.strip() for dim in served_dims.split(',')]
        
        memory_levels.append(mem_config)


    # ### Mapping Configuration ###
    zigzag_mapping = """
    Loop ordering for default_analysis_by_zigzag
    =====================================================================================
    Temporal Loops                 O                 W                 I                 
    =====================================================================================
    for FX in [0, 3):              sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
      for FY in [0, 3):            sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
        for OX in [0, 3):          sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
          for OY in [0, 3):        sram_2Mb_I2_O     sram_2Mb_I2_O     dram_8Mb_I1       
    -------------------------------------------------------------------------------------
    =====================================================================================
    Spatial Loops                                                                        
    =====================================================================================
            parfor K in [0, 8):                                                          
    -------------------------------------------------------------------------------------
    """
    parsed_mapping = parse_zigzag_mapping(zigzag_mapping)

    # ### Display Configurations ###
    input_config = {'Batch': B, 'Input height': IY, 'Input width': IX, 'Input channels': C}
    weight_config = {'Input channels': C, 'Weight height': FY, 'Weight width': FX, 'Output channels': K}
    output_config = {'Batch': B, 'Output height': output_height, 'Output width': output_width, 'Output channels': K}

    print("Test case configuration:\n")
    print("Input Matrix:\n", input_config)
    print("Weight Matrix:\n", weight_config)
    print("Output Size:\n", output_config)

    print("\nMemory configuration:")
    for i, mem_config in enumerate(memory_levels, start=1):
        print(f"\nMemory Level {i}:")
        for key, value in mem_config.items():
            print(f"  {key:<25}: {value}")

    print("\nMapping configuration:")
    print(parsed_mapping)

    # ### Compute Latency ###
    latency = calc_latency(B, C, IX, IY, K, FX, FY, P, S, total_unit_count, cycles_per_mac, memory_levels, parsed_mapping)

    word_access = 0

    print(f"\nTotal Latency: {latency} cycles")
    print(f"\nWord access: {word_access}")

if __name__ == "__main__":
    main()

Using this first calculate the word access for each operand for each direction

"Word access represents the number of memory words read or written per operand at each memory level, calculated in `calc_memory_word_access`. It accounts for four directions: `wr_in_by_low`, `rd_out_to_low`, `rd_out_to_high`, `wr_in_by_high`.

### Formula (per direction, per operand, per memory level)
\[
\text{word_access} = \lceil \frac{\text{data_elem_move_per_period} \times \text{data_precision}}{\text{min_bw}} \rceil \times \left( \frac{\text{min_bw}}{\text{max_bw}} \right) \times \text{total_period_count} \times \text{unit_count}
\]

Where:
- **data_elem_move_per_period**: Number of data elements moved per period.
- **data_precision**: Bits per data element.
- **min_bw**: Minimum memory bandwidth (granularity of access).
- **max_bw**: Maximum memory bandwidth.
- **total_period_count**: Number of periods for the data transfer.
- **unit_count**: Number of memory units at the level above (mem_lv + 1).

### Notes
- If `data_elem_move_per_period` or `data_precision` is 0, `word_access` = 0.
- The formula adjusts for memory access granularity by using `min_bw` and scales by the bandwidth ratio and parallelism (`unit_count`).

### Excel Implementation
- **Inputs**: Per operand and memory level: `data_elem_move_per_period`, `data_precision`, `min_bw`, `max_bw`, `total_period_count`, `unit_count`.
- **Output**: Four values per operand per level (one per direction).
"

Which is required for the effective latency calculations exactly like zigzag before getting the final latency calculations
