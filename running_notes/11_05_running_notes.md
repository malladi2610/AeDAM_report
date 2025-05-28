
# DSE process

For a *Selected DSE point* and a *Selected workload* the expected outputs are *Best mappings for all the layer (opt: latency)*, *Best mapping for all layers (opt: Energy)*, *Best mapping for all the layer (opt: EDP)*, To achieve this the following flow is followed

1. The zigzag is explored for all the three valid splitting for an event driven accelerator i.e unrolling across K, unrolling across FX and unrolling across FY this each produces individual Temporal orderings across each splitting.
2. Applying the filters for all the temporal orderings where only the event driven map space is filtered out and also the pooling layer is handled seperately to generate a mapspace.
3. Combining this entire map space of splitting across K, FX and FY to form a pure event driven mapspace for all the layers.
4. Applying the zigzag cost model to all the available mapping and then printing out the mappings which are better interms of latency, energy and energy delay product in three seperate files, along with their individual stats.

This will give the first point in the pareto curve and this process needs to be repeated to find the optimal DSE point to each of the optimisation criteria i.e latency, energy and the EDP.

The following case studies are being perfomed:
1. The amount of levels of the temporal ordeing also known as (lpf_limits in zigzag terms) - Goal: Find the optimal amount of temporal ordering
2. The effect of the varaition to the SRAM size - Goal: Find the best SRAM configuration for each of the optimisation criteria
3. The effect of the PE ratio - Goal is to find the best PE configuration for each of the optimisation criteria
4. The effect of replacing the SRAM with MRAM - Find if it's beneficial 

Do the above process for VGGNEt, Mobilenet V1 and Resnet -50:

Finally give the best Single core event driven architecture possible to run all these models for optimisation criteria of latency, energy and EDP.So, there will be three different configurations of the event driven accelerator in the end with proper validations.

# Validation process

Once the values are given by zigzag a seperate validation process needs to be setup to verify the results for each point produced during the DSE. This process takes the following inputs and it replicates the behaviour of the event driven accelerator to verify if the results are close to zigzag or not.

Inputs to the validator:
1. Architecture configuration
2. Workload configuration
3. Mapping loop orderings
4. Partial sum generated - Each input contributes to which outputs
5. Critical path algorithm set for the event driven accelerator
6. Calculate the word access in four way direction for each variable [I, W, O]- using zigzag's logic (With corrected output calculations)

From the above inputs
1. Calcuate the latency [Onloading, offloading and computations] - Using zigzag formulas but computed event by event
2. Calculate the energy [Overall energy] - Using zigzag formulas but computed event by event


