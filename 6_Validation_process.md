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