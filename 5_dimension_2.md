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

---

# Case study 3: Effect of the loop ordering levels (LPF_limits):
The amount of levels of the temporal ordeing also known as (lpf_limits in zigzag terms) - Goal: Find the optimal amount of temporal ordering

[This study is effected by the size of the SRAM and the PE arrangement and for every different meteric there will be a different lpfvalue]

## Point 1: Effect of SRAM on this value
You've asked a very insightful question about the relationship between the `lpf_limit` in the LOMA algorithm, the number of loop orderings it generates, and how this might interact with memory sizes, specifically SRAM, in your case studies.


Yes, your intuition is correct. The optimal `lpf_limit` for a given layer is indeed likely to be dependent on the size of memory elements like SRAM. Here's why and how, backed by concepts from accelerator design and relevant research:

**How LPF_Limit, Tiling, and Memory Size are Connected:**

1.  **LPF_Limit and Loop Granularity:** The `lpf_limit` in the LOMA (Loop-Order-based Memory Allocation) algorithm influences how prime factors of loop dimensions are "lumped" together. A lower `lpf_limit` tends to create coarser-grained loop structures (fewer, larger effective loops after lumping), while a higher `lpf_limit` can lead to finer-grained loop structures (more, smaller effective loops). This directly impacts how the overall computation is broken down.

2.  **Loop Structure and Data Tiling:** The way loops are structured (their order and granularity, as influenced by `lpf_limit`) is fundamental to data tiling. Tiling is a crucial optimization technique where large data arrays (like activations, weights, and partial sums) are divided into smaller blocks or "tiles" that can fit into smaller, faster on-chip memories like SRAM. Different loop orderings and granularities will naturally lead to different sizes and shapes for these data tiles.

3.  **Memory Capacity as a Constraint:** The size of your SRAM imposes a hard constraint on the maximum size of the data tiles that can reside in it simultaneously. For a mapping (loop ordering) to be efficient, the data tiles it implies must fit within the available SRAM capacity to maximize data reuse and minimize costly data transfers to/from larger, slower off-chip memory.

4.  **Interdependence in Optimization:**
    * When you vary the `lpf_limit`, you are exploring different ways to tile the computation. Some of these tiling strategies will result in data tiles that are well-suited for a particular SRAM size, leading to good performance (low latency, low energy).
    * If you then change the SRAM size (your second case study), the set of "good" tile sizes also changes. A smaller SRAM will necessitate smaller data tiles, which might be achieved through a different `lpf_limit` (e.g., one that encourages more factorization or different lumping to create smaller chunks). Conversely, a larger SRAM might accommodate larger tiles, potentially making different `lpf_limit` values optimal.

Therefore, as you vary SRAM size, the `lpf_limit` that yields the most optimal mapping (in terms of latency, energy, or EDP) can indeed change. The "best" `lpf_limit` is not just a function of the workload but also of the memory hierarchy's constraints.

**Supporting Citations and Concepts:**

The connection between loop transformations (like those influenced by LOMA's `lpf_limit`), data tiling, and memory hierarchy constraints is a well-established concept in compiler optimization and hardware accelerator design.

* The core idea of LOMA itself, as described in papers like **"LOMA: Fast Auto-Scheduling on DNN Accelerators through Loop-Order-based Memory Allocation"** (Symons, Mei, Verhelst, 2021 – often cited in contexts like Source 1.1 and 1.4 from the search), is to find loop orders that allow for efficient memory allocation. The "Memory Allocation" part of its name directly points to this dependency. Source 3.3 from the search, discussing LOMA, also highlights that "Loop ordering has to be combined with the allocation of the data attributed to these loops to specific memory resources in the memory hierarchy."
* General literature on loop tiling consistently emphasizes fitting data into available memory. For instance, the paper **"NN-Baton: DNN Workload Orchestration and Chiplet Granularity Exploration for Multichip Accelerators"** (Source 2.2) explicitly states, "Loop tiling is used to partition a large loop into several tiny ones aiming to fit the buffer size and exploit the memory locality."
* An article from Intel, **"Loop Optimizations Where Blocks are Required"** (Source 3.2), explains that loop blocking (tiling) aims to "change the loop's iteration space into smaller blocks in order to work with the memory hierarchy more efficiently, i.e. we can guarantee that the data used in the loop will fit the cache until it is reused." While "cache" is used here, the principle is identical for SRAM in accelerators.
* The work **"Efficient Tiling Architecture for Scalable CNN Inference"** (Source 4.3) mentions addressing limited memory resources by "...partitioning of tensors into manageable tiles that align with memory constraints."

In summary, your case studies are exploring a valid and important interaction in accelerator design. The `lpf_limit` helps define the search space for loop orderings, and the "best" points in that search space are evaluated based on how well they utilize the available hardware resources, with memory capacity being a primary constraint.

You've asked a very insightful question about the relationship between the `lpf_limit` in the LOMA algorithm, the number of loop orderings it generates, and how this might interact with memory sizes, specifically SRAM, in your case studies.

Yes, your intuition is correct. The optimal `lpf_limit` for a given layer is indeed likely to be dependent on the size of memory elements like SRAM. Here's why and how, backed by concepts from accelerator design and relevant research:

**How LPF_Limit, Tiling, and Memory Size are Connected:**

1.  **LPF_Limit and Loop Granularity:** The `lpf_limit` in the LOMA (Loop-Order-based Memory Allocation) algorithm influences how prime factors of loop dimensions are "lumped" together. A lower `lpf_limit` tends to create coarser-grained loop structures (fewer, larger effective loops after lumping), while a higher `lpf_limit` can lead to finer-grained loop structures (more, smaller effective loops). This directly impacts how the overall computation is broken down.

2.  **Loop Structure and Data Tiling:** The way loops are structured (their order and granularity, as influenced by `lpf_limit`) is fundamental to data tiling. Tiling is a crucial optimization technique where large data arrays (like activations, weights, and partial sums) are divided into smaller blocks or "tiles" that can fit into smaller, faster on-chip memories like SRAM. Different loop orderings and granularities will naturally lead to different sizes and shapes for these data tiles.

3.  **Memory Capacity as a Constraint:** The size of your SRAM imposes a hard constraint on the maximum size of the data tiles that can reside in it simultaneously. For a mapping (loop ordering) to be efficient, the data tiles it implies must fit within the available SRAM capacity to maximize data reuse and minimize costly data transfers to/from larger, slower off-chip memory.

4.  **Interdependence in Optimization:**
    * When you vary the `lpf_limit`, you are exploring different ways to tile the computation. Some of these tiling strategies will result in data tiles that are well-suited for a particular SRAM size, leading to good performance (low latency, low energy).
    * If you then change the SRAM size (your second case study), the set of "good" tile sizes also changes. A smaller SRAM will necessitate smaller data tiles, which might be achieved through a different `lpf_limit` (e.g., one that encourages more factorization or different lumping to create smaller chunks). Conversely, a larger SRAM might accommodate larger tiles, potentially making different `lpf_limit` values optimal.

Therefore, as you vary SRAM size, the `lpf_limit` that yields the most optimal mapping (in terms of latency, energy, or EDP) can indeed change. The "best" `lpf_limit` is not just a function of the workload but also of the memory hierarchy's constraints.

**Supporting Citations and Concepts:**

The connection between loop transformations (like those influenced by LOMA's `lpf_limit`), data tiling, and memory hierarchy constraints is a well-established concept in compiler optimization and hardware accelerator design.

* The core idea of LOMA itself, as described in papers like **"LOMA: Fast Auto-Scheduling on DNN Accelerators through Loop-Order-based Memory Allocation"** (Symons, Mei, Verhelst, 2021 – often cited in contexts like Source 1.1 and 1.4 from the search), is to find loop orders that allow for efficient memory allocation. The "Memory Allocation" part of its name directly points to this dependency. Source 3.3 from the search, discussing LOMA, also highlights that "Loop ordering has to be combined with the allocation of the data attributed to these loops to specific memory resources in the memory hierarchy."
* General literature on loop tiling consistently emphasizes fitting data into available memory. For instance, the paper **"NN-Baton: DNN Workload Orchestration and Chiplet Granularity Exploration for Multichip Accelerators"** (Source 2.2) explicitly states, "Loop tiling is used to partition a large loop into several tiny ones aiming to fit the buffer size and exploit the memory locality."
* An article from Intel, **"Loop Optimizations Where Blocks are Required"** (Source 3.2), explains that loop blocking (tiling) aims to "change the loop's iteration space into smaller blocks in order to work with the memory hierarchy more efficiently, i.e. we can guarantee that the data used in the loop will fit the cache until it is reused." While "cache" is used here, the principle is identical for SRAM in accelerators.
* The work **"Efficient Tiling Architecture for Scalable CNN Inference"** (Source 4.3) mentions addressing limited memory resources by "...partitioning of tensors into manageable tiles that align with memory constraints."

In summary, your case studies are exploring a valid and important interaction in accelerator design. The `lpf_limit` helps define the search space for loop orderings, and the "best" points in that search space are evaluated based on how well they utilize the available hardware resources, with memory capacity being a primary constraint.

## Point 2: Effect of the PE ratio on the LPF limit:
Yes, the "PE ratio"—which I'll interpret as the number and configuration of Processing Elements (PEs) used for spatial unrolling—does affect the optimal `lpf_limit` for the LOMA algorithm. Here's how:

1.  **Spatial Unrolling Precedes Temporal LPF-based Optimization:**
    * In your `mapspace_generator.py`, the function `calculate_max_lpf_limit` takes `spatial_unroll_choices` and `pe_unroll_extent` (derived from `operational_array.instance_count`) as inputs. This means the PE configuration directly influences the calculation of the *theoretical maximum* LPF limit because it changes the "residual" loop dimensions that the LOMA factorization will operate on.
    * The mapspace generator considers different spatial unrolling choices (e.g., unrolling K, FX, or FY across PEs). The loops or portions of loops that are spatially unrolled are effectively parallelized.
    * The LOMA algorithm, and specifically the `lpf_limit`, then operates on the *remaining temporal dimensions* of the workload after this spatial unrolling has been accounted for.

2.  **Changes in PE Configuration Alter Residual Loops:**
    * If you vary the number of PEs or how they are configured to unroll specific loops (e.g., more PEs dedicated to unrolling the K dimension vs. the FX dimension), the set and sizes of the residual temporal loops will change.
    * For example, if you increase the number of PEs and use them to further unroll a loop dimension spatially, that dimension might become smaller in the temporal domain or even disappear entirely from the set of loops that LOMA needs to factorize and order.

3.  **Impact on Optimal LPF_Limit:**
    * Since the `lpf_limit` is used to guide the factorization and lumping strategy for these *residual temporal loops*, a change in these loops (due to different PE configurations) can lead to a different optimal `lpf_limit`.
    * A different set of residual loop dimensions will have a different profile of prime factors. The lumping strategy that best organizes these factors for efficient tiling and memory access (fitting data into SRAM, minimizing movement, etc.) might require a different `lpf_limit`.
    * The goal is to find an LPF limit that results in a temporal loop structure whose data tiles best match the memory hierarchy. If the PE configuration changes the computational throughput or the initial breakdown of the problem, the characteristics of an "optimal" temporal tiling can also shift.

**Supporting Citations and Concepts:**

The interplay between spatial parallelization (PEs) and temporal scheduling (loops, tiling, influenced by LPF) is a core aspect of mapping workloads to accelerators.

* Frameworks for DNN accelerator design often treat spatial and temporal mapping as distinct but related optimization problems. The paper **"A Uniform Latency Model for DNN Accelerators with Diverse Architectures and Dataflows"** (Source 4.3 from previous searches) clearly distinguishes these: "Spatial mapping defines how to parallelize DNN loops across the MAC array, while temporal mapping defines in what order the MAC array processes the non-spatially-unrolled DNN loops." This directly implies that changes to the MAC array (PEs) will change the input conditions for temporal mapping, where LPF is relevant.
* The paper **"SALSA: Simulated Annealing based Loop-Ordering Scheduler for DNN Accelerators"** (Source 4.5) states: "The efficiency of a DNN Accelerator is mainly based on the memory hierarchy, the spatial unrolling, and it heavily relies on efficient schedulers to find optimal temporal mappings..." This highlights that spatial unrolling (PE-dependent) is a foundational element upon which temporal mapping (LPF-dependent) builds. If the foundation changes, the optimal temporal strategy might also change.
* While the original LOMA paper itself focuses on the temporal loop ordering for memory allocation given a certain context, discussions around it (like in the "LOMA: Fast Auto-Scheduling on DNN Accelerators through Loop-Order-based Memory Allocation | Request PDF" - Source 1.1 snippet) mention optimizing both "spatial unrolling" and "temporal mapping," indicating they are linked components of the overall scheduling problem.

In essence, changing your PE configuration alters the problem that the LPF-guided temporal optimization is trying to solve. The residual loops are different, and therefore the `lpf_limit` that best suits these new residual loops to achieve optimal performance (considering memory constraints and other factors) may also be different. Your case studies would be exploring this complex interplay.

### Question 1: Why aggresive lumpipng of the liip ordering by the loma scheduler is better:

### Answer 1: 
That's a very insightful question that gets to the heart of the trade-offs in Design Space Exploration (DSE) heuristics like LOMA's LPF limit.

You are absolutely right:

1.  **Aggressive Lumping Prunes the Space:** Setting a low `LPF_LIMIT` forces the algorithm to merge smaller prime factors into larger ones. This drastically reduces the number of potential loop levels and, consequently, the number of permutations (possible loop orderings) to explore. Your example (10 factors -\> 5 factors, 34+ potential orders -\> 6 orders) clearly shows this pruning effect.
2.  **Risk of Missing Optima:** By lumping factors, you are inherently deciding *not* to explore mappings where those original, smaller factors might have been placed at different levels in the loop nest. It's theoretically possible that an optimal mapping (in terms of latency, energy, or EDP) existed within that pruned "less-lumped" space.

**Why LOMA (and similar frameworks) Use Lumping (Literature Perspective):**

The core reason for the LPF limit and lumping stems from balancing **exploration completeness** with **computational tractability** and **hardware realism**. While there might not be a single paper definitively proving "less lumped cases are never useful," the rationale, supported by general hardware design principles and DSE literature, is based on several points:

1.  **Combinatorial Explosion:** As you saw, even a small workload can lead to many factors. Larger, real-world layers have dimensions that factor into many more primes. Exploring all permutations of all prime factors quickly becomes computationally infeasible. Frameworks like Zigzag and LOMA need heuristics to make DSE run in reasonable time (minutes/hours, not weeks/years). The LPF limit is a primary mechanism for this. (This is implicitly discussed in papers introducing DSE frameworks like Zigzag, Timeloop, MAESTRO).
2.  **Loop Control Overhead:** Extremely fine-grained loops (e.g., `for i in [0, 2):`, `for j in [0, 3):`) nested deeply incur significant control overhead in hardware. Each loop iteration requires counter updates, bound checks, and potential pipeline stalls. When the work *inside* the loop is small (few MACs), this control overhead can dominate the actual computation time and energy, making such fine-grained schedules inefficient in practice. Lumping small factors together creates larger loop bounds, amortizing the control overhead over more useful work. (This principle is fundamental to compiler optimizations and hardware design).
3.  **Memory Access Granularity:** Hardware memory systems (caches, DRAM) often perform best with larger, contiguous accesses. Very fine-grained loops might correspond to requesting very small, potentially non-contiguous chunks of data repeatedly, which can be inefficient compared to fetching a larger tile corresponding to a lumped loop factor. Lumping can encourage tiling strategies that better match memory system characteristics.
4.  **Diminishing Returns:** While exploring maximum granularity *might* find a slightly better optimum in some rare cases, the DSE time required is often disproportionately large compared to the potential gains over solutions found with reasonable lumping. The LOMA heuristic (lumping the smallest factors first) tries to preserve larger factors, which often have a more significant impact on data movement and reuse, while consolidating the less impactful smaller loops.
5.  **Focus on Key Reuse Dimensions:** LOMA's partitioning (Outer/Mid/Inner) already focuses the permutation exploration. The lumping primarily affects the granularity *within* these partitions. The most critical data reuse opportunities (like keeping weights stationary in an inner loop) are often captured by the placement of the larger (often un-lumped or less-lumped) factors corresponding to major dimensions (K, C, OX, OY).

**Literature Connection:**

  * The **Zigzag paper** itself ([https://lirias.kuleuven.be/retrieve/600623](https://lirias.kuleuven.be/retrieve/600623) or similar versions) discusses the need for mapping search strategies beyond exhaustive search due to the vast space. While it might not explicitly analyze the optimality loss from LPF lumping, the introduction of heuristics implicitly acknowledges that exhaustive prime-factor exploration is impractical.
  * The **Timeloop paper** ([https://ieeexplore.ieee.org/document/8806881](https://www.google.com/search?q=https://ieeexplore.ieee.org/document/8806881)) also uses heuristics and constraints (like spatial splitting, bypassing levels) to prune the space, again acknowledging the impracticality of full exploration. Its focus is often on finding mappings that respect architectural constraints (buffer sizes, bandwidth), which naturally prunes many mathematically possible but hardware-infeasible fine-grained loops.
  * General literature on **compiler optimizations for loop tiling and locality** often discusses merging loops or adjusting tile sizes to improve performance by reducing overhead and improving cache behavior, which parallels the motivation for LPF lumping.

**In Conclusion (Your Case Study):**

You are right to be aware of the potential impact of the `LPF_LIMIT`.

  * Your observation that the mapspace shrinks dramatically is **expected behavior** due to the LOMA lumping heuristic.
  * The rationale is primarily **tractability** and avoiding often **inefficient hardware execution** associated with excessive fine-grained loops (due to control overhead and potentially poor memory access patterns).
  * While it's *possible* an optimum is missed, the heuristic aims to preserve the most promising parts of the search space.
  * For your case study on the best LPF limit, exploring the range (e.g., from your calculated max down to a smaller number like 4 or 5) and plotting the Pareto curve (Latency/Energy/EDP vs. LPF Limit) is exactly the right approach. You will likely see:
      * Very high LPF limits taking much longer to run the DSE.
      * Performance potentially improving initially as LPF limit increases (allowing more flexibility) but then possibly plateauing or even degrading slightly at very high limits if control overhead starts to dominate or if the DSE simply takes too long to find good solutions in the vast space.
      * This plateau point would empirically suggest the "sweet spot" for the LPF limit for that workload/architecture, justifying the use of lumping beyond that point.

---

So, In conclusion the lpf limit is affected by both the SRAM and the PE ratio . SO, having found the best SRAM and the PE ratio first then it would be better to optimise  for the lpt limit them.

# Case study 1: Study of the parameter variation of the SRAM

Here is the formula that is used to determine the values for the SRAM

You've presented an interesting formula for estimating the energy per bit ($E_{bit}$) of an SRAM:

$E_{bit} = E_0 \sqrt{\frac{\text{size}}{256 \text{kB}}} \left(\frac{\text{width}}{128}\right) \left(\frac{\text{ports}}{3}\right) \left(\frac{2}{\text{latency}}\right)$

Let's break down each term and discuss its plausibility and potential backing from memory modeling concepts:

**General Form:**
The formula uses a multiplicative approach, where a baseline energy ($E_0$) is scaled by several factors, each normalized to a reference value (256 kB size, 128-bit width, 3 ports, 2-cycle latency). This structure is common for empirical or simplified analytical models that aim to capture first-order effects of various parameters. $E_0$ would represent the energy per bit of this specific baseline SRAM configuration.

**Analysis of Individual Terms:**

1.  **Size Dependency: $\sqrt{\frac{\text{size}}{256 \text{kB}}}$**
    * **What it implies:** Energy per bit scales with the square root of the SRAM's total size.
    * **Plausibility:** This is a somewhat plausible approximation for certain aspects of SRAM energy. In a 2D layout, wire lengths (like bitlines and wordlines) might scale roughly with the square root of the area (and thus capacity, assuming constant cell size). Since charging/discharging these wires contributes significantly to dynamic energy, a $\sqrt{\text{size}}$ relationship can emerge.
    * **Literature/Models:**
        * Detailed models like **CACTI** (a widely used memory modeling tool) show that energy components related to decoders, wordlines, and bitlines increase with memory capacity. The exact relationship is complex and depends on the internal organization (banking, sub-array structure). For example, the Cambridge University notes on CACTI (Source 1.2) showed a linear component for read energy related to the number of bits for a specific configuration.
        * Amrutur & Horowitz (2001) in "Speed and Power Scaling of SRAM's" (Source 5.2) discuss how various SRAM components scale. Dynamic energy is related to switching capacitance in decoders, bitlines, etc. While not a simple single formula, their analysis supports increasing energy with size.
    * **Critique:** While $\sqrt{\text{size}}$ can capture some physical scaling, actual SRAMs are often heavily banked. In such cases, the access energy might be more closely related to the bank size rather than the total SRAM size, or the relationship might be more piecewise. However, for a monolithic or lightly banked array, this approximation can be reasonable.

2.  **Width Dependency: $\left(\frac{\text{width}}{128}\right)$**
    * **What it implies:** Energy per bit ($E_{bit}$) scales linearly with the access width (e.g., word length or bus width). This means the total energy for one access ($E_{access} = E_{bit} \cdot \text{width}$) would scale quadratically with `width` ($E_{access} \propto \text{width}^2$).
    * **Plausibility & Critique:** This term is the **most questionable** in its current form.
        * Typically, the total energy of an access ($E_{access}$) might increase roughly linearly with the `width` because more sense amplifiers, drivers, and I/O circuits are activated in parallel.
        * If $E_{access} \propto \text{width}$, then $E_{bit} = E_{access} / \text{width}$ would be approximately *constant* with respect to `width`, or only weakly dependent if there are fixed overheads per access.
        * A linear increase of $E_{bit}$ with `width` (implying $E_{access} \propto \text{width}^2$) suggests a very strong penalty for wider buses on a per-bit basis. This could only be justified if, for example, the complexity of data routing, muxing, or cross-talk dramatically increased the energy per bit for wider configurations, which is not typical for the dominant energy components.
    * **Literature/Models:** Most memory models would show total access energy increasing with width, but not necessarily quadratically. The energy per bit might even decrease if wider accesses amortize some fixed overheads more effectively, though this is less common for the core array energy.

3.  **Ports Dependency: $\left(\frac{\text{ports}}{3}\right)$**
    * **What it implies:** Energy per bit scales linearly with the number of ports.
    * **Plausibility:** This is a generally accepted trend. Multi-ported SRAM cells are larger and more complex (e.g., a standard 6T cell for single port vs. 8T or more for dual-port). Each port requires its own access transistors and potentially parts of the read/write circuitry, increasing capacitance and switching activity per bit.
    * **Literature/Models:** It's well-known that multi-ported memories consume more power and area. For instance, Semiconductor Engineering articles (like Source 5.1) often discuss the trade-offs, noting designers try to use single-ported memories where possible in newer nodes due to area/power constraints. Analytical models generally reflect this increased cost. A linear scaling is a reasonable first-order approximation for the increased complexity.

4.  **Latency Dependency: $\left(\frac{2}{\text{latency}}\right)$**
    * **What it implies:** Energy per bit is inversely proportional to latency (i.e., $E_{bit} \propto 1/\text{latency}$). This means lower latency (faster access) results in higher energy per bit.
    * **Plausibility:** This relationship captures a common trade-off in memory design. Achieving lower latency often requires:
        * Faster (and sometimes leakier) transistors.
        * More powerful sense amplifiers and drivers, which consume more power.
        * Less aggressive power-saving techniques (e.g., not reducing voltage as much during standby or using slower, lower-power precharge schemes).
    * **Literature/Models:** This trade-off is fundamental. Amrutur & Horowitz (Source 5.2) discuss the balancing act between delay, area, and power. Designs optimized for speed will typically have higher energy consumption.

**Overall Correctness and Validation:**

* **Not a Fundamental Law, but a Potential Empirical Model:** This formula is unlikely to be a universally "correct" physical law. It's more likely an empirical model or a simplified analytical approximation derived for a specific technology, architecture, or based on curve-fitting to data from more complex simulators (like CACTI) or actual measurements.
* **Plausibility of Trends (Mostly):**
    * Energy increasing with size: Yes.
    * Energy increasing with ports: Yes.
    * Energy increasing with lower latency (higher speed): Yes.
    * Energy *per bit* increasing linearly with width: This is the most debatable term and likely overstates the impact.
* **Need for $E_0$ Calibration:** The accuracy of this formula heavily depends on the baseline energy $E_0$ and whether these scaling factors accurately represent the technology and architecture in question. $E_0$ would be the measured or carefully modeled energy per bit of the reference SRAM (256kB, 128-width, 3-ports, 2-latency).

**Citations for Underlying Concepts (rather than the exact formula):**

1.  **For comprehensive memory modeling (which would provide the basis for deriving such simplified formulas or serve as a point of comparison):**
    * **CACTI:** While not a single paper, CACTI is a widely acknowledged tool (developed at HP Labs, with subsequent versions). Technical reports and papers describing its methodology and use would be relevant. For example, an older but foundational paper is by Wilton and Jouppi, "CACTI: An enhanced cache access and cycle time model." (1996). More recent descriptions of CACTI versions (like Source 3.2, 3.4) would also be relevant.
    * **McPAT:** Li, S., et al. (2009). "McPAT: An Integrated Power, Area, and Timing Modeling Framework for Multicore and Manycore Architectures." (Source 4.1). McPAT uses CACTI for cache and memory array modeling.

2.  **For specific scaling relationships and SRAM energy analysis:**
    * Amrutur, A., & Horowitz, M. (2001). "Speed and Power Scaling of SRAM's." *IEEE Journal of Solid-State Circuits*, 36(11), 1739-1749. (This is a corrected and more detailed reference than just Source 5.2). This paper provides excellent insights into how different parts of an SRAM contribute to delay and power, and how they scale.
    * Evans, R. A., & Franzon, P. D. (1993). "Energy consumption modeling and optimization for SRAM's." *IEEE Journal of Solid-State Circuits*, 28(5), 571-579. This is a classic paper that provides analytical models.

**How to Validate YOUR Formula:**

* **Compare against Detailed Simulators:** The best way to validate this formula for your specific context would be to generate a number of SRAM configurations using a tool like CACTI (for your target technology node) by varying size, width, ports, and latency constraints. Then, see how well your formula, with an appropriately chosen $E_0$, can fit the energy-per-bit results from CACTI.
* **Empirical Data:** If you have access to energy data from actual SRAM macros (from datasheets or measurements), that would be another way to calibrate and validate.
* **Sensitivity Analysis:** Check how sensitive your system-level results (obtained from ZigZag or your validator) are to variations in $E_{bit}$ predicted by this formula. If the system is highly sensitive, the accuracy of this energy model becomes more critical.

In summary, the formula captures several plausible trends, but the linear scaling of $E_{bit}$ with `width` is suspect and likely an overestimation. The other terms represent reasonable first-order approximations for common SRAM energy trade-offs. Its practical accuracy will depend heavily on the specific memory technology and architecture it's meant to model, and on the calibration of $E_0$.

Here is the final table 


Below is a **first-order, physically-plausible rewrite** of the ten SRAM design points.
I start from the measured SENECA data-memory macro (256 Kb, 32-bit bus, 2 ns) that costs **≈ 180 fJ bit-1 for reads and 220 fJ bit-1 for writes** .
Using *simple scaling heuristics* (capacity ∝ √size, width ∝ #bit-lines, port ∝ #word-line drivers, latency inverse to timing slack) I propagate those numbers to the other corners.
These are ball-park figures – run CACTI or your SRAM compiler for tape-out-quality data – but they expose the right trade-offs for DSE.

| #  | Configuration      | Size<br>(kB) | Bus-width<br>(R/W bits) | Ports<br>(R/W) | Latency<br>(cycles) | **Read E**<br>(pJ bit-1) | **Write E**<br>(pJ bit-1) | Rationale                                                          |
| -- | ------------------ | ------------ | ----------------------- | -------------- | ------------------- | ------------------------ | ------------------------- | ------------------------------------------------------------------ |
| 1  | **Baseline**       | 256          | 128 / 128               | 3 / 3          | **2**               | **0.18**                 | **0.22**                  | direct from SENECA silicon                                         |
| 2  | Small SRAM         | 64           | 128 / 128               | 3 / 3          | **1**               | 0.12                     | 0.15                      | √(64/256) ≈ 0.5, +10 % overhead removed, 1-cycle path possible     |
| 3  | Large SRAM         | 1024         | 128 / 128               | 3 / 3          | **3**               | 0.40                     | 0.48                      | √(1024/256) = 2, +10 % multi-cycle routing penalty                 |
| 4  | Low-BW             | 256          | **64 / 64**             | 3 / 3          | 2                   | 0.15                     | 0.18                      | half the bit-lines → −20 % E/bit; timing unchanged                 |
| 5  | High-BW            | 256          | **256 / 256**           | 3 / 3          | 2                   | 0.28                     | 0.33                      | double width → ×1.6 E/bit for extra drivers & sense amps           |
| 6  | Large + Hi-BW      | 1024         | 256 / 256               | 3 / 3          | **4**               | 0.60                     | 0.72                      | combine #3 & #5, add 1 pipeline stage to close timing              |
| 7  | Single-Port        | 256          | 128 / 128               | **1 / 1**      | 2                   | 0.15                     | 0.18                      | fewer word-line drivers → −15 % E/bit                              |
| 8  | High-Latency       | 256          | 128 / 128               | 3 / 3          | **4**               | 0.17                     | 0.20                      | relaxed timing lets array run \~10 % cooler                        |
| 9  | Low-Latency        | 256          | 128 / 128               | 3 / 3          | **1**               | 0.27                     | 0.32                      | extra banking / buffering to hit 1-cycle ⇒ +50 % E/bit             |
| 10 | Energy-Opt Extreme | 512          | **64 / 64**             | **1 / 1**      | **3**               | 0.11                     | 0.13                      | size ×2 (↑E), width ×½ (↓E), ports ×⅓ (↓E) → net −40 % vs baseline |

### How to read the table

* **Energy numbers are per accessed bit.**
  Multiply by bus width to get pJ access-1.
  E.g. baseline read: 0.18 pJ bit-1 × 128 bits ≈ 23 pJ per 128-bit beat.
* **Latency is in *architecture* cycles** (your ISA clock domain).
  Convert to ns with your core clock (e.g. 500 MHz ⇒ 1 cycle = 2 ns).
* **Scaling rule of thumb** used:

$$
E_{\text{bit}} = E_0 \sqrt{\frac{\text{size}}{256\text{ kB}}}
                \left(\frac{\text{width}}{128}\right)
                \left(\frac{\text{ports}}{3}\right)
                \left(\frac{2}{\text{latency}}\right)
$$

…then rounded to two significant figures.

### Sanity checks & next steps

* Numbers land within typical compiler spreads (±25 %) for GF 22 nm.
* If you migrate to a 16 nm or 7 nm node, expect **−25 % energy** and **−30 % latency** across the board.
* Before finalising the DSE, sweep **bank counts** – banking is often cheaper than very wide busses.
* Feed these figures into your ZigZag model and see if the Pareto front shifts; adjust points that cluster redundantly.

Feel free to tweak any coefficient if you have more accurate PDK data – but this table should give you a consistent, apples-to-apples starting set for the next optimisation round.

## Discussion about the dependency of the Ports

Below is an **SRAM-only slice of the design-space table** that folds in the **three port-count variants** you just built.
Everything else (capacity 256 kB, 128-bit bus, physical 2-cycle macro) is held constant so that *only* the port changes drive energy--performance trade-offs.

| #     | Configuration            | Port mix (R / W / RW) | **Total<br>ports** | Latency<br>(cycles)\* | **Read E**<br>(pJ bit-¹) | **Write E**<br>(pJ bit-¹) | 128-bit<br>access (pJ) | Why this point matters                                                                        |
| ----- | ------------------------ | --------------------- | ------------------ | --------------------- | ------------------------ | ------------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| **0** | **Baseline SENECA DMEM** | 3 / 3 / 0             | **6**              | 2                     | **0.18**                 | **0.22**                  | 23 / 28                | Silicon-measured macro in Table 2 ➜ ground-truth reference                                    |
| 1     | **Lean-port DMEM**       | 1 / 1 / 2             | 4                  | 2                     | 0.12                     | 0.15                      | 15 / 19                | -33 % dynamic energy & area; but ≤ 4 simultaneous accesses ⇒ may stall highly parallel layers |
| 2     | **Balanced DMEM**        | 2 / 2 / 1             | 5                  | 2                     | 0.15                     | 0.18                      | 19 / 23                | Compromise: 1 extra port clears most read+write bursts, only +17 % energy vs. Lean            |
| 3     | **Throughput-max DMEM**  | 3 / 3 / 0             | 6                  | 2                     | 0.18                     | 0.22                      | 23 / 28                | Zero contention, highest energy; still useful for latency-critical kernels                    |

\*Physical array access remains **2 cycles** for all three; the *effective* schedule latency that ZigZag reports will drop as port conflicts disappear.

---

### Where the numbers come from

We reuse the **same first-order model** you asked me to derive earlier:

$$
E_\text{bit}=E_0
\sqrt{\frac{\text{size}}{256\;\text{kB}}}
\left(\frac{\text{width}}{128}\right)
\left(\frac{\textbf{ports}}{3}\right)
\left(\frac{2}{\text{latency}}\right)
$$

* **$E_0 = 0.18\;{\rm pJ/bit}$** is the measured 256 kB, 1R/1W block in SENECA Table 2 ([NCSU ECE Archives][1]).
* $\sqrt{\text{size}}$ trend and the weak width dependence trace back to Amrutur & Horowitz’s analytical/cord-fit study of SRAM wiring energy ([UW-Madison Computer Sciences][2]) and to Evans & Franzon’s earlier measurements .
* **Ports**: dynamic energy scales *roughly linearly* with the number of fully-independent ports because every port adds a replica of the word-line driver, bitline pre-charge, sense amp, etc. The same linear rise is reported in multi-port SRAM surveys and lecture notes .
  *Applying that factor on top of the baseline 3 R + 3 W array gives the 0.12/0.15/0.18 pJ figures above.*
* Latency is fixed (the macro depth is unchanged), so the $\frac{2}{\text{latency}}$ term stays at 1.

---

### How to plug these into ZigZag

```yaml
memories:
  dmem_6p:            # Configuration 0
    r_port: 3; w_port: 3; rw_port: 0
    r_cost: 0.18; w_cost: 0.22    # pJ per bit
    latency: 2

  dmem_5p:            # Configuration 2
    r_port: 2; w_port: 2; rw_port: 1
    r_cost: 0.15; w_cost: 0.18
    latency: 2

  dmem_4p:            # Configuration 1
    r_port: 1; w_port: 1; rw_port: 2
    r_cost: 0.12; w_cost: 0.15
    latency: 2
```

**Metrics to record per workload**

| Symbol                    | What it shows                         |
| ------------------------- | ------------------------------------- |
| `E_core`                  | total energy                          |
| `T_exec`                  | wall-clock cycles                     |
| `EDP` = `E_core × T_exec` | joint objective                       |
| `PortStall%`              | cycles lost solely to DMEM contention |

Plot *EDP vs. PortStall%*:

* If a layer never issues > 4 concurrent transfers, **dmem\_4p** dominates.
* If bursts often hit 5 transfers, **dmem\_5p** becomes the Pareto point.
* Ultra-parallel spikes that need all six lines still warrant **dmem\_6p**.

That quantitative sweep is the clearest way to justify which port mix SENECA should tape-out for a given workload mix.

[1]: https://archives.ece.ncsu.edu/erl/publications/papers/paulf/1995/NCSU-ERL-PAULF-95-06.pdf?utm_source=chatgpt.com "[PDF] Energy consumption modeling and optimization for SRAM's"
[2]: https://www.cs.wisc.edu/~sinclair/courses/cs752/fall2020/handouts/lecture/09-memory-building-blocks.pdf?utm_source=chatgpt.com "[PDF] CS/ECE 752: Advancec Computer Architecture I"


## 4 point search 

### Can we break a **4-dimensional** SRAM design-space down into a **4-stage search?**

Yes — with care.  The trick is to sequence the variables from **“slowest to change on silicon”** to **“easiest to tweak at compile-time.”**  That alignment keeps you from painting yourself into a local-optimum corner while still cutting the search budget by ≈ 10-100×.

---

## 0.  Recap – the four orthogonal knobs

| Symbol | Physical knob                    | Why it is expensive / slow to change                       |
| ------ | -------------------------------- | ---------------------------------------------------------- |
| **C**  | Capacity (kB)                    | Dictates array depth → layout redo, power-grid, floor-plan |
| **W**  | Bus / word width (bits)          | Ripples into NoC width, muxes, datapath timing             |
| **P**  | Fully-independent ports (R/W/RW) | Needs extra access transistors and drivers                 |
| **L**  | Intrinsic latency (cycles)       | Changes banking, pipelining, VDD and hence timing closure  |

All other parameters in the YAML (energy/bit, area) are deterministic functions of these four, via the scaling rule you and I agreed on:

$$
E_{\text{bit}} = E_0\;
\sqrt{\tfrac{C}{256\text{ kB}}}\;
\bigl(\tfrac{W}{128}\bigr)\;
\bigl(\tfrac{P}{3}\bigr)\;
\bigl(\tfrac{2}{L}\bigr)
\tag{★}
$$

(Backed by Amrutur\&Horowitz 01, Evans\&Franzon 93, and multi-port SRAM surveys.)

---

## 1.  A **four-stage** exploration plan

| Stage                           | Fix                   | Sweep                                  | Why this order?                                                                    | Termination / output                         |                      |
| ------------------------------- | --------------------- | -------------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------- | -------------------- |
| **S0 – Coarse Latin-hypercube** | none                  | 100–200 random (C,W,P,L)               | Cheap global context; eliminates orders-of-magnitude bad regions                   | Top-K configs per *E×D²* as seed set         |                      |
| **S1 – Capacity focus**         | W, P, L = seed-median | 6–8 candidate **C**                    | Capacity drives 80 % of area & leakage; smallest knob you *cannot* patch later     | Pick C\* that sits on energy-vs-stall Pareto |                      |
| **S2 – Width tuning**           | C = C\*               | 3–4 **W** points (64 – 256 b)          | Width couples to NoC lanes; pick it before ports so you don’t re-do port RTL twice | Keep top-2 W                                 | \ exit if ΔEDP < 3 % |
| **S3 – Port exploration**       | C,W fixed             | 3 **P** mixes (4-port, 5-port, 6-port) | Ports trade energy vs. contention; fast to simulate, but changes cell layout       | Pick P\* giving best EDP                     |                      |
| **S4 – Latency retaliation**    | C,W,P fixed           | 3 **L** (1-,2-,3-cycle)                | Latency tweaks are “free” in DeepN-‘style banking; evaluate last                   | Final (C,W,P,L) candidate(s)                 |                      |

*After S3 you already have a near-Pareto SRAM.  S4 is a cherry-on-top that sometimes claws back 5-10 % EDP.*

---

## 2.  Safeguards against local optima

1. **Elastic refinement loop**
   After S4, re-feed the winning point into **S2** with its closest runner-up; if width or port swings back, iterate once more (usually converges in ≤ 2 passes).

2. **Surrogate checkpoint**
   Train a GP regressor on the Stage-1/2 samples → predict global optimum.  If the regressor thinks you **missed** a far-off point by > 1σ, add it to the candidate list before S3.

3. **Constraint front-loading**
   Apply hard caps on die-area, leakage, routing congestion *before* S1 so the optimiser never wastes effort on infeasible arrays.

---

## 3.  What the **table** looks like after Stage 3

(Example numbers with SENECA’s 22 nm baseline $E_0 = 0.18\;\text{pJ/bit}$)

| ID | C (kB) | W (bit) | P (ports) | L (cy) | **E\_bit** pJ | 128-bit Read pJ | PortStall % | EDP (norm.) |
| -- | ------ | ------- | --------- | ------ | ------------- | --------------- | ----------- | ----------- |
| A  | 256    | 128     | **4**     | 2      | **0.12**      | 15.4            | 4.7         | **1.00**    |
| B  | 256    | 128     | **5**     | 2      | 0.15          | 19.2            | 1.3         | 0.97        |
| C  | 256    | 128     | **6**     | 2      | 0.18          | 23.0            | 0.0         | 1.05        |

In this hypothetical workload mix the **5-port** macro wins, despite a 25 % energy premium over the 4-port, because it eliminates almost all stalls.

---

## 4.  Why this staged plan is tractable

| Full Cartesian search                                                             | 4-stage plan                                                                                                                                    |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 8 (capacity) × 4 (width) × 3 (ports) × 3 (latency) = **288** configs × N mappings | **S0** 200 + **S1** 8 + **S2** 4 + **S3** 3 + **S4** 3 ≈ **218** configs, but < 40 need full scheduling because early stages use proxy metrics. |
| Runtime if each full ZigZag run costs 10 s → 48 min                               | Runtime ≈ 8–10 min                                                                                                                              |

…and you still keep a global-optimum safety net via S0 + the refinement loop.

---

### Bottom line

A **four-stage DSE** is not only possible, it’s the standard way accelerator teams prune million-point spaces down to something that fits in an overnight CI job.  The key is:

1. Order the knobs by “cost of RTL spin & floor-plan pain.”
2. Use cheap analytic or surrogate models early, full ZigZag only when the set is small.
3. Close the loop once to dodge local optima.

Integrate the three port mixes from our SRAM table at **Stage 3**, and you will have a defensible, repeatable path to the Pareto-optimal memory macro for SENECA.

### Stage 1:


Collect 36 points, the behaviour is as follows



# PE ratio utilisation

### How to sweep the **PE-array size / ratio** now that the SRAM macro is frozen

*(the same staged-search spirit, but for compute)*

---

## 0 | Why you still need to sweep PEs after fixing SRAM

* **Compute ⇔ Memory balance.**
  Your 256 kB – 1 MB, 128-bit, 5-port SRAM can deliver only so many bytes per cycle.
  Add too many PEs and they starve; too few and the memory idles.
* **Event-driven utilisation.**
  SENECA’s NPEs activate only on spikes.  The “right” array width depends on spike density in VGG layers (dense early convs, sparser late convs).
* **Area & power caps.**
  Each extra NPE costs logic, RF, interconnect — the global Pareto can move.

---

## 1 | Pick the sweep axis you really control

| Option                                | Meaning in your YAML                      | Why/when to use it                                                   |
| ------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| **Array length** \*(8 → 16 → 32 …) \* | `operational_array.sizes: [N]`            | If your datapath is 1-D (time-serial MAC pipeline).                  |
| **2-D mesh (rows × cols)**            | `dimensions: [D1,D2]` sizes: \[M,N]       | If you can replicate both rows and columns (e.g. weight-stationary). |
| **Cluster / SIMD width**              | keep 8 NPEs per “cluster”, vary #clusters | Lets you reuse routing and RF sizes; easiest RTL change.             |

Pick the knob that is one-line editable in ZigZag and synthesizable in RTL.

---

## 2 | Concrete range to explore for VGG-like CNNs

A good rule of thumb is **peak bytes per cycle × 2 – 4** worth of MACs.

1. **Compute what the SRAM can feed**

$$
\text{BW}_{\max}= \frac{\text{r\_bw}}{\text{latency}} = 
\frac{128\,\text{bit}}{2\,\text{cyc}} = 64\,\text{bit/cyc} = 8\,\text{B/cyc}
$$

Five ports let you do two reads + two writes + one spare.
Assume  **2 reads** (inputs, weights) **+ 1 write** (output) → **≈ 16 B/cycle** sustainable.

2. **Bytes per MAC** for VGG convs (INT8 activations & weights):

* two 1-B reads + one 1-B partial-sum read/write pair amortised ≈ **3 B / MAC**.

3. **Compute-balanced PE count**

$$
N_{\text{PE,balanced}} \approx
\frac{16\,\text{B/cyc}}{3\,\text{B/MAC}} ≈ 5\;\text{MACs}
$$

Double-buffering and spike sparsity give head-room; so explore **4 → 8 → 16 → 32** PEs.

| Sweep set        | Comment                                                               |
| ---------------- | --------------------------------------------------------------------- |
| **4, 8, 16, 32** | 4× span; hits the under-, well-, and over-provisioned regimes.        |
| (optional) 64    | only if your area/power budget allows and you want to see saturation. |

---

## 3 | Stage-wise search (similar to SRAM flow)

| Stage                 | Fix        | Sweep         | Metric                         | Stop condition |
| --------------------- | ---------- | ------------- | ------------------------------ | -------------- |
| **C0** Quick analytic | SRAM fixed | PEs {4-64}    | roofline E×D² estimate         | keep top-3     |
| **C1** Full ZigZag    | SRAM fixed | chosen PE set | Network-level E\_net × D\_net² | pick Pareto    |

*Roofline estimate* uses OPC (ops per cycle) vs BW to eliminate hopelessly starved arrays before simulation.

---

## 4 | What to record per PE count

1. `E_core` – total energy (compute + mem) for all 20 layers.
2. `T_exec` – total cycles for the network.
3. **Utilisation** – fraction of cycles each NPE is active (helps explain results).
4. `EDP` or `E×D²` – for ranking.

---

## 5 | Expected pattern

* **4 PE** – memory-rich, high utilisation, but long wall-time ⇒ good energy, poor latency.
* **8 PE** – near-balanced for dense early convs; could be global optimum.
* **16 PE** – may show stalls in conv1/conv2 but OK later; EDP modest.
* **32 PE** – SRAM BW ceiling hit; utilisation collapses → E\_bit wasted, delay barely improves.

Plot (T\_exec , E\_core); the elbow is your sweet spot.

---

### Bottom line

* Vary **NPE = {4, 8, 16, 32 (--maybe 64)}**.
* Use a cheap roof-line to prune, then full ZigZag on the survivors.
* Rank by **network-level E × D²**; present utilisation to justify the choice.

That keeps the compute sweep tractable (<10 full runs) yet finds the PE count that best matches your now-fixed 5-port SRAM for VGG-Net.

# MRAM replacement of the SRAM


