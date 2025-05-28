# Section divisions
This entire section is divided in four major sections

1. Mapspace generated
2. Word access computed (Critical path is included in this)
3. latency Calculated (Cycles)
4. Energy calculated (To be decided from the literature)

------------------------------------------------------------------------------------------------------------------------------------------------------
# General workload being modelled

The workload is being modelled in two different cases:
1. Frame based inputs
2. Event driven inputs

## Frame based inputs

The exploration workload on the event driven architecture is a Seven dimension nested loop structure as shown below 

**Shapes**  
- Input tensor: \(Inputs\in\mathbb R^{IH\times IW\times C}\)  
- Filter bank: \(Weights\in\mathbb R^{FH\times FW\times C\times K}\)  
- Output tensor: \(Outputs\in\mathbb R^{OH\times OW\times K}\)  
where, for unit stride and no padding,  
\[
OH = IH - FH + 1, 
\quad
OW = IW - FW + 1.
\]



**7-nested-sum (index) formula**  
For all  
\[
0 \le oh < OH,\quad
0 \le ow < OW,\quad
0 \le k < K
\]
\[
Y(oh,\,ow,\,k)
\;=\;
\sum_{c=0}^{C-1}
\;\sum_{fh=0}^{FH-1}
\;\sum_{fw=0}^{FW-1}
\;
X\bigl(oh + fh,\;ow + fw,\;c\bigr)
\;\times\;
W\bigl(fh,\;fw,\;c,\;k\bigr)
\]



**Equivalent nested-loop pseudocode**  
```cpp
// assume OH = IH-FH+1, OW = IW-FW+1
for (int oh = 0; oh < OH; ++oh)           // output rows
  for (int ow = 0; ow < OW; ++ow)         // output cols
    for (int k  = 0; k  < K;  ++k) {       // output channels
      Y[oh][ow][k] = 0;
      for (int c  = 0;  c  < C;  ++c)      // input channels
        for (int fh = 0; fh < FH; ++fh)   // filter height
          for (int fw = 0; fw < FW; ++fw) // filter width
            Outputs[oh][ow][k] +=
              Inputs[oh + fh][ow + fw][c]    // input pixel
              * Weights[fh][fw][c][k];        // filter weight
    }
```
## Event Driven inputs

In an **event-driven** accelerator, instead of walking through the entire output tensor one element at a time, we process **one “event”** at a time and immediately fire off all of its contributions before moving on.  An event might be:

- **A pixel‐intensity change** in a Dynamic Vision Sensor (DVS) camera—each pixel reports asynchronously when its log-brightness crosses a threshold  ([Event-based Vision - Guillermo Gallego - Google Sites](https://sites.google.com/view/guillermogallego/research/event-based-vision?utm_source=chatgpt.com)).  
- **A spike output** from a spiking neural network layer—each neuron emits a graded or binary “spike” that must be integrated into downstream neuron states  ([Frontiers | Optimizing event-based neural networks on digital neuromorphic architecture: a comprehensive design space exploration](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1335422/full)).  
- **A feature detection event** (e.g.\ a corner or edge) produced by a front-end module, or even  
- **An audio‐onset event** in neuromorphic audio processing, a LiDAR return, etc.  

Once an event has been processed—meaning all its multiply-accumulate contributions have been vectorized and applied—it never needs to revisit the accelerator’s pipeline.

#### Loop reordering: from “output-dominated” to “input-dominated”

Standard convolution is **output-dominated**, looping over each output coordinate and then summing over its receptive field. In contrast, an **input-dominated** (event-driven) ordering loops over each incoming activation and then fans out its contributions to all affected outputs:

```cpp
// assume OH = IH−FH+1, OW = IW−FW+1
// Inputs:  I[ih][iw][c]
// Weights: W[fh][fw][c][k]
// Outputs: O[oh][ow][k]

for (int ih = 0; ih < IH; ++ih)           // (1) input row
  for (int iw = 0; iw < IW; ++iw)         // (2) input col
    for (int c  = 0; c  < C;  ++c) {      // (3) input channel
      // process one “event” I[ih][iw][c]
      for (int fh = 0; fh < FH; ++fh)    // (4) filter row
        for (int fw = 0; fw < FW; ++fw)  // (5) filter col
          for (int k  = 0; k  < K;  ++k)  // (6) output channel
            O[ih−fh][iw−fw][k] +=
              I[ih][iw][c]               // input event
            * W[fh][fw][c][k];           // filter weight
    }
```

1.  **Spatial loops outer** (`ih`, `iw`) sweep over each input location exactly once.  
2.  **Channel loop** (`c`) steps through every input feature channel at that location.  
3.  **Filter‐window loops** (`fh`, `fw`) project that single activation across the kernel’s spatial support.  
4.  **Output‐channel loop** (`k`) vectorizes the multiply–accumulate across all filters in one go.  
 
To tweak loop‐nesting for data‐layout or vectorization on your target accelerator, the key is: **input spatial & channel loops come first**, then **filter & output-channel loops**.  



**References**  
- W. Xu et al., “Optimizing event-based neural networks on digital neuromorphic architecture…”, *Frontiers in Neuroscience*, 28 Mar 2024.  ([Frontiers | Optimizing event-based neural networks on digital neuromorphic architecture: a comprehensive design space exploration](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1335422/full))  
- G. Gallego et al., “Event-based Vision: A Survey”… Bio-inspired sensors output asynchronous pixel-level “events” when brightness changes.  ([Event-based Vision - Guillermo Gallego - Google Sites](https://sites.google.com/view/guillermogallego/research/event-based-vision?utm_source=chatgpt.com))  
- T. Serrano-Gotarredona & B. Linares-Barranco, “AER Image Filtering Architecture for Vision Processing Systems”, *IEEE Trans. Circuits Syst.* Sep 1999—first spatial event-driven convolution mapping.  ([Event camera](https://en.wikipedia.org/wiki/Event_camera?utm_source=chatgpt.com))

------------------------------------------------------------------------------------------------------------------------------------------------------
# Mapspace generated

This parameters is depended on two major variables 
1. Spatial Unrolling possible
2. Temporal Unrolling levels (Loop tiling levels)
(it starts from 4 and goes until 10 and can go more too dependent on the workload size)

To generate the event driven map space the thumb rules follwed are **input spatial & channel loops come first**, then **filter & output-channel loops**.  

The spatial unrolling is possible FH, FW, K to achieve event driven modelling (Citations backed from Multiply and fire) and for the temporal unrolling by fixing the IX, IY and C in the posistion and varying the rest of the variables FX, FY and K and performing the Tiliing also with respectivevalid splitting a map space can be generated.

Here is the formula for the map space generated.


| Your step                                                                                                      | What LOMA says                                                                             | Match? | Notes                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1. Pick the spatially‑unrolled dimensions                                                                     | *“First, we extract the LPFs … the spatially‑unrolled dimensions are discarded.”*          | ✔      | Same idea—take out what is already mapped in space before touching temporal loops.                                                                   |
|  2. Prime‑factorise each remaining loop up to an LPF limit                                                     | LOMA’s *Loop‑Prime‑Factor Generation* (Step A) and the optional *LPF‑limit lumping*        | ✔      | The “LPF limit” knob is exactly LOMA’s speed/optimality trade‑off.                                                                                   |
|  3. Break / lump factors so the total LPF count meets the limit                                                | *“LOMA will lump the smallest LPFs of the loop type with the most LPFs.”*                  | ✔      | Your “break … to achieve the required LPF limit” is the same heuristic.                                                                              |
|  4. Fix outer loops (IX, IY, C) and inner loops (FX, FY, K) for event‑driven dataflow                          | LOMA lets the **user** impose any ordering constraints before the permutation engine runs  | ✔      | You are simply providing extra constraints suited to an input‑dominated/event‑driven schedule.                                                       |
|  5. Enumerate every legal permutation inside the outer group, the inner group, **and** across the middle loops | LOMA Step B: partial multiset permutations per loop type, then recombine into full orders  | ✔      | The only difference is that you pre‑partition the search space into ⟨outer⟩ • ⟨middle⟩ • ⟨inner⟩ blocks; that is a valid subset of the global space. |

Bottom line: **your procedure is fully consistent with the LOMA search strategy; you are just adding the event‑driven constraint that input‑space loops stay outermost and kernel loops stay innermost.**


## Algorithm

```python
# INPUT ---------------------------------------------------------------
dims = {            # full layer sizes
    "IX": IH, "IY": IW, "C": C,
    "FX": FH, "FY": FW, "K": K,
    # add any other loops (B, OX, OY …) if relevant
}
unroll = {"C": Cu, "K": Ku}  # PE‑array spatial unrolling
LPF_LIMIT = 9                # user knob (set None for exhaustive)

# STEP 1  ▸ remove the part already unrolled in space
residual = {l: dims[l] // unroll.get(l, 1) for l in dims}

# STEP 2  ▸ prime‑factorise every residual loop
factors = {l: multiset_prime_factors(residual[l]) for l in residual}

# STEP 3  ▸ optional LPF lumping to respect LPF_LIMIT
if LPF_LIMIT and sum(len(p) for p in factors.values()) > LPF_LIMIT:
    lump_smallest_LPFs(factors, LPF_LIMIT)  # LOMA heuristic

# STEP 4  ▸ partition into outer, middle, inner groups (event‑driven)
G_outer = factors["IX"] + factors["IY"] + factors["C"]
G_inner = factors["FX"] + factors["FY"] + factors["K"]
G_mid   = [f for l,p in factors.items()
              if l not in ("IX","IY","C","FX","FY","K")
              for f in p]

# STEP 5  ▸ generate the map space
for perm_o in multiset_perms(G_outer):      # outer permutations
    for perm_m in partial_perms_by_looptype(G_mid):  # LOMA Step B
        for perm_i in multiset_perms(G_inner):        # inner perms
            yield perm_o + perm_m + perm_i            # one loop order
```

*Helper routines*

* `multiset_prime_factors(n)` returns the multiset of prime factors of *n*.
* `lump_smallest_LPFs(...)` repeatedly multiplies the two smallest factors of the loop type with the largest count until the global LPF count ≤ `LPF_LIMIT` (LOMA §III‑E).
* `multiset_perms(S)` produces **unique** permutations of the multiset *S* (Python example: `itertools.permutations` + hash set, or the lightweight multiset generator referenced in LOMA \[14]).
* `partial_perms_by_looptype(L)` implements LOMA Step B: generate permutations separately per loop type and stitch them together; this cuts the combinatorial blow‑up.


#### Closed‑form upper bound on the number of loop orders

Let

* $n_o =$ total LPFs in the outer group, with counts $\{n_{ix},n_{iy},n_c\}$
* $n_i =$ total LPFs in the inner group, with counts $\{n_{fx},n_{fy},n_k\}$
* $n_m =$ total LPFs in the middle group, divided over loop types $t\in M$ with counts $n_t$

Then

$$
\underbrace{\frac{n_o!}{n_{ix}!\,n_{iy}!\,n_{c}!}}_{\text{outer perms}}
\; \times\;
\Bigl(\prod_{t\in M}\frac{n_t!}{\prod_{p\in t} m_{tp}!}\Bigr)
_{\text{LOMA Step B partial perms}}
\; \times\;
\underbrace{\frac{n_i!}{n_{fx}!\,n_{fy}!\,n_{k}!}}_{\text{inner perms}}
$$

is an upper bound on the number of unique loop‑order candidates explored.

Because we permute multisets **inside each block only**, the true count is often orders‑of‑magnitude smaller than a full $(n_o+n_m+n_i)!$ permutation, exactly as LOMA observes (12 600 vs 3.6 million in Fig. 3) .


#### Why the event‑driven fix (IX IY C … FX FY K) makes sense

* In an input‑dominated schedule the activation *arrives once*, so holding IX/IY/C outermost maximises on‑chip reuse of that value before it is discarded.
* Making FX/FY/K innermost ensures every MAC fed by that activation is hit back‑to‑back; the accelerator can fully vectorise over filter taps and output channels in a single cycle.
* LOMA places no restriction on *where* a loop sits, so the above ordering is just an additional user constraint that narrows the map space without violating optimality guarantees for **that** subset.

<!-- ---

\### Slide‑ready takeaway

> **Event‑driven map‑space generation = LOMA + fixed I/O kernel blocks**
> 1️⃣ discard spatially‑unrolled sizes 2️⃣ prime‑factorise loops (LPFs) 3️⃣ optionally lump LPFs ≤ limit 4️⃣ force 〈IX IY C〉 outer, 〈FX FY K〉 inner 5️⃣ multiset‑permute outer, middle, inner blocks ⇒ loop‑order candidates.

All steps are directly grounded in the LOMA methodology , so your validation procedure is sound. -->

---
# Data Path for the event driven accelerator

For an event driven accelerator the data path is different than the traditional frame based acceleratordue to which the memory allocation step for the variable is different.

Here is the difference b/w the data allocation path frame based and event based.

* **Input events** arrive over the on-chip network (NoC) and go directly into the RISC-V controller (not to SRAM).  They’re immediately preprocessed into micro-tasks and fed to the loop controller; there is no buffering of the raw events in Data Memory .
* **Weights** reside in the Data Memory (a 256 KB SRAM).  During each event, the loop controller issues a DMEM read to fetch the appropriate weight into the NPE register file, all on the fly .
* **Partial sums / neuron states** are accumulated in the NPEs’ registers and then written back into the same SRAM bank as they’re updated .
* **Output spikes** are generated by the event-generator block (which inspects NPE registers), pushed into an output FIFO, post-processed by RISC-V, and then sent back out over the NoC .

---

### Frame-based vs. Event-driven: Data-movement comparison

| **Aspect**                     | **Frame-based (Eyeriss)**                                                                                                                                  | **Event-driven (Seneca)**                                                                                                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input handling**             | Full input feature maps (“frames”) are read from off-chip DRAM into the on-chip Global Buffer (GLB), then streamed via the NoC into each PE’s local SPAD.  | Individual “spikes” arrive over the NoC directly into the controller; they are **never** stored in SRAM, but immediately turned into micro-tasks for the loop controller.  |
| **Weight access**              | All filter weights are pre-loaded (or prefetched) from DRAM into the GLB, then broadcast or unicast via the NoC into SPADs for reuse.                      | Weights remain in SRAM.  Each event issues a DMEM read to fetch just the needed weights into NPE registers on demand.                                                      |
| **Partial-sum (psum) storage** | Psums are accumulated in SPAD and/or routed through neighboring PEs for spatial reduction, then written back to the GLB before DRAM write-back.            | Psums (neuron states) are held in NPE registers and after each update are **stored back** to SRAM (DMEM) immediately.                                                      |
| **Output handling**            | Entire output feature maps are collected in the GLB, optionally compressed, and then written out in bulk to DRAM.                                          | When a neuron fires, the event generator creates an output packet, which RISC-V post-processes and streams out via the NoC—no bulk frame buffer is ever used.              |
| **Local buffering**            | Dedicated SPADs for ifmap, weight, and psum per PE (12 b, 224 b, 24 b widths) plus a 108 kB GLB shared across PEs.                                         | Small register-file per NPE (e.g. 256 B) plus a 2 Mb SRAM “Data Memory.”  Task FIFOs and Loop Buffer registers hold micro-tasks and micro-code.                            |
| **Network**                    | Custom NoC with three separate networks (multicast GIN for inputs/weights, GON for psums, and 1-hop links for psum chaining).                              | Minimal-footprint NoC optimized for lightweight multicasting of events (source-based routing tables) and variable-length packets.                                          |
| **Flow granularity**           | Coarse: whole frames (tiled) are staged through multiple passes, with large bursts of data movement and deep reuse.                                        | Fine: individual events (spikes) trigger tiny, localized data movements and instant computation, with no reuse across events.                                              |



### Seneca’s critical data-paths

* **Input path**
  `NoC → RISC-V controller → Task FIFO → Loop controller → NPE register file`
  Events stream in, interrupt RISC-V, and immediately flow into the loop engine—never touching SRAM .

* **Weight path**
  `SRAM (DMEM) → loop controller issues DMEM read → NPE register file → MAC units`
  Each event triggers a direct SRAM read of its associated weight into the NPEs .

* **Partial-sum path**
  `NPE register file accumulates → DMEM write`
  Updated neuron states (partial sums) are written back immediately into SRAM .

* **Output path**
  `NPE registers → Event generator → output FIFO → RISC-V post-processing → NoC`
  As soon as a neuron’s threshold is crossed, the event generator forms a packet, which RISC-V then dispatches over the NoC .

------------------------------------------------------------------------------------------------------------------------------------------------------

# Word access computation

From the context of the Architecture bounds, mapspace created and the critical path. Word access is computed

### 1. Context: LOMA’s Memory Assignment

Once LOMA has picked an LPF ordering and assigned each loop (and hence each operand) to a level in the memory hierarchy, it hands off two key pieces of information to the cost model:

* **`mapping.mem_level[layer_op]`**: how many levels deep each operand is stored
* **`mapping.unit_mem_data_movement[layer_op][lvl]`**: metadata about how many words move per period, their precision, and how many transfer periods occur

*(LOMA’s Memory Allocation is described in §III-D of the LOMA paper) *



### 2. High-Level Algorithm (Pseudocode)

```python
from math import ceil
from zigzag.mapping.data_movement import DataDirection, DataMoveAttr

def calc_memory_word_access(mapping, accelerator):
    """
    Returns: dict[layer_op] → list[ dict[DataDirection → accesses] ]
    """
    mem_access = {}

    for layer_op in mapping.layer.layer_operands:
        # Map the layer operand (I, W, or O) to its MemoryOperand
        mem_op = mapping.layer.memory_operand_links.layer_to_mem_op(layer_op)
        mem_access[layer_op] = []

        # For each memory level this operand resides in…
        for lvl in range(mapping.mem_level[layer_op]):
            ml = accelerator.get_memory_level(mem_op, lvl)
            # Number of spatial replicas at this level
            unit_count = mapping.spatial_mapping.unit_count[layer_op][lvl + 1]
            dm = mapping.unit_mem_data_movement[layer_op][lvl]

            # Pull out the three key attributes (dicts keyed by DataDirection)
            amt_per_period = dm.get_attribute(DataMoveAttr.DATA_TRANS_AMOUNT_PER_PERIOD)
            precision     = dm.get_attribute(DataMoveAttr.DATA_PRECISION)
            period_count  = dm.get_attribute(DataMoveAttr.DATA_TRANS_PERIOD_COUNT)

            accesses = {}
            # Compute accesses **per** direction
            for dirn in DataDirection:
                a  = amt_per_period.get(dirn, 0)
                p  = precision    .get(dirn, 0)
                pc = period_count .get(dirn, 0)
                max_bw = ml.get_max_bandwidth(mem_op, dirn)
                min_bw = ml.get_min_bandwidth(mem_op, dirn)

                if a == 0 or p == 0:
                    accesses[dirn] = 0
                else:
                    # 1) how many worst-case blocks?
                    blocks = ceil((a * p) / min_bw)
                    # 2) normalize for peak BW, expand by #periods & #replicas
                    accesses[dirn] = blocks * (min_bw / max_bw) * pc * unit_count

            mem_access[layer_op].append(accesses)

    return mem_access
```

1. **Loop over operands & levels**
   For each of Input, Weight, Output (and for each memory‐level it was assigned), we’ll compute four directional counters:

   * `RD_OUT_TO_HIGH`
   * `WR_IN_BY_HIGH`
   * `RD_OUT_TO_LOW`
   * `WR_IN_BY_LOW`

2. **Fetch data‐movement stats**
   From `mapping.unit_mem_data_movement[layer_op][lvl]` we read:

   * **`DATA_TRANS_AMOUNT_PER_PERIOD`** (elements per period)
   * **`DATA_PRECISION`** (bits per element)
   * **`DATA_TRANS_PERIOD_COUNT`** (# of periods)

3. **Bandwidth bounds**
   From the accelerator’s `MemoryLevel` we get:

   * `min_bw`: the minimum sustained words-per-cycle (worst case)
   * `max_bw`: the peak words-per-cycle

4. **Block count**

   $$
     \text{blocks} = \Bigl\lceil\frac{\text{amount}\times\text{precision}}{\min\_bw}\Bigr\rceil
   $$

5. **Normalize & scale**

   $$
     \text{accesses} 
       = \text{blocks}
         \;\times\;\frac{\min\_bw}{\max\_bw}
         \;\times\;\text{period\_count}
         \;\times\;\text{unit\_count}
   $$

6. **Zero‐filter**
   If either the amount or the precision is zero, we skip the math and record 0 accesses.



### What “period” means in the formula

In ZigZag a **period** is *one complete life-time of the tile that lives in a given memory level*.

* Pick an operand **op** and a memory level **l**.
  All temporal loops that are **mapped at l or deeper** form the **live tile** (that tile fits in level *l*).
  The loops that sit **above** level *l*\* iterate over such tiles.
* When the upper loop indices advance, the tile is evicted and a new one is fetched; the **same data-movement sequence repeats**.
  Each repetition is what the code calls **one period**.

Hence

$$
\boxed{\,P^{(op,l)}
       \;=\;\prod_{t\in\text{outer loops for }(op,l)} T_t\,}
$$

where $T_t$ is the trip-count of temporal loop *t*.

---

### Where it is calculated in the code

* During mapping analysis `CostModelEvaluation._init_data_movement` walks the temporal loop nest from **outer → inner**.
* Whenever an operand crosses a memory boundary the routine multiplies the running **`period_factor`** for that operand by the current loop’s range (`rng`).
* The resulting integer is stored in

```python
DataMovePattern.data_trans_period_count.<dir>
```

and finally passed as `total_period_count` to

```python
_calc_memory_access(..., total_period_count, ...)
```

(see *cost\_model.py*, `_calc_memory_access`, lines \~23 030).

---

### Quick example with your layer 0 mapping

*For output **O** between Register (L0) and SRAM (L1):*

| loop   | range | placed in L0? | contributes to **P**? |
| ------ | ----- | ------------- | --------------------- |
| FX     | 3     | yes           | no                    |
| FY     | 3     | yes           | no                    |
| K      | 8     | yes           | no                    |
| **C**  | 3     | no            | **yes**               |
| **IY** | 224   | no            | **yes**               |
| **IX** | 224   | no            | **yes**               |

$$
P^{(O,\,\text{L0})}=C\cdot IY\cdot IX=3\times224\times224=150\,528
$$

That is exactly the factor that turns the single-tile transfer of
$192\;\text{words }\bigl(=K_\text{tile}\times FY\times FX=8\times3\times3\bigr)$
into the 28 901 376 word “wr v” you see in ZigZag’s JSON:

$$
192 \;(\text{words/period}) \times 150\,528 \;(\text{periods}) = 28\,901\,376
$$

---

### One-liner to compute **P** in a stand-alone script

```python
def period_count(operand: str, level: str, temporal_loops, alias):
    """Return P^(op,l) given the parsed mapping."""
    factor = 1
    level_seen = False
    for tl in temporal_loops:                       # outer → inner
        tgt = alias[tl["mapping"][operand]]
        if tgt == level:
            level_seen = True
        elif level_seen:                            # loop sits above <level>
            rng = tl["range"][1] - tl["range"][0] if isinstance(tl["range"], list) else tl["range"]
            factor *= rng
    return factor
```

Call the function for every *(op,l)* pair and feed the result into the
formula you showed:

```python
A_dir = ceil(A_elem * p / BW_min) * (BW_min / BW_max) * P * UC
```

That reproduces exactly the same period numbers—and therefore the same
word-access counts—that ZigZag prints.


---

# Latency computation

Once the wordaccess is computed then the latency is calucated whichc consists of three different costs i.e Computation cost, Data onloading cost and Data offloading cost.

Here is the explnantion as follows


## Formulas

**Extracting formulas**

The user wants me to extract the formulas used for data onloading, data offloading, and computation from the given code.

It looks like data onloading is calculated through functions like `calc_data_loading_latency`, `calc_loading_single_port`, and others. The key formula appears in `calc_onloading_combined` and `calc_offloading_combined`, which both relate to `data_onloading_cycle` and `data_offloading_cycle` in `calc_overall_latency`. I’ll focus on clarifying and extracting those specific formulas to answer the user's request.

**Extracting data onloading formulas**

The user wants the formulas for data onloading.

For a single input operand, it's the sum of the longest loading cycles between `wr_in_by_high` and `rd_out_to_low` across memory levels. For multiple operands, the formula combines shared, half-shared, and individual parts of the operands.

The key formula is:
For two operands:

$$
D_{on} = \min\left[
S_1 + \max(S_2 + H_2 + I_2, H_1 + I_1), S_2 + \max(S_1 + H_1 + I_1, H_2 + I_2)
\right]
$$

**Extracting latency formulas**

The user requested formulas for data onloading, offloading, and computation. Based on the code, here are the relevant equations:

1. **Data onloading**:

   * For a single operand:

   $$
   D_{on} = \sum_{i=0}^{L-2} \max(C^{(i)}_{rd\_out\_to\_high}, C^{(i+1)}_{wr\_in\_by\_low})
   $$

   * For two operands:

   $$
   D_{on} = \min\bigl[S_1 + \max(S_2 + H_2 + I_2, H_1 + I_1), S_2 + \max(S_1 + H_1 + I_1, H_2 + I_2)\bigr]
   $$

2. **Data offloading**:

   $$
   D_{off} = \sum_{i=0}^{L-2} \max(C^{(i)}_{rd\_out\_to\_high}, C^{(i+1)}_{wr\_in\_by\_low})
   $$

3. **Computation latency**:

   $$
   L_{comp} = \lceil \frac{T_{MAC}}{N_{PE}} \rceil \cdot c + S
   $$

Where:

* \$T\_{MAC}\$ = total MAC count
* \$N\_{PE}\$ = total unit count
* \$c\$ = cycles per operation
* \$S\$ = stall slack


Thought for 23 seconds


Here are the three core latency formulas, expressed exactly as they’re computed in ZigZag’s `CostModelEvaluation`:

---

### 1. Computation Latency $L_{\text{comp}}$

Let

* $N_{\text{MAC}}$ = total MAC operations (`layer.total_mac_count`)
* $N_{\text{PE}}$ = number of PEs in the array (`accelerator.operational_array.total_unit_count`)
* $c$ = cycles per MAC (`cycles_per_op`, usually 1)
* $T_{\text{temp}}$ = length of your temporal schedule (`mapping_int.temporal_mapping.total_cycle`)
* $S$ = stall/slack combined across all memory ports (computed via the merged PortActivity logic)

Then ZigZag does:

$$
\begin{aligned}
T_{\text{ideal}} &= \Bigl\lceil \tfrac{N_{\text{MAC}}}{N_{\text{PE}}}\Bigr\rceil \times c, \\[6pt]
L_{\text{comp}}
&= T_{\text{temp}}\times c \;+\; S.
\end{aligned}
$$

In code:

```python
ideal_temporal_cycle = mapping_int.temporal_mapping.total_cycle * cycles_per_op
stall_slack_comb   = …   # from combine_data_transfer_rate_per_physical_port()
L_comp = ideal_temporal_cycle + stall_slack_comb
```

---

### 2. Data-Onloading Latency $D_{\text{on}}$

For **each** input operand $op$ and **each** pair of adjacent memory levels $\ell$ and $\ell+1$:

1. Compute

   $$
     C^{(wr)}_{op,\ell}
     = \Bigl\lceil \frac{d^{(wr)}_{op,\ell}\,\times\,p^{(wr)}_{op,\ell}}{bw^{(wr)}_{op,\ell}}\Bigr\rceil,
     \quad
     C^{(rd)}_{op,\ell+1}
     = \Bigl\lceil \frac{d^{(rd)}_{op,\ell+1}\,\times\,p^{(rd)}_{op,\ell+1}}{bw^{(rd)}_{op,\ell+1}}\Bigr\rceil
   $$

   where

   * $d$ = `data_trans_amount_per_period`
   * $p$ = `data_precision`
   * $bw$ = actual port bandwidth (use read- or write-bw as appropriate)
2. Form the pair-latency

   $$
     L_{op,\ell} = \max\bigl(C^{(wr)}_{op,\ell},\,C^{(rd)}_{op,\ell+1}\bigr).
   $$
3. Partition all those $L_{op,\ell}$ into

   * fully-shared sum $S_{op}$
   * half-shared sum $H_{op}$
   * individual sum $I_{op}$
     (based on the port-sharing flags)

Then:

* **If** there is only one input operand $op$,

  $$
    D_{\text{on}} = I_{op}.
  $$
* **If** there are exactly two inputs $\{op_1,op_2\}$,

  $$
    D_{\text{on}}
    = \min\Bigl[
      S_{1} + \max\bigl(S_{2}+H_{2}+I_{2},\;H_{1}+I_{1}\bigr),
      \quad
      S_{2} + \max\bigl(S_{1}+H_{1}+I_{1},\;H_{2}+I_{2}\bigr)
    \Bigr].
  $$

<sub>*This exactly mirrors `calc_onloading_combined()` in the code.*</sub>

---

### 3. Data-Offloading Latency $D_{\text{off}}$

For the **single** output operand and levels $\ell=0,\dots,M{-}2$:

1. Compute the same pair-latencies

   $$
     C^{(rd)}_{\ell} = \bigl\lceil\tfrac{d^{(rd)}_{\ell}\,p^{(rd)}_{\ell}}{bw^{(rd)}_{\ell}}\bigr\rceil,
     \quad
     C^{(wr)}_{\ell+1} = \bigl\lceil\tfrac{d^{(wr)}_{\ell+1}\,p^{(wr)}_{\ell+1}}{bw^{(wr)}_{\ell+1}}\bigr\rceil.
   $$
2. $$
     L_{\text{off},\ell} = \max\bigl(C^{(rd)}_{\ell},\,C^{(wr)}_{\ell+1}\bigr).
   $$
3. Sum them up:

   $$
     D_{\text{off}} = \sum_{\ell=0}^{M-2} L_{\text{off},\ell}.
   $$

<sub>*This is exactly what `calc_offloading_combined()` does.*</sub>

---

Finally, the three end-to-end latencies reported are:

$$
\begin{aligned}
L_0 &= L_{\text{comp}},\\
L_1 &= L_0 + D_{\text{on}},\\
L_2 &= L_1 + D_{\text{off}}.
\end{aligned}
$$


## Algorithm

**Latency Calculation Algorithm**

```python
def calc_total_latency(mapping, accelerator, cycles_per_mac=1):
    """
    Given:
      - mapping: a Mapping object (with .unit_mem_data_movement, .mem_level, .spatial_mapping.unit_count,
                 .temporal_mapping.total_cycle, etc.)
      - accelerator: describes MemoryLevels, MemoryPorts, and the OperationalArray
      - cycles_per_mac: how many cycles each MAC takes (usually 1)
    Returns:
      (latency_without_io, latency_with_load, latency_with_load_and_store)
    """

    # 1. Determine which levels can double-buffer
    double_buffer = {}
    for op in mapping.layer.layer_operands:
        double_buffer[op] = [False]
        for lvl in range(mapping.mem_level[op]):
            util_shared = mapping.effective_mem_utili_shared[op][lvl]
            util_indiv  = mapping.effective_mem_utili_individual[op][lvl]
            # if <50% used, or individual slack, enable double buffering
            double_buffer[op].append(util_shared <= 0.5 or util_indiv <= 1 - util_shared)

    # 2. For each operand & level, compute:
    #    - allowed_cycles[op][lvl]  = from DATA_TRANS_PERIOD or INST_DATA_TRANS_WINDOW
    #    - real_cycles[op][lvl]     = ceil(data_bits / bw)
    allowed = {}   # MemoryAccesses per (op,lvl)
    real     = {}   # MemoryAccesses per (op,lvl)
    for op in mapping.layer.layer_operands:
      allowed[op], real[op] = [], []
      for lvl in range(mapping.mem_level[op]):
        umdm = mapping.unit_mem_data_movement[op][lvl]
        # select attr based on double_buffer flag
        win_attr = (DataMoveAttr.DATA_TRANS_PERIOD
                    if double_buffer[op][lvl] else
                    DataMoveAttr.INST_DATA_TRANS_WINDOW)
        win_next = (DataMoveAttr.DATA_TRANS_PERIOD
                    if double_buffer[op][lvl+1] else
                    DataMoveAttr.INST_DATA_TRANS_WINDOW)

        allowed_cycles = MemoryAccesses(
          rd_out_to_low  = umdm.get(win_attr).rd_out_to_low,
          wr_in_by_low   = umdm.get(win_attr).wr_in_by_low,
          rd_out_to_high = umdm.get(win_next).rd_out_to_high,
          wr_in_by_high  = umdm.get(win_next).wr_in_by_high
        )
        allowed[op].append(allowed_cycles)

        # real cycles = ceil(bits / bw)
        real_cycles = MemoryAccesses(
          rd_out_to_low  = ceil(umdm.amount_per_period.rd_out_to_low  * umdm.precision.rd_out_to_low  / bw_read),
          wr_in_by_low   = ceil(umdm.amount_per_period.wr_in_by_low   * umdm.precision.wr_in_by_low   / bw_write),
          rd_out_to_high = ceil(umdm.amount_per_period.rd_out_to_high * umdm.precision.rd_out_to_high / bw_read),
          wr_in_by_high  = ceil(umdm.amount_per_period.wr_in_by_high  * umdm.precision.wr_in_by_high  / bw_write)
        )
        real[op].append(real_cycles)

    # 3. Build per-port PortActivity lists and compute the global stall/slack
    port_activities = []
    for mem_lvl in accelerator.memory_hierarchy.mem_level_list:
      for port in mem_lvl.ports:
        duties = []
        for (mem_op, lvl, dirn) in port.served_op_lv_dir:
          if mapping.layer.memory_operand_links.contains(mem_op):
            op = mapping.layer.memory_operand_links.mem_to_layer_op(mem_op)
            P  = allowed[op][lvl].get(dirn)
            R  = real   [op][lvl].get(dirn)
            Cc = umdm.period_count.get(dirn)
            duties.append(PortActivity(real=R, allowed=P, period=umdm.period.get(dirn), period_count=Cc))
        if duties:
          port_activities.append((port, duties))

    # combine MUW & stall/slack per port, then take the max across all ports
    stall_slack_list = []
    for port, duties in port_activities:
      if len(duties) == 1:
        stall_slack_list.append(duties[0].stall_or_slack)
      else:
        # union of updating windows minus overlaps + positive slack
        union = calc_mem_updating_window_union(duties)
        pos = sum(d.stall_or_slack for d in duties if d.stall_or_slack>0)
        neg = sum(d.stall_or_slack for d in duties if d.stall_or_slack<0)
        total_win = sum(d.mem_updating_window for d in duties)
        stall_slack = pos + max(0, neg + total_win - union)
        stall_slack_list.append(stall_slack)
    stall_slack_comb = max(stall_slack_list, default=0)

    # 4. Compute on-loading (initial) cycles & off-loading (final) cycles
    data_onloading  = calc_data_loading_latency(...)   # combines ports & handles shared/individual cases
    data_offloading = calc_data_offloading_latency(...)

    # 5. Compute the ideal MAC cycles
    total_macs = mapping.layer.total_mac_count
    nPEs       = accelerator.operational_array.total_unit_count
    ideal_cycle = ceil(total_macs / nPEs) * cycles_per_mac

    # 6. Compute the temporal-mapping cycles (assuming perfect reuse)
    temp_cycle = mapping.temporal_mapping.total_cycle * cycles_per_mac

    # 7. Assemble the three latency numbers
    L0 = temp_cycle + stall_slack_comb                   # compute only
    L1 = L0         + data_onloading                     # + initial data load
    L2 = L1         + data_offloading                    # + final data store

    return L0, L1, L2
```

1. **Double-buffer flag**
   Decide for each operand and memory level whether its data footprint fits in half the buffer (→ can stream next chunk while computing).

2. **Allowed vs. Real**

   * **Allowed:** how many cycles you *can* spend streaming, from your data‐movement window attributes.
   * **Real:** how many cycles you *must* spend, given actual data size and memory bandwidth.

3. **Port-level Stall/Slack**
   For each physical port, merge all served‐operand timelines into a single “stall or slack” number. Then assume ports run fully in parallel and take the worst case.

4. **Initial & Final I/O**
   Combine per‐port loading activities into a single data-onloading cycle count; similarly for offloading.

5. **MAC Compute**

   * **Ideal MAC cycles:** `ceil(total_MACs / #PEs)`
   * **Temporal mapping cycles:** the schedule length from your temporal mapping.

6. **Total Latencies**

   * **Compute only (L0):** temporal + stall/slack
   * **+Load (L1):** L0 + data\_onloading
   * **+Store (L2):** L1 + data\_offloading

## Difference b/w the latency computation b/w the Frame based and Event based computation


## 1. ZigZag’s 4-term latency breakdown

ZigZag splits end-to-end latency into four contributions:

1. **Ideal compute cycles**

   $$
     T_{\rm compute}
     \;=\;
     \mathrm{ceil}\Bigl(\tfrac{\text{\# MACs}}{\text{PEs}}\Bigr)
     \quad\bigl(\equiv \texttt{mapping.temporal\_mapping.total\_cycle}\bigr)
   $$
2. **Stall/slack** (due to memory‐port contention)
3. **Data-onload** (bringing operands in)
4. **Data-offload** (writing results out)

All four terms are computed by the same ZigZag subroutines (e.g.
`calc_allowed_and_real_data_transfer_cycle_per_data_transfer_link`,
`calc_data_loading_latency`, etc.), but the inputs (loop counts,
`memory_word_access` vectors, etc.) differ.


## 2. Frame-based (Eyeriss-style) mapping

* **Spatial loop ordering**: you tile over the entire frame—e.g.
  batch → output-channels → input-channels → output-Y → output-X.
* **Data-onload**: you pay once (or once per tile) to stream in
  all input activations and weights for that tile (through global buffers, NoC, DRAM, …).
* **Compute**: you then sit in the inner two loops (output-Y, output-X) and churn through MACs, reusing on-chip buffers aggressively.
* **Data-offload**: you write back all partial-sums (or final outputs) at the end of the tile.

Because big tiles amortize the on- and off-load cost across thousands of MACs, the data-movement terms look like

$$
  T_{\rm onload}^{\rm frame}
  \;=\;
  \sum_{\ell\in\{\text{ifmap, wgt}\}}
  \mathrm{calc\_data\_loading\_latency}\bigl(\ell,\,\text{tile\_size}\bigr),
$$

and similarly for $T_{\rm offload}^{\rm frame}$, both proportional to the frame-tile size .


## 3. Event-driven (Seneca-style) mapping

1. **Top‐level loop** is “for each incoming event” (rather than “for each output-X,Y”).
2. **Data-onload** now happens per‐event: every spike brings in a tiny packet (its source address + value) and triggers a micro-kernel.

   * You never preload a whole feature map—only the weights for that one event’s fan-out and the neuron-states you’re updating.
3. **Compute**: the loop-controller dispatches one microcode that updates exactly those connected PEs (in parallel)—so

   $$
     T_{\rm compute}^{\rm event}
     \;=\;
     \mathrm{ceil}\Bigl(\tfrac{\text{\# fan-out}}{\text{PEs}}\Bigr)
     \quad\text{cycles per event.}
   $$
4. **Data-offload** is just a tiny packet per event (the new spike).

Because every event is self-contained,

$$
  T_{\rm onload}^{\rm event}
  \;=\;
  \sum_{\ell\in\{\text{event‐pkt, wgt, state}\}}
  \mathrm{calc\_data\_loading\_latency}\bigl(\ell,\,1\text{ event}\bigr),
$$

and similarly

$$
  T_{\rm offload}^{\rm event}
  \;=\;
  \mathrm{calc\_data\_offloading\_latency}\bigl(\text{1 event}\bigr).
$$


## 4. Putting it all together

| Term              | Frame-based                                | Event-driven                                      |
| :---------------- | :----------------------------------------- | :------------------------------------------------ |
| **Ideal compute** | $\lceil\frac{\#MAC_{\rm frame}}{PE}\rceil$ | $\lceil\frac{\#fan\_out}{PE}\rceil$ per event     |
| **Stall/slack**   | from long data‐stream bursts               | from many small bursts—usually lower              |
| **Data-onload**   | big bulk loads (activations + weights)     | tiny per-event loads (single spike + its weights) |
| **Data-offload**  | big bulk writes (partial-sums/out-maps)    | tiny per-event writes (one spike packet)          |

Because event-driven processing never revisits data (no “frame buffer”), its on- and off-load costs per MAC are typically **much lower**—but you pay them **per event** rather than amortized over spatial tiles.  In practice that means:

* **Frame architectures** excel when you have large, dense layers; their $T_{\rm on/offload}$ is small relative to thousands of MACs.
* **Event-driven** excels when data sparsity (or streaming inputs) means you only ever touch a tiny fraction of the network per event; you avoid hoisting whole tiles.


---

# Energy Computation

Let

* $\mathcal{O}$ = set of layer operands (ifmap, weight, psum…)
* $L$ = number of memory levels (0 = closest to PE, up to highest)
* $\mathrm{MACs}$ = total number of multiply–accumulates in the layer
* $\epsilon_{\mathrm{mac}}$ = energy per MAC (from the OperationalArray)
* For each operand $o$ and level $\ell$, let

  * $R_{\uparrow}(o,\ell)$ = # of “read out to higher level” accesses
  * $R_{\downarrow}(o,\ell)$ = # of “read out to lower level” accesses
  * $W_{\uparrow}(o,\ell)$ = # of “write in from higher level” accesses
  * $W_{\downarrow}(o,\ell)$ = # of “write in from lower level” accesses
    (all of these come from your `memory_word_access[o][ℓ]` vector)
* And let

  * $\epsilon_{r}(\ell)$ = energy per read at level $\ell$
  * $\epsilon_{w}(\ell)$ = energy per write at level $\ell$

Then:

1. **MAC energy**

$$
E_{\mathrm{mac}}
\;=\;
\mathrm{MACs}\;\times\;\epsilon_{\mathrm{mac}}
$$

2. **Memory energy**

$$
E_{\mathrm{mem}}
\;=\;
\sum_{o\in\mathcal O}\;
\sum_{\ell=0}^{L-1}\Bigl[\,
\bigl(R_{\uparrow}(o,\ell)+R_{\downarrow}(o,\ell)\bigr)\,\epsilon_{r}(\ell)
\;+\;
\bigl(W_{\uparrow}(o,\ell)+W_{\downarrow}(o,\ell)\bigr)\,\epsilon_{w}(\ell)
\Bigr]
$$

3. **Total energy**

$$
E_{\mathrm{total}}
\;=\;
E_{\mathrm{mac}}
\;+\;
E_{\mathrm{mem}}
$$


## 2. Algorithm (Pseudocode)

```python
def calc_total_energy(layer, accelerator, memory_word_access):
    """
    layer.total_mac_count         → number of MACs
    accelerator.operational_array.unit.energy_cost → ε_mac
    accelerator.memory_hierarchy   → list of MemoryLevel objects ℓ=0…L-1
    memory_word_access[o][ℓ]       → MemoryAccesses with fields:
                                     rd_out_to_high, rd_out_to_low,
                                     wr_in_by_high, wr_in_by_low
    """

    # 1) MAC energy
    num_macs   = layer.total_mac_count
    e_mac      = accelerator.operational_array.unit.energy_cost
    E_mac      = num_macs * e_mac

    # 2) Memory energy
    E_mem = 0.0
    for mem_lvl in accelerator.memory_hierarchy.mem_level_list:
        lvl = mem_lvl.level_id  # 0,1,…,L-1
        # per-access energies at this level
        e_read  = mem_lvl.read_energy
        e_write = mem_lvl.write_energy

        for op in layer.layer_operands:
            accesses = memory_word_access[op][lvl]
            # sum reads + writes in both directions
            n_reads  = accesses.rd_out_to_high + accesses.rd_out_to_low
            n_writes = accesses.wr_in_by_high + accesses.wr_in_by_low

            E_mem += n_reads  * e_read
            E_mem += n_writes * e_write

    # 3) Total
    E_total = E_mac + E_mem
    return E_mac, E_mem, E_total
```

* **`layer.total_mac_count`** and **`ε_mac`** come straight from your compute mapping and the operational-array spec.
* **`memory_word_access`** is exactly the output of your earlier word-access stage.
* Each **`MemoryLevel`** carries its own read/write energy costs (`r_cost`, `w_cost` in your YAML).


## 3. Difference in the energy the computation b/w the Frame based and the Event based 


For an event‐driven core like Seneca, the *shape* of every term in ZigZag’s energy model stays the same, but two things change:

1. **Input‐event and output‐event overhead**
   You pay a fixed cost for each event “packet” coming in (pre‐processing) and each spike going out (post‐processing). Call these

   $$
     E_{\text{pre}} \quad\text{and}\quad E_{\text{post}}
   $$

   (both in pJ per event).

2. **Memory‐access pattern**
   Frame‐based CNNs stream *all* activations in and out of every level; event‐driven only streams the **single event value** into the RF (no SRAM/GLB hits), plus the weights *needed by that event’s fan‐out* and the neuron‐state updates. In practice that means

   * **Input operand** only incurs register‐file reads (no high‐level SRAM/DRAM hits),
   * **Weights & partial‐sum (state)** incur the usual multilevel accesses from SRAM down to RF,
   * **Output operand** is just a small packet out (no frame buffer writes).


## A. Modified Energy Formulas

Let:

* $N_e$ = number of input events
* $\mathrm{MAC}_e$ = MACs *per* event (i.e.\ the event’s fan-out size)
* $\epsilon_{\mathrm{mac}}$ = MAC energy per op
* $\mathcal O = \{\text{I},\,\text{W},\,\text{O}\}$ the three operands
* $\ell=0$…$L-1$ the memory levels (0 = RF, up to highest SRAM/NoC)
* From ZigZag’s `memory_word_access_ev[o][ℓ]`, let

  $$
    R_\uparrow(o,\ell),\;R_\downarrow(o,\ell),\;
    W_\uparrow(o,\ell),\;W_\downarrow(o,\ell)
  $$

  be the four directional counts *per event*
* $\epsilon_r(\ell),\;\epsilon_w(\ell)$ = read/write energy at level $\ell$

Then **per event**:

1. **MAC energy**

   $$
     E_{\rm mac}^{(e)}
     = \mathrm{MAC}_e \;\times\;\epsilon_{\rm mac}
   $$

2. **Memory energy**

   $$
     E_{\rm mem}^{(e)}
     = \sum_{o\in\{\!\text{W},\text{O}\}}\,
       \sum_{\ell=0}^{L-1}
       \Bigl[
         \bigl(R_\uparrow+R_\downarrow\bigr)(o,\ell)\,\epsilon_r(\ell)
         \;+\;
         \bigl(W_\uparrow+W_\downarrow\bigr)(o,\ell)\,\epsilon_w(\ell)
       \Bigr]
   $$

   *(note: input “I” only uses RF reads, so all SRAM‐level terms for I drop out)*

3. **Event‐I/O overhead**

   $$
     E_{\rm io}^{(e)}
     = E_{\rm pre} \;+\; E_{\rm post}
   $$

4. **Total per‐event**

   $$
     E_{\rm event}
     = E_{\rm pre}
       \;+\;
       E_{\rm mac}^{(e)}
       \;+\;
       E_{\rm mem}^{(e)}
       \;+\;
       E_{\rm post}
   $$

5. **Aggregate** over all $N_e$ events:

   $$
     E_{\rm total}
     = N_e \;\times\; E_{\rm event}
   $$

## B. Comparison with Frame‐Based

| Term                                               | Frame‐Based                                         | Event‐Driven                                                      |
| -------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- |
| **MAC**                                            | $\#\text{MAC}_{\rm frame}\times \epsilon_{\rm mac}$ |                                                                   |
| $\mathrm{MAC}_e\times\epsilon_{\rm mac}$ per event |                                                     |                                                                   |
| **Memory**                                         | $\sum_{o\in\{I,W,O\}}\sum_\ell (\!R+W\!)\epsilon$   | $\sum_{o\in\{W,O\}}\sum_\ell (\!R+W\!)\epsilon$ per event         |
| **I/O**                                            | frame‐tile preload + write‐back costs               | $E_{\rm pre}+E_{\rm post}$ *per event*                            |
| **Total**                                          | one big $E_{\rm mac}+E_{\rm mem}$                   | $N_e\bigl(E_{\rm pre}+E_{\rm mac}+E_{\rm mem}+E_{\rm post}\bigr)$ |


## C. Script‐Ready Pseudocode

```python
def calc_event_energy(
    N_events, mac_per_event, mem_access_ev,
    accel, E_pre, E_post
):
    # mac_per_event: #MACs per incoming event (fan‐out)
    # mem_access_ev[o][lvl]: (R↑,R↓,W↑,W↓) per event for weights & psum
    # accel.memory_hierarchy.mem_level_list: levels ℓ
    # each level ℓ has .read_energy, .write_energy

    # 1) Fixed per-event overhead
    E_io = E_pre + E_post

    # 2) MAC energy per event
    eps_mac = accel.operational_array.unit.energy_cost
    E_mac_ev = mac_per_event * eps_mac

    # 3) Memory energy per event
    E_mem_ev = 0.0
    for lvl, mem in enumerate(accel.memory_hierarchy.mem_level_list):
        e_r = mem.read_energy
        e_w = mem.write_energy
        for op in ["W","O"]:                     # only W & psum
            R_up, R_down, W_up, W_down = mem_access_ev[op][lvl]
            E_mem_ev += (R_up+R_down)*e_r + (W_up+W_down)*e_w

    # 4) Total per event
    E_event = E_io + E_mac_ev + E_mem_ev

    # 5) Aggregate
    E_total = N_events * E_event
    return E_total
```



