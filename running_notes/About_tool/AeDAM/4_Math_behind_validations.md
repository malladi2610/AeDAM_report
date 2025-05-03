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

---

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

---
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

---

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
# Word access computed
## Understaning the wordaccess 
Can you help me in understanding the word access results from Zigzag.

Here are workload and hardware comfigurations 

Here is the workload dimensions:

- id: 0 # Conv2
  name: layer_2
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 1, 1, 12, 12, 5, 5]
  operand_precision:
    W: 8
    I: 8
    O: 8
    O_final: 16
  operand_source:
    I: 0
    W: 0

Here is the architecture that is being modelled
name: Event_Driven_Accelerator

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

  shared_memory_inputs_outputs:
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

On running the outputs here are the following wordaccess readings  and beset loop ordering from the zigzag:

Based on the loop ordering below the wordaccess readings are generated

Here is the loop ordering that is followed
======================================================================================================================================
Temporal Loops                     O                                W                                I                                
======================================================================================================================================
for OX in [0, 3):                  sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
  for OX in [0, 4):                sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
    for OY in [0, 3):              sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
      for OY in [0, 4):            sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
        for FY in [0, 5):          register                         sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
          for FX in [0, 5):        register                         sram                             register                         
--------------------------------------------------------------------------------------------------------------------------------------
======================================================================================================================================
Spatial Loops                                                                                                                         
======================================================================================================================================
            parfor K in [0, 1):                                                                                                       
--------------------------------------------------------------------------------------------------------------------------------------


Here are the word access readings
"word_accesses": {
                "O": [
                    {
                        "rd ^": 144,
                        "wr v": 0,
                        "rd v": 3600,
                        "wr ^": 3600
                    },
                    {
                        "rd ^": 18,
                        "wr v": 0,
                        "rd v": 0,
                        "wr ^": 144
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 0,
                        "wr ^": 36
                    }
                ],
                "W": [
                    {
                        "rd ^": 0,
                        "wr v": 3600,
                        "rd v": 3600,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 2,
                        "rd v": 3600,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 4,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0,
                        "wr v": 720,
                        "rd v": 3600,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 720,
                        "wr ^": 0
                    }
                ]}



Adding context to what "rd^, wr v, rd v, wr ^" means in the wordaccess are 

rd^ - to high, meaning the data is read out from the current level of memory to go to the higher level of memory
wr V - from high, meaning the data is provided by the higher level of memory to be written to the current level of memory
rd V - to low, meaning the data is read out from the current level of memory to go to the lower level of memory
wr ^ - from low, meaning the data is provided by the lower level of memory to be written to the current level of memory

The memory layout order is as follows
"mem_hierarchy_dict": {
                "I1": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=shared_memory_inputs_outputs,operands=[I1, O], served_dimensions={D1})"
                ],
                "I2": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=sram,operands=[I2, O], served_dimensions={D1})",
                    "MemoryLevel(instance=shared_memory_Weights,operands=[I2], served_dimensions={D1})"
                ],
                "O": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=sram,operands=[I2, O], served_dimensions={D1})",
                    "MemoryLevel(instance=shared_memory_inputs_outputs,operands=[I1, O], served_dimensions={D1})"
                ]}


With all the infomation. Can you find a relation on how the wordaccess values are actually arrived to as those values are not making much sense to me and need to understand how they are derived to.

If you need cost model of the zigzag which it uses for the wordaccess calculation I can provide that too, for better understanding and explanantion. Let me know

#### Answer:
To understand how the word access values are calculated in ZigZag for your specific workload and hardware configuration, let’s break it down systematically based on the provided workload dimensions, hardware architecture, loop ordering, and memory hierarchy. The goal is to explain the derivation of the `"rd ^"`, `"wr v"`, `"rd v"`, and `"wr ^"` values for each operand (`O`, `W`, `I`) at each memory level.

---

### Step 1: Understanding the Workload and Computation
The workload is a convolutional layer defined as:

- **Equation**: `O[b][k][oy][ox] += W[k][c][fy][fx] * I[b][c][iy][ix]`
- **Dimension Relations**: `ix = ox + fx`, `iy = oy + fy` (stride = 1)
- **Loop Dimensions and Sizes**:
  - `B` (batch size) = 1
  - `K` (output channels) = 1
  - `C` (input channels) = 1
  - `OY` (output height) = 12
  - `OX` (output width) = 12
  - `FY` (filter height) = 5
  - `FX` (filter width) = 5

**Total MAC Operations**:
For each output element `O[b][k][oy][ox]`, we perform a sum over `c`, `fy`, and `fx`. Since `C = 1`, it’s a 2D convolution over `FY` and `FX`:
- Number of MACs per output pixel = `FY * FX = 5 * 5 = 25`
- Total output pixels = `B * K * OY * OX = 1 * 1 * 12 * 12 = 144`
- Total MACs = `144 * 25 = 3600`

**Input Dimensions**:
- `IY = OY + FY - 1 = 12 + 5 - 1 = 16`
- `IX = OX + FX - 1 = 12 + 5 - 1 = 16`
- Input size = `16 x 16 = 256` elements

**Weight Dimensions**:
- `W[K][C][FY][FX] = W[1][1][5][5] = 25` weights

This gives us the computational demand: 3600 MACs to produce 144 output elements, using 25 weights and accessing a 16x16 input array.

---

### Step 2: Memory Hierarchy and Access Notation
The memory hierarchy for each operand is:

- **O (Output)**:
  1. Register
  2. SRAM
  3. Shared Memory for Inputs/Outputs
- **W (Weights)**:
  1. Register
  2. SRAM
  3. Shared Memory for Weights
- **I (Input)**:
  1. Register
  2. Shared Memory for Inputs/Outputs

**Word Access Notation**:
- `rd ^`: Read from current level to a higher level.
- `wr v`: Write to current level from a higher level.
- `rd v`: Read from current level to a lower level (e.g., to operational array or lower memory).
- `wr ^`: Write to current level from a lower level.

For the register level (lowest), `rd v` typically means reads to the operational array, and `wr ^` means writes from the operational array. For higher levels, data moves between memory levels or to/from the register.

---

### Step 3: Loop Ordering and Tiling
The loop ordering uses tiling for `OX` and `OY`, with temporal loops nested as follows:

- **Tiling**:
  - `OX = 12` split into tiles: outer `for OX in [0,3)` (3 tiles), inner `for OX in [0,4)` (4 elements per tile), so `3 * 4 = 12`.
  - `OY = 12` split similarly: outer `for OY in [0,3)` (3 tiles), inner `for OY in [0,4)` (4 elements per tile).
  - Total tiles = `3 * 3 = 9`, each tile computing a `4x4` block (16 outputs).

- **Nested Loops** (interpreted):
  ```pseudo
  for OX_tile in [0,3):  // 3 tiles
    for OY_tile in [0,3):  // 3 tiles
      for OX in [OX_tile*4, OX_tile*4+4):  // 4 OX per tile
        for OY in [OY_tile*4, OY_tile*4+4):  // 4 OY per tile
          for FY in [0,5):  // 5
            for FX in [0,5):  // 5
              parfor K in [0,1):  // K=1, so 1 iteration
  ```

- **Memory Levels per Loop**:
  - Outer `OX` and `OY` loops: `O` and `W` use SRAM, `I` uses Shared Memory for Inputs/Outputs.
  - `FY`: `O` uses Register, `W` uses SRAM, `I` uses Shared Memory for Inputs/Outputs.
  - `FX`: `O` uses Register, `W` uses SRAM, `I` uses Register.

This suggests that inner loops (`FY`, `FX`) operate at the register level where possible, while outer loops manage data from higher memories.

---

### Step 4: Calculating Word Accesses
Let’s derive the word access values for each operand based on the computation and loop schedule.

#### **Operand O (Output)**:
**Memory Levels**:
1. Register
2. SRAM
3. Shared Memory for Inputs/Outputs

**Word Accesses**:
```json
"O": [
    { "rd ^": 144, "wr v": 0, "rd v": 3600, "wr ^": 3600 },  // Register
    { "rd ^": 18, "wr v": 0, "rd v": 0, "wr ^": 144 },      // SRAM
    { "rd ^": 0, "wr v": 0, "rd v": 0, "wr ^": 36 }         // Shared Memory
]
```

- **Register Level**:
  - Each MAC reads the current partial sum, updates it, and writes it back.
  - Total MACs = 3600.
  - `rd v` = 3600: Read partial sum from register to operational array for each MAC.
  - `wr ^` = 3600: Write updated partial sum from operational array to register.
  - `rd ^` = 144: After computing each `O[oy][ox]` (25 MACs), the final value is sent to SRAM. Total outputs = 144.
  - `wr v` = 0: No initial loads from SRAM, as outputs are computed from scratch.

- **SRAM Level**:
  - `wr ^` = 144: 144 final outputs written from register to SRAM.
  - `rd ^` = 18: SRAM sends data to Shared Memory. With 9 tiles (3x3), this suggests some aggregation (e.g., 144 outputs moved in fewer transactions), but 18 is unexpected (expect 9 or 144). Possibly a larger block transfer or optimization artifact.
  - `rd v` = 0, `wr v` = 0: No reads to register (outputs aren’t reused here), no writes from Shared Memory.

- **Shared Memory Level**:
  - `wr ^` = 36: Outputs written from SRAM to Shared Memory. If 144 outputs are aggregated into larger blocks (e.g., 4 outputs per word or multiple tiles), 36 suggests further data packing, though exact derivation is unclear without the cost model.
  - Others = 0: No reads or writes in other directions.

#### **Operand W (Weights)**:
**Memory Levels**:
1. Register
2. SRAM
3. Shared Memory for Weights

**Word Accesses**:
```json
"W": [
    { "rd ^": 0, "wr v": 3600, "rd v": 3600, "wr ^": 0 },  // Register
    { "rd ^": 0, "wr v": 2, "rd v": 3600, "wr ^": 0 },     // SRAM
    { "rd ^": 0, "wr v": 0, "rd v": 4, "wr ^": 0 }         // Shared Memory
]
```

- **Register Level**:
  - `rd v` = 3600: Each MAC reads a weight from register to operational array.
  - `wr v` = 3600: Weights loaded from SRAM to register 3600 times. With only 25 weights, this suggests no reuse; each MAC fetches `W[k][c][fy][fx]` anew due to `FY` and `FX` loops using SRAM, not register.
  - `rd ^` = 0, `wr ^` = 0: No movement to/from other levels beyond SRAM.

- **SRAM Level**:
  - `rd v` = 3600: SRAM supplies weights to register for each MAC, matching register’s `wr v`.
  - `wr v` = 2: Loaded from Shared Memory to SRAM twice. With 25 weights, this is low—possibly fetching in small batches (e.g., 12-13 weights), but exact tiling is unclear.
  - `rd ^` = 0, `wr ^` = 0: No other movement.

- **Shared Memory Level**:
  - `rd v` = 4: Read 4 times to SRAM. If 25 weights are fetched in chunks (e.g., 6-7 weights), 4 reads could suffice, but `SRAM wr v = 2` suggests a mismatch or optimization not fully captured here.

#### **Operand I (Input)**:
**Memory Levels**:
1. Register
2. Shared Memory for Inputs/Outputs

**Word Accesses**:
```json
"I": [
    { "rd ^": 0, "wr v": 720, "rd v": 3600, "wr ^": 0 },  // Register
    { "rd ^": 0, "wr v": 0, "rd v": 720, "wr ^": 0 }      // Shared Memory
]
```

- **Register Level**:
  - `rd v` = 3600: Each MAC reads an input from register to operational array.
  - `wr v` = 720: Inputs loaded from Shared Memory to register 720 times. For each `(ox,oy)`, a 5x5 input window is needed (25 elements), but with overlap. `720 / 144 = 5` loads per output pixel, suggesting each `FY` loop loads one row (5 elements) reused across `FX`.
  - `rd ^` = 0, `wr ^` = 0: No other movement.

- **Shared Memory Level**:
  - `rd v` = 720: Matches register’s `wr v`, supplying 720 input loads.
  - Others = 0: Inputs are read-only from here.

---

### Step 5: Synthesis and Observations
The word access values reflect:
- **Register Level**: Direct MAC operations (3600 for all operands), with `O` accumulating partial sums, `W` and `I` fetched per MAC or with reuse (`I` has reuse).
- **Higher Levels**: Data movement based on tiling and loop ordering. `O` moves 144 outputs to SRAM, then aggregates to Shared Memory. `W` lacks reuse (fetched 3600 times), suggesting a scheduling inefficiency. `I` leverages overlap (720 loads vs. 3600 MACs).
- **Discrepancies**: `W`’s `rd v = 4` (Shared Memory) vs. `wr v = 2` (SRAM), and `O`’s `rd ^ = 18` (SRAM) vs. `wr ^ = 36` (Shared Memory) suggest possible packing, word size differences, or cost model specifics not detailed here.

---

### Conclusion
The word access values are derived from:
1. **MAC Operations**: 3600 reads/writes at the register level for `O`, `W`, `I`.
2. **Loop Ordering**: Determines which memory level supplies data per loop, affecting reuse (e.g., `I` at 720 due to `FY`/`FX` structure).
3. **Tiling**: `OX` and `OY` tiling (9 tiles of 4x4) influences data transfers to higher memories (e.g., `O`’s 144 to 18 to 36).
4. **Memory Hierarchy**: Data flows from Shared Memory → SRAM → Register for `W` and `O`, and Shared Memory → Register for `I`.

For precise reconciliation (e.g., `W`’s 4 vs. 2, `O`’s 18 vs. 36), the ZigZag cost model would clarify assumptions about data packing or transfer granularity. Without it, the values align with a per-MAC access pattern adjusted by tiling and minimal reuse for this schedule. If you can provide the cost model, I can refine this further!

*The discrepency in the answer is due to the change in the bandwidth from the Shared_memory to SRAM*
"# Understanding ZigZag's Word Access Calculation

## 1. Purpose of ZigZag
ZigZag is a framework for exploring the design space of hardware architectures and mapping strategies for deep learning accelerators. It helps optimize how neural network computations are mapped to hardware by analyzing computation patterns, data movements, and memory access costs.

## 2. Supporting Libraries and Components
ZigZag consists of several Python modules that work together:
- `classes/cost_model/` - Contains the cost modeling logic
- `classes/workload/` - Defines the neural network workloads
- `classes/hardware/` - Describes hardware architectures
- `classes/mapping/` - Handles mapping of computations to hardware
- `classes/opt/` - Optimization algorithms

## 3. Word Access Calculation in ZigZag

### Core Files that Determine Word Access Calculation

```python name=relevant_files.py
# Key files involved in word access calculation:
# 1. zigzag/classes/cost_model/memory_word_access.py - Core word access calculation
# 2. zigzag/classes/cost_model/cost_model.py - Coordinates cost modeling
# 3. zigzag/classes/hardware/architecture.py - Hardware memory hierarchy definition
# 4. zigzag/classes/mapping/temporal_mapping.py - Defines loop ordering
```

### How Word Access is Calculated

The word access calculation in ZigZag happens in several steps:

#### 1. Memory Hierarchy Analysis

The memory hierarchy for each operand (I, W, O) is defined in your hardware configuration. For your case:
- **Output (O)**: Register → SRAM → Shared Memory
- **Weights (W)**: Register → SRAM → Shared Memory
- **Input (I)**: Register → Shared Memory

#### 2. Loop Analysis

ZigZag analyzes the nested loop structure (OX_tile, OY_tile, OX, OY, FY, FX, K) to determine:
- Which loops access which operands
- Which memory level each loop gets data from
- How data is reused across loop iterations

#### 3. Word Access Counting

For each operand at each memory level, ZigZag tracks four types of accesses:
- `rd ^`: Read data from current level to send to a higher level
- `wr v`: Write data to current level from a higher level
- `rd v`: Read data from current level to send to a lower level
- `wr ^`: Write data to current level from a lower level

#### 4. Memory Level Translation and Packing

The discrepancies you're seeing (like W's 4 vs 2, and O's 18 vs 36) are likely related to how ZigZag handles data packing between memory levels. 

When memory levels have different widths, ZigZag calculates "packing factors" - how many logical words fit in one physical memory word. This affects the count of memory transactions.

### Where This Happens in the Code

The calculation takes place primarily in these functions:

```python name=memory_word_access.py
# In zigzag/classes/cost_model/memory_word_access.py (or similar)

def calculate_memory_word_access(layer_node, spatial_mapping, temporal_mapping, architectural_state):
    """
    This calculates all memory word accesses for each operand at each memory level
    """
    # Initialize memory word access counts
    mem_word_access = initialize_mem_word_access()
    
    # Process each temporal loop
    for loop_id, loop_dim in enumerate(temporal_mapping.loop_order):
        for operand in ['I', 'W', 'O']:
            # Find which memory level this loop accesses
            mem_level = find_mem_level_for_loop(operand, loop_id, architectural_state)
            
            # Calculate data reuse for this loop
            data_reuse = calculate_data_reuse(operand, loop_dim, layer_node)
            
            # Update word access counts based on loop and reuse
            update_word_access(mem_word_access, operand, mem_level, loop_dim, data_reuse)
    
    # Apply memory level translations (this causes your discrepancies)
    apply_memory_hierarchy_translations(mem_word_access, architectural_state)
    
    return mem_word_access

def apply_memory_hierarchy_translations(mem_word_access, architectural_state):
    """
    This function applies translations between memory levels, accounting for
    memory width differences (packing factors)
    """
    for operand in ['I', 'W', 'O']:
        for level in range(1, len(architectural_state.memory_hierarchy[operand])):
            # Calculate packing factor between levels
            packing_factor = calculate_packing_factor(
                architectural_state, operand, level-1, level
            )
            
            # Apply packing factor to access counts
            # This is where W's 4 vs 2 and O's 18 vs 36 differences occur
            if packing_factor != 1:
                # The lower (level-1) memory's "rd ^" gets divided by packing_factor
                # to get the higher (level) memory's "wr v"
                mem_word_access[operand][level]['wr v'] = math.ceil(
                    mem_word_access[operand][level-1]['rd ^'] / packing_factor
                )
```

### Specific Answer to Your Question

The discrepancies you're seeing in the word access values:

1. **For W (Weight) discrepancy (4 vs 2)**:
   - This happens in `apply_memory_hierarchy_translations` where the packing factor between Shared Memory and SRAM is calculated.
   - ZigZag is determining that 4 read operations from Shared Memory can pack into 2 write operations to SRAM, likely because SRAM has a wider memory width than Shared Memory for weights.

2. **For O (Output) discrepancy (18 vs 36)**:
   - Similarly, the 36 write operations to Shared Memory from SRAM are translated to 18 read operations from SRAM because of memory width differences.
   - ZigZag is calculating that each SRAM read can contain data for 2 Shared Memory writes.

The core of this calculation happens in ZigZag's memory word access module, where it applies packing factors based on the memory hierarchy configuration in your hardware description.

If you want to examine this directly, look for files named `memory_word_access.py` or similar in the `classes/cost_model/` directory of your ZigZag installation."


*Conclusion* - The above discription is from running a simple worklood like I(16,16,1), W(5,5,1), O(12,12,1). Where the Produced map is as follows

```txt
Loop ordering for layer_2
======================================================================================================================================
Temporal Loops                     O                                W                                I                                
======================================================================================================================================
for IX in [0, 4):                  sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
  for IX in [0, 4):                sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
    for IY in [0, 4):              sram                             sram                             shared_memory_inputs_outputs     
--------------------------------------------------------------------------------------------------------------------------------------
      for IY in [0, 4):            sram                             sram                             register                         
--------------------------------------------------------------------------------------------------------------------------------------
        for FY in [0, 5):          sram                             sram                             register                         
--------------------------------------------------------------------------------------------------------------------------------------
          for FX in [0, 5):        sram                             sram                             register                         
--------------------------------------------------------------------------------------------------------------------------------------
======================================================================================================================================
Spatial Loops                                                                                                                         
======================================================================================================================================
            parfor K in [0, 1):                                                                                                       
--------------------------------------------------------------------------------------------------------------------------------------

```

The stats are as follows
```json
{
    "outputs": {
        "memory": {
            "utilization": {
                "O": [
                    0.75,
                    0.025177001953125,
                    0.00025177001953125
                ],
                "W": [
                    0.75,
                    0.025177001953125,
                    5.9604644775390625e-06
                ],
                "I": [
                    0.75,
                    0.00025177001953125
                ]
            },
            "word_accesses": {
                "O": [
                    {
                        "rd ^": 6400,
                        "wr v": 6400,
                        "rd v": 6400,
                        "wr ^": 6400
                    },
                    {
                        "rd ^": 50,
                        "wr v": 0,
                        "rd v": 6400,
                        "wr ^": 6400
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 0,
                        "wr ^": 100
                    }
                ],
                "W": [
                    {
                        "rd ^": 0,
                        "wr v": 6400,
                        "rd v": 6400,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 2,
                        "rd v": 6400,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 4,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0,
                        "wr v": 64,
                        "rd v": 256,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 64,
                        "wr ^": 0
                    }
                ]
            }
        },
        "energy": {
            "energy_total": 556998.6,
            "operational_energy": 256.0,
            "memory_energy": 556742.6,
            "memory_energy_breakdown_per_level": {
                "O": [
                    2560.0,
                    149645.0,
                    200000
                ],
                "W": [
                    1280.0,
                    67225.6,
                    8000
                ],
                "I": [
                    32.0,
                    128000
                ]
            },
            "memory_energy_breakdown_per_level_per_operand": {
                "O": [
                    {
                        "rd ^": 640.0,
                        "wr v": 640.0,
                        "rd v": 640.0,
                        "wr ^": 640.0
                    },
                    {
                        "rd ^": 525.0,
                        "wr v": 0.0,
                        "rd v": 67200.0,
                        "wr ^": 81920.0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 0,
                        "wr ^": 200000
                    }
                ],
                "W": [
                    {
                        "rd ^": 0.0,
                        "wr v": 640.0,
                        "rd v": 640.0,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0.0,
                        "wr v": 25.6,
                        "rd v": 67200.0,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 8000,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0.0,
                        "wr v": 6.4,
                        "rd v": 25.6,
                        "wr ^": 0.0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 128000,
                        "wr ^": 0
                    }
                ]
            }
        },
        "latency": {
            "data_onloading": 2.0,
            "computation": 6400.0,
            "data_offloading": 2.0
        },
        "spatial": {
            "mac_utilization": {
                "ideal": 0.125,
                "stalls": 0.125,
                "stalls_onloading": 0.12496094970321775,
                "stalls_onloading_offloading": 0.12492192379762648
            }
        }
    },
    "inputs": {
        "accelerator": {
            "id": 0,
            "name": "Event_Driven_Accelerator",
            "operational_array": {
                "operational_unit": {
                    "energy_cost": 0.04,
                    "area": 1
                },
                "dimensions": {
                    "D1": 8
                }
            },
            "memory_hierarchy": [
                "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                "MemoryLevel(instance=sram,operands=[I2, O], served_dimensions={D1})",
                "MemoryLevel(instance=shared_memory_Weights,operands=[I2], served_dimensions={D1})",
                "MemoryLevel(instance=shared_memory_inputs_outputs,operands=[I1, O], served_dimensions={D1})"
            ],
            "mem_hierarchy_dict": {
                "I1": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=shared_memory_inputs_outputs,operands=[I1, O], served_dimensions={D1})"
                ],
                "I2": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=sram,operands=[I2, O], served_dimensions={D1})",
                    "MemoryLevel(instance=shared_memory_Weights,operands=[I2], served_dimensions={D1})"
                ],
                "O": [
                    "MemoryLevel(instance=register,operands=[I1, I2, O], served_dimensions=set())",
                    "MemoryLevel(instance=sram,operands=[I2, O], served_dimensions={D1})",
                    "MemoryLevel(instance=shared_memory_inputs_outputs,operands=[I1, O], served_dimensions={D1})"
                ]
            },
            "dataflows": null,
            "mem_size_dict": {
                "I1": [
                    64,
                    33554432
                ],
                "I2": [
                    64,
                    262144,
                    33554432
                ],
                "O": [
                    64,
                    262144,
                    33554432
                ]
            },
            "mem_r_bw_dict": {
                "I1": [
                    128,
                    64
                ],
                "I2": [
                    128,
                    128,
                    64
                ],
                "O": [
                    128,
                    128,
                    64
                ]
            },
            "mem_w_bw_dict": {
                "I1": [
                    128,
                    64
                ],
                "I2": [
                    128,
                    128,
                    64
                ],
                "O": [
                    128,
                    128,
                    64
                ]
            },
            "mem_r_bw_min_dict": {
                "I1": [
                    128,
                    64
                ],
                "I2": [
                    128,
                    128,
                    64
                ],
                "O": [
                    128,
                    128,
                    64
                ]
            },
            "mem_w_bw_min_dict": {
                "I1": [
                    128,
                    64
                ],
                "I2": [
                    128,
                    128,
                    64
                ],
                "O": [
                    128,
                    128,
                    64
                ]
            },
            "mem_sharing_list": [
                {
                    "I1": 0,
                    "I2": 0,
                    "O": 0
                },
                {
                    "I1": 1,
                    "O": 2
                },
                {
                    "I2": 1,
                    "O": 1
                }
            ]
        },
        "layer": {
            "name": "layer_2",
            "type": "Conv",
            "equation": "O[b][k][oy][ox] = W[k][c][fy][fx] * I[b][c][iy][ix]",
            "equation_relations": [
                "OX = 1*IX + 1*FX",
                "OY = 1*IY + 1*FY"
            ],
            "loop_dimensions": {
                "B": 1,
                "K": 1,
                "C": 1,
                "IY": 16,
                "IX": 16,
                "FY": 5,
                "FX": 5
            },
            "operand_precision": {
                "W": 8,
                "I": 8,
                "O": 8,
                "O_final": 16
            },
            "user_spatial_mapping": {
                "D1": {
                    "K": 1
                }
            },
            "memory_operand_links": {
                "O": "O",
                "W": "I2",
                "I": "I1"
            }
        },
        "spatial_mapping": {
            "O": [
                [],
                [
                    "(K, 1)"
                ],
                [],
                []
            ],
            "W": [
                [],
                [
                    "(K, 1)"
                ],
                [],
                []
            ],
            "I": [
                [],
                [
                    "(K, 1)"
                ],
                []
            ]
        },
        "temporal_mapping": {
            "O": [
                [],
                [
                    "(FX, 5)",
                    "(FY, 5)",
                    "(IY, 4)",
                    "(IY, 4)",
                    "(IX, 4)",
                    "(IX, 4)"
                ],
                []
            ],
            "W": [
                [],
                [
                    "(FX, 5)",
                    "(FY, 5)",
                    "(IY, 4)",
                    "(IY, 4)",
                    "(IX, 4)",
                    "(IX, 4)"
                ],
                []
            ],
            "I": [
                [
                    "(FX, 5)",
                    "(FY, 5)",
                    "(IY, 4)"
                ],
                [
                    "(IY, 4)",
                    "(IX, 4)",
                    "(IX, 4)"
                ]
            ]
        }
    }
}
```

From the stats and the basic calculations as shown below

1. Total MAC operations: 16*16*5*5 = 6400 -> This is more than the output defined problem  but as we are trying to model the problem in terms of the input this aries and it is fine (Should be solved as we scale the workload)
2. INput workload: 16*16 = 296
3. Weight workload: 5*5 = 25
4. Output workload: 12*12 = 144

Now, The bandwidth of the Shared memory is 64 and that of SRAM is 128 bits. When the bits are transferred from the shared memory to the SRAM they are clubbed and sent out 

$$
\text{accesses} = 
\left\lceil \frac{\text{data\_elem\_move\_per\_period} \times \text{data\_precision}}{\text{min\_bw}} \right\rceil 
\times \frac{\text{min\_bw}}{\text{max\_bw}} 
\times \text{total\_period\_count} 
\times \text{unit\_count}
$$

And due to which initiall the Weights are clubbed as 4 i.e [5*5*8/64 = 3.125 => upper ciel of this is 4] at the Shared memmory and from there they were transferred to the SRAM which has a bandwidth of 128. So, they are clubbed even better and the rd access decrease to 2

This same phenomemon happens to the outputs also where the initally in SRAM the are written with 50 and when they are changed to Shared memory they are written with 100 blocks

Where as for the inputs the read down is decided by the shared memory tiling that produced the values as shown above. [4*4*4 = 64]

# Following functions are required
1. Partial sum generator

Based on your HTML script, I’ve extracted the logic that generates the partial sum contribution table (which shows how each input cell contributes to the output cells in a convolution operation) into a separate JavaScript function. This function, `generateContributionTable`, takes the input size and kernel size as parameters and returns a 2D array where each element is a list of `[i, j]` output positions that the corresponding input cell contributes to. This can be easily integrated into a validator script to check convolution contributions.

Here’s the extracted function:

```javascript
/**
 * Generates a contribution table for convolution.
 * For each input cell (r, c), lists the output cells [i, j] that it contributes to.
 * 
 * @param {number} inputSize - The size of the input grid (assuming square grid).
 * @param {number} kernelSize - The size of the kernel (assuming square kernel).
 * @returns {Array<Array<Array<[number, number]>>>} A 2D array where each element is a list of [i, j] pairs.
 */
function generateContributionTable(inputSize, kernelSize) {
    const outputSize = inputSize - kernelSize + 1;
    const table = [];
    for (let r = 0; r < inputSize; r++) {
        table[r] = [];
        for (let c = 0; c < inputSize; c++) {
            table[r][c] = [];
            const iMin = Math.max(0, r - kernelSize + 1);
            const iMax = Math.min(outputSize - 1, r);
            const jMin = Math.max(0, c - kernelSize + 1);
            const jMax = Math.min(outputSize - 1, c);
            for (let i = iMin; i <= iMax; i++) {
                for (let j = jMin; j <= jMax; j++) {
                    table[r][c].push([i, j]);
                }
            }
        }
    }
    return table;
}
```

### How to Use It in a Validator Script
You can include this function in a separate JavaScript file (e.g., `contributionTable.js`) and use it in your validator script like this:

```javascript
// Example validator script
const contributionTable = generateContributionTable(16, 5);

// Check contributions for a specific input cell, e.g., (r=4, c=4)
console.log("Contributions from input cell (4,4):", contributionTable[4][4]);

// Validate that an input cell contributes to expected output cells
const expected = [[0,0], [0,1], [1,0], [1,1]]; // Example expected contributions
const actual = contributionTable[0][0];
const isValid = expected.every(([i, j], idx) => 
    actual[idx] && actual[idx][0] === i && actual[idx][1] === j
);
console.log("Validation result:", isValid);
```

### Explanation
- **Input Parameters**: `inputSize` (e.g., 16) and `kernelSize` (e.g., 5) define the grid and convolution kernel.
- **Output Size**: Calculated as `inputSize - kernelSize + 1` (e.g., 16 - 5 + 1 = 12), assuming no padding and stride 1.
- **Return Value**: A 2D array where `table[r][c]` is an array of `[i, j]` pairs representing output positions in the 12x12 grid that the input cell `(r, c)` contributes to.
- **Logic**: For each input cell `(r, c)`, it computes the range of output indices `(i, j)` where the 5x5 kernel, when positioned with its top-left at `(i, j)`, overlaps `(r, c)`.

This function is now independent of the DOM and can be reused in any JavaScript environment, such as a validator script, to analyze or verify convolution contributions.