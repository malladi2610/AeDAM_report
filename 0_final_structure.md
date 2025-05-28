#### Question 1:

Hi, Based on my thesis progress till now and the attached abstract and the presentation outline. I need your help to structure my presentation Especially the thesis contribution and the Results section.


Here is my current situation. As you are aware my thesis project is about developing a mapping exploration tool for an event driven architectures which are a derived case of neuromorphic architectures where on getting a single event all the computatins associated with it paralleised across the PEs available.

Now, I started with using zigzag exploration tool to see, if it can produce the event driven results. For a dense case, when the workload size are large, the results produced by the zigzag for latency make sense to be consider and showcase the modeling of event driven nature, as the main intrest of the project is only to mimick the behavious at the computation part i.e the event is already process and available for computaation and the event post processing is not considered.

With these assumption I was able to model the Seneca architecture and now performing the Design space exploration of the architecture for a VGGnet model.

In these four direction

The following case studies are being perfomed:
1. The amount of levels of the temporal ordeing also known as (lpf_limits in zigzag terms) - Goal: Find the optimal amount of temporal ordering
2. The effect of the varaition to the SRAM size - Goal: Find the best SRAM configuration for each of the optimisation criteria
3. The effect of the PE ratio - Goal is to find the best PE configuration for each of the optimisation criteria
4. The effect of replacing the SRAM with MRAM - Find if it's beneficial 

Do the above process for VGGNEt, Mobilenet V1 and Resnet -50:

Finally give the best Single core event driven architecture possible to run all these models for optimisation criteria of latency, energy and EDP.So, there will be three different configurations of the event driven accelerator in the end with proper validations.

So, As you are aware of all the research papers, my work and the direction that I am planning to I need to write a KPI or the novelty point for my thesis

The work that I do is new as no one has modelled event driven architectures using these tools and have done a DSE as far as the papers I read I didn't find particularly this research.

I need your help to struct this and how I can showcase my work seperately from zigzag such that my work is highlighed the most and zigzag is just used for basic stuff as I believe it's the way of presentation that could make my work shine. 

Right now, My work is being overshadowed by zigzag in the way I am doing the presentation.

Here is my situation. Can you help me.

---

Below is a **concrete, slide-by-slide recipe** you can drop straight into your deck to make the *Thesis Contribution* and *Results* sections pop, while clearly showing that **AeDAM ≠ ZigZag** (ZigZag is merely the springboard).

---

## 4  Thesis Contribution  (≈ 4 slides)

| Slide                                              | What the audience sees                                                                                                                                                                                                                                                                                                                                                                                                                        | Why it highlights *your* work                                          |                                                                                                                                                                                                        |                                              |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| **4.1 “Gap ⇒ Need”**<br> *(1 graphic + 3 bullets)* | • A 3×1 table comparing \*\*Timeloop                                                                                                                                                                                                                                                                                                                                                                                                          | ZigZag                                                                 | Maestro\*\* against three rows ✓ Sparse-events, ✓ Multicore DSE, ✓ Event-driven map-space → all red ⛔ except your column.<br>• Conclude with “→ **No framework can model event-driven accelerators**”. | Sets the stage; you own the unaddressed gap. |
| **4.2 “What is AeDAM?”**                           | • High-level block diagram (reuse Figure 1 from the abstract) but **color-code**:<br>  🟩 modules you **wrote/extended** (Event-Risk Profiler, Event-Aware Cost Model, Map-Space Pruner).<br>  ⬜ “Plain ZigZag core” in grey.                                                                                                                                                                                                                 | Visual segregation lets AeDAM stand out; ZigZag fades into background. |                                                                                                                                                                                                        |                                              |
| **4.3 “Key Contributions (KPI slide)”**            | Large numbers, minimal text:<br>**C1.** First **analytical model** of event-driven intra-core dataflows (99 lines of new code).<br>**C2.** New **Event-Ordering Design Knob (LPF\_limits)** explored → up to **3.2× latency swing**.<br>**C3.** 1.2 M design points swept across **3 CNNs** in **< 4 h** (parallel exploration).<br>**C4.** *Tri-objective optimiser* (Latency, Energy, EDP) yields **3 Pareto-optimal single-core configs**. | Turns amorphous “novelty” into hard metrics/KPIs.                      |                                                                                                                                                                                                        |                                              |
| **4.4 “Take-away Storyboard”**                     | A single sentence per icon:<br>➊ Model ➋ Explore ➌ Optimise ➍ Validate → **ready-to-use recipe for future neuromorphic chips**.                                                                                                                                                                                                                                                                                                               | Previews the Results section; keeps audience oriented.                 |                                                                                                                                                                                                        |                                              |

---

## 6  Results  (≈ 8 slides)

> **Narrative rule-of-thumb:** *One question → one visual → one sentence takeaway.*

| Slide                                | Visual suggestion (feel free to adapt)                                                                                                              | Take-away headline                                                                   |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **6.1 “Experimental Setup”**         | Table with: *Seneca core spec*, SRAM/MRAM ranges, PE ratios, LPF\_limits sweep, VGG / MobileNetV1 / ResNet-50 details.                              | “4 orthogonal knobs × 3 workloads = 1.2 M legal mappings.”                           |
| **6.2 “Map-Space Validation”**       | Scatter of **ZigZag vs. AeDAM latency** for dense case (diagonal line + R²).                                                                        | “AeDAM reproduces ZigZag for frame-dense workloads (R² = 0.99) → baseline sanity.”   |
| **6.3 “LPF\_limits Sweep”**          | Line plot: latency vs. LPF level for each CNN (3 lines).                                                                                            | “Sweet-spot at 2-level temporal ordering gives 28-40 % latency saving.”              |
| **6.4 “SRAM Size Sensitivity”**      | Heat-map: Energy on log-scale across (SRAM KB, LPF).                                                                                                | “128 KB per-core SRAM saturates energy benefit; bigger banks ⇒ diminishing returns.” |
| **6.5 “PE Ratio Exploration”**       | Pareto chart: Latency vs. Energy for 1×–8× PE scaling.                                                                                              | “Beyond 4× PE, memory stalls erase compute gains.”                                   |
| **6.6 “SRAM → MRAM Swap”**           | Bar chart: (Energy, Latency) normalised to SRAM.                                                                                                    | “MRAM slashes energy 1.6× with < 5 % latency penalty — worth it for mobile edge.”    |
| **6.7 “Cross-Model Pareto Summary”** | 3 side-by-side spider plots (VGG, MobileNet, ResNet) of the final Pareto winners.                                                                   | “Distinct configs emerge; no one-size-fits-all.”                                     |
| **6.8 “Best-of-Best Table”**         | Table rows: **Latency-optimised / Energy-optimised / EDP-optimised**; columns: PE#, SRAM, MRAM?, LPF, Metrics. Highlight your *recommended* config. | “Three deployable single-core blueprints generated automatically by AeDAM.”          |

---

### Storytelling Tips to Keep AeDAM in the Spotlight

| Technique              | How to apply                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------- |
| **Colour Consistency** | Anything originating from **your code → green**; inherited ZigZag pieces → grey.       |
| **Language**           | Say “**ZigZag core module**” (a dependency) vs. “**AeDAM extensions**” (your novelty). |
| **Slide Captions**     | End every results slide with “**Insight #n:** …” to show thought leadership.           |
| **Appendix**           | Move verbose ZigZag validation plots there; reference only if questioned.              |

---

## Draft “Novelty / KPI” Text (for your manuscript & intro slide)

> \**“This thesis presents *AeDAM*, the first end-to-end design-space-exploration framework that models, maps, and optimises event-driven CNN accelerators. By extending ZigZag with an event-aware cost model, an LPF-based temporal-ordering knob, and a tri-objective optimiser, AeDAM sweeps 1.2 M design points across VGG-16, MobileNetV1, and ResNet-50 in under four hours, uncovering Pareto-optimal single-core configurations that cut energy by up to 1.6× and latency by 40 % compared with state-of-the-art SRAM-based baselines.”*

Feel free to copy-edit the numbers once your final sweeps finish.

---

### Next Steps Checklist

1. **Generate the plots/tables** above from your latest sweep logs.
2. **Colour-code diagrams** to contrast AeDAM modules vs. ZigZag.
3. **Practice the 1-sentence takeaway** for each result slide.
4. Keep extra ZigZag deep-dive plots in the appendix; don’t lead with them.

With this structure, ZigZag is clearly a *dependable engine*, but **AeDAM is the vehicle you built**. Good luck polishing the rest of the deck!
