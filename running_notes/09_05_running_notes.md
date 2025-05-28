# Goal:

To finalise the flow of the scripts for validation process and the DSE process and finish the development.

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


---

# Final DSE process

**Developing integrated flow scripts**

For the DSE testing process, the user wants to develop scripts that work together in a logical flow. I'll need to break down this task into key components: creating scripts for each processing step (like zigzag exploration, mapping filters, and optimizing cost models), collecting outputs for each CNN model, and ensuring the best configuration for each optimization criterion. They also want a main function that manages and assembles these scripts, collecting results for analysis. It’ll be important to structure everything to integrate seamlessly.

**Planning DSE testing flow**

I’ll create a detailed plan, starting with software architecture, dataset organization, and file structure for the DSE testing process. This includes a main script that orchestrates everything: reusable modules, optimal parameterization, and config files (YAML/JSON). I’ll also address concurrency with joblib, using HDF5 for results, and pandas for organizing data. The plan will focus on event-driven accelerator mapping, include functions like "LayerGraphMapper" and "CostModelEvaluation," and ensure integration with version control. I'll also mention packaging options and CLI usage.

**Searching for zigzag documentation**

It seems like I could answer based on what I know, but the instructions suggest I should browse for up-to-date or niche information. It might be better to confirm by looking up the zigzag documentation, especially to verify the API names. It'll only take a moment to search and I'm sure it'll help ensure my response is accurate. Better to be cautious and up-to-date!


**Checking zigzag documentation**

While I could answer based on what I know, it's better to browse and confirm, especially since the instructions suggest browsing for up-to-date or niche information. The GitHub readme may point to the API, and checking the docs should give clarity on relevant classes like "accelerator," "workload," and "mapping." I’ll quickly scan the docs to confirm these details, just to be sure everything’s up to date.


**Checking zigzag docs and structure**

I could probably answer based on what I know, but it’s safer to browse the zigzag documentation to confirm API names and details. The GitHub readme might direct me to the API, and I’m expecting to see classes like "accelerator," "workload," and "mapping" in the docs. To be thorough, I’ll open the readme and explore the zigzag directory for the most accurate and up-to-date information. This will make sure everything is aligned with the latest details.


**Checking zigzag docs structure**

It’s probably safe to answer based on what I know, but it’s better to confirm. I’ll browse the zigzag documentation to be thorough, as instructions suggest browsing for up-to-date info. The GitHub readme should show the API’s structure, with classes like "accelerator," "workload," and "mapping." I’ll scroll through the directory to get a clearer picture and make sure I'm on the right track with the most current details from the docs.


**Checking docs for details**

I could probably handle this without browsing, but it's safer to check for the latest, specific info. The GitHub readme might show the API path, especially with classes like “accelerator,” “workload,” and “mapping.” I’m scrolling through the zigzag docs now but haven't found clear details yet. It feels like I'm nearing the bottom where things are less clear, so I definitely need to open more sections of the docs for a complete picture.


**Exploring API docs**

I could probably answer this from knowledge, but browsing is still safer for the latest or niche details. The GitHub readme might give insights on the API path, and the docs likely mention "accelerator," "workload," and "mapping" classes. I’m scrolling through now, but it looks like I need to dive deeper into the API docs to fully understand the structure. Time to open those docs! They should provide the specific guidance I need to clarify things.


**Exploring zigzag API**

I feel confident based on what I know, but it's safer to browse for up-to-date information. Scrolling through the zigzag docs now—looks like the GitHub readme could indicate the API path, and the docs mention things like "accelerator," "workload," and "mapping." We need to dive deeper into the API docs to explore the mapping generation details. It’s looking like zigzag.api.get\_hardware\_performance\_zigzag could work for generating the performance map, but the user’s process for splits and filtering might require more.

**Outlining script modules**

I’m proposing a structured set of modules to handle the workflow. The config directories include YAML files for specifying the workload, accelerator templates, and their attributes. Scripts will cover tasks from generating mapspace and evaluating costs, to extracting Pareto-optimal solutions and running case studies. The pipeline will utilize tools like Hydra for config management, joblib for parallelization, and possibly Snakemake for orchestration. We’ll also have utility scripts for logging, space analysis, and parameter sweeps. The plan will cover environment setup, version control, and testing.


Thought for 56 seconds


Below is a software-engineering blueprint you can turn directly into a small, well-tested code base.
It focuses only on **Part 1 – the DSE testing process**, and it is organised so that a single *entry-point* script (`main.py`) can launch the whole flow or any subset of case-studies you want.

---

## 1. High-level workflow

```
 ┌───────────────┐ 1. sweep over DSE‐points
 │   main.py     │────────────────────────────────────────────┐
 └───────────────┘                                             │
          ▼                                                    │
┌────────────────┐ 2. prepare workload & accel config          │
│   setup_run    │                                             │
└────────────────┘                                             │
          ▼                                                    │
┌────────────────┐ 3. generate event-driven map-space (K/FX/FY)│
│ mapspace_gen   │                                             │
└────────────────┘                                             │
          ▼                                                    │
┌────────────────┐ 4. cost-model evaluation with ZigZag        │
│ cost_eval      │  (energy, latency, EDP per mapping)         │
└────────────────┘                                             │
          ▼                                                    │
┌────────────────┐ 5. extract per-layer best mappings          │
│ pareto_extract │  (3 metrics → 3 CSV/YAML files)             │
└────────────────┘                                             │
          ▼                                                    │
┌────────────────┐ 6. aggregate across layers & runs           │
│   summarise    │  → final results table & artefacts          │
└────────────────┘                                             │
```

---

## 2. Repository & script layout

```
thesis-dse/
│
├─ configs/                   # pure data, no code
│   ├─ workloads.yaml         # VGGNet, MobileNet-V1, ResNet-50
│   ├─ base_accelerator.yaml  # single-core template
│   ├─ sweeps/
│   │    ├─ lpf_limits.yaml
│   │    ├─ sram_sizes.yaml
│   │    ├─ pe_ratios.yaml
│   │    └─ mram_options.yaml
│   └─ logging.yaml
│
├─ scripts/
│   ├─ main.py                # CLI orchestrator
│   ├─ 00_setup_run.py        # copies & patches configs
│   ├─ 01_mapspace_gen.py     # produces *.mapspace pickle
│   ├─ 02_cost_eval.py        # calls ZigZag API
│   ├─ 03_pareto_extract.py   # writes best-mapping CSV
│   ├─ 04_summarise.py        # collects per-study metrics
│   └─ utils/
│        ├─ logger.py
│        ├─ search_space.py   # dataclasses for Mapping, Metrics
│        └─ pareto.py
│
├─ results/                   # auto-generated, git-ignored
│   ├─ <timestamped run>/
│   │    ├─ logs/
│   │    ├─ mapspaces/
│   │    ├─ evaluations/
│   │    ├─ pareto/
│   │    └─ summary/
│   └─ latest → symlink
│
└─ requirements.txt
```

### Why separate tiny scripts?

* **Loose coupling / restartability** – you can rerun *only* the cost model or *only* the summariser.
* **Parallelism** – `main.py` can dispatch multiple DSE points to workers (e.g. Python `multiprocessing` or SLURM array).
* **Deterministic artefacts** – every stage writes a well-defined file so later stages never recompute unless you delete the file.

---

## 3. Key modules in detail

| Script / module            | Core responsibility                                                                                                                                                                                    | Important functions / cli                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| **00\_setup\_run.py**      | Take a sweep item (e.g. `sram_size=128 kB`), patch the base accelerator YAML (and possibly the global `lpf_limit`) and copy the chosen workload ONNX into the run folder.                              | `prepare_run(dse_id, workload_id)`               |
| **01\_mapspace\_gen.py**   | For each layer generate *three* temporal orderings (split-K, split-FX, split-FY); filter out non-event-driven maps and pooling layers; merge into one pickle (`mapspace.pkl`).                         | `generate_event_driven_mapspace(run_dir)`        |
| **02\_cost\_eval.py**      | Load the pickle, loop through mappings, call `zigzag.api.get_hardware_performance_zigzag(opt=…)` for *latency*, *energy*, *EDP* and cache results (`evaluations.jsonl`).                               | `evaluate_costs(run_dir)` (uses graceful `tqdm`) |
| **03\_pareto\_extract.py** | Parse the JSONL, pick the single best mapping per layer under each optimisation target; emit three CSVs (`best_latency.csv`, …) containing mapping id + stats.                                         | `extract_pareto(run_dir)`                        |
| **04\_summarise.py**       | Across all DSE points of a sweep, build a dataframe and rank configurations. Produce: <br>• `summary.csv` (one row per DSE point, three metric columns) <br>• optional plots (EDP vs SRAM size, etc.). | `summarise_sweep(sweep_name)`                    |

Utility modules

* **search\_space.py** – dataclass `Mapping`, `LayerMetrics`, helper to serialize/deserialize.
* **pareto.py** – generic `get_pareto_front(points, objectives)`.
* **logger.py** – returns a colourised `logging.Logger` based on `configs/logging.yaml`.

---

## 4. The orchestrator (`main.py`)

```python
#!/usr/bin/env python3
"""
CLI entry-point. Examples
  $ python main.py --all
  $ python main.py --sweep lpf_limits --workers 8
"""
import argparse
from pathlib import Path
from multiprocessing import Pool
from scripts import (
    setup_run, mapspace_gen, cost_eval, pareto_extract, summarise
)
# 1. parse args → list of (dse_point, workload) tuples
# 2. in Pool: call setup_run → mapspace_gen → cost_eval → pareto_extract
# 3. after Pool join: summarise each sweep
# 4. print / log final “best config” table
```

Design notes

* Uses **argparse** with mutually exclusive groups (`--all`, `--resume`, `--only-mapspace`, …).
* Every sub-step is wrapped in `try/except`; failures are logged and collected so the run can continue.
* `--workers N` enables local parallelism; for HPC you can replace Pool with a SLURM launcher.

---

## 5. Configuration strategy

* **YAML everywhere** so files are human-readable and version-controllable.
* **`OmegaConf`** (or plain `yaml.safe_load`) to merge default + sweep-specific overrides.
* Example `sweeps/pe_ratios.yaml`

```yaml
- id: pe_x1
  patch:
    accelerator:
      PE_DIM: [16, 16]
- id: pe_x2
  patch:
    accelerator:
      PE_DIM: [32, 32]
```

`00_setup_run.py` simply deep-merges `patch` into `base_accelerator.yaml`.

---

## 6. Output artefacts

* **Per-run folder**: everything needed for debugging a single (workload, DSE-point) pair.

  * `mapspace.pkl`
  * `evaluations.jsonl`
  * `best_latency.csv`, `best_energy.csv`, `best_edp.csv`
  * `run.log`
* **Per-sweep folder**:

  * `summary.csv` – columns: `dse_id`, `best_latency`, `best_energy`, `best_edp`
  * `plots/` – automatically named PNGs.
* These can later feed the *Validation process* (Part 2).

---

## 7. Coding against ZigZag

Key call:

```python
from zigzag import api
energy, latency, _ = api.get_hardware_performance_zigzag(
        workload=workload_yaml_or_onnx,
        accelerator=acc_yaml,
        mapping=mapping_yaml,
        opt="energy",                       # or "latency" / "EDP"
        dump_folder=run_dir/"zz_outputs",
)
```

API docs: ﻿cite﻿turn5view0﻿cite﻿

> **Tip**: cache the `cme` pickles – they are large but let you post-process without re-simulation.

---

## 8. Parallelism & performance tips

| Task                 | Strategy                                                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Map-space generation | Fast; serial execution is fine.                                                                                               |
| Cost evaluation      | Dominant runtime – parallelise at *mapping* granularity (`joblib.Parallel`) or at *run* granularity (`multiprocessing.Pool`). |
| Pareto extraction    | O(#mappings log #layers) – negligible.                                                                                        |

Add a `--max-maps` arg when prototyping.

---

## 9. Quality & reproducibility

* **Version-pin** ZigZag and your own repo via `requirements.txt`.
* Each run writes a `git-hash.txt` for both ZigZag and your code.
* `pytest` folder with small unit tests:

  * sanity check that `mapspace.pkl` is not empty
  * check that Pareto extraction truly dominates.

---

## 10. Deliverables for your thesis

1. **Three YAML files** describing the *best single-core* event-driven accelerator for *latency*, *energy*, *EDP* respectively (direct output of `04_summarise.py`).
2. **Plots & tables** auto-exported to `results/latest/`.
3. **Appendix scripts** – a Makefile (or `bash run_all.sh`) that re-runs the entire study on any fresh machine.

---

### Next step

Once you agree on the structure, we can flesh out concrete code skeletons or move to **Part 2 – the Validation process**.


---

Here is the information about the execution process I have now.

There is a zigzag main.py that is modified to be *main_monkey_patch.py* 

THis script can extract the spatial and temporal mapping by taking the architecture, workload and the mapping constraint as the input. 

Here is the process I have in the mind

1. This script for a selected DSE point has to run three times with spatial unrolling across K, FX and FY and collect all the temporal mappings for each of the spatial mappings
As the workload being tested is VGGnet it has 21 layer and the zigzag produced possible temporal ordering combinations for all the layer. This is stored in a folder called output_mappings
2. This all_mapping.json would be created for each of the spatial unrolling and this needs to pass from two filters which is the *json_filter_v2" and the 
*pooling_filter*
3. This gives the temporal mappings which are event driven for each of the spatial splitting. and they need to be merged together to create a single event driven mapspace which can be passed to the cost model of zigzag
4. To pass the mapspace to the cost model some special conversions needs to be perfomed which are yet to be figured out.
5. This will produce the output with a cost estimate for all the event driven mappings from this based on the optimisation criteria Latency and energy and the EDP is a drived cased and needs some processing seperately
6. This will from the scanning of the cost ematimated map space with give three outputs with the relevant stats of each of the layer loop orderings for latency, energy and the EDP which is calculated for each of the mapspace to find the best combination.
7. This will add a single point in the three pareto graphs.

This process repeats for currently 10 points in the DSE, which needs to be extended to find the platue of the curve.

Here are the scripts I already have

1. main_monkey_patch.py

import json
import logging
import os
import sys
from functools import wraps

# ZigZag imports
from zigzag.api import get_hardware_performance_zigzag
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.cost_model.cost_model import CostModelEvaluation

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format)

sys.path.insert(0, os.getcwd())  # For importing zigzag

# Define output paths
base_dir = os.path.expanduser("~/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/May_experiments/results-08-05/outputs_mappings")
os.makedirs(base_dir, exist_ok=True)
output_json = os.path.join(base_dir, "all_mappings.json")
latency_log_path = os.path.join(base_dir, "latency_components.log")
stall_log_path = os.path.join(base_dir, "stall_components.log")

# Create global dictionaries to store mappings and latency data
all_mappings = {}
latency_data = {}
spatial_count = 0
temporal_count = 0

# Create or clear log files
with open(latency_log_path, 'w') as f:
    f.write("LATENCY COMPONENTS LOG\n")
    f.write("=====================\n\n")

with open(stall_log_path, 'w') as f:
    f.write("STALL COMPONENTS LOG\n")
    f.write("===================\n\n")

# Helper function to log to file
def log_to_file(filepath, message):
    with open(filepath, 'a') as f:
        f.write(f"{message}\n")

# Create helper conversion functions (same as your original script)
def convert_mapping_to_dict(mapping):
    """Convert any mapping to a dictionary format."""
    # Same implementation as in your script
    try:
        # Handle MappingSingleOADim objects specifically
        if hasattr(mapping, "__class__") and mapping.__class__.__name__ == "MappingSingleOADim":
            if hasattr(mapping, "items"):
                return {str(dim): int(size) for dim, size in mapping.items()}
            elif hasattr(mapping, "__dict__"):
                return {k: convert_mapping_to_dict(v) for k, v in mapping.__dict__.items() 
                        if not k.startswith("_")}
            else:
                return str(mapping)
            
        # Handle normal mapping_dict attribute
        if hasattr(mapping, "mapping_dict"):
            return {str(k): {str(dim): int(size) for dim, size in v.items()} 
                    for k, v in mapping.mapping_dict.items()}
            
        # Handle loop_dim_size attribute
        elif hasattr(mapping, "loop_dim_size"):
            return {str(k): {str(dim): int(size) for dim, size in v.items()} 
                    for k, v in mapping.loop_dim_size.items()}
                    
        # Handle dictionary-like objects
        elif hasattr(mapping, "items"):
            result = {}
            for k, v in mapping.items():
                # Convert both key and value
                key = str(k)
                if hasattr(v, "items") or hasattr(v, "__dict__") or hasattr(v, "mapping_dict"):
                    value = convert_mapping_to_dict(v)
                else:
                    # Try to convert to basic Python types
                    try:
                        value = int(v)
                    except (TypeError, ValueError):
                        try:
                            value = float(v)
                        except (TypeError, ValueError):
                            value = str(v)
                result[key] = value
            return result
            
        # Handle any object with __dict__
        elif hasattr(mapping, "__dict__"):
            return {k: convert_mapping_to_dict(v) for k, v in mapping.__dict__.items() 
                    if not k.startswith("_")}
                    
        # Default to string representation
        return str(mapping)
        
    except Exception as e:
        return f"Error converting mapping: {str(e)}"

# Function to safely save mappings to JSON (same as your original script)
def save_mappings_to_json(mappings, filepath):
    # Same implementation as in your script
    # Convert all objects to JSON-serializable dictionaries
    serializable_mappings = {}
    for key, value in mappings.items():
        if isinstance(value, dict):
            # Handle nested dictionaries
            processed_value = {}
            for k, v in value.items():
                if k == "spatial_mapping" or k == "temporal_mappings":
                    if isinstance(v, list):
                        processed_value[k] = [convert_mapping_to_dict(item) for item in v]
                    else:
                        processed_value[k] = convert_mapping_to_dict(v)
                else:
                    processed_value[k] = v
            serializable_mappings[key] = processed_value
        else:
            serializable_mappings[key] = convert_mapping_to_dict(value)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_mappings, f, indent=2)

# Store original methods
original_spatial_run = SpatialMappingGeneratorStage.run
original_temporal_run = TemporalMappingGeneratorStage.run
original_combine_data_transfer_rate = CostModelEvaluation.combine_data_transfer_rate_per_physical_port
original_calc_overall_latency = CostModelEvaluation.calc_overall_latency

# Monkey patch the stall calculation method to log stall components
def patched_combine_data_transfer_rate(self):
    # First call the original method
    original_combine_data_transfer_rate(self)
    
    # Now log the stall components
    layer_id = getattr(self.layer, "id", "unknown")
    message = f"\n==== STALL COMPONENTS FOR LAYER {layer_id} ====\n"
    
    # Log port activity collect
    message += "PORT ACTIVITIES:\n"
    for idx, mem_ports in enumerate(self.port_activity_collect):
        message += f"  Memory Instance {idx}:\n"
        for port_name, port_activity in mem_ports.items():
            message += f"    Port {port_name}:\n"
            for act in port_activity:
                message += f"      {act.served_op_lv_dir}: real={act.real_cycle}, allowed={act.allowed_cycle}, period={act.period}, count={act.period_count}, stall/slack={act.stall_or_slack}\n"
    
    # Log memory updating window union
    message += "\nMEMORY UPDATING WINDOW UNION:\n"
    for idx, mem_ports in enumerate(self.mem_updating_window_union_collect):
        message += f"  Memory Instance {idx}:\n"
        for port_name, value in mem_ports.items():
            message += f"    Port {port_name}: {value}\n"
    
    # Log stall/slack combination
    message += "\nSTALL/SLACK COMBINATION:\n"
    for idx, mem_ports in enumerate(self.stall_slack_comb_collect):
        message += f"  Memory Instance {idx}:\n"
        for port_name, value in mem_ports.items():
            message += f"    Port {port_name}: {value}\n"
    
    message += f"\nFINAL STALL_SLACK_COMB: {self.stall_slack_comb}\n"
    message += "=========================================\n"
    
    # Log to file
    log_to_file(stall_log_path, message)
    
    return

# Monkey patch the latency calculation method to log latency components
@wraps(original_calc_overall_latency)
def patched_calc_overall_latency(self, *args, **kwargs):
    """
    This is a patched version of calc_overall_latency that logs additional information.
    It properly handles the original method's signature using *args and **kwargs.
    """
    # First call the original method correctly (passing only what it expects)
    original_calc_overall_latency(self)
    
    # Now log all the latency components
    layer_id = getattr(self.layer, "id", "unknown")
    message = f"\n==== LATENCY COMPONENTS FOR LAYER {layer_id} ====\n"
    
    # Log all the latency components
    message += f"Layer total MAC operations: {self.layer.total_mac_count}\n"
    message += f"Accelerator operational array units: {self.accelerator.operational_array.total_unit_count}\n"
    message += f"Ideal cycle count: {self.ideal_cycle}\n"
    message += f"Ideal temporal cycle: {self.ideal_temporal_cycle}\n"
    message += f"MAC spatial utilization: {self.mac_spatial_utilization}\n"
    message += f"Stall/slack combined: {self.stall_slack_comb}\n"
    message += f"Data onloading cycle: {self.data_onloading_cycle}\n"
    message += f"Data offloading cycle: {self.data_offloading_cycle}\n"
    
    # Calculate and log the breakdown
    message += "\nLATENCY BREAKDOWN:\n"
    message += f"Latency total0 (computation): {self.latency_total0}\n"
    message += f"  = ideal_temporal_cycle ({self.ideal_temporal_cycle}) + stall_slack_comb ({self.stall_slack_comb})\n"
    message += f"Latency total1 (+ onloading): {self.latency_total1}\n"
    message += f"  = latency_total0 ({self.latency_total0}) + data_onloading_cycle ({self.data_onloading_cycle})\n"
    message += f"Latency total2 (+ offloading): {self.latency_total2}\n"
    message += f"  = latency_total1 ({self.latency_total1}) + data_offloading_cycle ({self.data_offloading_cycle})\n"
    
    message += "\nMAC UTILIZATION:\n"
    message += f"MAC utilization0: {self.mac_utilization0}\n"
    message += f"MAC utilization1: {self.mac_utilization1}\n"
    message += f"MAC utilization2: {self.mac_utilization2}\n"
    
    # Log data loading details
    message += "\nDATA LOADING DETAILS:\n"
    for layer_op, data in self.data_loading_cc_per_op.items():
        message += f"  Layer operand {layer_op.name}:\n"
        for key, (cycle, shared) in data.items():
            message += f"    {key}: cycles={cycle}, shared={shared}\n"
    
    # Log data offloading details
    message += "\nDATA OFFLOADING DETAILS:\n"
    for layer_op, data in self.data_offloading_cc_per_op.items():
        message += f"  Layer operand {layer_op.name}:\n"
        for key, (cycle, shared) in data.items():
            message += f"    {key}: cycles={cycle}, shared={shared}\n"
    
    message += "=========================================\n"
    
    # Log to file
    log_to_file(latency_log_path, message)
    
    # Store the data for later use
    latency_data[f"layer_{layer_id}"] = {
        "ideal_cycle": self.ideal_cycle,
        "ideal_temporal_cycle": self.ideal_temporal_cycle,
        "stall_slack_comb": self.stall_slack_comb,
        "data_onloading_cycle": self.data_onloading_cycle,
        "data_offloading_cycle": self.data_offloading_cycle,
        "latency_total0": self.latency_total0,
        "latency_total1": self.latency_total1,
        "latency_total2": self.latency_total2
    }

# Monkey patch methods from your original script (spatial and temporal mapping generation)
def patched_spatial_run(self):
    global spatial_count
    layer_id = getattr(self.layer, "id", getattr(self.layer, "name", "unknown"))
    print(f"Running patched spatial mapping generator for layer: {layer_id}")
    
    # Collect all spatial mappings first
    generated_mappings = list(self.generate_spatial_mappings())
    print(f"Generated {len(generated_mappings)} spatial mappings")
    
    # Capture each spatial mapping
    for i, generated_mapping in enumerate(generated_mappings):
        # Extract mapping info before yielding it
        spatial_id = f"spatial_{spatial_count}"
        spatial_count += 1
        
        # Convert to dictionary - will be converted again when saving
        mapping_dict = generated_mapping  # Store original for now
        
        # Save in our global dictionary
        all_mappings[spatial_id] = {
            "layer_id": layer_id,
            "spatial_mapping": mapping_dict,
            "temporal_mappings": []
        }
        
        print(f"Captured spatial mapping {spatial_id}: {generated_mapping}")
        
        # Set on layer as the original method does
        self.layer.spatial_mapping = generated_mapping
        
        # Save periodically
        if len(all_mappings) % 5 == 0:
            try:
                save_mappings_to_json(all_mappings, output_json)
            except Exception as e:
                print(f"Error saving spatial mappings: {e}")
    
    # Run the original method to continue the pipeline
    for result in original_spatial_run(self):
        yield result

# Monkey patch the temporal mapping generator
def patched_temporal_run(self):
    global temporal_count
    layer_id = getattr(self.layer, "id", getattr(self.layer, "name", "unknown"))
    print(f"Running patched temporal mapping generator for layer: {layer_id}")
    
    # Find the corresponding spatial mapping
    current_spatial_id = None
    for spatial_id, mapping_data in all_mappings.items():
        if mapping_data["layer_id"] == layer_id:
            # This is a simple approximation - in reality we'd want to match the exact mapping
            current_spatial_id = spatial_id
            break
    
    print(f"Processing temporal mappings for spatial mapping: {current_spatial_id}")
    
    if current_spatial_id is None:
        print("WARNING: Could not find matching spatial mapping - creating one")
        spatial_id = f"spatial_fallback_{spatial_count}"
        all_mappings[spatial_id] = {
            "layer_id": layer_id,
            "spatial_mapping": self.spatial_mapping,  # Will be converted when saving
            "temporal_mappings": []
        }
        current_spatial_id = spatial_id
    
    # Get all temporal mappings
    temporal_mappings = list(self.generate_temporal_mappings())
    print(f"Generated {len(temporal_mappings)} temporal mappings")
    
    # Save each temporal mapping
    for i, temporal_mapping in enumerate(temporal_mappings):
        # Store original mapping object - will be converted when saving
        all_mappings[current_spatial_id]["temporal_mappings"].append(temporal_mapping)
        temporal_count += 1
        
        # Save periodically
        if temporal_count % 1000 == 0:
            try:
                save_mappings_to_json(all_mappings, output_json)
                print(f"Saved {temporal_count} temporal mappings")
            except Exception as e:
                print(f"Error saving temporal mappings: {e}")
    
    # Run the original method to continue the pipeline
    for result in original_temporal_run(self):
        yield result

# Apply all monkey patches
SpatialMappingGeneratorStage.run = patched_spatial_run
TemporalMappingGeneratorStage.run = patched_temporal_run
CostModelEvaluation.combine_data_transfer_rate_per_physical_port = patched_combine_data_transfer_rate
CostModelEvaluation.calc_overall_latency = patched_calc_overall_latency
print("Monkey patching applied successfully")

# Define the experiment id and pickle name
hw_name = "single_core_8_NPE_event_driven_accelerator"
workload_name = "lenet_5_second_layer_model_frame"
experiment_id = f"{hw_name}-{workload_name}"
pickle_name = f"{experiment_id}-saved_list_of_cmes"

# Define main input paths
accelerator_path = os.path.expanduser("~/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/May_experiments/results-08-05/inputs/architecture/event_driven_accelerator.yaml")
workload_path = os.path.expanduser("~/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/May_experiments/results-08-05/inputs/workload/VGGnet_event_first_layer.yaml")
mapping_path = os.path.expanduser("~/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/May_experiments/results-08-05/inputs/mapping/VGGnet_event_first_layer_map.yaml")

# Define other inputs of api call
temporal_mapping_search_engine = "lowa"
optimization_criterion = "latency"
dump_folder = os.path.expanduser(f"~/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/May_experiments/results-08-05/outputs/{experiment_id}")
pickle_filename = os.path.expanduser(f"~/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/May_experiments/results-08-05/outputs/{pickle_name}.pickle")

print("Starting ZigZag run...")
# Get the hardware performance through api call
result = get_hardware_performance_zigzag(
    accelerator=accelerator_path,
    workload=workload_path,
    mapping=mapping_path,
    temporal_mapping_search_engine=temporal_mapping_search_engine,
    opt=optimization_criterion,
    dump_folder=dump_folder,
    pickle_filename=pickle_filename,
    nb_spatial_mappings_generated=6,  # Increase for more spatial mappings
    lpf_limit=6,  # Increase for more temporal mappings
)
print("ZigZag run completed")

if len(result) == 3:
    energy, latency, results = result
elif len(result) == 5:
    energy, latency, _, _, results = result
else:
    raise ValueError("Unexpected result length from get_hardware_performance_zigzag")

# Extract the CMEs from the results
cmes = [result[0] for result in results[0][1]]

# Save a bar plot of the cost model evaluations breakdown
save_path = os.path.expanduser("~/Thesis/zigzag/Extras/zigzag_tutorials/zigzag-tutorial/Self_tests/May_experiments/results-08-05/outputs/breakdown.png")
bar_plot_cost_model_evaluations_breakdown(cmes, save_path=save_path)
print_mapping(cmes[0])

# Save final mappings and latency data
try:
    save_mappings_to_json(all_mappings, output_json)
    
    # Save latency data to a separate file
    latency_json_path = os.path.join(base_dir, "latency_data.json")
    with open(latency_json_path, 'w') as f:
        json.dump(latency_data, f, indent=2)
    
    print(f"Final mapping count: {len(all_mappings)} spatial mappings")
    print(f"Total temporal mappings: {sum(len(m.get('temporal_mappings', [])) for m in all_mappings.values())}")
    print(f"All mappings saved to: {output_json}")
    print(f"Latency data saved to: {latency_json_path}")
    print(f"Detailed latency components log: {latency_log_path}")
    print(f"Detailed stall components log: {stall_log_path}")
except Exception as e:
    print(f"Error saving final data: {e}")

2. Json_parser_v2
import simdjson as sj
import json

def filter_mappings(json_file, output_file):
    # Initialize the SIMD JSON parser
    parser = sj.Parser()
    
    # Parse the entire JSON file
    with open(json_file, 'rb') as f:
        data = parser.parse(f.read())
    
    # Dictionary to store filtered data
    filtered_data = {}
    
    # Process each spatial class independently
    for spatial_key in data.keys():
        spatial_dict = data[spatial_key]
        layer_id = spatial_dict['layer_id']  # Assuming primitive type
        # Convert spatial_mapping to Python dict if it's a simdjson Object
        spatial_mapping = spatial_dict['spatial_mapping'].as_dict() if isinstance(spatial_dict['spatial_mapping'], sj.Object) else spatial_dict['spatial_mapping']
        temporal_mappings = spatial_dict['temporal_mappings']
        
        # If there are temporal mappings, find the max 'I' and filter
        if temporal_mappings:
            max_I = max(mapping['mac_level_data_stationary_cycle']['I'] for mapping in temporal_mappings)
            # Convert each filtered mapping to a Python dict for JSON serialization
            filtered_mappings = [mapping.as_dict() for mapping in temporal_mappings if mapping['mac_level_data_stationary_cycle']['I'] == max_I]
        else:
            filtered_mappings = []
        
        # Store the filtered data for this spatial class
        filtered_data[spatial_key] = {
            'layer_id': layer_id,
            'spatial_mapping': spatial_mapping,
            'temporal_mappings': filtered_mappings
        }
    
    # Write the filtered data to the output file
    with open(output_file, 'w') as f_out:
        json.dump(filtered_data, f_out, indent=4, ensure_ascii=False)

# Usage
if __name__ == "__main__":
    json_file = 'all_mappings.json'    # Replace with your input file path
    output_file = 'filtered.json'  # Replace with your output file path
    filter_mappings(json_file, output_file)

3. Pooling_parser

This is used to remove the pooling mappings as they are currently not correct as they are produced by the tool

import simdjson as sj
import json

def filter_spatial_classes(input_file, output_file, exclude_list):
    # Initialize the SIMD JSON parser
    parser = sj.Parser()
    
    # Read and parse the input JSON file, converting it to a Python dictionary
    with open(input_file, 'rb') as f:
        data = parser.parse(f.read()).as_dict()
    
    # Filter out the specified spatial classes using a dictionary comprehension
    filtered_data = {k: v for k, v in data.items() if k not in exclude_list}
    
    # Write the filtered data to the output file with proper formatting
    with open(output_file, 'w') as f_out:
        json.dump(filtered_data, f_out, indent=4, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    input_file = 'filtered.json'
    output_file = 'filtered_spatial.json'
    # Define the spatial classes to exclude (modify this list as needed)
    exclude_list = ['spatial_2', 'spatial_5', 'spatial_9', 'spatial_13', 'spatial_17']
    filter_spatial_classes(input_file, output_file, exclude_list)

4. Once the required mapspace is produced here is the idea of how it can be fed to the cost model for the calculation

This "reconstruction" script is all about taking the data you saved in all_mappings.json and turning it back into the live Python objects that ZigZag's brain (specifically, its cost model) understands.

Let's break down how you can build this.

1. Purpose of this Program (The "Reconstruction Script")

The main goal of this new Python script will be to:

Read a specific mapping configuration (one spatial mapping and one of its associated temporal mappings) from your all_mappings.json file.
Recreate the necessary ZigZag Python objects for the Accelerator, LayerNode, SpatialMapping, and TemporalMapping using the data from your JSON.
Feed these reconstructed objects into ZigZag's CostModelEvaluation to calculate its energy and latency.
(Optionally) Do this for many or all mappings in your JSON to compare them or find the best one according to the cost model.
Think of it like taking a blueprint (your JSON data) and building the actual machine components (Python objects) so you can test the machine (run the cost model).

2. All the Supporting and Depending Libraries/Classes of that Program

You'll be using a mix of standard Python libraries and, importantly, several classes directly from the KULeuven-MICAS/zigzag repository.

Standard Python:

json: To load your all_mappings.json file.
os: For path manipulation.
logging (optional but good): For informative messages.
ZigZag Framework (you'll need to import these):

For Loading Initial Config:
zigzag.inputs.load_accelerator_from_yaml (or similar function): To load your hardware accelerator definition from its YAML file.
Core Data Structures to Reconstruct:
zigzag.workload.layer_node.LayerNode: The class representing a neural network layer.
zigzag.workload.layer_attributes: This module contains many dataclasses that LayerNode uses, such as:
LayerEquation
LayerDimSizes
LayerOperandPrecision
LayerDimRelation
LayerPadding
InputOperandSource
SpatialMappingHint
MemoryOperandLinks
LayerTemporalOrdering
zigzag.spatial.mapping.SpatialMapping (or potentially SpatialMappingInternal depending on what the cost model expects – you'll need to verify this from the ZigZag source. SpatialMapping is common).
zigzag.temporal.mapping.TemporalMapping.
For Cost Evaluation:
zigzag.cost_model.cost_model.CostModelEvaluation: The class that performs the detailed energy and performance estimation.
(Alternatively) zigzag.stages.CostModelEvaluationStage if you want to run it as part of a pipeline (though direct instantiation of CostModelEvaluation is often simpler for this use case).
Other ZigZag Utilities/Datatypes:
zigzag.datatypes: For basic types like LayerOperand, LayerDim, etc.
zigzag.hardware.accelerator.Accelerator: The type of object returned by load_accelerator_from_yaml.
Important Note on ZigZag Versions: The exact class names, their module paths, and their constructor arguments (__init__ methods) can change between ZigZag versions. You MUST refer to the specific version of the ZigZag codebase you are using. My explanations will be based on common patterns, but verification is key.


reconstruct_layer_node_attributes(ln_data): This is the most complex part. You need to go through every attribute that LayerNodeAttributes expects. For each attribute that is an object itself (like LayerEquation, LayerDimSizes, LayerPadding, etc.), you must:

Find that class in the ZigZag source code (usually in zigzag.workload.layer_attributes).
Look at its __init__ method to see what arguments it takes.
Extract the corresponding data from your ln_data dictionary (which came from layer_node in your JSON) and instantiate that attribute object.
Example: If LayerEquation takes a string eq_str in its constructor, you'd do eq_data = ln_data.get("equation", {}).get("data"); equation_obj = LayerEquation(eq_data).
reconstruct_mapping_attributes_for_layer_node(ln_data_for_layer, acc) (Helper for LayerNode):

Similar to above, this function will create the MappingAttributes object that LayerNode's constructor needs.
Its spatial_mapping field will be a SpatialMapping object. This SpatialMapping object must be created from the spatial_mapping dictionary found inside ln_data_for_layer (e.g., ln_data_for_layer["spatial_mapping"]).
This means this helper will also call reconstruct_spatial_mapping but with the LayerNode's own spatial mapping data. This is where the circular dependency I mentioned needs careful handling. The SpatialMapping constructor needs a LayerNode.
How ZigZag often handles this internally: When a LayerNode is created from a workload file, its __init__ method takes the dictionary for spatial_mapping (and other mapping attributes), and then inside __init__ (or in the MappingAttributes constructor), it creates the SpatialMapping object, passing self (the LayerNode instance being created) to the SpatialMapping constructor.
For your reconstruction, you might need to pass the ln_data_for_layer to reconstruct_spatial_mapping and let it know it's creating the "internal" SM for the LayerNode. This is advanced.
Simplification: The LayerNode constructor in zigzag/workload/layer_node.py takes node_attr: LayerNodeAttributes and mapping_attr: MappingAttributes. Your reconstruct_layer_node_attributes and reconstruct_mapping_attributes functions need to correctly prepare these objects. The MappingAttributes dataclass itself takes a spatial_mapping: SpatialMapping object. So, yes, you first need to create this SpatialMapping object (from the spatial_mapping dict within the layer_node JSON section) before you can create MappingAttributes.
reconstruct_spatial_mapping(sm_data, layer_node_obj, accelerator):

Find the SpatialMapping class in ZigZag (likely zigzag.spatial.mapping.SpatialMapping).
Check its __init__. It typically takes the mapping dictionary (your sm_data), the LayerNode object it applies to, and often some information about the accelerator core (like a core_id). You'll need to get this core_id from your loaded accelerator object.
The sm_data here is the chosen spatial mapping for the current evaluation (e.g., {"D1": {"K": 8}}).
reconstruct_temporal_mapping(tm_data, layer_node_obj, spatial_mapping_obj, accelerator):

Find TemporalMapping in zigzag.temporal.mapping.TemporalMapping.
Check its __init__. It needs the LayerNode, the chosen SpatialMapping object, the Accelerator object, and usually the mapping_dic_origin (the loop tiling/ordering structure from your tm_data). It might also take mapping_dic_stationary or other specific loop information.
spatial_mapping_int for CostModelEvaluation:

The CostModelEvaluation constructor often takes spatial_mapping and spatial_mapping_int. If your spatial mappings are always integer-based (like K: 8), you can often pass the same spatial_mapping_obj to both. If they can be fractional, you'll need to figure out how spatial_mapping_int is derived or if it's also in your JSON.
Error Handling and Logging: Add more robust error checking and logging. If an object fails to reconstruct, log which one and why.

Looping: My main() function only processes one example mapping. You'll need to add loops to iterate through all spatial_id entries in all_mappings_data and then through each temporal mapping in the temporal_mappings list for each spatial entry.

This is a significant but very rewarding task. You're essentially rebuilding a part of ZigZag's internal state. Take it one step at a time, focusing on correctly instantiating each object based on its class definition in the ZigZag source code. Use print(type(obj)) and print(vars(obj)) extensively to debug.

This would need some reconstruction of the mapspace to be read by the cost model

With this information, Can you develop a complete integration software flow utilising the exisitng scripts and builidng new scripts. So, the process is seemless and meeting the requirement to perform the DSE effectively.


---


Finishing the Testing process

This plan looks good, Based on your current plan here is the current situation of the file structure of the map space created that will be input to the tool.

Right now the mapspace that goes as an input to the zigzag is segregated as follows for each of the three spatial unrollings, under a folder name of each layer, all the temporal mappings corresponding to that particular spatial unrolling are stored.

For Ex: Unrolling_across_K -> Layer_0 -> Temporal_0, Temporal_1, Temporal_2 etc..,

This is create by this script

"import os
import json
import yaml
import re
import argparse
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mapping_converter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_temporal_mapping(mapping_str):
    """
    Parse a temporal mapping string into a list of dimension, size pairs.
    
    Args:
        mapping_str (str): String representation of temporal mapping
                         e.g. "[[(FX, 3), (K, 8)], [(FY, 3), (IY, 224)], [(IX, 224), (C, 3)]]"
    
    Returns:
        list: List of [dimension, size] pairs in the correct loop order
    """
    # Remove outer brackets and quotes
    clean_str = mapping_str.strip("'[]")
    
    # Split the string by level (separated by '], [')
    levels = clean_str.split('], [')
    
    # Initialize result list
    result = []
    
    # Process each level
    for level in levels:
        # Clean up any remaining brackets
        level = level.strip('[]')
        
        # Split by '), (' to get each dimension-size pair
        pairs = level.split('), (')
        
        # Process each pair
        for pair in pairs:
            # Clean up any remaining parentheses
            pair = pair.strip('()')
            
            # Split by comma to get dimension and size
            parts = pair.split(', ')
            
            if len(parts) == 2:
                dim, size = parts
                # Add as a list in the format ZigZag expects: [dimension, size]
                result.append([dim, int(size)])
    
    return result

def get_spatial_split_type(mapping_key):
    """
    Extract the spatial split type from the mapping key.
    
    Args:
        mapping_key (str): Key of the mapping (e.g. "K_unroll_spatial_0")
    
    Returns:
        str: The spatial split type (e.g. "K_unroll")
    """
    # Extract the part before "_spatial_"
    parts = mapping_key.split("_spatial_")
    if len(parts) > 1:
        return parts[0]
    return "unknown_split"

def get_default_spatial_dimension(spatial_split_type):
    """
    Get the default spatial dimension based on the spatial split type.
    
    Args:
        spatial_split_type (str): Type of spatial split (e.g. "K_unroll")
    
    Returns:
        str: The dimension to use in default mapping
    """
    # Extract dimension from split type (e.g. K from K_unroll)
    if "_unroll" in spatial_split_type:
        return spatial_split_type.split("_")[0]
    return "K"  # Default to K if we can't determine

def convert_temporal_mapping_to_yaml(mapping_key, layer_id, spatial_mapping, temporal_mapping, tm_index, output_dir):
    """
    Convert a temporal mapping entry to a ZigZag YAML file.
    
    Args:
        mapping_key (str): Key of the spatial mapping (e.g. "K_unroll_spatial_0")
        layer_id (str): ID of the layer
        spatial_mapping (dict): Spatial mapping information
        temporal_mapping (dict): Temporal mapping information
        tm_index (int): Index of the temporal mapping
        output_dir (str): Directory to save the YAML file
    
    Returns:
        str: Path to the created YAML file
    """
    try:
        # Extract mapping_dic_origin
        if "mapping_dic_origin" not in temporal_mapping:
            logger.warning(f"No mapping_dic_origin found for {mapping_key}, temporal mapping {tm_index}")
            return None
        
        # Use the "I" operand mapping to determine loop order
        i_mapping_str = temporal_mapping["mapping_dic_origin"].get("I", "")
        if not i_mapping_str:
            logger.warning(f"No I mapping found for {mapping_key}, temporal mapping {tm_index}")
            return None
        
        # Parse the temporal mapping string
        temporal_ordering = parse_temporal_mapping(i_mapping_str)
        
        # Extract spatial split type
        spatial_split_type = get_spatial_split_type(mapping_key)
        
        # Get default spatial dimension and size
        default_dimension = get_default_spatial_dimension(spatial_split_type)
        default_size = 8  # Default size
        
        # Try to get the actual size from the spatial mapping
        for dim, values in spatial_mapping.items():
            if default_dimension in values:
                default_size = values[default_dimension]
                break
        
        # Create directory structure: spatial_split/layer_X/
        split_dir = os.path.join(output_dir, spatial_split_type)
        layer_dir = os.path.join(split_dir, f"layer_{layer_id}")
        os.makedirs(layer_dir, exist_ok=True)
        
        # Create the output file path with temporal mapping index
        output_file = os.path.join(layer_dir, f"temporal_{tm_index}.yaml")
        
        # Write the YAML file with customized formatting to match the expected format
        with open(output_file, 'w') as f:
            # First, write the default mapping
            f.write("- name: default\n")
            f.write("  spatial_mapping:\n")
            f.write("    D1:\n")
            f.write(f"      - {default_dimension}, {default_size}\n")
            f.write("  memory_operand_links:\n")
            f.write("    O: O\n")
            f.write("    W: I2\n")
            f.write("    I: I1\n")
            
            # Add the actual mapping
            f.write("- name: " + f"layer_{layer_id}" + "\n")
            
            # Write spatial mapping in the exact format required
            f.write("  spatial_mapping:\n")
            for dim, values in spatial_mapping.items():
                f.write(f"    {dim}:\n")
                for var, size in values.items():
                    f.write(f"      - {var}, {size}\n")
            
            # Write temporal ordering in the exact format required
            f.write("  temporal_ordering:\n")
            for dim_pair in temporal_ordering:
                f.write(f"    - [{dim_pair[0]}, {dim_pair[1]}]\n")
            
            # Write memory operand links
            f.write("  memory_operand_links:\n")
            f.write("    O: O\n")
            f.write("    W: I2\n")
            f.write("    I: I1\n")
        
        logger.debug(f"Created mapping file with default case: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error converting mapping {mapping_key}, temporal mapping {tm_index}: {str(e)}")
        return None

def process_mapping(args):
    """
    Process a single mapping with all its temporal mappings.
    Used for parallel processing.
    
    Args:
        args (tuple): (mapping_key, mapping_data, output_dir)
    
    Returns:
        dict: Results from processing
    """
    mapping_key, mapping_data, output_dir = args
    
    # Extract layer ID
    layer_id = mapping_data.get("layer_id", "0")
    
    # Extract spatial mapping
    spatial_mapping = mapping_data.get("spatial_mapping", {})
    
    # Get all temporal mappings
    temporal_mappings = mapping_data.get("temporal_mappings", [])
    if not temporal_mappings:
        logger.warning(f"No temporal mappings found for {mapping_key}")
        return {
            "mapping": mapping_key,
            "success": False,
            "error": "No temporal mappings found"
        }
    
    # Process each temporal mapping
    successful_files = []
    for tm_index, temporal_mapping in enumerate(temporal_mappings):
        yaml_path = convert_temporal_mapping_to_yaml(
            mapping_key,
            layer_id,
            spatial_mapping,
            temporal_mapping,
            tm_index,
            output_dir
        )
        
        if yaml_path:
            successful_files.append(yaml_path)
    
    # Return success if at least one temporal mapping was successfully processed
    if successful_files:
        return {
            "mapping": mapping_key,
            "success": True,
            "yaml_paths": successful_files,
            "total_temporal_mappings": len(temporal_mappings),
            "successful_temporal_mappings": len(successful_files)
        }
    else:
        return {
            "mapping": mapping_key,
            "success": False,
            "error": "Failed to convert any temporal mappings",
            "total_temporal_mappings": len(temporal_mappings)
        }

def main():
    """
    Main function to parse arguments and process mappings.
    """
    parser = argparse.ArgumentParser(description="Convert JSON mappings to ZigZag YAML format")
    parser.add_argument("--input", required=True, help="Input JSON file with mappings")
    parser.add_argument("--output_dir", required=True, help="Output directory for YAML files")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON mappings
    try:
        with open(args.input, 'r') as f:
            mappings_data = json.load(f)
        
        # Check if it's a dictionary (as expected)
        if not isinstance(mappings_data, dict):
            logger.error("Input JSON should be a dictionary of mappings")
            return
        
        logger.info(f"Loaded {len(mappings_data)} spatial mappings from {args.input}")
        
        # Count total temporal mappings
        total_temporal_mappings = sum(
            len(data.get("temporal_mappings", [])) 
            for data in mappings_data.values()
        )
        logger.info(f"Found {total_temporal_mappings} total temporal mappings across all spatial mappings")
            
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return
    
    # Process mappings (convert to YAML)
    all_results = []
    
    # Use parallel processing if workers > 1
    if args.workers > 1:
        logger.info(f"Using {args.workers} worker processes")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Create list of arguments for each mapping
            process_args = [(key, mappings_data[key], output_dir) for key in mappings_data]
            
            # Process mappings in parallel with progress bar
            results = list(tqdm(
                executor.map(process_mapping, process_args),
                total=len(process_args),
                desc="Processing mappings"
            ))
            
            all_results.extend(results)
    else:
        # Process sequentially with progress bar
        for key in tqdm(mappings_data.keys(), desc="Processing mappings"):
            result = process_mapping((key, mappings_data[key], output_dir))
            all_results.append(result)
    
    # Count successful conversions
    successful_spatial = sum(1 for r in all_results if r["success"])
    successful_temporal = sum(r.get("successful_temporal_mappings", 0) for r in all_results)
    
    # Save results summary
    results_summary = {
        "total_spatial_mappings": len(mappings_data),
        "successful_spatial_mappings": successful_spatial,
        "failed_spatial_mappings": len(mappings_data) - successful_spatial,
        "total_temporal_mappings": total_temporal_mappings,
        "successful_temporal_mappings": successful_temporal,
        "failed_temporal_mappings": total_temporal_mappings - successful_temporal,
        "details": all_results
    }
    
    with open(os.path.join(output_dir, "processing_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Processed {len(mappings_data)} spatial mappings")
    logger.info(f"Spatial mappings - Successful: {successful_spatial}, Failed: {len(mappings_data) - successful_spatial}")
    logger.info(f"Temporal mappings - Successful: {successful_temporal}, Failed: {total_temporal_mappings - successful_temporal}")
    logger.info(f"YAML files saved to {output_dir}")
    
    # Create a simple README file explaining the structure
    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write("# ZigZag Mapping Files\n\n")
        f.write("This directory contains mapping files for ZigZag hardware evaluation.\n\n")
        f.write("## Directory Structure\n\n")
        f.write("- `<spatial_split_type>/`: Folders for each spatial splitting strategy (e.g., K_unroll, FX_unroll)\n")
        f.write("  - `layer_<id>/`: Folders for each layer\n")
        f.write("    - `temporal_<index>.yaml`: Each temporal mapping variation\n\n")
        f.write("## File Structure\n\n")
        f.write("Each YAML file contains two mappings:\n")
        f.write("1. A default mapping required by ZigZag\n")
        f.write("2. The actual layer mapping with spatial and temporal information\n\n")
        f.write("## Running ZigZag\n\n")
        f.write("To evaluate a mapping with ZigZag:\n\n")
        f.write("```bash\n")
        f.write("python -m zigzag.main --workload <workload_file> --accelerator <accelerator_file> --mapping <mapping_file>\n")
        f.write("```\n")

if __name__ == "__main__":
    main()"

Now, based on your flow

Phase 1: Data Collection and Storage Strategy
Create a master CSV/Excel file structure:

Columns: mapping_file, spatial_split, layer_id, temporal_id, energy, latency, edp, success, error
This becomes our "database" of results
Implement a file-based checkpoint system:

Track which mappings have been evaluated
Allow resuming from where we left off if the program crashes
Implement an efficient evaluation loop:

Code
For each mapping file:
  - Check if already evaluated (using checkpoint)
  - If not, evaluate the mapping
  - Append result directly to CSV/Excel file
  - Update checkpoint


Phase 2: Batch Processing Improvement

Implement batch processing:

Group mappings by layer for more efficient processing
Evaluate mappings for one layer at a time
This reduces the context switching overhead between layers


Optimize ZigZag API calls:

Load workload and accelerator models once per batch
Minimize redundant calculations


Phase 3: Analysis Pipeline
Create a separate analysis module:

Takes the master CSV/Excel file as input
For each layer, finds best mappings based on latency, energy, and EDP
Outputs selection results to a summary file
Results Organization:

Create best_latency, best_energy, best_edp directories
Copy only the selected mapping files to these directories
Generate summary JSON files with statistics


Phase 4: Progress Monitoring and Reporting
Real-time metrics tracking:

Show progress percentage
Display current layer being processed
Show estimated time remaining
Interactive reporting option:
Generate interim reports without stopping evaluation
Show current best mappings discovered so far


Implementation Steps
Step 1: Create the Database Structure

Implement CSV/Excel file with appropriate columns
Test read/write operations for single records
Step 2: Implement Checkpointing

Create checkpointing mechanism
Test resume functionality
Step 3: Modify Evaluation Loop

Rewrite to use file-based storage instead of memory
Test with small subset of mappings
Step 4: Batch Processing

Implement layer-based batching
Measure performance improvement
Step 5: Analysis Module

Implement data analysis on CSV/Excel file
Test best mapping selection logic
Step 6: Create Results Organization

Implement directory creation and file copying
Test with sample data

This plan that you proposed is good and by the help of the exisitng zigzag validator which already has a lot of scripts that can be reused to build the final script.

So, Let's first modify the reconstruct script that I shared. Once that is tested we will improve the zigzag validator that I will share accordingly.
