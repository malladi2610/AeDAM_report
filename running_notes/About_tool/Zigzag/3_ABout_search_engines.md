# Complete Analysis of LOWA and SALSA Search Engines in ZigZag

## 1. Purpose of these programs

**LOWA (Loop-Order based Workload Allocation)** and **SALSA (Simulated Annealing-based Loop Structure Allocator)** are search engines within the ZigZag framework that solve different but related problems:

- **LOWA**: Finds efficient temporal mappings (loop orderings) to organize computations in time
- **SALSA**: Explores the spatial mapping space to distribute computations across hardware processing elements

Together, they help ZigZag find optimal ways to map deep learning workloads onto hardware accelerators.

## 2. All the supporting and depending libraries

### LOWA Dependencies
- `zigzag.opt.loma.memory_allocator`: Handles memory allocation for temporal mappings
- `zigzag.opt.loma.multipermute`: Generates loop permutations
- `zigzag.hardware.architecture.memory_hierarchy`: Models memory levels
- `zigzag.mapping.temporal_mapping`: Represents temporal mappings

### SALSA Dependencies
- `zigzag.opt.salsa.generator`: Generates spatial mapping candidates
- `zigzag.opt.salsa.cost_model`: Evaluates mapping efficiency
- `zigzag.mapping.spatial_mapping`: Represents spatial mappings
- `numpy`: For numerical operations
- `random`: For stochastic search

## 3. Detailed explanation of each function

### LOWA Engine (Loop-Order based Workload Allocation)

#### Location: `zigzag/opt/lowa/engine.py`

1. **`__init__(self, accelerator, layer, spatial_mapping, **kwargs)`**
   - **Purpose**: Initializes the LOWA engine with the necessary components
   - **How it works**: Sets up the hardware model, layer details, and spatial mapping to use
   - **Parameters**: Accepts optional parameters like `lpf_limit` to control search space

2. **`run(self)`**
   - **Purpose**: Main function that executes the search for optimal temporal mappings
   - **How it works**: 
     - Generates possible loop permutations
     - Applies constraints (if any)
     - Allocates loops to memory levels
     - Returns best temporal mappings
   - **Returns**: Generator of TemporalMapping objects

3. **`get_temporal_loops(self)`**
   - **Purpose**: Extracts all loop dimensions that need temporal scheduling
   - **How it works**: Analyzes the layer and removes any spatially-scheduled dimensions
   - **Returns**: Dictionary of loop dimensions and their sizes

4. **`set_constraints(self, constraints)`**
   - **Purpose**: Adds constraints to the search process
   - **How it works**: Takes a list of PermutationConstraint objects that specify which loops must be adjacent
   - **Used for**: Guiding the search towards specific loop ordering patterns

5. **`_permute_spaces(self)`**
   - **Purpose**: Core algorithm that generates loop permutations
   - **How it works**: 
     - Creates an undirected graph of loop dimensions
     - Assigns weights to edges based on hardware and workload characteristics
     - Finds minimum spanning tree to create good permutations
   - **Key innovation**: Uses graph theory to intelligently reduce the permutation space

#### LOWA's Key Parameters:
- **`lpf_limit`**: Maximum number of loop permutation factors to explore (limits the search space)
- **`use_heuristic`**: When True, uses smarter edge weights for better permutations
- **`loop_edge_weights`**: Custom weights for controlling the permutation generation

### SALSA Engine (Simulated Annealing-based Loop Structure Allocator)

#### Location: `zigzag/opt/salsa/engine.py`

1. **`__init__(self, accelerator, layer, **kwargs)`**
   - **Purpose**: Initializes the SALSA engine
   - **How it works**: Sets up the hardware model, layer, and search parameters
   - **Parameters**: Includes temperature scheduling for simulated annealing

2. **`run(self)`**
   - **Purpose**: Main function that executes the simulated annealing search
   - **How it works**: 
     - Starts with an initial spatial mapping
     - Iteratively explores neighboring mappings
     - Accepts improvements immediately
     - Sometimes accepts worse solutions with a probability that decreases over time
   - **Returns**: Generator of SpatialMapping objects

3. **`_initialize_mapping(self)`**
   - **Purpose**: Creates a starting point for the search
   - **How it works**: Either uses a user-provided mapping or generates a reasonable default

4. **`_generate_neighbor(self, current_mapping)`**
   - **Purpose**: Creates a slightly modified version of the current mapping
   - **How it works**: Makes small changes like moving a loop dimension from one PE to another

5. **`_calculate_cost(self, mapping)`**
   - **Purpose**: Evaluates how good a mapping is
   - **How it works**: Uses a cost model to estimate energy, latency, or other metrics

#### SALSA's Key Parameters:
- **`initial_temperature`**: Controls how likely the algorithm accepts worse solutions early on
- **`cooling_rate`**: How quickly the temperature decreases
- **`max_iterations`**: Maximum number of search steps
- **`cost_function`**: What to optimize for (energy, latency, or combined)

### How They Work Together

1. **Typical workflow**:
   - SALSA first finds a good spatial mapping
   - LOWA then finds the best temporal mapping for that spatial mapping

2. **Search space reduction**:
   - Both engines use clever heuristics to avoid exhaustive search
   - LOWA uses graph theory to generate smart permutations
   - SALSA uses simulated annealing to efficiently explore the spatial mapping space

3. **Trade-offs**:
   - LOWA is faster but may miss some optimal solutions
   - SALSA can find better solutions with enough time but takes longer to run

This combination of search strategies allows ZigZag to find good mappings in a reasonable amount of time, balancing exploration thoroughness with practical runtime constraints.