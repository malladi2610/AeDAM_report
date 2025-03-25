#### Question 1:

It is possible to get all the possible combinations of the mapping constriant possible when the workload has the following parameters [B, K, C, IY, IX, FY, FX], in which the position of the IX, IY C are fixed in the mapping and rest are left to be oriented and tested with zigzag to find the mapping that is best.

Here is the template mapping

spatial_mapping:
    D1:
      - K, 6 [This can be any of the parameters from FX, FY, K]
  temporal_ordering:
    # The below two [FX, 5] and [FY,5] it can be either K, FX and FX based on the situation where which variable is present in the spatial ordering
    - [FX, 5] #Innermost
    - [FY, 5]

    # The position and orientation of the below three are fixed
    - [C ,1]
    - [IX, 32]
    - [IY, 32] #Outermost


With this can you list out on all the possible combination possible with the remaining parameters



#### Answer 1

To determine all possible combinations of the mapping constraints for the workload with parameters [B, K, C, IY, IX, FY, FX], we need to adhere to the rules provided in the template mapping. The template specifies a spatial mapping and a temporal ordering, with some parts fixed and others variable. Let's break it down and systematically generate all possible combinations.

### Step 1: Understand the Template Mapping

#### **Spatial Mapping**
- **D1**: Assigned one parameter from the set {K, FX, FY} with a factor of 6.
  - Possible choices: [K, 6], [FX, 6], or [FY, 6].
- This means there are 3 possible options for the spatial mapping.

#### **Temporal Ordering**
The temporal ordering consists of five levels, with the following specifications:
- **Fixed Levels**: The last three levels are explicitly fixed in both position and orientation:
  - [C, 1]
  - [IX, 32]
  - [IY, 32] (outermost among these three)
  - These must appear in this exact order relative to each other in the temporal ordering.
- **Variable Innermost Levels**: The first two levels (innermost) are shown as:
  - [FX, 5]
  - [FY, 5]
  - However, the comment states: "The below two [FX, 5] and [FY, 5] it can be either K, FX and FX based on the situation where which variable is present in the spatial ordering."
  - This phrasing is slightly confusing, but interpreting it in context, it likely means that these two levels should be assigned the two parameters from {K, FX, FY} that are *not* used in the spatial mapping, each with a factor of 5. The typo ("FX and FX") probably intended "FX or FY," implying flexibility based on the spatial choice.
- **Parameter B**: The workload includes B, but the template's temporal ordering only shows five levels without B. Since the query states that "the position of the IX, IY, C are fixed in the mapping and rest are left to be oriented," B must be included in the temporal ordering as an additional level, with its position flexible relative to the fixed levels.

### Step 2: Define the Constraints
- **Workload Parameters**: [B, K, C, IY, IX, FY, FX]
- **Spatial Mapping**: Uses one of {K, FX, FY}.
- **Temporal Ordering**: Must include all remaining parameters (B, K, FX, FY minus the one used spatially, plus C, IX, IY).
  - Innermost two levels: The two parameters from {K, FX, FY} not used in the spatial mapping, each with factor 5.
  - Fixed sequence: [C, 1], [IX, 32], [IY, 32] in that relative order.
  - B: Must be included, position flexible, factor unspecified (we’ll assume [B, B] for simplicity, representing the full batch size, as the factor isn’t provided).

Since the template shows five levels but we have six parameters to place (after accounting for the spatial mapping), we’ll assume B adds a sixth level, and we need to determine its position. However, to keep the answer focused and manageable, let’s first consider the minimal interpretation aligned with the template and then adjust for B.

### Step 3: Generate Combinations Without B’s Position Variability (Initial Interpretation)
First, let’s assume B is fixed (e.g., as the outermost level) and focus on the permutations of {K, FX, FY}. For each spatial mapping choice:
- **Spatial Mapping**: Choose one of {K, FX, FY}.
- **Innermost Temporal Levels**: Assign the remaining two from {K, FX, FY} to the first two temporal levels, with factor 5. Since order matters in temporal ordering, consider both permutations.

#### **Case 1: Spatial Mapping D1: [K, 6]**
- Remaining from {K, FX, FY}: {FX, FY}
- Innermost temporal permutations:
  1. [FX, 5], [FY, 5]
  2. [FY, 5], [FX, 5]
- Temporal ordering (without B for now):
  1. [FX, 5], [FY, 5], [C, 1], [IX, 32], [IY, 32]
  2. [FY, 5], [FX, 5], [C, 1], [IX, 32], [IY, 32]

#### **Case 2: Spatial Mapping D1: [FX, 6]**
- Remaining from {K, FX, FY}: {K, FY}
- Innermost temporal permutations:
  1. [K, 5], [FY, 5]
  2. [FY, 5], [K, 5]
- Temporal ordering (without B for now):
  1. [K, 5], [FY, 5], [C, 1], [IX, 32], [IY, 32]
  2. [FY, 5], [K, 5], [C, 1], [IX, 32], [IY, 32]

#### **Case 3: Spatial Mapping D1: [FY, 6]**
- Remaining from {K, FX, FY}: {K, FX}
- Innermost temporal permutations:
  1. [K, 5], [FX, 5]
  2. [FX, 5], [K, 5]
- Temporal ordering (without B for now):
  1. [K, 5], [FX, 5], [C, 1], [IX, 32], [IY, 32]
  2. [FX, 5], [K, 5], [C, 1], [IX, 32], [IY, 32]

- **Total without B’s variability**: 3 spatial choices × 2 permutations = 6 combinations.

### Step 4: Incorporate Parameter B
The template shows five levels, but the workload includes B, and the query implies all parameters should be oriented except the fixed [C, IX, IY]. Thus, the temporal ordering should have six levels:
- Innermost two: From {K, FX, FY} (remaining two), factor 5.
- Fixed: [C, 1], [IX, 32], [IY, 32].
- B: Position flexible among these levels.

If B’s position is variable, we’d insert [B, B] in any of the 6 possible positions (before innermost, between each level, or outermost) for each of the 6 combinations, yielding 6 × 6 = 36 combinations. However, this may be excessive for the query’s intent, which mentions testing with “zigzag” (possibly a search method), suggesting a practical number of combinations.

Given standard practice (e.g., in convolutional neural network mappings), B (batch size) is often the outermost loop. Let’s assume B is placed as the outermost temporal level, after [IY, 32], for simplicity and consistency, with factor [B, B] (full batch size, as no tiling factor is specified).

### Step 5: List All Possible Combinations
Here are the 6 combinations, including B as the outermost temporal level:

#### **Combination 1**
- **Spatial Mapping:**
  - D1: [K, 6]
- **Temporal Ordering:**
  - [FX, 5] # Innermost
  - [FY, 5]
  - [C, 1]
  - [IX, 32]
  - [IY, 32]
  - [B, B] # Outermost

#### **Combination 2**
- **Spatial Mapping:**
  - D1: [K, 6]
- **Temporal Ordering:**
  - [FY, 5] # Innermost
  - [FX, 5]
  - [C, 1]
  - [IX, 32]
  - [IY, 32]
  - [B, B] # Outermost

#### **Combination 3**
- **Spatial Mapping:**
  - D1: [FX, 6]
- **Temporal Ordering:**
  - [K, 5] # Innermost
  - [FY, 5]
  - [C, 1]
  - [IX, 32]
  - [IY, 32]
  - [B, B] # Outermost

#### **Combination 4**
- **Spatial Mapping:**
  - D1: [FX, 6]
- **Temporal Ordering:**
  - [FY, 5] # Innermost
  - [K, 5]
  - [C, 1]
  - [IX, 32]
  - [IY, 32]
  - [B, B] # Outermost

#### **Combination 5**
- **Spatial Mapping:**
  - D1: [FY, 6]
- **Temporal Ordering:**
  - [K, 5] # Innermost
  - [FX, 5]
  - [C, 1]
  - [IX, 32]
  - [IY, 32]
  - [B, B] # Outermost

#### **Combination 6**
- **Spatial Mapping:**
  - D1: [FY, 6]
- **Temporal Ordering:**
  - [FX, 5] # Innermost
  - [K, 5]
  - [C, 1]
  - [IX, 32]
  - [IY, 32]
  - [B, B] # Outermost

### Step 6: Verify and Conclude
- **Parameters Used**: Each combination uses all 7 parameters: one of {K, FX, FY} in spatial mapping, the other two in the innermost temporal levels, plus B, C, IX, IY in the temporal ordering.
- **Fixed Constraint**: [C, 1], [IX, 32], [IY, 32] maintain their relative order.
- **Total Combinations**: 3 spatial choices × 2 permutations of innermost levels = 6.

If B’s position were variable, we’d have more combinations, but fixing B as the outermost aligns with typical mappings and keeps the problem tractable. Thus, there are 6 possible mapping combinations as listed above. These can be tested with the “zigzag” method (not detailed here) to find the best mapping, as per the query.


#### Question 2:

Can you help me on how I can present this information nicely in the graph

Here is the table on the Actual frame based execution stats

|                | Description                         |
|----------------|-------------------------------------|
| **Word Access**| Inputs: 81, Weights: 9, Outputs: 81 |
| **Partial Sum**| 8 * 81 = 648 (across 8 NPEs)        |
| **Latency**    | Off: 0, Comp: 81, On: 0             |

Here is the table on the Zigzag input stationary stats
|                | Description                           |
|----------------|---------------------------------------|
| **Word Access**| Inputs: 9, Weights: 225, Outputs: 225 |
| **Partial Sum**| 8 * 225 = 1800 (across 8 NPEs)        |
| **Latency**    | Off: 0, Comp: 225, On: 0              |

Here is the final table on what the actual event driven stats

|                | Description                           |
|----------------|---------------------------------------|
| **Word Access**| Inputs: 9, Weights: 225, Outputs: 81  |
| **Partial Sum**| 8 * 81 = 648 (across 8 NPEs)          |
| **Latency**    | Off: 0, Comp: 81, On: 0               |

Here is the workload

Input - [1,1,5,5]
Weights(Kernels) - [8,1,3,3]
Outputs - [1,8,3,3]

Now, I want to show the comparision of the Frame based execution, Zigzag execution and actual event based execution stats with a difference clearly notified especially for the wordaccess and the latency.