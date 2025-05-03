# About
This is the json parser development for filtering the required map space from the Zigzag

# Questions 

#### Question 1:

I have a 720 elements mapspace, each with 20 sub classes. I want a python script that reads this json file, then filters the elements with certain combinations from the entire mapspace and stores them in the seperate file.

Here is how a single element in the mapspace looks

```json
//It starts with the spatial_mappings block
"spatial_0": {
    "layer_id": 0,
    "spatial_mapping": {
      "D1": {
        "C": 6
      }
    },
    //Then comes the temporal mappings block and rest of the maps are part of this block
    "temporal_mappings": [
      {
        "mapping_dic_origin": {
          "O": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]",
          "W": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]",
          "I": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]"
        },
        "layer_node": {
          "id": "0",
          "name": "layer_2",
          "type": "Conv",
          "equation": {
            "data": "O[b][k][oy][ox] = W[k][c][fy][fx] * I[b][c][iy][ix]",
            "disassembly": "['O', 'b', 'k', 'oy', 'ox', '=', 'W', 'k', 'c', 'fy', 'fx', '*', 'I', 'b', 'c', 'iy', 'ix']"
          },
          "layer_dim_sizes": {
            "B": 1,
            "K": 16,
            "C": 6,
            "IY": 70,
            "IX": 70,
            "FY": 5,
            "FX": 5
          },
          "operand_precision": {
            "data": {
              "W": 16,
              "I": 16,
              "O": 16,
              "O_final": 16
            }
          },
          "dimension_relations": "[OX = 1*IX + 1*FX, OY = 1*IY + 1*FY]",
          "padding": {
            "data": {}
          },
          "constant_operands": "[I, W]",
          "input_operand_source": {},
          "spatial_mapping": {
            "D1": {
              "C": 6
            }
          },
          "spatial_mapping_hint": {
            "data": {
              "D1": "{FX, B, IY, IX, C, FY, K}"
            }
          },
          "memory_operand_links": {
            "data": {
              "O": {},
              "W": {},
              "I": {}
            },
            "layer_operands": "[O, W, I]",
            "mem_operands": "[O, I2, I1]"
          },
          "temporal_ordering": {
            "data": "[]"
          },
          "layer_operands": "[O, W, I]",
          "output_operand": {},
          "input_operands": "[W, I]",
          "layer_dims": "[B, K, C, IY, IX, FY, FX]",
          "pr_loop": {
            "OX": "(IX, FX)",
            "OY": "(IY, FY)"
          },
          "pr_scaling_factors": {
            "OX": "((IX, 1), (FX, 1))",
            "OY": "((IY, 1), (FY, 1))"
          },
          "pr_layer_dim_sizes": {
            "OX": 74,
            "OY": 74
          },
          "loop_relevancy_info": {
            "r_dims": {
              "O": "[B, K]",
              "W": "[K, C, FY, FX]",
              "I": "[C, IY, IX]"
            },
            "ir_dims": {
              "O": "[C]",
              "W": "[IY, IX]",
              "I": "[FX, FY, K]"
            },
            "pr_dims": {
              "O": {
                "OX": "(IX, FX)",
                "OY": "(IY, FY)"
              },
              "W": {},
              "I": {}
            },
            "orig_pr_loop": {
              "OX": "(IX, FX)",
              "OY": "(IY, FY)"
            }
          },
          "pr_decoupled_relevancy_info": {
            "r_dims": {
              "O": "[B, K, OX_R, OY_R]",
              "W": "[K, C, FY, FX, OX_R, OY_R]",
              "I": "[C, IY, IX, OX_R, OY_R]"
            },
            "ir_dims": {
              "O": "[C, OX_IR, OY_IR]",
              "W": "[IY, IX, OX_IR, OY_IR]",
              "I": "[FX, FY, K, OX_IR, OY_IR]"
            },
            "pr_dims": {}
          },
          "operand_size_elem": {
            "O": 87616,
            "W": 2400,
            "I": 29400
          },
          "total_mac_count": "11760000",
          "operand_size_bit": {
            "O": 1401856,
            "W": 38400,
            "I": 470400
          },
          "operand_data_reuse": {
            "O": 134,
            "W": 4900,
            "I": 400
          }
        },
        "operand_list": "[O, W, I]",
        "mem_level": {
          "O": 1,
          "W": 1,
          "I": 1
        },
        "mapping_dic_stationary": {
          "O": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]",
          "W": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]",
          "I": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]"
        },
        "mac_level_data_stationary_cycle": {
          "O": 1,
          "W": 1,
          "I": 16
        },
        "cycle_cabl_level": {
          "O": "[1960000]",
          "W": "[1960000]",
          "I": "[1960000]"
        },
        "total_cycle": "1960000",
        "top_r_loop_size": {
          "O": "[1, 1]",
          "W": "[1, 25]",
          "I": "[1, 1]"
        },
        "top_ir_loop_size": {
          "O": "[1, 1]",
          "W": "[1, 1]",
          "I": "[1, 25]"
        }
      },]
}
```

Now, from this map I want to filter the combinations the cell "mac_level_data_stationary_cycle" where the "I": is the highest. For this particular case of 720 mappings it is 80. So, all the cases where this particular variable is 80 the elements need to be extracted and saved in the seperate file.

While developing the code, make sure this code is modular and the condition variable ["mac_level_data_stationary_cycle" where the "I": is the highest], can be changed in future. also right now the json file is small but later it can be as large as 1.2 GB. Build the script accordingly.



#### Answer 1:

import json
import ijson

def filter_mappings(json_file, output_file, condition):
    """
    Filters mappings from a JSON file based on a condition and saves them to a new file.
    
    Args:
        json_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
        condition (callable): Function that takes a mapping dict and returns a boolean.
    """
    with open(output_file, 'w') as f_out:
        f_out.write('[')  # Start of JSON list
        first = True
        
        with open(json_file, 'rb') as f_in:
            # Iterate through each "spatial_X" dictionary incrementally
            for spatial_key, spatial_dict in ijson.kvitems(f_in, ''):
                # Process each mapping in "temporal_mappings"
                for mapping in spatial_dict['temporal_mappings']:
                    if condition(mapping):
                        if not first:
                            f_out.write(',')  # Add comma before subsequent entries
                        json.dump(mapping, f_out)  # Write the mapping as JSON
                        first = False
        
        f_out.write(']')  # End of JSON list

# Define the filtering condition
def condition(mapping):
    """
    Checks if the 'I' value in 'mac_level_data_stationary_cycle' equals 80.
    
    Args:
        mapping (dict): A single mapping from "temporal_mappings".
    
    Returns:
        bool: True if the condition is met, False otherwise.
    """
    return mapping['mac_level_data_stationary_cycle']['I'] == 80

# Usage example
if __name__ == "__main__":
    json_file = 'input.json'    # Replace with your input file path
    output_file = 'filtered.json'  # Replace with your output file path
    filter_mappings(json_file, output_file, condition)


#### Question 2: The output diplayed is not according to the needs as it doesn't match the actual output format and it needs to. So, modification needs to be done to the code how  to do it.


Here is the format in which I want
```json
//It starts with the spatial_mappings block
"spatial_0": {
    "layer_id": 0,
    "spatial_mapping": {
      "D1": {
        "C": 6
      }
    },
    //Then comes the temporal mappings block and rest of the maps are part of this block
    "temporal_mappings": [
      {
        "mapping_dic_origin": {
          "O": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]",
          "W": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]",
          "I": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]"
        },
        "layer_node": {
          "id": "0",
          "name": "layer_2",
          "type": "Conv",
          "equation": {
            "data": "O[b][k][oy][ox] = W[k][c][fy][fx] * I[b][c][iy][ix]",
            "disassembly": "['O', 'b', 'k', 'oy', 'ox', '=', 'W', 'k', 'c', 'fy', 'fx', '*', 'I', 'b', 'c', 'iy', 'ix']"
          },
          "layer_dim_sizes": {
            "B": 1,
            "K": 16,
            "C": 6,
            "IY": 70,
            "IX": 70,
            "FY": 5,
            "FX": 5
          },
          "operand_precision": {
            "data": {
              "W": 16,
              "I": 16,
              "O": 16,
              "O_final": 16
            }
          },
          "dimension_relations": "[OX = 1*IX + 1*FX, OY = 1*IY + 1*FY]",
          "padding": {
            "data": {}
          },
          "constant_operands": "[I, W]",
          "input_operand_source": {},
          "spatial_mapping": {
            "D1": {
              "C": 6
            }
          },
          "spatial_mapping_hint": {
            "data": {
              "D1": "{FX, B, IY, IX, C, FY, K}"
            }
          },
          "memory_operand_links": {
            "data": {
              "O": {},
              "W": {},
              "I": {}
            },
            "layer_operands": "[O, W, I]",
            "mem_operands": "[O, I2, I1]"
          },
          "temporal_ordering": {
            "data": "[]"
          },
          "layer_operands": "[O, W, I]",
          "output_operand": {},
          "input_operands": "[W, I]",
          "layer_dims": "[B, K, C, IY, IX, FY, FX]",
          "pr_loop": {
            "OX": "(IX, FX)",
            "OY": "(IY, FY)"
          },
          "pr_scaling_factors": {
            "OX": "((IX, 1), (FX, 1))",
            "OY": "((IY, 1), (FY, 1))"
          },
          "pr_layer_dim_sizes": {
            "OX": 74,
            "OY": 74
          },
          "loop_relevancy_info": {
            "r_dims": {
              "O": "[B, K]",
              "W": "[K, C, FY, FX]",
              "I": "[C, IY, IX]"
            },
            "ir_dims": {
              "O": "[C]",
              "W": "[IY, IX]",
              "I": "[FX, FY, K]"
            },
            "pr_dims": {
              "O": {
                "OX": "(IX, FX)",
                "OY": "(IY, FY)"
              },
              "W": {},
              "I": {}
            },
            "orig_pr_loop": {
              "OX": "(IX, FX)",
              "OY": "(IY, FY)"
            }
          },
          "pr_decoupled_relevancy_info": {
            "r_dims": {
              "O": "[B, K, OX_R, OY_R]",
              "W": "[K, C, FY, FX, OX_R, OY_R]",
              "I": "[C, IY, IX, OX_R, OY_R]"
            },
            "ir_dims": {
              "O": "[C, OX_IR, OY_IR]",
              "W": "[IY, IX, OX_IR, OY_IR]",
              "I": "[FX, FY, K, OX_IR, OY_IR]"
            },
            "pr_dims": {}
          },
          "operand_size_elem": {
            "O": 87616,
            "W": 2400,
            "I": 29400
          },
          "total_mac_count": "11760000",
          "operand_size_bit": {
            "O": 1401856,
            "W": 38400,
            "I": 470400
          },
          "operand_data_reuse": {
            "O": 134,
            "W": 4900,
            "I": 400
          }
        },
        "operand_list": "[O, W, I]",
        "mem_level": {
          "O": 1,
          "W": 1,
          "I": 1
        },
        "mapping_dic_stationary": {
          "O": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]",
          "W": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]",
          "I": "[[(K, 16), (IY, 70), (IX, 10), (IX, 7), (FY, 5), (FX, 5)]]"
        },
        "mac_level_data_stationary_cycle": {
          "O": 1,
          "W": 1,
          "I": 16
        },
        "cycle_cabl_level": {
          "O": "[1960000]",
          "W": "[1960000]",
          "I": "[1960000]"
        },
        "total_cycle": "1960000",
        "top_r_loop_size": {
          "O": "[1, 1]",
          "W": "[1, 25]",
          "I": "[1, 1]"
        },
        "top_ir_loop_size": {
          "O": "[1, 1]",
          "W": "[1, 1]",
          "I": "[1, 25]"
        }
      },]
}
```

But here is the output I recieved from the code

```json
[{"mapping_dic_origin": {"O": "[[(FX, 5), (K, 16), (FY, 5), (IY, 70), (IX, 10), (IX, 7)]]", "W": "[[(FX, 5), (K, 16), (FY, 5), (IY, 70), (IX, 10), (IX, 7)]]", "I": "[[(FX, 5), (K, 16), (FY, 5), (IY, 70), (IX, 10), (IX, 7)]]"}, "layer_node": {"id": "0", "name": "layer_2", "type": "Conv", "equation": {"data": "O[b][k][oy][ox] = W[k][c][fy][fx] * I[b][c][iy][ix]", "disassembly": "['O', 'b', 'k', 'oy', 'ox', '=', 'W', 'k', 'c', 'fy', 'fx', '*', 'I', 'b', 'c', 'iy', 'ix']"}, "layer_dim_sizes": {"B": 1, "K": 16, "C": 6, "IY": 70, "IX": 70, "FY": 5, "FX": 5}, "operand_precision": {"data": {"W": 16, "I": 16, "O": 16, "O_final": 16}}, "dimension_relations": "[OX = 1*IX + 1*FX, OY = 1*IY + 1*FY]", "padding": {"data": {}}, "constant_operands": "[I, W]", "input_operand_source": {}, "spatial_mapping": {"D1": {"C": 6}}, "spatial_mapping_hint": {"data": {"D1": "{FX, B, IY, IX, C, FY, K}"}}, "memory_operand_links": {"data": {"O": {}, "W": {}, "I": {}}, "layer_operands": "[O, W, I]", "mem_operands": "[O, I2, I1]"}, "temporal_ordering": {"data": "[]"}, "layer_operands": "[O, W, I]", "output_operand": {}, "input_operands": "[W, I]", "layer_dims": "[B, K, C, IY, IX, FY, FX]", "pr_loop": {"OX": "(IX, FX)", "OY": "(IY, FY)"}, "pr_scaling_factors": {"OX": "((IX, 1), (FX, 1))", "OY": "((IY, 1), (FY, 1))"}, "pr_layer_dim_sizes": {"OX": 74, "OY": 74}, "loop_relevancy_info": {"r_dims": {"O": "[B, K]", "W": "[K, C, FY, FX]", "I": "[C, IY, IX]"}, "ir_dims": {"O": "[C]", "W": "[IY, IX]", "I": "[FX, FY, K]"}, "pr_dims": {"O": {"OX": "(IX, FX)", "OY": "(IY, FY)"}, "W": {}, "I": {}}, "orig_pr_loop": {"OX": "(IX, FX)", "OY": "(IY, FY)"}}, "pr_decoupled_relevancy_info": {"r_dims": {"O": "[B, K, OX_R, OY_R]", "W": "[K, C, FY, FX, OX_R, OY_R]", "I": "[C, IY, IX, OX_R, OY_R]"}, "ir_dims": {"O": "[C, OX_IR, OY_IR]", "W": "[IY, IX, OX_IR, OY_IR]", "I": "[FX, FY, K, OX_IR, OY_IR]"}, "pr_dims": {}}, "operand_size_elem": {"O": 87616, "W": 2400, "I": 29400}, "total_mac_count": "11760000", "operand_size_bit": {"O": 1401856, "W": 38400, "I": 470400}, "operand_data_reuse": {"O": 134, "W": 4900, "I": 400}}, "operand_list": "[O, W, I]", "mem_level": {"O": 1, "W": 1, "I": 1},"mapping_dic_stationary": {"O": "[[(FX, 5), (K, 16), (FY, 5), (IY, 70), (IX, 10), (IX, 7)]]", "W": "[[(FX, 5), (K, 16), (FY, 5), (IY, 70), (IX, 10), (IX, 7)]]", "I": "[[(FX, 5), (K, 16), (FY, 5), (IY, 70), (IX, 10), (IX, 7)]]"}, "mac_level_data_stationary_cycle": {"O": 1, "W": 1, "I": 400}, "cycle_cabl_level": {"O": "[1960000]", "W": "[1960000]", "I": "[1960000]"}, "total_cycle": "1960000", "top_r_loop_size": {"O": "[1, 1]", "W": "[1, 1]", "I": "[1, 4900]"}, "top_ir_loop_size": {"O": "[1, 1]", "W": "[1, 4900]", "I": "[1, 1]"}},```


So, it needs to be improved further to meet the requirements


