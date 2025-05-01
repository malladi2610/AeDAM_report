#### Question 1:

Get a json_parser to read the json data which is structures as follows as these json files are really large, I need them to extract the data only when the specifice parameter is greater than a certain number

Here is the structure of the json this just a single data element

"spatial_0": {
    "layer_id": 0,
    "spatial_mapping": {
      "D1": {
        "FY": 5
      }
    },
    "temporal_mappings": [
      {
        "mapping_dic_origin": {
          "O": "[[(K, 16), (IY, 14), (IX, 7), (IX, 2), (FX, 5), (C, 6)]]",
          "W": "[[(K, 16), (IY, 14), (IX, 7), (IX, 2), (FX, 5), (C, 6)]]",
          "I": "[[(K, 16), (IY, 14), (IX, 7), (IX, 2), (FX, 5), (C, 6)]]"
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
            "IY": 14,
            "IX": 14,
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
              "FY": 5
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
            "OX": 18,
            "OY": 18
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
            "O": 5184,
            "W": 2400,
            "I": 1176
          },
          "total_mac_count": "470400",
          "operand_size_bit": {
            "O": 82944,
            "W": 38400,
            "I": 18816
          },
          "operand_data_reuse": {
            "O": 90,
            "W": 196,
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
          "O": "[[(K, 16), (IY, 14), (IX, 7), (IX, 2), (FX, 5), (C, 6)]]",
          "W": "[[(K, 16), (IY, 14), (IX, 7), (IX, 2), (FX, 5), (C, 6)]]",
          "I": "[[(K, 16), (IY, 14), (IX, 7), (IX, 2), (FX, 5), (C, 6)]]"
        },
        "mac_level_data_stationary_cycle": {
          "O": 1,
          "W": 1,
          "I": 16
        },
        "cycle_cabl_level": {
          "O": "[94080]",
          "W": "[94080]",
          "I": "[94080]"
        },
        "total_cycle": "94080",
        "top_r_loop_size": {
          "O": "[1, 1]",
          "W": "[1, 30]",
          "I": "[1, 6]"
        },
        "top_ir_loop_size": {
          "O": "[1, 6]",
          "W": "[1, 1]",
          "I": "[1, 1]"
        }
      },

I need a complete script that takes the json file as input, when tunable variables in the program to input the parameters which needs to be used for filterning "I will always be filtering with the parameter "        "mac_level_data_stationary_cycle": {
          "O": 1,
          "W": 1,
          "I": 16"

Where I would be with filtering with a case where O > 10 or W > 10 or I > 10 any of these cases and then I would be saving this data in a seperate file.

Can you help me with the json file which meets the above requirements


#### Answer: 

