See, I will point out a problem here where zigzag is doing correctly and our script is now.


Here are the word access values from zigzag

"word_accesses": {
                "O": [
                    {
                        "rd ^": 28901376,
                        "wr v": 28901376,
                        "rd v": 86704128,
                        "wr ^": 86704128
                    },
                    {
                        "rd ^": 1214976,
                        "wr v": 1214976,
                        "rd v": 10838016,
                        "wr ^": 10838016
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 2429952,
                        "wr ^": 2429952
                    }
                ],
                "W": [
                    {
                        "rd ^": 0,
                        "wr v": 28901376,
                        "rd v": 86704128,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 216,
                        "rd v": 10838016,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 216,
                        "wr ^": 0
                    }
                ],
                "I": [
                    {
                        "rd ^": 0,
                        "wr v": 401408,
                        "rd v": 1204224,
                        "wr ^": 0
                    },
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 50176,
                        "wr ^": 0
                    }
                ]
            }

1. They are not multiplies with the operand precision and are always in words

2. If you see the inputs

For a workload as follows

 workload = {
        'dimensions': {'K': 64, 'C': 3, 'FX': 3, 'FY': 3, 'IX': 224, 'IY': 224},
        'calculated_output_dimensions': {'OX': 224, 'OY': 224},
        'operand_precision': {'W': 16, 'O': 16, 'I': 16}
    }

With NPE of 8 and a restricted data path as follows

  Input Path ('I'): ['NOC_inputs_outputs', 'register']
DEBUG (get_weight_path): No mapping data for psum. Psum est. based on full output: 12544.00 KB
Warning: Estimated Psum buffer (12544.00KB) exceeds/equals total SRAM (512.00KB). Assuming no SRAM for weights.
Info: Weights (3456B) will use 'shared_memory_Weights' due to available SRAM (0B after psums).
  Weight Path ('W'): ['shared_memory_Weights', 'sram', 'register']
  Partial Sum Write Path ('O_WR'): ['register', 'sram']
  Final Output Path ('O_FINAL'): ['register', 'NOC_inputs_outputs']

The data is already assumed to be present in the places where it is supposed to be 

Here is my explanation of the wordaccess for inputs

NOC level

{
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 50176,
                        "wr ^": 0
                    }


rd V - From NOC to RF with is 224 * 224 elements

To RF

 {
                        "rd ^": 0,
                        "wr v": 401408,
                        "rd v": 1204224,
                        "wr ^": 0
                    },

wr V - THis is from the NOC to RF in all NPES: 224*224*3*16*8 - Input size * size of the operand * Noo.f NPES = 19267584/48 (W_bw of RF) = 401408

rd V - This is from RF to NPEs : 224*224*3*8 -  Input size * Noo.f NPES = 1204224 - Which indicates each input is procesed one event at a time and until all it's computations are over it stayed there and never repeated [THis is the event driven definsion for me]


Now for the weights
RF
 {
                        "rd ^": 0,
                        "wr v": 28901376,
                        "rd v": 86704128,
                        "wr ^": 0
                    },

SRAM
                    {
                        "rd ^": 0,
                        "wr v": 216,
                        "rd v": 10838016,
                        "wr ^": 0
                    },

Shared memory
                    {
                        "rd ^": 0,
                        "wr v": 0,
                        "rd v": 216,
                        "wr ^": 0}

Shared memory: 

rd V : From Shared memeory to SRAM (Initial load): 3*3*3*64*16 (all the weights * weight operand size)= 27648/128 (Read bw of Shared memory) = 216

SRAM:



