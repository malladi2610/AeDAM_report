I have a question here regarding the path that is being decided



"--- Retrieving Data Paths (Demonstration) ---

  Input Path ('I'): ['NOC_inputs_outputs', 'register']

Warning: Estimated Psum buffer (12544.00KB) exceeds total SRAM (256.00KB). Assuming no SRAM for weights.

Info: Weights (3456B) estimated to not fit in available SRAM (0B considering psums). Using 'shared_memory_Weights'.

  Weight Path ('W'): ['shared_memory_Weights', 'sram', 'register']

  Partial Sum Write Path ('O_WR'): ['register', 'sram']

  Final Output Path ('O_FINAL'): ['register', 'NOC_inputs_outputs']"





Here are some clarification regarding the architecture



"name: Event_Driven_Accelerator



operational_array:

  unit_energy: 0.04  # for ANN MAC operations

  unit_area: 1

  dimensions: [D1]

  sizes: [8]  # 8 NPEs



memories:

  register:

    size: 1024  # 1024 bits for NPE register files

    r_bw: 48  # high bandwidth for fast access

    w_bw: 48

    r_cost: 8  # low energy cost, e.g., 8-12 pJ

    w_cost: 8

    area: 0.01  # small area per instance

    r_port: 3

    w_port: 2

    rw_port: 0

    latency: 1  # <1 ns (Has to been an integer as per zigzag modelling, So 1ns)

    operands: [I1,I2,O]

    ports:

      - tl: r_port_1

        fh: w_port_1

      - tl: r_port_2

        fh: w_port_2

      - th: r_port_2

        fh: w_port_2

        tl: r_port_3

    served_dimensions: []



  sram:

    size: 2097152  # 2 Mbits Data Memory

    r_bw: 128

    w_bw: 128

    r_cost: 1.6  # medium energy cost, e.g., 180-220 pJ, current unit is pj/bits

    w_cost: 1.6

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

    size: 1677721600  # 1600 Mbits, e.g., for STT-MRAM

    r_bw: 128  

    w_bw: 128

    r_cost: 15  

    w_cost: 15

    area: 2

    r_port: 1

    w_port: 0

    rw_port: 0

    latency: 13  

    operands: [I2]

    ports:

      - tl: r_port_1

    served_dimensions: [D1]



  NOC_inputs_outputs:

    size: 1073741824   # 1024 Mbits, e.g., for STT-MRAM 

    r_bw: 64  

    w_bw: 64

    r_cost: 0.06  

    w_cost: 0.06  

    area: 2

    r_port: 1

    w_port: 1

    rw_port: 0

    latency: 10  

    operands: [I1, O]

    ports:

      - tl: r_port_1

      - fl: w_port_1

    served_dimensions: [D1]



"



All the data of sizes are in bits here.



Now coming to the workload



I = 224*224*3 (IH, IW, C) * 16 bits (Size of the data) = 2408448 bits

W = 3 * 3 * 64 * 3 (FX, FY, K, C) * 16 (Size of the data) = 27648 bits

C = 224 * 224 * 64 (OX, OY, K) * 16 (Size of the data) = 51380224 bits

Total partial sums = NO.of MAC operation = 3 * 3 * 3 * 224 * 224 * 64 * 16 = 1387266048 bits

Now with this data and the above mentioned architecture. Can you explain the data_path output that We are getting from the script?