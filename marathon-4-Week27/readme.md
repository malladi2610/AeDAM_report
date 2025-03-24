<!-- # Next hackathon - 23rd March - 24th March
1. If multicore results are presentable, Work on sparsity integration using sparseloop
2. Run the same workloads and do the multicore dense case and the Sparse case
3. Bring entire thesis together with all the implementations of Single core, Multi core and Sparsity and get strong literature on the components which are worth exploring and make a question of them and present them to review for kanishkan -->

# Marathon goals - 23rd (12:00 PM) - 24th (12:00 PM)

1. [Yes] Running the 2 core accelerator design using the stream for Lenet model and extrating the results from it
2. [No] Build an validation tool to the stream and zigzag to validate the difference in the results
4. [No] Build a script that gives all the possible maps to run without missing out on any combination (As this process needs to be done manually)
3. [No]Study NAAS to see on how to solve the problem on the result variation
4. Send two documents to the Kanishkan and Bishnoi
    4.1. An abstract which is the combination of the Introduction and Basics of Background
    4.2. The results from the Dimension-1 exploration with graphs and tables
    4.3 (For kanishkan) Push the code in the Zigzag stream and AeDAM for his validation

# Marathon closing 

1. Intalled the Stream correctly and tried to run the single core NPE architecture with two cores and lenet-5 model and lenet-5 model mapping from the zigzag

This didn't work right way as the stream 
    1. Doesn't accept the yaml file as input and only ONNX files, the modification attempt to read the yaml file was done but it is complex and requires a lot of efforts

In simple words, Stream didn't work right out of the box

2. The validation tool was not built as the entire efforts were focused on to getting the stream working 

3. The mapping constraint script was not built as same stream focus

4. NAAS study was also not done same stream reason

5. But the documentation will be written in the rest of the day and present to respected people

# Next week goals

1. Get a basic running of the stream with our architecture and simple model/ Lenet would be great

2. Zigzag validation and also study NAAS to find the solution for the extra partial sum count and see how stream can take advantage of it

3. Make AeDAM repo such that Zigzag is the base and the stream uses the same Zigzag base and push the latest changes to the repo

4. Complete (1st draft) the Introduction and Background Study of the report and the dimension 1 exploration (draft) just waiting for the result in coming two weeks

5. (If possible) - Understand how the dimension exploration can be extended the multi dataflow exploration.


