It is a 24 hrs hackathon starting from Sunday till Monday - 16th to 17th March

# Goals to achieve are as follows:
1. Zigzag cost model replication on the excel sheet/Python script with Technology dependent and independent stats Understanding the cost model to manipulate it with the parameters to get desired results -> Stopped this for now as it is not leading anywhere and it is getting complicated. Any way this is used for validation and once we have the results we can performa validation
2. Select two workloads (Small and Big), Consider a multi-layer workload for better exploration results comparision
3. Run the results on Zigzag and make it give input stationary/event driven results with multiple different configuration
4. Make a simulator newly by building upon the previous implementation to validate the results of the zigzag model after verification
5. Clean the repo(Separating the inputs, ouput, verifications, validations(Simulation) of single core and multi core results) and add the stream integration to the repo and push the code

# Expected Output of the hackathon
1. Single core, dimension 1 (Single dataflow event driven exploration) with verification and validation
2. A working simulator and visualiser to verify and validate the results

# Conclusion of the hackathon
1. The Zigzag cost model replication was difficult and was getting complicated - So, Dropped that plan and will move with validation once the multicore is done
2. Started running the workload on the Single core architecture given by Kanishkan the cost model is working fine. Now, Should present the results in a manner the are presentable and make a meaningful debuctions

# Plan for the next week
1. Continue with the single core results with more menaing ful workloads of a model and run it completely
2. Do a meaningful graphs and table to make it presentable
3. Replicate the same thing with the multicore and show improvements from single core to multicore 


# Next hackathon
1. If multicore results are presentable, Work on sparsity integration using sparseloop
2. Run the same workloads and do the multicore dense case and the Sparse case
3. Bring entire thesis together with all the implementations of Single core, Multi core and Sparsity and get strong literature on the components which are worth exploring and make a question of them and present them to review for kanishkan