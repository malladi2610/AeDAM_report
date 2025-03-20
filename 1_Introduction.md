# Introduction

Developing a workload mapping exploration tool for Event driven accelerators. (AeDAM: An event Driven Mapping Exploration tool) [DSE of the Event driven accelerators]


## Motivation

AI as a technology has been transforming our lives in many ways, it is preset everywhere helping mankind to explore the depts of the ocean in the search of new species to understanding our universe from deriving itelligent correlation from the peta bytes of data to detecting a frand in a fraction of a second.
These applications are possible by the AI operating at Different scales. 


- Question: The Starting to work on the Edge AI applications is not that correct, needs to be improved?
But the area of focus here is the edge applications which deals with running the AI models on the Edge to achieve lower latency where fast inferencese are required.

- Question: The transition of working on event driven accelerators, from the Edge AI applications to the event driven accelerators is required?

Tranditional frame based accelerators are slow and have disadvantage when of executing every single value even if there are sparse cases but in event driven case as each event is passed at once then only the events which are non zero are processed which leads to better performace of the same workload when it is executed on both Traditional and event based accelrators

So, working on event based accelerators is benficial.

## Problem Statment
Exploration of workloads on any accelerator is difficult due varied structre of each one of them and the exploration tools exisitng cater mainly for the traditional accelrators and not for event driven accelerators and there is a need for and approach and research which can perform exploration for them in single core, multi core and also taking sparisity of workload into consideration.

## Need for this research 
There has been exploration done for the transitional architectures to find the best configurationss possible for a given workload and the tools are buld around them, but when it comes to event driven accelerators there has been no research done to find the best configurations possible when the execution is done in this (event driven) manner.

- (Literature papers to support it are needed)

## Contribution of this research
This thesis utilises exiting tools, improves/tweaks them to perform the exploration of the workloads onto the event driven accelerators to optimise for the latency, energy and EDP

- On a high level the basic math behind the computation remains the same but how is utilised and modified to get the best output is the research

## Question to be part of this topic


