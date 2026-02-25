
🚗 Project Overview
Parallel Traffic Density Simulation with Multi-Lane and Weather Effects using Matrix-Based Computation
1️⃣ Real-World Motivation (Simple Explanation)
In real cities, traffic conditions change continuously due to:
•	Number of vehicles on roads
•	Multiple lanes on each road
•	External conditions such as rain or accidents
Analyzing traffic over a large city involves processing huge amounts of data repeatedly over time, which becomes slow if done on a single processor.
This makes traffic analysis a suitable application for High Performance Computing (HPC).
2️⃣ How the Real World is Simplified for Computation
To make the problem computationally manageable, the real traffic system is simplified, not fully simulated.
Real World → Computational Model
Real World	Computational Model
City road network	2D grid (matrix)
Road segment	One cell in the matrix
Cars on the road	Traffic density value
Multiple lanes	Extra matrix dimension
Weather / accidents	Modifier values
Time passing	Iterative updates
This approach is commonly used in large-scale simulations such as traffic flow, heat diffusion, and weather modeling.
3️⃣ Matrix-Based Traffic Representation
The city is represented as a 2D grid, where each grid cell represents a road segment.
To improve realism:
🔹 Multi-Lane Roads
Each road segment contains multiple lanes, modeled as:
traffic[row][column][lane]
Each lane stores its own traffic density.
🔹 Weather Effects
A separate weather modifier matrix is used:
weather[row][column]
Example values:
•	1.0 → normal conditions
•	0.8 → rain
•	0.5 → accident or obstruction
Weather affects how traffic density changes but does not change the structure of computation.

4️⃣ Traffic Update Process (Core Computation)
Traffic evolves over discrete time steps.
For each time step:
•	Each road segment is updated
•	Each lane is updated independently
•	Weather conditions modify the final value
Update Logic (Conceptual)
1.	Read current traffic density
2.	Read neighboring road densities
3.	Compute base traffic update
4.	Apply weather modifier
5.	Store updated traffic density
All operations are simple:
•	Addition
•	Multiplication
•	Loop iterations
➡️ This makes the computation matrix-based and parallel-friendly.
5️⃣ Why This Problem Fits High Performance Computing
🔑 Key Observation
Traffic updates for each road segment and lane are independent of other updates within the same time step.
This independence allows the computation to be executed in parallel.
6️⃣ Mapping the Problem to HPC Models
🔹 Serial Execution
•	One processor updates all road segments sequentially
•	Used as a baseline for correctness and performance comparison
🔹 Shared Memory Parallelism (OpenMP)
•	Multiple CPU cores update different road segments at the same time
•	All threads share the same traffic grid in memory
•	Parallelism is applied to grid update loops
Real-world analogy:
One traffic control center with many engineers working on different road sections simultaneously.
🔹 Distributed Memory Parallelism (MPI)
•	The city grid is divided into blocks
•	Each MPI process handles a portion of the grid
•	Neighboring blocks exchange boundary information
Real-world analogy:
Multiple traffic control centers, each responsible for part of the city, coordinating at boundaries.
🔹 Hybrid Parallelism (MPI + OpenMP)
•	MPI divides the city among processes
•	OpenMP parallelizes computation within each process
Multiple control centers, Multiple engineers in each center. This model provides better scalability for large simulations.
7️⃣ What the Project Produces (Outputs)
1.	Final traffic density matrices
o	For each lane and road segment
2.	Performance measurements
o	Execution time
o	Speedup
o	Efficiency
3.	Accuracy comparison
o	Parallel results compared with serial output
4.	Scalability analysis
o	Effect of increasing threads and processes
8️⃣ Why This is a Strong HPC Project
✔ Real-world relevance (traffic systems)
✔ Simple and clean computation
✔ Excellent parallelism potential
✔ Supports OpenMP, MPI, and Hybrid models
✔ Easy performance and accuracy evaluation
✔ High marks with low implementation risk
 One-Line Summary (Very Important)
This project demonstrates how a simplified real-world traffic system can be efficiently modeled using matrix-based computation and accelerated using shared-memory and distributed-memory high-performance computing techniques.



🌍 Level 1: MPI → Multiple Cities / Regions
Imagine a national traffic authority.
•	The country is divided into regions or cities
•	Each region has its own traffic control center
•	Each center is responsible only for its region’s roads
Example:
Colombo Control Center → Colombo roads
Kandy Control Center   → Kandy roads
Galle Control Center   → Galle roads

This is MPI:
•	Each control center = one MPI process
•	Each process has its own memory
🏢 Level 2: OpenMP → Engineers inside one Control Center
Inside each traffic control center:
•	There are multiple traffic engineers
•	All engineers work on the same city map
•	Each engineer updates different road segments or lanes
Example:
Colombo Control Center:
  Engineer 1 → Main roads
  Engineer 2 → Side roads
  Engineer 3 → Highway lanes
👉 Engineers:
•	Share the same information
•	Do not send messages to each other
•	Just work in parallel
This is OpenMP:
•	Engineers = threads
•	Shared city map = shared memory

•	🔀 Hybrid Model = Both Levels Together
Put both together:
Country
 ├─ Colombo Control Center (MPI Process)
 │    ├─ Engineer 1 (OpenMP Thread)
 │    ├─ Engineer 2 (OpenMP Thread)
 │
 ├─ Kandy Control Center (MPI Process)
 │    ├─ Engineer 1 (OpenMP Thread)
 │    ├─ Engineer 2 (OpenMP Thread)
•	MPI divides work across regions
•	OpenMP divides work within each region

