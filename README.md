# AIDRA – Adaptive Intelligent Disaster Response Agent

## Project Description

**AIDRA (Adaptive Intelligent Disaster Response Agent)** is an AI-based disaster management system designed to support emergency response operations during natural or man-made disasters. The system simulates a disaster environment and applies multiple Artificial Intelligence techniques to optimize rescue operations, resource allocation, and decision-making.

The project integrates **Machine Learning**, **Search Algorithms**, **Constraint Satisfaction Problems (CSP)**, and **Fuzzy Logic** to create an intelligent and adaptive rescue system capable of handling dynamic disaster situations.


## Objectives

* Simulate a disaster environment using a grid-based model
* Predict patient triage priority using Machine Learning
* Find optimal rescue paths using search algorithms
* Allocate emergency resources efficiently using CSP
* Handle uncertainty using fuzzy logic
* Adapt to environmental changes through dynamic replanning


## Features

* Grid-based disaster environment simulation
* Victim classification based on severity levels
* Pathfinding using BFS, DFS, Greedy Best First Search, and A*
* Resource allocation using Constraint Satisfaction Problem (CSP)
* Machine Learning models for medical triage prediction
* Fuzzy logic-based priority evaluation
* Dynamic route replanning during environmental changes
* Real-time adaptive rescue decision-making


## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Scikit-fuzzy
* Matplotlib


### 1. Environment Module

Creates a grid-based disaster simulation environment containing:

* Victims
* Hazards
* Blocked roads
* Hospitals
* Rescue base stations


### 2. Machine Learning Module

Uses classification algorithms to predict medical triage priority based on patient information.

Implemented models:

* K-Nearest Neighbors (KNN)
* Naive Bayes



### 3. Search Algorithms Module

Implements intelligent pathfinding algorithms for rescue operations.

Algorithms included:

* Breadth First Search (BFS)
* Depth First Search (DFS)
* Greedy Best First Search
* A* Search Algorithm



### 4. CSP Module

Allocates victims to available ambulances and rescue teams while satisfying operational constraints.

Constraints include:

* Ambulance capacity limits
* Medical kit availability
* Rescue team scheduling


### 5. Fuzzy Logic Module

Handles uncertainty in disaster conditions and computes rescue priority scores using fuzzy inference.

Factors considered:

* Injury severity
* Risk level
* Distance
* Environmental danger



### 6. Replanning Module

Dynamically updates rescue paths when:

* Roads become blocked
* Hazards appear
* Environmental conditions change

This ensures adaptive and intelligent decision-making during rescue operations.



## System Workflow

1. Generate disaster environment
2. Detect and classify victims
3. Predict triage priority using Machine Learning
4. Calculate rescue priority using Fuzzy Logic
5. Find optimal rescue paths using search algorithms
6. Allocate resources using CSP
7. Dynamically replan routes if environment changes





## Example Applications

AIDRA can be applied in:

* Earthquake response systems
* Flood rescue management
* Fire emergency operations
* Smart city disaster management
* AI-based emergency planning systems

---

## Future Improvements

* Integration with real-time sensor data
* Drone-assisted rescue planning
* Reinforcement Learning for autonomous decision-making
* Real-world GIS map integration
* Multi-agent coordination systems
* Web dashboard for live monitoring

---

## Conclusion

AIDRA demonstrates how multiple Artificial Intelligence techniques can be integrated into a single intelligent system for efficient disaster response management. The project combines:

* Learning
* Reasoning
* Optimization
* Uncertainty handling
* Adaptive replanning

to simulate realistic and intelligent emergency rescue operations.


