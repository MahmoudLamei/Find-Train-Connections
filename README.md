# Assignment 1.1: Find train connections

This repository contains the solution for the "Find train connections" project.

You can find the code inside the file `main.py`

I used the Networkx library to turn the dataset to a graph.
The stations were used as  nodes and the edges between them had all the information needed to go from node A to B.

Due to the large size of the data set it took about 5 minutes to turn the dataset to a graph, so I decided to save it as a pickle file.

The test() method creates the `solutins.csv` file that is used for testing, using the provided script.

The method format() takes the returned value from the dijkstra() method and turns it to the required format.

To make the dataset easier to use I created a new one with the relevant columns only.