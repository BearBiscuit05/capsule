#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <iostream>
#include "globalConfig.h"

// Define the Edge class
class Edge {
public:
    int srcVId;
    int destVId;
    int weight;

public:
    Edge();
    Edge(int srcVId, int destVId, int weight);
    bool operator!=(const Edge& other) const {
        return (weight != other.weight) && (srcVId != other.srcVId) && (destVId != other.destVId);
    }
    int getSrcVId() const;
    int getDestVId() const;
    int getWeight() const;
    void addWeight();
};

// Define the Graph class
class Graph {
public:
    std::vector<Edge> edgeList;
    int vCount;
    int eCount;
    std::ifstream fileStream;
    std::string graphpath;


    Graph();
    Graph(GlobalConfig config);
    ~Graph();
    Graph(const Graph& other);
    Graph& operator=(const Graph& other);
    int readStep(Edge& edge);
    void readGraphFromFile();
    void addEdge(int srcVId, int destVId);
    std::vector<Edge> getEdgeList();
    int getVCount();
    int getECount();
    void clear();
};

#endif // GRAPH_H