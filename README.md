# Euler Eulerchain - Generative AI Framework 📊

![Euler Eulerchain](https://via.placeholder.com/800x400)

## Author ✍️
**Name**: Prashant Verma  
**Email**: prashant27050@gmail.com

---

## Table of Contents 📚

1. [Introduction](#introduction-ℹ️)
2. [Installation](#installation-💻)
3. [Project Structure](#project-structure-📁)
4. [Running the Application](#running-the-application-🚀)
5. [User Guide](#user-guide-📝)
    - [Main Interface](#main-interface-🖥️)
    - [Loading a Graph](#loading-a-graph-📂)
    - [Visualizing a Graph](#visualizing-a-graph-🔍)
    - [Adding Nodes](#adding-nodes-➕)
    - [Adding Edges](#adding-edges-🔗)
    - [Saving a Graph](#saving-a-graph-💾)
    - [Executing Queries](#executing-queries-📋)
    - [Saving Queries](#saving-queries-💼)
    - [Loading Queries](#loading-queries-📄)
6. [FAQ](#faq-❓)
7. [Support](#support-🙋‍♂️)

---

## Introduction ℹ️

The Euler Eulerchain is a knowledge graph viewer that allows users to create, visualize, and manage knowledge graphs. It provides an interactive graphical user interface (GUI) for performing various operations on the knowledge graph such as adding nodes and edges, executing queries, and visualizing the graph.

## Installation 💻

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/euler_graph_database.git
    cd euler_graph_database
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure 📁


## Running the Application 🚀

To run the application, use the following command:
```bash
python -m gui.app
```

This command ensures that the Python interpreter recognizes the package structure and resolves imports correctly.

## User Guide 📝

### Main Interface 🖥️

The main interface of the application consists of the following components:
- **Header**: Displays the application name.
- **Navbar**: Provides options to load, visualize, add nodes, add edges, save the graph, and display help/about information.
- **Query Entry**: A text area to enter queries.
- **Buttons**: 
    - `Execute Query`: Executes the entered query.
    - `Save Query`: Saves the entered queries to a file.
    - `Load Query`: Loads queries from a file.
- **Output Areas**:
    - `Query Output`: Displays the results of executed queries.
    - `JSON Output`: Displays the JSON representation of the current graph.

### Loading a Graph 📂

1. Click on the `Load Graph` button in the navbar.
2. Select the file containing the graph you want to load.

### Visualizing a Graph 🔍

1. After loading a graph, click on the `Visualize Graph` button in the navbar.
2. The graph will be displayed in the visualization area.

### Adding Nodes ➕

1. Click on the `Add Node` button in the navbar.
2. Enter the node ID and label in the prompt that appears.
3. The node will be added to the graph.

### Adding Edges 🔗

1. Click on the `Add Edge` button in the navbar.
2. Enter the edge ID, source node ID, target node ID, and edge label in the prompt that appears.
3. The edge will be added to the graph.

### Saving a Graph 💾

1. Click on the `Save Graph` button in the navbar.
2. Choose the location to save the graph file.

### Executing Queries 📋

1. Enter your query in the `Query Entry` area.
2. Click the `Execute Query` button.
3. The result of the query will be displayed in the `Query Output` area.

### Saving Queries 💼

1. Enter your queries in the `Query Entry` area.
2. Click the `Save Query` button.
3. Choose the location to save the queries file (with `.euler` extension).

### Loading Queries 📄

1. Click the `Load Query` button.
2. Select the file containing the queries.
3. The queries will be loaded into the `Query Entry` area.

## FAQ ❓

**Q: What file formats are supported for saving graphs?**  
A: The application supports saving graphs in JSON format.

**Q: How do I fix module import errors?**  
A: Ensure you are running the application using `python -m gui.app` to correctly set up the module paths.

**Q: Can I visualize large graphs?**  
A: Yes, but performance may vary depending on the size of the graph and the capabilities of your system.

## Support 🙋‍♂️

For any issues or questions, please contact:

**Name**: Prashant Verma  
**Email**: prashant27050@gmail.com

Or visit the GitHub repository for more information and updates.