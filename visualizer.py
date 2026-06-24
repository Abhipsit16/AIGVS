import networkx as nx
import matplotlib.pyplot as plt

def export_colored_tree_diagram(pipeline_ref, filename="aigvs_tree.png"):
    print("\n[Visualizer] Generating graphical tree diagram...")
    G = nx.DiGraph()
    
    # Define colors for different node types
    color_map = []
    
    def add_edges_recursively(node, parent_id=None, depth=0):
        # Determine node type and capacity
        if hasattr(node, 'children') and node.children:
            node_label = f"BRANCH\n{node.node_id[-6:]}"
            node_color = "lightblue" if depth > 0 else "gold" # Gold for Root
        else:
            capacity = node._count if hasattr(node, '_count') else len(getattr(node, 'chunk_ids', []))
            node_label = f"LEAF\n{node.node_id[-6:]}\n({capacity} items)"
            node_color = "lightgreen"
            
        G.add_node(node.node_id, label=node_label, color=node_color)
        
        if parent_id:
            G.add_edge(parent_id, node.node_id)
            
        if hasattr(node, 'children'):
            for child in node.children.values():
                add_edges_recursively(child, node.node_id, depth + 1)

    # Build the graph starting from the root
    add_edges_recursively(pipeline_ref.root)
    
    # Extract labels and colors in the correct order
    labels = nx.get_node_attributes(G, 'label')
    colors = [nx.get_node_attributes(G, 'color').get(node, 'gray') for node in G.nodes()]

    # Calculate a hierarchical layout (Top-Down)
    plt.figure(figsize=(16, 10))
    
    try:
        # Use PyGraphviz or standard spring layout if hierarchy fails
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except ImportError:
        print("[Visualizer] Graphviz not found, falling back to planar layout...")
        pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw the tree
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, 
            node_size=3000, font_size=8, font_weight="bold", edge_color="gray", 
            arrows=True, arrowsize=20)
    
    plt.title("AIGVS Macro-Community Architecture", fontsize=16)
    plt.savefig(filename, format="PNG", dpi=300, bbox_inches="tight")
    print(f"[Visualizer] Saved beautifully colored tree to {filename}!")
    plt.close()