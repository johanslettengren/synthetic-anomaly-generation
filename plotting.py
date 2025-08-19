import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def plot_network(wn, node_size=18, label_offset=8, with_labels=True):
    """
    Plot a WNTR WaterNetworkModel with clean styling:
    - No axes or ticks
    - Labels above nodes with white outline (skips names beginning 'LEAK')
    - LEAK nodes drawn as dark red triangles, other nodes as steel blue circles
    """
    # --- collect positions
    pos = {}
    for name in wn.node_name_list:
        node = wn.get_node(name)
        try:
            x, y = node.coordinates
        except AttributeError:
            x, y = node.x, node.y
        pos[name] = (x, y)

    # --- edges
    xlines, ylines = [], []
    for lname in wn.link_name_list:
        link = wn.get_link(lname)
        x0, y0 = pos[link.start_node_name]
        x1, y1 = pos[link.end_node_name]
        xlines += [x0, x1, None]
        ylines += [y0, y1, None]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xlines, ylines, linewidth=1.2, alpha=0.8, zorder=1, color="gray")

    # --- split nodes into regular vs leak
    regular_nodes = [n for n in wn.node_name_list if not n.startswith("LEAK")]
    leak_nodes    = [n for n in wn.node_name_list if n.startswith("LEAK")]

    # --- plot regular nodes (steel blue circles)
    xs = [pos[n][0] for n in regular_nodes]
    ys = [pos[n][1] for n in regular_nodes]
    ax.scatter(xs, ys, s=node_size**2/10, alpha=0.95, zorder=2,
               color="black", marker="o", label="Regular nodes")

    # --- plot leak nodes (dark red triangles)
    xs_leak = [pos[n][0] for n in leak_nodes]
    ys_leak = [pos[n][1] for n in leak_nodes]
    ax.scatter(xs_leak, ys_leak, s=node_size**2/10, alpha=0.95, zorder=2,
               color="gray", marker="o", label="Leak nodes")

    # --- labels (skip LEAK nodes)
    if with_labels:
        for n in regular_nodes:
            x, y = pos[n]
            ax.annotate(
                n, (x, y),
                xytext=(0, label_offset), textcoords='offset points',
                ha='center', va='bottom',
                zorder=3,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")]
            )

    # --- clean background
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Optional: add legend if you want
    # ax.legend(frameon=False)

    plt.show()
