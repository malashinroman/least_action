import os

from graphviz import Digraph


def visualize_trajectories_double(
    trajectories_R, trajectories_F, terminal_nodes=[], output=None
):
    """
    Visualize probabilistic computational graph
    Edge strength is probability to take action
    Node is classifier index
    """

    def traj_freq(traj_list):
        # get frequency of each trajectory
        dtraj_freq = {}
        for t in traj_list:
            key = repr(t)
            new_key = key[1:-1]
            if new_key in dtraj_freq:
                dtraj_freq[new_key] += 1
            else:
                dtraj_freq[new_key] = 1
        return dtraj_freq

    def filter_rare_trajectories(dict, limit):
        # remove noise statistics
        filtered_dict = {}
        for key, value in dict.items():
            if value >= limit:
                filtered_dict[key] = value
        return filtered_dict

    trajectory_freq_R = traj_freq(trajectories_R)
    trajectory_freq_F = traj_freq(trajectories_F)

    svR = sorted(trajectory_freq_R.values())[::-1]
    svF = sorted(trajectory_freq_F.values())[::-1]
    if len(svR) > 20:
        limit = svR[20]
        trajectory_freq_R = filter_rare_trajectories(trajectory_freq_R, limit)
    if len(svF) > 20:
        limit = svF[20]
        trajectory_freq_F = filter_rare_trajectories(trajectory_freq_F, limit)

    def key2nodes_edges(key):
        nodes = []
        edges = []
        elements = key.split(", ")
        node = "s"
        prev_e = "s"
        for e in elements:
            if prev_e != e:
                new_node = node + "," + e
            else:
                new_node = node
            edges.append((node, new_node))
            nodes.append(new_node)
            node = new_node
            prev_e = e
        return nodes, edges

    def nodes_edges_value(trajectory_freq, trajectories):
        all_nodes = {"s": len(trajectories)}
        all_edges = {}
        for key, val in trajectory_freq.items():
            nodes, edges = key2nodes_edges(key)
            for i, n in enumerate(nodes):
                if n not in all_nodes:
                    all_nodes[n] = val
                else:
                    if i > 0:
                        if nodes[i - 1] == n:
                            continue
                    all_nodes[n] += val
            for edge in edges:
                if edge not in all_edges:
                    all_edges[edge] = val
                else:
                    all_edges[edge] += val
        return all_nodes, all_edges

    all_nodes_R, all_edges_R = nodes_edges_value(trajectory_freq_R, trajectories_R)
    all_nodes_F, all_edges_F = nodes_edges_value(trajectory_freq_F, trajectories_F)
    terminal_nodes.append("s")

    def nodes_edges2_double_graph(nodesR, edgesR, nodesF, edgesF, terminal_nodes, file):
        dot = Digraph(comment="finite_state_machine")
        for n in sorted(nodesR.keys()):
            if n.split(",")[-1] in terminal_nodes:
                dot.attr("node", shape="square")
            else:
                dot.attr("node", shape="circle")
            # dot.node(n, n.split(',')[-1] + '/{}'.format(nodes[n]))
            dot.node(n, n.split(",")[-1])
        for n in sorted(nodesF.keys()):
            if n.split(",")[-1] in terminal_nodes:
                dot.attr("node", shape="square")
            else:
                dot.attr("node", shape="circle")
            # dot.node(n, n.split(',')[-1] + '/{}'.format(nodes[n]))
            dot.node(n, n.split(",")[-1])

        for e, v in edgesR.items():
            # norm = nodes[e[0]]
            norm = nodesR["s"] + nodesF["s"]
            dot.edge(
                e[0], e[1], label="{:.4g}".format(float(v) / norm), color="#3399ff"
            )
        for e, v in edgesF.items():
            # norm = nodes[e[0]]
            norm = nodesR["s"] + nodesF["s"]
            dot.edge(
                e[0], e[1], label="{:.4g}".format(float(v) / norm), color="#ffcccc"
            )
        dot.save(os.path.basename(file), os.path.dirname(file))

    nodes_edges2_double_graph(
        all_nodes_R, all_edges_R, all_nodes_F, all_edges_F, terminal_nodes, output
    )


def visualize_trajectories(trajectories, terminal_nodes=[], output=None):
    trajectory_freq = {}
    for t in trajectories:
        key = repr(t)
        new_key = key[1:-1]
        if new_key in trajectory_freq:
            trajectory_freq[new_key] += 1
        else:
            trajectory_freq[new_key] = 1

    def filter_rare_trajectories(dict, limit):
        filtered_dict = {}
        for key, value in dict.items():
            if value >= limit:
                filtered_dict[key] = value
        return filtered_dict

    sv = sorted(trajectory_freq.values())[::-1]
    if len(sv) > 20:
        limit = sv[20]

        trajectory_freq = filter_rare_trajectories(trajectory_freq, limit)

    def key2nodes_edges(key):
        nodes = []
        edges = []
        elements = key.split(", ")
        node = "s"
        prev_e = "s"
        for e in elements:
            if prev_e != e:
                new_node = node + "," + e
            else:
                new_node = node
            edges.append((node, new_node))
            nodes.append(new_node)
            node = new_node
            prev_e = e

        return nodes, edges

    all_nodes = {"s": len(trajectories)}
    all_edges = {}
    for key, val in trajectory_freq.items():
        nodes, edges = key2nodes_edges(key)
        for i, n in enumerate(nodes):
            if n not in all_nodes:
                all_nodes[n] = val
            else:
                if i > 0:
                    if nodes[i - 1] == n:
                        continue
                all_nodes[n] += val

        for edge in edges:
            if edge not in all_edges:
                all_edges[edge] = val
            else:
                all_edges[edge] += val
    terminal_nodes.append("s")

    def nodes_edges2graph(nodes, edges, terminal_nodes, file):
        dot = Digraph(comment="finite_state_machine")
        for n in sorted(nodes.keys()):
            if n.split(",")[-1] in terminal_nodes:
                dot.attr("node", shape="square")
            else:
                dot.attr("node", shape="circle")
            # dot.node(n, n.split(',')[-1] + '/{}'.format(nodes[n]))
            dot.node(n, n.split(",")[-1])

        for e, v in edges.items():
            # norm = nodes[e[0]]
            norm = nodes["s"]
            dot.edge(e[0], e[1], label="{:.2g}".format(float(v) / norm))
        dot.render(file)

    nodes_edges2graph(all_nodes, all_edges, terminal_nodes, output)
