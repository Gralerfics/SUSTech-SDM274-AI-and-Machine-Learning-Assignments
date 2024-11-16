import pygraphviz as pgv


class DecisionTreePlotter:
    def __init__(self, tree, feature_names = None, label_names = None) -> None:
        self.tree = tree
        self.feature_names = feature_names
        self.label_names = label_names
        self.graph = pgv.AGraph(strict = True, directed = True)

    def _build(self, dt_node):
        if dt_node.children:
            feature_desc = self.feature_names[dt_node.feature_index]
            if self.feature_names:
                label = feature_desc['name']
            else:
                label = str(dt_node.feature_index)

            self.graph.add_node(str(id(dt_node)), label=label, shape="box")

            for feature_value, dt_child in dt_node.children.items():
                self._build(dt_child)
                feature_value_desc = feature_desc.get("value_names")
                if feature_value_desc:
                    edge_label = feature_value_desc[feature_value]
                else:
                    edge_label = str(feature_value)

                self.graph.add_edge(str(id(dt_node)), str(id(dt_child)), label=edge_label)
        else:
            if self.label_names:
                label = self.label_names[dt_node.value]
            else:
                label = str(dt_node.value)

            self.graph.add_node(str(id(dt_node)), label = label, shape = "ellipse")

    def plot(self, output_file = "decision_tree.png"):
        self._build(self.tree)
        self.graph.layout(prog = "dot")
        self.graph.draw(output_file)

