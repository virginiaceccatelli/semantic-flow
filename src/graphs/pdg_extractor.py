"""Program Dependence Graph (PDG) combining data-flow and control-dependence edges.

The PDG is the union of:
  - def-use edges (from DataFlowGraph)
  - control-dependency edges (from ControlFlowGraph.control_dependencies)

This is the minimal structure needed for taint-flow and vulnerability reasoning.
A full Code Property Graph would add the AST and call-graph edges on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .cfg_extractor import CFGExtractor, ControlFlowGraph
from .dfg_extractor import DefUseExtractor, DataFlowGraph


@dataclass
class PDGNode:
    node_id: str    # "cfg:{id}" or "dfg:{name}:{line}:{col}"
    kind: str       # "stmt" or "var"
    line: int
    col: int
    label: str = ""


class ProgramDependenceGraph:
    """Union of data-flow and control-dependence edges over Python source."""

    def __init__(self):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.dfg: DataFlowGraph | None = None
        self.cfg: ControlFlowGraph | None = None

    def taint_paths(self, source_line: int, sink_line: int) -> list[list]:
        """Return all simple paths between a taint source node and a sink node.

        Nodes are identified by line number (approximate).
        """
        source_nodes = [
            n for n in self.graph.nodes
            if isinstance(n, str) and f":{source_line}:" in n
        ]
        sink_nodes = [
            n for n in self.graph.nodes
            if isinstance(n, str) and f":{sink_line}:" in n
        ]
        paths = []
        for s in source_nodes:
            for t in sink_nodes:
                try:
                    for path in nx.all_simple_paths(self.graph, s, t, cutoff=20):
                        paths.append(path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        return paths

    def __repr__(self) -> str:
        return (
            f"ProgramDependenceGraph("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )


class PDGExtractor:
    """Build a ProgramDependenceGraph from Python source."""

    def __init__(self):
        self._cfg_extractor = CFGExtractor()
        self._dfg_extractor = DefUseExtractor()

    def extract(self, source: str) -> ProgramDependenceGraph:
        pdg = ProgramDependenceGraph()

        # Data-flow component
        dfg = self._dfg_extractor.extract(source)
        pdg.dfg = dfg
        for edge in dfg.edges:
            d, u = edge.definition, edge.use
            def_key = f"var:{d.name}:{d.line}:{d.col}"
            use_key = f"var:{u.name}:{u.line}:{u.col}"
            pdg.graph.add_node(def_key, kind="var", line=d.line, col=d.col, label=f"{d.name}(def)")
            pdg.graph.add_node(use_key, kind="var", line=u.line, col=u.col, label=f"{u.name}(use)")
            pdg.graph.add_edge(def_key, use_key, kind="data_dep")

        # Control-flow / control-dependency component
        cfg = self._cfg_extractor.extract(source)
        pdg.cfg = cfg
        cdg = self._cfg_extractor.control_dependencies(cfg)
        for u, v, data in cdg.edges(data=True):
            u_node = cfg.graph.nodes[u].get("cfg_node") if u in cfg.graph.nodes else None
            v_node = cfg.graph.nodes[v].get("cfg_node") if v in cfg.graph.nodes else None
            if u_node and v_node:
                uk = f"stmt:{u_node.line}:{u_node.col}"
                vk = f"stmt:{v_node.line}:{v_node.col}"
                pdg.graph.add_node(uk, kind="stmt", line=u_node.line, col=u_node.col,
                                   label=u_node.stmt_type)
                pdg.graph.add_node(vk, kind="stmt", line=v_node.line, col=v_node.col,
                                   label=v_node.stmt_type)
                pdg.graph.add_edge(uk, vk, kind="control_dep")

        return pdg
