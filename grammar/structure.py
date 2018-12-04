"""Module to generate Heierarchical Hidden Markov Model structure:
- using JavaParser.py (Alex Wong)
"""
import logging
import networkx as nx
import numpy as np
from antlr4.atn.ATNState import ATNState
from antlr4.atn.Transition import Transition

from analyze.parser import SourceCodeParser
from .JavaParser import JavaParser

EPSILON = 1e-7

class StructureBuilder():
    """Take JavaParser.py (outputted by Antlr4) and make HHMM/HMM compatible structure
    - ATN is Augmented Transition Network
    """
    logger = logging.getLogger(__name__)
    parser = JavaParser(None)

    def __init__(self):
        self.atn_graph = nx.DiGraph()
        self.rule_graph = nx.DiGraph()

        self.state_token_emissions = dict()
        self.rule_token_emissions = dict()

        self.structure_parsed_states = set()
        self._parse_structure()

        # state 1 always has EOF
        self.state_token_emissions[1].append(
            self.parser.symbolicNames.index("T_EOF"))

        print("parsed {0}/{1} ATN states".format(
            len(self.structure_parsed_states), len(self.parser.atn.states)))

        self.logger.info("atnG nodes %d", self.atn_graph.number_of_nodes())
        self.logger.info("atnG edges %d", self.atn_graph.number_of_edges())
        nx.drawing.nx_pydot.write_dot(self.atn_graph, "atn_graph.dot")

        self.logger.info("ruleG nodes %d", self.rule_graph.number_of_nodes())
        self.logger.info("ruleG edges %d", self.rule_graph.number_of_edges())
        nx.drawing.nx_pydot.write_dot(self.rule_graph, "rule_graph.dot")

    def _parse_structure(self):
        """iterate through all augmented transition network states, find state and token mapping
        """
        for state_idx in range(0, len(self.parser.atn.states)):
            start_state = self.parser.atn.states[state_idx]

            # keep track of the transition states we need to parse still
            state_stack = [start_state]

            while len(state_stack):
                cur_state = state_stack.pop()
                if cur_state.stateNumber in self.structure_parsed_states:
                    continue

                rule_idx = cur_state.ruleIndex
                rule_str = self.parser.ruleNames[rule_idx]
                state_rule_str = " ".join((
                    str(cur_state.stateNumber),
                    "S:", ATNState.serializationNames[cur_state.stateType],
                    "R:", rule_str))
                self.logger.debug(
                    "%d %s", cur_state.stateNumber, state_rule_str)
                self.structure_parsed_states.add(cur_state.stateNumber)
                self.atn_graph.add_node(
                    cur_state.stateNumber, label=state_rule_str)
                self.rule_graph.add_node(rule_idx, label=rule_str)

                for state_transition in cur_state.transitions:
                    target_state = state_transition.target
                    next_rule_str = self.parser.ruleNames[target_state.ruleIndex]
                    next_state_rule_str = " ".join(("S:", ATNState.serializationNames[target_state.stateType],
                                                    "R:", next_rule_str))
                    self.atn_graph.add_edge(
                        cur_state.stateNumber, target_state.stateNumber)
                    self.rule_graph.add_edge(rule_idx, target_state.ruleIndex)
                    if target_state.stateNumber in self.structure_parsed_states or target_state in state_stack:
                        self.logger.debug(" \t(dup)> %s %d %s", Transition.serializationNames[state_transition.serializationType],
                                          target_state.stateNumber,
                                          next_state_rule_str)
                    else:
                        self.logger.debug(
                            " \t> %s %d %s", Transition.serializationNames[
                                state_transition.serializationType],
                            target_state.stateNumber,
                            next_state_rule_str)
                        state_stack.append(target_state)

                # get state expected context free grammar token emissions
                cfg_token_emissions = StructureBuilder.parser.atn.getExpectedTokens(
                    cur_state.stateNumber, None)
                if cfg_token_emissions.intervals is not None:
                    state_token_emission = self.state_token_emissions.get(
                        cur_state.stateNumber, [])
                    rule_token_emission = self.rule_token_emissions.get(
                        cur_state.ruleIndex, [])
                    for interval in cfg_token_emissions.intervals:
                        for token_id in interval:
                            state_token_emission.append(token_id)
                            rule_token_emission.append(token_id)
                    self.state_token_emissions[cur_state.stateNumber] = state_token_emission
                    self.rule_token_emissions[cur_state.ruleIndex] = rule_token_emission
                else:
                    # state 1381, rule 103 is messed up
                    # raise ValueError("No token emissions for state {}".format(cur_state.stateNumber))
                    pass
        for state_id, token_emissions in self.state_token_emissions.items():
            self.logger.debug("\tSTATE %d EMITS %s", state_id, " ".join(
                list(map(lambda s: str(s), token_emissions))))
        for rule_id, token_emissions in self.rule_token_emissions.items():
            self.logger.debug("\tRULE %s EMITS %s", self.parser.ruleNames[rule_id], " ".join(
                list(map(lambda s: str(s), token_emissions))))

    @staticmethod
    def _build_hmm_matrices(graph, emissions):
                # prune emissions, remove all -1 values and get all tokenless nodes
        tokenless_states = []
        for state, tokens in emissions.items():
            pruned_tokens = [x for x in tokens if x != -1]
            emissions[state] = pruned_tokens

        for node in graph.nodes:
            pruned_tokens = emissions.get(node, [])
            if len(pruned_tokens) == 0:
                tokenless_states.append(node)

        while tokenless_states:
            tokenless_state = tokenless_states.pop()
            # remove the state from the graph, but connect the in and out edges
            in_edges = graph.in_edges(tokenless_state)
            out_edges = graph.out_edges(tokenless_state)
            for (parent, _) in in_edges:
                for (_, child) in out_edges:
                    graph.add_edge(parent, child)
            graph.remove_node(tokenless_state)

        num_nodes = graph.number_of_nodes()
        num_token_types = len(SourceCodeParser.JAVA_TOKEN_TYPE_MAP.keys())
        javac_token_map = dict()
        emission_probs = np.zeros((num_nodes, num_token_types))
        node_counter = 0
        for node in graph.nodes:
            javac_token_map[node] = javac_token_map.get(node, [0 for x in range(0, num_token_types)])
            pruned_tokens = emissions.get(node, [])
            if len(pruned_tokens) == 0:
                raise ValueError("node with no token emissions still exist!")

            for antlr_token in pruned_tokens:
                # map to Antlr symbolic name
                token_name = StructureBuilder.parser.symbolicNames[antlr_token]
                if token_name == "T_EOF":
                    token_name = "EOF"
                # map to JavaC symbolic name
                javac_id = SourceCodeParser.JAVA_TOKEN_TYPE_MAP.get(
                    token_name, -1)
                if javac_id == -1:
                    StructureBuilder.logger.debug(
                        "JavaC Invalid token found in node %d", node)
                    continue
                javac_token_map[node][javac_id] += 1
            emission_probs[node_counter] = javac_token_map[node]
            node_counter += 1
        
        # normalize the emission probs with some epsilon
        np.place(emission_probs, emission_probs == 0, [EPSILON])
        em_sums = emission_probs.sum(axis=1)
        norm_em_probs = emission_probs / em_sums[:, np.newaxis]

        # normalize the transition matrix
        trans_matrix = nx.adjacency_matrix(graph).todense()
        np.place(trans_matrix, trans_matrix == 0, [EPSILON])
        norm_trans_matrix = trans_matrix.astype(float)
        for row_idx in range(0, len(norm_trans_matrix)):
            row = norm_trans_matrix[row_idx]
            row_sum = row.sum()
            if row_sum > 0:
                norm_trans_matrix[row_idx] = row / row_sum
            else:
                StructureBuilder.logger.warning("No transitions from %d, set trans to 1", row_idx)
                # invalid to have EOF termination I guess
                norm_trans_matrix[row_idx, 0] = 1
                row = norm_trans_matrix[row_idx]
                norm_trans_matrix[row_idx] = row / row.sum()

        return norm_trans_matrix, norm_em_probs

    def build_atn_hmm_matrices(self):
        """
        transition matrix using the nodes and edges in augmented transition network
        """
        return self._build_hmm_matrices(self.atn_graph, self.state_token_emissions)

    def build_rule_hmm_matrices(self):
        """
        transition matrix using the nodes and edges in rule transition network
        """
        return self._build_hmm_matrices(self.rule_graph, self.rule_token_emissions)

