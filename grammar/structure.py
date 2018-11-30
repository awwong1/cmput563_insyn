"""Module to generate Heierarchical Hidden Markov Model structure:
- using JavaParser.py (Alex Wong)
"""
import logging
import networkx as nx
import numpy as np
from antlr4.atn.ATNState import ATNState
from antlr4.atn.Transition import Transition

from .JavaParser import JavaParser


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
        print("parsed {0}/{1} ATN states".format(
            len(self.structure_parsed_states), len(self.parser.atn.states)))

        self.logger.info("atnG nodes %d", self.atn_graph.number_of_nodes())
        self.logger.info("atnG edges %d", self.atn_graph.number_of_edges())
        nx.drawing.nx_pydot.write_dot(self.atn_graph, "atn_graph.dot")
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
    def _build_trans_matrix(graph):
        n = graph.number_of_nodes()
        trans_matrix = np.zeros((n, n,))
        for idx in graph.nodes:
            node = graph.nodes[idx]
            out_edges = graph.out_edges(idx)
            StructureBuilder.logger.debug(node.get("label", idx), out_edges)
            if out_edges:
                norm_prob = 1/len(out_edges)
                for (from_node, to_node) in out_edges:
                    trans_matrix[from_node][to_node] = norm_prob
        return trans_matrix

    def build_atn_transition_matrix(self):
        """
        transition matrix using the nodes and edges in augmented transition network
        """
        return self._build_trans_matrix(self.atn_graph)

    def build_rule_transition_matrix(self):
        """
        transition matrix using the nodes and edges in rule transitions
        """
        return self._build_trans_matrix(self.rule_graph)

    def build_atn_antlr_emission_probs(self, use_avg=True):
        """
        atn state to emission probabilities
        """
        num_hidden_states = self.atn_graph.number_of_nodes()
        num_token_types = len(self.parser.symbolicNames)
        emission_probs = np.zeros((num_hidden_states, num_token_types))
        for idx in self.atn_graph.nodes:
            possible_tokens = self.state_token_emissions.get(idx, [])
            pass
