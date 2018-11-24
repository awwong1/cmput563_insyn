"""Module to generate Heierarchical Hidden Markov Model structure:
- using JavaParser.py (Alex Wong)
"""
import networkx as nx
from antlr4.atn.ATNState import ATNState
from antlr4.atn.Transition import Transition

from .JavaParser import JavaParser


class StructureGenerator():
    """Take JavaParser.py (outputted by Antlr4) and make HHMM compatible structure
    - ATN is Augmented Transition Network
    """
    parser = JavaParser(None)

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.rule_to_states = dict()

        self.atn_graph = nx.DiGraph()
        self.rule_graph = nx.DiGraph()
        self.decision_graph = nx.DiGraph()

        self.structure_parsed_states = set()
        self._parse_structure()
        self._parse_decision()
        print("parsed {0}/{1} ATN states".format(
            len(self.structure_parsed_states), len(self.parser.atn.states)))
        self._verbose("atnG nodes", self.atn_graph.number_of_nodes())
        self._verbose("atnG edges", self.atn_graph.number_of_edges())
        nx.drawing.nx_pydot.write_dot(self.atn_graph, "atn_graph.dot")
        self._verbose("ruleG nodes", self.rule_graph.number_of_nodes())
        self._verbose("ruleG edges", self.rule_graph.number_of_edges())
        nx.drawing.nx_pydot.write_dot(self.rule_graph, "rule_graph.dot")
        self._verbose("decG nodes", self.decision_graph.number_of_nodes())
        self._verbose("decG edges", self.decision_graph.number_of_edges())
        nx.drawing.nx_pydot.write_dot(self.decision_graph, "decision_graph.dot")

    def _parse_structure(self):
        """iterate through all augmented transition network states, find rule and token mapping
        """
        for state_idx in range(0, len(self.parser.atn.states)):
            self._parse_state(state_idx)

    def _parse_state(self, init_state_idx):
        """called within structure parse, step through the states & state transitions
        """
        # while loop iterations (debug)
        counter = 1
        state_idx = init_state_idx
        start_state = self.parser.atn.states[state_idx]

        # keep track of the transition states we need to parse still
        state_stack = [start_state]

        while len(state_stack):
            cur_state = state_stack.pop()
            if cur_state.stateNumber in self.structure_parsed_states:
                continue

            rule_idx = cur_state.ruleIndex
            state_rule_str = " ".join((
                "S:", ATNState.serializationNames[cur_state.stateType],
                "R:", self.parser.ruleNames[rule_idx]))
            self._verbose(cur_state.stateNumber, state_rule_str,
                          cur_state.nextTokenWithinRule)
            self.structure_parsed_states.add(cur_state.stateNumber)
            rule_states = self.rule_to_states.get(rule_idx, set())
            rule_states.add(cur_state)
            self.rule_to_states[rule_idx] = rule_states
            self.atn_graph.add_node(
                cur_state, id=cur_state.stateNumber, rule_id=rule_idx, label=state_rule_str)
            self.rule_graph.add_node(
                rule_idx, label="{0}: {1}".format(rule_idx, self.parser.ruleNames[rule_idx])
            )

            for state_transition in cur_state.transitions:
                target_state = state_transition.target
                next_state_rule_str = " ".join(("S:", ATNState.serializationNames[target_state.stateType],
                                                "R:", self.parser.ruleNames[target_state.ruleIndex]))
                self.atn_graph.add_edge(
                    cur_state, target_state, trans=Transition.serializationNames[state_transition.serializationType])
                self.rule_graph.add_edge(
                    rule_idx, target_state.ruleIndex
                )
                if target_state.stateNumber in self.structure_parsed_states or target_state in state_stack:
                    self._verbose(" \t(dup)>", Transition.serializationNames[state_transition.serializationType],
                                  target_state.stateNumber,
                                  next_state_rule_str, target_state.nextTokenWithinRule)
                else:
                    self._verbose(
                        " \t>", Transition.serializationNames[state_transition.serializationType],
                        target_state.stateNumber,
                        next_state_rule_str, target_state.nextTokenWithinRule)
                    state_stack.append(target_state)
            counter += 1

    def _parse_decision(self):
        atn = self.parser.atn
        for decision_idx in range(0, len(atn.decisionToState)):
            state = atn.decisionToState[decision_idx]
            self.decision_graph.add_node(state, label="{0}:".format(state.stateNumber))
            for transition in state.transitions:
                next_state = transition.target
                self.decision_graph.add_edge(state, next_state)

    def _verbose(self, *print_args):
        if self.verbose:
            print(*print_args)
