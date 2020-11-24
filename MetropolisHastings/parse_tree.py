import random
import codecs

class Node:
    def __init__(self, node_id, value, is_terminal, parent):
        self.node_id = node_id
        self.value = value
        self.children = []
        self.is_terminal = is_terminal
        self.parent = parent

class ParseTree:
    """ Parse Tree implementation given the self.dsl given. """
    
    def __init__(self, dsl, max_nodes):
        """
        - dsl is the domain specific language used to create this parse tree.
        - max_nodes is a relaxed max number of nodes this tree can hold.
        - current_id is an auxiliary field used for id'ing the nodes. 
        """

        self.dsl = dsl
        self.root = Node(
                        node_id = 0, 
                        value = self.dsl.start, 
                        is_terminal = False, 
                        parent = ''
                        )
        self.max_nodes = max_nodes
        self.current_id = 0

    def build_glenn_tree(self, is_yes_no):
        if is_yes_no:
            child_1 = Node(node_id = 1, value = r"\t", is_terminal = True, parent = self.root.value)
            child_2 = Node(node_id = 2, value = r"\t", is_terminal = True, parent = self.root.value)
            child_header = Node(node_id = 3, value = r"header", is_terminal = False, parent = self.root.value)

            # First weights
            child_4 = Node(node_id = 4, value = r"progress_value=[0,0,", is_terminal = True, parent = child_header.value)
            child_5 = Node(node_id = 5, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_6 = Node(node_id = 6, value = r"7", is_terminal = True, parent = child_5.value)
            child_5.children.append(child_6)
            child_7 = Node(node_id = 7, value = r",", is_terminal = True, parent = child_header.value)

            child_8 = Node(node_id = 8, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_9 = Node(node_id = 9, value = r"7", is_terminal = True, parent = child_8.value)
            child_8.children.append(child_9)
            child_10 = Node(node_id = 10, value = r",", is_terminal = True, parent = child_header.value)

            child_11 = Node(node_id = 11, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_12 = Node(node_id = 12, value = r"3", is_terminal = True, parent = child_11.value)
            child_11.children.append(child_12)
            child_13 = Node(node_id = 13, value = r",", is_terminal = True, parent = child_header.value)

            child_14 = Node(node_id = 14, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_15 = Node(node_id = 15, value = r"2", is_terminal = True, parent = child_14.value)
            child_14.children.append(child_15)
            child_16 = Node(node_id = 16, value = r",", is_terminal = True, parent = child_header.value)

            child_17 = Node(node_id = 17, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_18 = Node(node_id = 18, value = r"2", is_terminal = True, parent = child_17.value)
            child_17.children.append(child_18)
            child_19 = Node(node_id = 19, value = r",", is_terminal = True, parent = child_header.value)

            child_20 = Node(node_id = 20, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_21 = Node(node_id = 21, value = r"1", is_terminal = True, parent = child_20.value)
            child_20.children.append(child_21)
            child_22 = Node(node_id = 22, value = r",", is_terminal = True, parent = child_header.value)

            child_23 = Node(node_id = 23, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_24 = Node(node_id = 24, value = r"2", is_terminal = True, parent = child_23.value)
            child_23.children.append(child_24)
            child_25 = Node(node_id = 25, value = r",", is_terminal = True, parent = child_header.value)

            child_26 = Node(node_id = 26, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_27 = Node(node_id = 27, value = r"2", is_terminal = True, parent = child_26.value)
            child_26.children.append(child_27)
            child_28 = Node(node_id = 28, value = r",", is_terminal = True, parent = child_header.value)

            child_29 = Node(node_id = 29, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_30 = Node(node_id = 30, value = r"3", is_terminal = True, parent = child_29.value)
            child_29.children.append(child_30)
            child_31 = Node(node_id = 31, value = r",", is_terminal = True, parent = child_header.value)

            child_32 = Node(node_id = 32, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_33 = Node(node_id = 33, value = r"7", is_terminal = True, parent = child_32.value)
            child_32.children.append(child_33)
            child_34 = Node(node_id = 34, value = r",", is_terminal = True, parent = child_header.value)

            child_35 = Node(node_id = 35, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_36 = Node(node_id = 36, value = r"7", is_terminal = True, parent = child_35.value)
            child_35.children.append(child_36)
            child_37 = Node(node_id = 37, value = r"]\n\t\tmove_value=[0,0,", is_terminal = True, parent = child_header.value)
            # Second weights
            child_38 = Node(node_id = 38, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_39 = Node(node_id = 39, value = r"7", is_terminal = True, parent = child_38.value)
            child_38.children.append(child_39)
            child_40 = Node(node_id = 40, value = r",", is_terminal = True, parent = child_header.value)

            child_41 = Node(node_id = 41, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_42 = Node(node_id = 42, value = r"0", is_terminal = True, parent = child_41.value)
            child_41.children.append(child_42)
            child_43 = Node(node_id = 43, value = r",", is_terminal = True, parent = child_header.value)

            child_44 = Node(node_id = 44, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_45 = Node(node_id = 45, value = r"2", is_terminal = True, parent = child_44.value)
            child_44.children.append(child_45)
            child_46 = Node(node_id = 46, value = r",", is_terminal = True, parent = child_header.value)

            child_47 = Node(node_id = 47, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_48 = Node(node_id = 48, value = r"0", is_terminal = True, parent = child_47.value)
            child_47.children.append(child_48)
            child_49 = Node(node_id = 49, value = r",", is_terminal = True, parent = child_header.value)

            child_50 = Node(node_id = 50, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_51 = Node(node_id = 51, value = r"4", is_terminal = True, parent = child_50.value)
            child_50.children.append(child_51)
            child_52 = Node(node_id = 52, value = r",", is_terminal = True, parent = child_header.value)

            child_53 = Node(node_id = 53, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_54 = Node(node_id = 54, value = r"3", is_terminal = True, parent = child_53.value)
            child_53.children.append(child_54)
            child_55 = Node(node_id = 55, value = r",", is_terminal = True, parent = child_header.value)

            child_56 = Node(node_id = 56, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_57 = Node(node_id = 57, value = r"4", is_terminal = True, parent = child_56.value)
            child_56.children.append(child_57)
            child_58 = Node(node_id = 58, value = r",", is_terminal = True, parent = child_header.value)

            child_59 = Node(node_id = 59, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_60 = Node(node_id = 60, value = r"0", is_terminal = True, parent = child_59.value)
            child_59.children.append(child_60)
            child_61 = Node(node_id = 61, value = r",", is_terminal = True, parent = child_header.value)

            child_62 = Node(node_id = 62, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_63 = Node(node_id = 63, value = r"2", is_terminal = True, parent = child_62.value)
            child_62.children.append(child_63)
            child_64 = Node(node_id = 64, value = r",", is_terminal = True, parent = child_header.value)

            child_65 = Node(node_id = 65, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_66 = Node(node_id = 66, value = r"0", is_terminal = True, parent = child_65.value)
            child_65.children.append(child_66)
            child_67 = Node(node_id = 67, value = r",", is_terminal = True, parent = child_header.value)

            child_68 = Node(node_id = 68, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_69 = Node(node_id = 69, value = r"7", is_terminal = True, parent = child_68.value)
            child_68.children.append(child_69)
            # Remaining of the header
            child_70 = Node(node_id = 70, value = r"]\n\t\todds=", is_terminal = True, parent = child_header.value)
            child_71 = Node(node_id = 71, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_72 = Node(node_id = 72, value = r"7", is_terminal = True, parent = child_71.value)
            child_71.children.append(child_72)

            child_73 = Node(node_id = 73, value = r"\n\t\tevens=", is_terminal = True, parent = child_header.value)
            child_74 = Node(node_id = 74, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_75 = Node(node_id = 75, value = r"1", is_terminal = True, parent = child_74.value)
            child_74.children.append(child_75)

            child_76 = Node(node_id = 76, value = r"\n\t\thighs=", is_terminal = True, parent = child_header.value)
            child_77 = Node(node_id = 77, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_78 = Node(node_id = 78, value = r"6", is_terminal = True, parent = child_77.value)
            child_77.children.append(child_78)

            child_79 = Node(node_id = 79, value = r"\n\t\tlows=", is_terminal = True, parent = child_header.value)
            child_80 = Node(node_id = 80, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_81 = Node(node_id = 81, value = r"5", is_terminal = True, parent = child_80.value)
            child_80.children.append(child_81)

            child_82 = Node(node_id = 82, value = r"\n\t\tmarker=", is_terminal = True, parent = child_header.value)
            child_83 = Node(node_id = 83, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_84 = Node(node_id = 84, value = r"6", is_terminal = True, parent = child_83.value)
            child_83.children.append(child_84)

            child_85 = Node(node_id = 85, value = r"\n\t\tthreshold=", is_terminal = True, parent = child_header.value)
            child_86 = Node(node_id = 86, value = r"INTEGERS", is_terminal = False, parent = child_header.value)
            child_87 = Node(node_id = 87, value = r"29", is_terminal = True, parent = child_86.value)
            child_86.children.append(child_87)

            child_header.children.extend((child_4, child_5, child_7, child_8, child_10, child_11, child_13, 
                child_14, child_16, child_17, child_19, child_20, child_22, child_23, child_25, child_26, child_28, 
                child_29, child_31, child_32, child_34, child_35, child_37, child_38, child_40, child_41, child_43, 
                child_44, child_46, child_47, child_49, child_50, child_52, child_53, child_55, child_56, child_58, 
                child_59, child_61, child_62, child_64, child_65, child_67, child_68, child_70, child_71, child_73, 
                child_74, child_76, child_77, child_79, child_80, child_82, child_83, child_85, child_86))
            # interlude
            child_88 = Node(node_id = 88, value = r"\n", is_terminal = True, parent = self.root.value)
            child_89 = Node(node_id = 89, value = r"\t", is_terminal = True, parent = self.root.value)
            child_90 = Node(node_id = 90, value = r"\t", is_terminal = True, parent = self.root.value)
            #body if
            child_if = Node(node_id = 91, value = r"body_if", is_terminal = False, parent = self.root.value)
            child_92 = Node(node_id = 92, value = r"if", is_terminal = True, parent = child_if.value)
            child_93 = Node(node_id = 93, value = r"actions[0]", is_terminal = True, parent = child_if.value)
            child_94 = Node(node_id = 94, value = r"in", is_terminal = True, parent = child_if.value)
            child_95 = Node(node_id = 95, value = r"['y','n']:", is_terminal = True, parent = child_if.value)
            child_96 = Node(node_id = 96, value = r"\n\t\t\tif", is_terminal = True, parent = child_if.value)
            child_97 = Node(node_id = 97, value = r"bool_template", is_terminal = False, parent = child_if.value)
            child_98 = Node(node_id = 98, value = r"bool_exp", is_terminal = False, parent = child_97.value)
            child_99 = Node(node_id = 99, value = r"functions_bool", is_terminal = False, parent = child_98.value)
            child_100 = Node(node_id = 100, value = r"ExperimentalDSL.are_there_available_columns_to_play(state)", is_terminal = True, parent = child_99.value)
            child_101 = Node(node_id = 101, value = r"BOOL_OP", is_terminal = False, parent = child_97.value)
            child_102 = Node(node_id = 102, value = r"and not", is_terminal = True, parent = child_101.value)
            child_103 = Node(node_id = 103, value = r"bool_template", is_terminal = False, parent = child_97.value)
            child_104 = Node(node_id = 104, value = r"bool_exp", is_terminal = False, parent = child_103.value)
            child_105 = Node(node_id = 105, value = r"ExperimentalDSL.will_player_win_after_n(state)", is_terminal = True, parent = child_104.value)
            child_106 = Node(node_id = 106, value = r"BOOL_OP", is_terminal = False, parent = child_103.value)
            child_107 = Node(node_id = 107, value = r"or", is_terminal = True, parent = child_106.value)
            child_108 = Node(node_id = 108, value = r"bool_exp", is_terminal = False, parent = child_103.value)
            child_109 = Node(node_id = 109, value = r"functions_yes_no", is_terminal = False, parent = child_108.value)
            child_110 = Node(node_id = 110, value = r"ExperimentalDSL.calculate_score(state,", is_terminal = True, parent = child_109.value)
            child_111 = Node(node_id = 111, value = r"vector", is_terminal = False, parent = child_109.value)
            child_112 = Node(node_id = 112, value = r"progress_value", is_terminal = True, parent = child_111.value)
            child_113 = Node(node_id = 113, value = r")", is_terminal = True, parent = child_109.value)
            child_114 = Node(node_id = 114, value = r"OP", is_terminal = False, parent = child_108.value)
            child_115 = Node(node_id = 115, value = r"+", is_terminal = True, parent = child_114.value)
            child_116 = Node(node_id = 116, value = r"functions_yes_no", is_terminal = False, parent = child_108.value)
            child_117 = Node(node_id = 117, value = r"ExperimentalDSL.calculate_difficulty_score(state, odds, evens, highs, lows)", is_terminal = True, parent = child_116.value)
            child_118 = Node(node_id = 118, value = r"COMP_OP", is_terminal = False, parent = child_108.value)
            child_119 = Node(node_id = 119, value = r"<", is_terminal = True, parent = child_118.value)
            child_120 = Node(node_id = 120, value = r"variable", is_terminal = False, parent = child_108.value)
            child_121 = Node(node_id = 121, value = r"threshold", is_terminal = True, parent = child_120.value)
            child_122 = Node(node_id = 122, value = r":\n\t\t\t\treturn", is_terminal = True, parent = child_if.value)
            child_123 = Node(node_id = 123, value = r"'y'\n\t\t\telse:\n\t\t\t\treturn", is_terminal = True, parent = child_if.value)
            child_124 = Node(node_id = 124, value = r"'n'", is_terminal = True, parent = child_if.value)

            child_97.children.extend((child_98, child_101, child_103))
            child_98.children.extend((child_99,))
            child_99.children.extend((child_100,))
            child_101.children.extend((child_102,))
            child_103.children.extend((child_104, child_106, child_108))
            child_104.children.extend((child_105,))
            child_106.children.extend((child_107,))
            child_108.children.extend((child_109, child_114, child_116, child_118, child_120))
            child_109.children.extend((child_110,child_111, child_113))
            child_111.children.extend((child_112,))
            child_114.children.extend((child_115,))
            child_116.children.extend((child_117,))
            child_118.children.extend((child_119,))
            child_120.children.extend((child_121,))
            child_if.children.extend((child_92, child_93, child_94, child_95, child_96, child_97, child_122, child_123, child_124))
            self.root.children.extend((child_1, child_2, child_header, child_88, child_89, child_90, child_if))

            self.current_id = 124
        else:
            child_1 = Node(node_id = 1, value = r"\n", is_terminal = True, parent = self.root.value)
            child_2 = Node(node_id = 2, value = r"\t", is_terminal = True, parent = self.root.value)
            child_3 = Node(node_id = 3, value = r"\t", is_terminal = True, parent = self.root.value)
            child_else = Node(node_id = 4, value = r"body_else", is_terminal = False, parent = self.root.value)
            child_5 = Node(node_id = 5, value = r"else:\n\t\t\tscores=np.zeros(len(actions))\n\t\t\tfor", is_terminal = True, parent = child_else.value)
            child_6 = Node(node_id = 6, value = r"i", is_terminal = True, parent = child_else.value)
            child_7 = Node(node_id = 7, value = r"in", is_terminal = True, parent = child_else.value)
            child_8 = Node(node_id = 8, value = r"range(len(scores)):\n\t\t\t\tfor", is_terminal = True, parent = child_else.value)
            child_9 = Node(node_id = 9, value = r"column", is_terminal = True, parent = child_else.value)
            child_10 = Node(node_id = 10, value = r"in", is_terminal = True, parent = child_else.value)
            child_11 = Node(node_id = 11, value = r"actions[i]:\n\t\t\t\t\tscores[i]+=", is_terminal = True, parent = child_else.value)
            child_12 = Node(node_id = 12, value = r"expression_col", is_terminal = False, parent = child_else.value)
            child_13 = Node(node_id = 13, value = r"expression_col", is_terminal = False, parent = child_12.value)
            child_14 = Node(node_id = 14, value = r"expression_col", is_terminal = False, parent = child_13.value)
            child_15 = Node(node_id = 15, value = r"variable_col", is_terminal = False, parent = child_14.value)
            child_18 = Node(node_id = 18, value = r"move_value[column]", is_terminal = True, parent = child_15.value)
            child_16 = Node(node_id = 16, value = r"OP", is_terminal = False, parent = child_14.value)
            child_19 = Node(node_id = 19, value = r"*", is_terminal = True, parent = child_16.value)
            child_17 = Node(node_id = 17, value = r"functions_col", is_terminal = False, parent = child_14.value)
            child_20 = Node(node_id = 20, value = r"ExperimentalDSL.advance(actions[i])", is_terminal = True, parent = child_17.value)
            child_21 = Node(node_id = 21, value = r"OP", is_terminal = False, parent = child_13.value)
            child_23 = Node(node_id = 23, value = r"-", is_terminal = True, parent = child_21.value)
            child_22 = Node(node_id = 22, value = r"variable_col", is_terminal = False, parent = child_13.value)
            child_24 = Node(node_id = 24, value = r"marker", is_terminal = True, parent = child_22.value)
            child_25 = Node(node_id = 25, value = r"OP", is_terminal = False, parent = child_12.value)
            child_27 = Node(node_id = 27, value = r"*", is_terminal = True, parent = child_25.value)
            child_26 = Node(node_id = 26, value = r"functions_col", is_terminal = False, parent = child_12.value)
            child_28 = Node(node_id = 28, value = r"ExperimentalDSL.is_new_neutral(column, state)", is_terminal = True, parent = child_26.value)
            child_29 = Node(node_id = 29, value = r"\n\t\t\tchosen_action=actions[np.argmax(scores)]", is_terminal = True, parent = child_else.value)
            child_30 = Node(node_id = 30, value = r"\n\t\t\treturn", is_terminal = True, parent = child_else.value)
            child_31 = Node(node_id = 31, value = r"chosen_action", is_terminal = True, parent = child_else.value)

            child_12.children.extend((child_13, child_25, child_26))
            child_13.children.extend((child_14, child_21, child_22))
            child_14.children.extend((child_15, child_16, child_17))
            child_15.children.extend((child_18,))
            child_16.children.extend((child_19,))
            child_17.children.extend((child_20,))
            child_21.children.extend((child_23,))
            child_22.children.extend((child_24,))
            child_25.children.extend((child_27,))
            child_26.children.extend((child_28,))
            child_else.children.extend((child_5, child_6, child_7, child_8, child_9, child_10, child_11, child_12, child_29, child_30, child_31))
            self.root.children.extend((child_1, child_2, child_3, child_else))

            self.current_id = 31


    def build_tree(self, start_node):
        """ Build the parse tree according to the self.dsl rules. """

        if self.current_id > self.max_nodes:
            self._finish_tree(self.root)
            return
        else:
            self._expand_children(
                                start_node, 
                                self.dsl._grammar[start_node.value]
                                )

            # It does not expand in order, so the tree is able to grow not only
            # in the children's ordering
            index_to_expand = [i for i in range(len(start_node.children))]
            random.shuffle(index_to_expand)

            for i in range(len(index_to_expand)):
                if not start_node.children[i].is_terminal:
                    self.build_tree(start_node.children[i])

    def _expand_children(self, parent_node, dsl_entry):
        """
        Expand the children of 'parent_node'. Since it is a parse tree, in 
        this implementation we choose randomly which child is chosen for a 
        given node.
        """

        dsl_children_chosen = random.choice(dsl_entry)
        children = self._tokenize_dsl_entry(dsl_children_chosen)
        for child in children:
            is_terminal = self._is_terminal(child)
            node_id = self.current_id + 1
            self.current_id += 1
            child_node = Node(
                            node_id = node_id, 
                            value = child, 
                            is_terminal = is_terminal, 
                            parent = parent_node.value
                            )
            parent_node.children.append(child_node)

    def _expand_children_finish_tree(self, parent_node):
        """
        "Quickly" expands the children of 'parent_node'. Avoids expanding nodes
        that are recursive. i.e.: A -> A + B 
        """

        dsl_children_chosen = random.choice(self.dsl.quickly_finish[parent_node.value])
        children = self._tokenize_dsl_entry(dsl_children_chosen)
        for child in children:
            is_terminal = self._is_terminal(child)
            node_id = self.current_id + 1
            self.current_id += 1
            child_node = Node(
                            node_id = node_id, 
                            value = child, 
                            is_terminal = is_terminal, 
                            parent = parent_node.value
                            )
            parent_node.children.append(child_node)

    def _finish_tree(self, start_node):
        """ 
        Finish expanding nodes that possibly didn't finish fully expanding.
        This can happen if the max number of tree nodes is reached.
        """

        tokens = self._tokenize_dsl_entry(start_node.value)
        is_node_finished = True
        finishable_nodes = self.dsl.finishable_nodes
        for token in tokens:
            if token in finishable_nodes and len(start_node.children) == 0:
                is_node_finished = False
                break
        if not is_node_finished:
            self._expand_children_finish_tree(start_node)
        for child_node in start_node.children:
            self._finish_tree(child_node)

    def _is_terminal(self, dsl_entry):
        """ 
        Check if the DSL entry is a terminal one. That is, check if the entry
        has no way of expanding.
        """

        tokens = self._tokenize_dsl_entry(dsl_entry)
        for token in tokens:
            if token in self.dsl._grammar:
                return False
        return True

    def _tokenize_dsl_entry(self, dsl_entry):
        """ Return the DSL value by spaces to generate the children. """

        return dsl_entry.split()

    def generate_program(self):
        """ Return a program given by the tree in a string format. """

        list_of_nodes = []
        self._get_traversal_list_of_nodes(self.root, list_of_nodes)
        whole_program = ''
        for node in list_of_nodes:
            # Since what is read from the tree is a string and not a character,
            # We have to convert the newline and tab special characters into 
            # strings.
            # This piece of code is needed for indentation purposes.
            newline = '\\' + 'n'
            tab = '\\' + 't'
            if node[0].value in [newline, tab]:
                whole_program += node[0].value
            else:
                whole_program += node[0].value + ' '
                    
        # Transform '\n' and '\t' into actual new lines and tabs
        whole_program = codecs.decode(whole_program, 'unicode_escape')

        return whole_program

    def _get_traversal_list_of_nodes(self, node, list_of_nodes):
        """
        Add to list_of_nodes the program given by the tree when traversing
        the tree in a preorder manner.
        """

        for child in node.children:
            # Only the values from terminal nodes are relevant for the program
            # synthesis (using a parse tree)
            if child.is_terminal:
                list_of_nodes.append((child, child.parent))
            self._get_traversal_list_of_nodes(child, list_of_nodes)

    def mutate_tree(self):
        """ Mutate a single node of the tree. """

        while True:
            index_node = random.randint(0, self.current_id - 1)
            #index_node = 0
            node_to_mutate = self.find_node(self.root, index_node)
            if node_to_mutate == None:
                self.print_tree(self.root, '  ')
                raise Exception('Node randomly selected does not exist.' + \
                    ' Index sampled = ' + str(index_node) + \
                    ' Tree is printed above.')
            if not node_to_mutate.is_terminal:
                break

        
        
        # delete its children and mutate it with new values
        node_to_mutate.children = []

        # Update the nodes ids. This is needed because when mutating a single 
        # node it is possible to create many other nodes (and not just one).
        # So a single id swap is not possible. This also prevents id "holes"
        # for the next mutation, this way every index sampled will be in the
        # tree.
        self.current_id = 0
        self._update_nodes_ids(self.root)

        # Build the tree again from this node (randomly as if it was creating
        # a whole new tree)
        self.build_tree(node_to_mutate)

        # Finish the tree with possible unfinished nodes (given the max_nodes
        # field)
        self._finish_tree(self.root)

        # Updating again after (possibly) finishing expanding possibly not
        # expanded nodes.
        self.current_id = 0
        self._update_nodes_ids(self.root)

    def find_node(self, node, index):
        """ Return the tree node with the corresponding id. """

        if node.node_id == index:
            return node
        else:
            for child_node in node.children:
                found_node = self.find_node(child_node, index)
                if found_node:
                    return found_node
        return None

    def _update_nodes_ids(self, node):
        """ Update all tree nodes' ids. Used after a tree mutation. """

        node.node_id = self.current_id
        self.current_id += 1
        for child_node in node.children:
            self._update_nodes_ids(child_node)

    def print_tree(self, node, indentation):
        """ Prints the tree in a simplistic manner. Used for debugging. """

        #For root
        if indentation == '  ':
            print(
                node.value, 
                ', id = ', node.node_id, 
                ', node parent = ', node.parent
                )
        else:
            print(
                indentation, 
                node.value, 
                ', id = ', node.node_id, 
                ', node parent = ', node.parent
                )
        for child in node.children:
            self.print_tree(child, indentation + '    ')