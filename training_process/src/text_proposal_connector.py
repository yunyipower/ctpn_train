import numpy as np
from other import clip_boxes
from text_proposal_graph_builder import TextProposalGraphBuilder

class TextProposalConnector:
    """
        Connect text proposals into text lines
    """
    def __init__(self):
        self.graph_builder=TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph=self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, xsides, im_size):
        tp_groups=self.group_text_proposals(text_proposals, scores, im_size)
        text_lines=np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]
            left_anchor = text_proposals[tp_indices[0]]
            right_anchor = text_proposals[tp_indices[-1]]

            w_a_left = left_anchor[2] - left_anchor[0] + 1.
            w_a_right = right_anchor[2] - right_anchor[0]+ 1.
            ctr_a_left = left_anchor[0] + w_a_left*0.5
            ctr_a_right = right_anchor[0] + w_a_right*0.5

            left_xside_pre  = xsides[tp_indices[0]][0]
            right_xside_pre = xsides[tp_indices[-1]][0]
            x0_refined = left_xside_pre * w_a_left+ctr_a_left
            x1_refined = right_xside_pre * w_a_right+ctr_a_right

            x0=np.min(text_line_boxes[:, 0])
            x1=np.max(text_line_boxes[:, 2])

            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5
            lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0_refined+offset, x1_refined-offset)
            lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0_refined+offset, x1_refined-offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score=scores[list(tp_indices)].sum()/float(len(tp_indices))

            text_lines[index, 0]=x0_refined
            text_lines[index, 1]=min(lt_y, rt_y)
            text_lines[index, 2]=x1_refined
            text_lines[index, 3]=max(lb_y, rb_y)
            text_lines[index, 4]=score

        text_lines=clip_boxes(text_lines, im_size)

        return text_lines
