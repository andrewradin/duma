#!/usr/bin/env python3

import sys
from builtins import range
class plot_scores:
    def __init__(self, **kwargs):
        self.named_scores = kwargs.get('named_scores',None)
        self.noi = kwargs.get('names_of_interest',None)
        self.under_color = kwargs.get('fill_color', 'blue')
        self.noi_color = kwargs.get('line_color', 'black')
    def run(self, filename):
        import matplotlib.pyplot as plt
        import numpy as np
        self.named_scores = list(reversed(sorted(self.named_scores, key = lambda x: float(x[1]))))
        names = [x[0] for x in self.named_scores]
        y = np.array([x[1] for x in self.named_scores])
        x = np.array(list(range(len(y))))
        plt.clf()
        plt.plot(x, y, marker = None, lw = 1, color = self.under_color)
        d = [0]*len(y)
        plt.fill_between(x, y, where = y >= d, color = self.under_color)
        max_drug_score = max(y)
        for n in self.noi:
            try:
                x_seg = names.index(n)
                y_seg_b = max_drug_score / 2 - max_drug_score / 15
                y_seg_t = max_drug_score / 2 + max_drug_score / 15
                plt.plot([x_seg, x_seg],
                        [y_seg_b, y_seg_t],
                        lw = 1,
                        color = self.noi_color,
                        )
            except ValueError:
                pass
        plt.ylabel('Method score')
        plt.xlabel('Drug rank')
        plt.title('Drug scores')
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse
    parser = argparse.ArgumentParser(description='Plot scores with specific scores highlighted by lines')
    args=parser.parse_args()
