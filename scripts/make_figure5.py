# -*- coding: utf-8 -*-
"""Stack the density/margin panels vertically so the figure fits one column.

Side by side the panel needed the full text width, which kept it in the
full-width float queue and delayed it by two pages past its citation.
"""
import io
p = r'E:\PythonProject4\scripts\plot_supp_figs.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace('fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))',
              'fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 5.0))  # one column, stacked')
s = s.replace('ax2.set_title("(b) Pruning margin"); ax2.legend(fontsize=8.5, loc="center right", framealpha=0.9)',
              'ax2.set_title("(b) Pruning margin")\n'
              'ax1.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.13),\n'
              '           ncol=2, frameon=False, columnspacing=1.0, handletextpad=0.4)')
s = s.replace('ax1.set_title("(a) Effective mask density"); ax1.set_ylim(0, 0.55)',
              'ax1.set_title("(a) Effective mask density", fontsize=10); ax1.set_ylim(0, 0.55)')
s = s.replace('ax2.set_title("(b) Pruning margin")',
              'ax2.set_title("(b) Pruning margin", fontsize=10)')
io.open(p, 'w', encoding='utf-8').write(s)
print('plot_supp_figs.py updated: density/margin stacked vertically')
