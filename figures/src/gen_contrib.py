import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
NAVY='#122b54'; TEAL='#127a7a'; GOLD='#c79a2e'; GREY='#8a9099'; DARK='#222428'
# baseline job-153 -> final job-180 win/tie/loss over 74 cells
cats=['Wins','Ties','Losses']
base=[48,4,22]; final=[70,1,3]
x=np.arange(len(cats)); w=0.38
fig,ax=plt.subplots(figsize=(8.4,4.2),dpi=200)
b1=ax.bar(x-w/2,base,w,label='Before this pass (job-153)',color=GREY,edgecolor='white')
b2=ax.bar(x+w/2,final,w,label='After optimization (job-180)',color=TEAL,edgecolor='white')
for bars in (b1,b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.8,str(int(b.get_height())),
                ha='center',va='bottom',fontsize=13,fontweight='bold',color=DARK)
ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=13)
ax.set_ylabel('Cells (of 74)',fontsize=12,color=DARK)
ax.set_ylim(0,80)
ax.set_title('Contribution vs the external reference model, across 74 (N, batch) cells',
             fontsize=13.5,color=NAVY,fontweight='bold',pad=10)
ax.legend(fontsize=11,frameon=False,loc='upper right')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.tick_params(length=0); ax.set_axisbelow(True); ax.yaxis.grid(True,color='#eceff3')
plt.tight_layout()
plt.savefig('contribution_wtl.png',dpi=200,bbox_inches='tight',facecolor='white')
from PIL import Image
print('contribution',Image.open('contribution_wtl.png').size)
