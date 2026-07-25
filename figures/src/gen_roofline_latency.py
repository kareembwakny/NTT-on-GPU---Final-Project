import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch, FancyBboxPatch
NAVY='#122b54'; STEEL='#33507e'; TEAL='#127a7a'; BLUE='#2e6cb5'; GOLD='#c79a2e'; GREY='#8a9099'; DARK='#222428'

# ================= ROOFLINE =================
BW=517e9; CROOF=6.38e12; ridge=CROOF/BW
fig,ax=plt.subplots(figsize=(9.2,5.5),dpi=200)
ai=np.logspace(np.log10(1.5),np.log10(60),400)
roof=np.minimum(ai*BW,CROOF)/1e12
ax.plot(ai,roof,color=NAVY,lw=3,zorder=3)
ax.axvline(ridge,color=GREY,lw=1.1,ls='--',zorder=2)
ax.text(ridge*1.05,8.3,'ridge  12.3 ops/B',fontsize=10,color=GREY,ha='left',va='center')
# roof labels (placed in clear zones)
ax.text(2.05,1.55,'DRAM bandwidth roof\n517 GB/s (achieved)',fontsize=10.5,color=NAVY,
        rotation=29,rotation_mode='anchor',ha='left',va='bottom',fontweight='bold')
ax.text(46,CROOF/1e12*1.05,'INT32 compute roof — 6.4 TOPS',fontsize=10.5,color=NAVY,ha='right',va='bottom',fontweight='bold')
# kernel points
pts=[('N=256',7.9,446),('N=512',7.2,430),('N=1024',6.6,402),
     ('N=2048',5.8,372),('N=4096',5.1,350),('N=8192',4.5,343)]
xs=[a for _,a,_ in pts]; ys=[a*b*1e9/1e12 for _,a,b in pts]
ax.plot(xs,ys,color=TEAL,lw=1.0,ls=(0,(2,2)),zorder=4,alpha=0.6)
ax.scatter(xs,ys,s=120,color=TEAL,edgecolor='white',lw=1.5,zorder=5)
# label only endpoints
ax.annotate('N=256  (86% of roof)',(7.9,ys[0]),xytext=(9.4,ys[0]+0.9),fontsize=9.6,color=DARK,fontweight='bold',
            ha='left',arrowprops=dict(arrowstyle='-',color=GREY,lw=1))
ax.annotate('N=8192  (66% of roof)',(4.5,ys[-1]),xytext=(2.0,ys[-1]-0.32),fontsize=9.6,color=DARK,fontweight='bold',
            ha='left',arrowprops=dict(arrowstyle='-',color=GREY,lw=1))
# memory-bound callout in clear lower-right
ax.text(15.5,1.35,'All measured sizes sit at\n66–86% of the achieved roof\n→ memory-bound & saturated',
        fontsize=10.4,color=TEAL,ha='left',va='center',fontweight='bold')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1.5,60); ax.set_ylim(0.9,9)
ax.set_xticks([2,5,10,20,50]); ax.set_yticks([1,2,3,5,8])
for axis in (ax.xaxis,ax.yaxis):
    axis.set_major_formatter(FuncFormatter(lambda v,_:('%g'%v)))
    axis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel('Operational intensity  (ops / byte)',fontsize=12,color=DARK)
ax.set_ylabel('Attainable performance  (INT32 TOPS)',fontsize=12,color=DARK)
ax.set_title('Roofline — RTX 2080 Ti (Turing, sm_75)',fontsize=14.5,color=NAVY,fontweight='bold',pad=10)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.grid(True,which='major',color='#eceff3',lw=0.8,zorder=0)
plt.tight_layout()
plt.savefig('roofline_poster.png',dpi=200,bbox_inches='tight',facecolor='white')
from PIL import Image
print('roofline',Image.open('roofline_poster.png').size)

# ================= LATENCY / EXECUTION-TIME BREAKDOWN =================
final=0.514
seg=[('Compute / irreducible',0.514,NAVY),
     ('Removed: barrier & sync  (radix-4/8, block size)',0.272,TEAL),
     ('Removed: memory layout  (padding, uint2)',0.199,BLUE),
     ('Removed: arithmetic  (Shoup, Montgomery)',0.015,GOLD)]
fig,ax=plt.subplots(figsize=(9.4,4.4),dpi=200)
left=0
for lab,frac,col in seg:
    ax.barh(0,frac*100,left=left*100,color=col,edgecolor='white',height=0.42,zorder=3)
    if frac>0.05:
        ax.text((left+frac/2)*100,0,('%d%%'%round(frac*100)),ha='center',va='center',
                fontsize=14,color='white',fontweight='bold')
    left+=frac
ax.annotate('2%',(99,0),xytext=(99,0.62),fontsize=10.5,color=GOLD,ha='center',fontweight='bold',
            arrowprops=dict(arrowstyle='-',color=GOLD,lw=1.2))
# bottom bracket: optimized
ax.plot([0,0,final*100,final*100],[-0.32,-0.40,-0.40,-0.32],color=NAVY,lw=1.7)
ax.text(final*50,-0.58,'optimized kernel  —  51% of baseline time',ha='center',va='top',fontsize=11,color=NAVY,fontweight='bold')
# top bracket: removed
ax.plot([final*100,final*100,100,100],[0.32,0.40,0.40,0.32],color=DARK,lw=1.7)
ax.text((final+(1-final)/2)*100,0.60,'time removed by tuning  →  1.94× faster',ha='center',va='bottom',fontsize=11,color=DARK,fontweight='bold')
leg=[Patch(fc=c,label=l) for l,f,c in seg]
ax.legend(handles=leg,loc='upper center',bbox_to_anchor=(0.5,-0.14),ncol=2,fontsize=9.8,frameon=False)
ax.set_xlim(-1,101); ax.set_ylim(-1.15,1.0); ax.axis('off')
ax.set_title('Where the 1.94× comes from — baseline kernel time by the bottleneck each optimization removed',
             fontsize=12,color=NAVY,fontweight='bold',pad=14)
plt.tight_layout()
plt.savefig('latency_breakdown.png',dpi=200,bbox_inches='tight',facecolor='white')
print('latency',Image.open('latency_breakdown.png').size)
