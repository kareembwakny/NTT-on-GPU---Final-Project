import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

NAVY='#122b54'; TEAL='#127a7a'; BLUE='#2e6cb5'; GOLD='#c79a2e'
GREY='#8a9099'; DARK='#222428'
fig,ax=plt.subplots(figsize=(12.6,7.0),dpi=200)
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')

def box(x,y,w,h,fc,ec='none',lw=0):
    ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw,zorder=2,joinstyle='round'))
def txt(x,y,s,size,color=DARK,w='normal',ha='center',va='center',it=False):
    ax.text(x,y,s,fontsize=size,color=color,fontweight=w,ha=ha,va=va,
            style='italic' if it else 'normal',zorder=5)
def arrow(x,y0,y1):
    ax.add_patch(FancyArrowPatch((x,y0),(x,y1),arrowstyle='<->',mutation_scale=12,color=GREY,lw=1.7,zorder=3))

# ---- chip outer boundary ----
box(1.5,3,97,92,'#f3f5f8',ec=NAVY,lw=2.4)
txt(50,91.2,'NVIDIA RTX 2080 Ti    ·    Turing TU102    ·    sm_75    ·    q = 998244353',13.5,NAVY,'bold')

# ---- DRAM ----
box(5,80,90,6.6,NAVY)
txt(50,83.3,'GDDR6 DRAM    ·    11 GB    ·    352-bit    ·    616 GB/s peak  (517 GB/s achieved)',12.5,'white','bold')
txt(50,77.7,'▶  Multi-block 4-step saturates this — reaches 86% of the achieved roofline',10.5,TEAL,'bold')
arrow(30,80,75.1); arrow(70,80,75.1)

# ---- L2 ----
box(5,68.6,90,6.0,'#33507e')
txt(50,71.6,'L2 cache    ·    5.5 MB    (shared by all 68 SMs)',12.5,'white','bold')
arrow(26,68.6,64.4); arrow(50,68.6,64.4); arrow(74,68.6,64.4)

# ============ LEFT: SM array ============
LX,LY,LW,LH=5,6.5,40,57.5
box(LX,LY,LW,LH,'#e7ecf3',ec='#c2cbd8',lw=1.4)
txt(LX+LW/2,LY+LH-2.8,'68 Streaming Multiprocessors',12.5,NAVY,'bold')
txt(LX+LW/2,LY+LH-5.4,'one transform spread across all 68  (multi-block 4-step)',9.6,GREY,it=True)
cols,rows=6,5; gx0=LX+2.6; gy0=LY+3.0; gw=(LW-5.2)/cols; gh=(LH-13.0)/rows
tiles=[]
for r in range(rows):
    for c in range(cols):
        tx=gx0+c*gw+0.6; ty=gy0+r*gh+0.6
        box(tx,ty,gw-1.2,gh-1.2,TEAL)
        tiles.append((tx,ty,gw-1.2,gh-1.2))
txt(LX+LW/2,LY+1.9,'SM 0    ·    SM 1    ·    …    ·    SM 67',8.8,GREY)

# ============ RIGHT: zoomed SM ============
ZX,ZY,ZW,ZH=52,6.5,43,57.5
box(ZX,ZY,ZW,ZH,'white',ec=NAVY,lw=2.0)
txt(ZX+ZW/2,ZY+ZH-2.8,'Inside one SM  —  and how each kernel uses it',12.0,NAVY,'bold')

# magnifier lines from top-right tile to zoomed box
src=tiles[cols-1]
ax.add_patch(FancyArrowPatch((src[0]+src[2],src[1]+src[3]),(ZX,ZY+ZH-6),arrowstyle='-',color=GREY,lw=1.1,ls=(0,(5,4)),zorder=1))
ax.add_patch(FancyArrowPatch((src[0]+src[2],src[1]),(ZX,ZY+4.5),arrowstyle='-',color=GREY,lw=1.1,ls=(0,(5,4)),zorder=1))

res=[
 ('Register file   ·   64 K × 32-bit', 'Warp-shuffle + Shoup: 5 stages held in registers via __shfl_xor', BLUE, True),
 ('Shared memory / L1   ·   96 KB unified  (≤64 KB shared)', 'Padded + radix-8: bank-conflict-free, barriers 13 → 5', TEAL, True),
 ('4 × warp schedulers   ·   dispatch', 'block size tuned for 4 blocks/SM → hides barrier cost', NAVY, True),
 ('INT32 datapath   ·   64 cores', 'all kernels: Shoup / Montgomery modular multiply', GOLD, True),
 ('2nd-gen Tensor Cores   ·   INT8 / INT4 IMMA', 'IDLE — 287 TOPS unused → future-work path', GREY, False),
]
n=len(res); pad=1.5; avail=ZH-8.4; bh=(avail-(n-1)*pad)/n; bx=ZX+2.2; bw=ZW-4.4
by=ZY+ZH-6.8-bh
for lab,tag,col,used in res:
    if used:
        box(bx,by,bw,bh,col)
        txt(bx+1.8,by+bh*0.66,lab,10.2,'white','bold',ha='left')
        tagcol='#3a2f10' if col==GOLD else '#eef4f4'
        txt(bx+1.8,by+bh*0.30,tag,8.6,tagcol,ha='left',it=True)
    else:
        box(bx,by,bw,bh,'#eef0f3',ec=GREY,lw=1.5)
        txt(bx+1.8,by+bh*0.66,lab,10.2,GREY,'bold',ha='left')
        txt(bx+1.8,by+bh*0.30,tag,8.6,GREY,ha='left',it=True)
    by-=bh+pad

plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
plt.savefig('gpu_interior.png',dpi=200,bbox_inches='tight',facecolor='white',pad_inches=0.06)
from PIL import Image
im=Image.open('gpu_interior.png'); print('saved',im.width,'x',im.height,'ratio %.3f'%(im.width/im.height))
