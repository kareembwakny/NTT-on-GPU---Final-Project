import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

NAVY='#122b54'; STEEL='#33507e'; TEAL='#127a7a'; BLUE='#2e6cb5'; GOLD='#c79a2e'
GREY='#8a9099'; DARK='#222428'; LBG='#eef2f7'
BLUET='#dbe8f7'; TEALT='#d3ecec'; NAVYT='#dce3ef'
fig,ax=plt.subplots(figsize=(11.6,9.0),dpi=200)
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')

def box(x,y,w,h,fc,ec='none',lw=0):
    ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw,zorder=2,joinstyle='round'))
def txt(x,y,s,size,color=DARK,w='normal',ha='center',va='center',it=False):
    ax.text(x,y,s,fontsize=size,color=color,fontweight=w,ha=ha,va=va,
            style='italic' if it else 'normal',zorder=6)
def darrow(x0,y0,x1,y1,color=NAVY,lw=2.6,ms=17):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=ms,color=color,lw=lw,zorder=4))

# ---- INPUTS ----
txt(5,96.0,'INPUTS',12.5,GREY,'bold',ha='left')
ins=[(6,'Polynomial a(x)','N coeffs (mod q)',BLUE),(38,'Problem shape','(N, batch)',STEEL),(70,'Constants','q · twiddles',GOLD)]
for x,t1,t2,c in ins:
    box(x,90.0,24,5.0,'white',ec=c,lw=2.4)
    txt(x+12,93.0,t1,13,c,'bold'); txt(x+12,91.1,t2,11,DARK)
darrow(50,89.8,50,87.0,STEEL); txt(52,88.4,'Host → Device (H2D)',11,STEEL,'bold',ha='left')

# ---- DISPATCH ----
box(14,80.6,72,6.4,NAVY)
txt(50,84.4,'Adaptive dispatch table',15,'white','bold')
txt(50,82.0,'argmin over measured (N, batch) grid  ·  picks best kernel  ·  ≈1.25 ns',11,'#d7deea')
box(2.0,80.6,10.5,6.4,'#fbf3df',ec=GOLD,lw=2.0)
txt(7.25,84.4,'Build-time',10.5,GOLD,'bold'); txt(7.25,82.6,'bit-exact',10.5,DARK); txt(7.25,81.2,'74/74',10.5,DARK)
ax.add_patch(FancyArrowPatch((12.5,83.0),(14,83.4),arrowstyle='-|>',mutation_scale=12,color=GOLD,lw=1.6,ls=(0,(3,2)),zorder=4))
darrow(50,80.4,50,77.6); txt(52,79.0,'selects one lane',11,NAVY,'bold',ha='left')

# ---- GPU BOX ----
GX,GY,GW,GH=2.0,11.5,96,65.5
box(GX,GY,GW,GH,LBG,ec=NAVY,lw=3.0)
txt(GX+GW/2,GY+GH-2.7,'GPU KERNEL EXECUTION  —  RTX 2080 Ti · Turing sm_75 · 68 SMs',14,NAVY,'bold')

# memory column
mx=GX+2.4; mw=21.5
def vlink(xc,yt,yb): ax.add_patch(FancyArrowPatch((xc,yt),(xc,yb),arrowstyle='<|-|>',mutation_scale=15,color=GREY,lw=2.0,zorder=4))
box(mx,66.0,mw,5.2,NAVY);  txt(mx+mw/2,68.9,'GDDR6 DRAM',13,'white','bold'); txt(mx+mw/2,67.0,'616 / 517 GB/s',10.5,'#cfd8e6')
vlink(mx+mw/2,65.9,62.4)
box(mx,57.4,mw,4.8,STEEL); txt(mx+mw/2,60.2,'L2 cache',13,'white','bold'); txt(mx+mw/2,58.4,'5.5 MB',10.5,'#cfd8e6')
vlink(mx+mw/2,57.3,53.8)
say=15.0; satop=53.6; sah=satop-say
box(mx,say,mw,sah,'#dfe6ef',ec='#c2cbd8',lw=1.5)
txt(mx+mw/2,satop-2.2,'68 SMs',13,NAVY,'bold')
cc,rr=5,4; g0x=mx+2.0; g0y=say+2.6; gwt=(mw-4.0)/cc; ght=(sah-7.0)/rr
for r in range(rr):
    for c in range(cc):
        box(g0x+c*gwt+0.4,g0y+r*ght+0.4,gwt-0.8,ght-0.8,TEAL)
txt(mx+mw/2,say+1.6,'64K regs · 96KB shared · 64 INT32',9.2,GREY)
txt(mx+mw/2,GY+1.4,'coalesced / uint4 load & store',10.5,STEEL,'bold',it=True)

ax.plot([GX+26.6,GX+26.6],[GY+1.8,GY+GH-4.4],color='#c2cbd8',lw=1.6,zorder=3)

lx0=GX+28.5; lx1=GX+GW-2.2
txt(lx0,GY+GH-5.0,'On-chip dataflow of the selected kernel',11.5,DARK,'bold',ha='left',it=True)

def lane(yc,h,badge,bcol,btint,steps):
    bw=16.5
    box(lx0,yc-h/2,bw,h,btint,ec=bcol,lw=2.2)
    for i,t in enumerate(badge):
        txt(lx0+bw/2, yc+(h/2-2.6)-i*2.7, t, 12.5 if i==0 else 11.5, bcol if i==0 else DARK, 'bold')
    sx0=lx0+bw+3.2; n=len(steps); gap=2.8
    sw=(lx1-sx0-(n-1)*gap)/n; x=sx0
    for i,s in enumerate(steps):
        box(x,yc-h/2+1.0,sw,h-2.0,'white',ec=bcol,lw=1.8)
        lines=s.split('|')
        for j,ln in enumerate(lines):
            txt(x+sw/2, yc+(len(lines)-1)*1.5-j*3.0, ln, 11.5, DARK, 'bold')
        if i<n-1:
            ax.add_patch(FancyArrowPatch((x+sw,yc),(x+sw+gap,yc),arrowstyle='-|>',mutation_scale=13,color=bcol,lw=2.0,zorder=4))
        x+=sw+gap

lane(GY+GH-12.5,12.5,['N ≤ 1024','Warp-shuffle'],BLUE,BLUET,
     ['Registers','__shfl_xor|+ Shoup','Output'])
lane(GY+GH-27.0,12.5,['N = 2K–8K','Padded'],TEAL,TEALT,
     ['Padded|shared mem','radix-8|+ Montgomery','Output'])
lane(GY+GH-42.0,13.5,['large N','Multi-block'],NAVY,NAVYT,
     ['Split|N₁×N₂','Sub-NTTs|×68 SMs','Transpose|+ twiddle','Sub-NTTs','Output'])

box(lx0,GY+2.0,lx1-lx0,3.8,'#eef0f3',ec=GREY,lw=1.6)
txt((lx0+lx1)/2,GY+3.9,'Tensor Cores (INT8/INT4) — IDLE → future-work datapath',11,GREY,'bold',it=True)

darrow(50,11.3,50,8.5,STEEL); txt(52,10.0,'Device → Host (D2H)',11,STEEL,'bold',ha='left')

# ---- OUTPUTS ----
txt(5,7.4,'OUTPUTS',12.5,GREY,'bold',ha='left')
box(6,1.6,46,5.2,'white',ec=BLUE,lw=2.4)
txt(29,5.0,'Â(x) — NTT-domain coefficients',12.5,BLUE,'bold'); txt(29,2.9,'natural or bit-reversed (per-call flag)',10.5,DARK)
box(56,1.6,38,5.2,'white',ec=TEAL,lw=2.4)
txt(75,5.0,'Fused: â·b̂ → INTT',12.5,TEAL,'bold'); txt(75,2.9,'one-kernel polynomial product c(x)',10.5,DARK)

plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
plt.savefig('pipeline_arch2.png',dpi=200,bbox_inches='tight',facecolor='white',pad_inches=0.08)
from PIL import Image
im=Image.open('pipeline_arch2.png'); print('saved',im.width,'x',im.height,'ratio %.3f'%(im.width/im.height))
