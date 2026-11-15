from pathlib import Path
import sys, json
import cv2
import numpy as np

from leaflens import run, analyze
root = Path(__file__).resolve().parent
h,w=480,640
mask=np.zeros((h,w),np.uint8)
points=np.array([[320,35],[385,83],[442,155],[465,235],[442,315],[386,385],[320,445],[254,385],[198,315],[175,235],[198,155],[255,83]])
cv2.fillPoly(mask,[points],255)
base=np.full((h,w,3),244,np.uint8)
y,x=np.mgrid[:h,:w]
shade=(8*np.sin(x/60)+8*np.cos(y/80)).astype(np.int16)
for c,v in enumerate([53,143,68]):base[:,:,c][mask>0]=np.clip(v+shade[mask>0],0,255)
cv2.line(base,(320,57),(320,420),(72,163,90),3,cv2.LINE_AA)
for yy in [120,175,230,285,340]:
    width=int(90*(1-abs(yy-230)/240))
    cv2.line(base,(320,yy+35),(320-width,yy),(72,160,87),2,cv2.LINE_AA)
    cv2.line(base,(320,yy+35),(320+width,yy),(72,160,87),2,cv2.LINE_AA)
spotted=base.copy()
for cx,cy,rx,ry,angle in [(282,135,19,15,20),(366,192,23,18,-20),(246,250,15,23,15),(350,300,26,20,15),(301,359,15,12,0)]:
    cv2.ellipse(spotted,(cx,cy),(rx,ry),angle,0,360,(37,74,112),-1,cv2.LINE_AA)
    cv2.ellipse(spotted,(cx,cy),(max(2,rx-5),max(2,ry-5)),angle,0,360,(28,49,80),-1,cv2.LINE_AA)
speckled=base.copy()
for xx,yy in [(260,150),(370,190),(240,250),(340,310),(300,355)]:
    cv2.circle(speckled,(xx,yy),1,(28,49,80),-1)
summary={}
for name,img in [('clean_leaf',base),('five_spots',spotted),('tiny_specks',speckled),('blank_background',np.full_like(base,244))]:
    p=root/'samples'/f'{name}.png'; cv2.imwrite(str(p),img)
    summary[name]=run(p,root/'results'/name)
assert summary['clean_leaf']['spot_count']==0
assert summary['five_spots']['spot_count']==5
assert summary['tiny_specks']['spot_count']==0
assert summary['blank_background']['affected_percent'] is None
for v in summary.values():
    if v['spot_pixels'] is not None: assert 0<=v['spot_pixels']<=v['leaf_pixels']
(root/'results'/'sample_results.json').write_text(json.dumps(summary,indent=2))
print('Four deterministic smoke checks passed. These are synthetic, not field accuracy tests.')
