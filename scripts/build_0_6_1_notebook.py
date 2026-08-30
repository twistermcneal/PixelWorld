import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
nb = json.loads((root / 'notebooks' / 'PixelWorld_0_6.ipynb').read_text())


def set_cell(index, text):
    nb['cells'][index]['source'] = text.splitlines(True)


set_cell(0, """# PixelWorld-0.6.1 — Terrainregionen und deterministische Vegetation

0.6.1 ersetzt absolute Objektkoordinaten durch `Terrainregion + Anchor`. Wichtige Landmarken erhalten dadurch stabile, kollisionsfreie Positionen. Normale Bäume werden nicht als knappe Object Slots vorhergesagt, sondern aus einer gelernten Waldregion und Vegetationsdichte deterministisch verteilt.
""")

set_cell(1, """# Notebook-Abhängigkeiten installieren.
%pip install -q ipympl

import hashlib, random
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED)
SIZE = 64
MAX_SLOTS = 8
SLOT_LATENT_DIM = 6
TERRAIN_LATENT_DIM = 7
LAYOUT_DIM = TERRAIN_LATENT_DIM + MAX_SLOTS*SLOT_LATENT_DIM

BIOMES = ['temperate', 'tropical', 'arid', 'tundra']
ORIENTATIONS = ['north', 'east', 'south', 'west']
TERRAINS = {'water': 0, 'sand': 1, 'grass': 2, 'dirt': 3, 'rock': 4, 'snow': 5}
REGIONS = ['beach', 'open_land', 'rock_field', 'forest']
LANDMARK_CLASSES = ['chest', 'npc', 'portal', 'ruin']
LANDMARK_SIZES = {'chest': (5,4), 'npc': (5,9), 'portal': (6,8), 'ruin': (8,8)}
ACTIONS = ['LOOK', 'USE', 'SCAN']
TRIGGER_TYPES = ['NONE', 'WORLD', 'STORY', 'SECRET']
TERRAIN_PALETTE = np.asarray([(25,85,145),(224,198,125),(58,125,65),(130,92,55),(105,105,110),(225,232,238)],np.uint8)
REGION_PALETTE = np.asarray([(235,204,126),(92,150,78),(115,110,112),(32,92,48)],np.uint8)

@dataclass
class Landscape:
    prompt: str
    seed: int
    biome: str
    terrain: np.ndarray
    regions: np.ndarray
    vegetation: np.ndarray
    rgb: np.ndarray
    object_map: np.ndarray
    walkable: np.ndarray
    interaction: np.ndarray
    terrain_params: tuple
    objects: dict

def world_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)

def transition_seed(current_seed,slot,trigger_type,story_state=0):
    return world_seed(f'{current_seed}:{slot}:{trigger_type}:{story_state}')

def layout_from_seed(seed):
    return np.random.default_rng(seed).random(LAYOUT_DIM).astype(np.float32)
""")

set_cell(2, """## 1. Terrainregionen und Scatter-Layer

Der Terrain Graph enthält zusätzlich Waldstufe und Vegetationsdichte. Eine Region Map unterscheidet Strand, offenes Land, Felsfeld und Wald. Landmark Slots sagen nur noch Region und einen von 16 kanonischen Anchors voraus.
""")

set_cell(3, """def terrain_params(prompt,seed):
    p=prompt.lower(); layout=layout_from_seed(seed)
    biome=next((b for b in BIOMES if b in p),BIOMES[int(layout[0]*4)%4])
    return (BIOMES.index(biome),int(layout[1]*4)%4,22+int(layout[2]*21),3+int(layout[3]*9),
            int(layout[4]*6),int(layout[5]*6),1+int(layout[6]*5))

def render_terrain(params,seed):
    biome_id,orientation,shoreline,beach_width,rockiness,_,_=map(int,params)
    yy,xx=np.mgrid[0:SIZE,0:SIZE]; along=xx if orientation in (0,2) else yy; across=yy if orientation in (0,2) else xx
    if orientation in (2,1): across=SIZE-1-across
    phase=(seed%10007)/10007*2*np.pi
    coast=shoreline+2.3*np.sin(along/7+phase)+1.1*np.sin(along/3.3+phase*1.7)
    terrain=np.full((SIZE,SIZE),TERRAINS['grass'],np.int64)
    terrain[across<coast]=TERRAINS['water']; terrain[(across>=coast)&(across<coast+beach_width)]=TERRAINS['sand']
    land=across>=coast+beach_width; base=['grass','grass','dirt','snow'][biome_id]; terrain[land]=TERRAINS[base]
    rock_field=np.sin(xx*.37+phase)+np.cos(yy*.29-phase)+np.sin((xx+yy)*.17)
    terrain[land&(rock_field>2.35-rockiness*.12)]=TERRAINS['rock']
    return terrain

def render_regions(terrain,params,seed):
    *_,forest_level,_=map(int,params); yy,xx=np.mgrid[0:SIZE,0:SIZE]; phase=(seed%8191)/8191*2*np.pi
    regions=np.full((SIZE,SIZE),REGIONS.index('open_land'),np.int64)
    regions[terrain==TERRAINS['sand']]=REGIONS.index('beach'); regions[terrain==TERRAINS['rock']]=REGIONS.index('rock_field')
    forest_field=np.sin(xx*.16+phase)+np.cos(yy*.14-phase*.7)+np.sin((xx-yy)*.09+phase)
    forestable=np.isin(terrain,[TERRAINS['grass'],TERRAINS['dirt'],TERRAINS['snow']])
    regions[forestable&(forest_field>1.65-forest_level*.18)]=REGIONS.index('forest')
    return regions

def scatter_vegetation(terrain,regions,params,seed):
    density=int(params[-1]); vegetation=np.zeros((SIZE,SIZE),np.uint8); rng=np.random.default_rng(world_seed(f'vegetation:{seed}'))
    ys,xs=np.where(regions==REGIONS.index('forest')); order=rng.permutation(len(xs)); accepted=[]; target=min(len(xs)//16,density*18)
    for idx in order:
        x,y=int(xs[idx]),int(ys[idx])
        if all((x-ax)**2+(y-ay)**2>=16 for ax,ay in accepted):
            accepted.append((x,y)); vegetation[y,x]=1
            if len(accepted)>=target: break
    return vegetation

def anchor_candidates(regions,terrain,region_id,w,h,seed,slot):
    # Integralbilder prüfen alle Rechtecke vektorisiert statt in Python-Schleifen.
    def window_sum(mask):
        integral=np.pad(mask.astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1)
        return integral[h:,w:]-integral[:-h,w:]-integral[h:,:-w]+integral[:-h,:-w]
    region_pixels=window_sum(regions==region_id); land_pixels=window_sum(terrain!=TERRAINS['water'])
    valid=(region_pixels>=int(np.ceil(.70*w*h)))&(land_pixels==w*h)
    valid[1::2,:]=False; valid[:,1::2]=False
    ys,xs=np.where(valid)
    if len(xs)==0: return []
    rng=np.random.default_rng(world_seed(f'anchors:{seed}:{slot}:{region_id}:{w}:{h}'))
    order=rng.permutation(len(xs)); return [(int(xs[i]),int(ys[i])) for i in order]

def resolve_anchor(regions,terrain,region_id,anchor_id,w,h,seed,slot,occupied):
    for fallback in [region_id,REGIONS.index('open_land'),REGIONS.index('beach'),REGIONS.index('forest'),REGIONS.index('rock_field')]:
        candidates=anchor_candidates(regions,terrain,fallback,w,h,seed,slot)
        if not candidates: continue
        for step in range(len(candidates)):
            x,y=candidates[(anchor_id+step)%len(candidates)]
            if np.all(occupied[y:y+h,x:x+w]==0): return x,y,fallback
    return None

def generate_landscape(prompt,seed=None):
    seed=world_seed(prompt) if seed is None else int(seed); layout=layout_from_seed(seed); params=terrain_params(prompt,seed)
    terrain=render_terrain(params,seed); regions=render_regions(terrain,params,seed); vegetation=scatter_vegetation(terrain,regions,params,seed)
    obj=np.zeros((SIZE,SIZE),np.int64); objects={}
    for slot in range(MAX_SLOTS):
        values=layout[TERRAIN_LATENT_DIM+slot*SLOT_LATENT_DIM:TERRAIN_LATENT_DIM+(slot+1)*SLOT_LATENT_DIM]
        if not (slot==0 or values[0]>.34): continue
        class_id=2 if slot==0 else int(values[1]*4)%4; kind=LANDMARK_CLASSES[class_id]; w,h=LANDMARK_SIZES[kind]
        region_id=int(values[2]*4)%4; anchor_id=int(values[3]*16)%16
        resolved=resolve_anchor(regions,terrain,region_id,anchor_id,w,h,seed,slot,obj)
        if resolved is None: continue
        x,y,resolved_region=resolved; action=ACTIONS[int(values[4]*3)%3]; trigger=TRIGGER_TYPES[int(values[5]*4)%4]; oid=slot+1
        obj[y:y+h,x:x+w]=oid; vegetation[y:y+h,x:x+w]=0
        objects[oid]={'class':kind,'bbox':[x,y,w,h],'region_id':region_id,'resolved_region_id':resolved_region,'anchor_id':anchor_id,
                      'action':action,'trigger_type':trigger,'next_seed':transition_seed(seed,slot,trigger)}
    rgb=TERRAIN_PALETTE[terrain].copy(); rgb[vegetation>0]=(25,72,38)
    walkable=np.isin(terrain,[TERRAINS['sand'],TERRAINS['grass'],TERRAINS['dirt'],TERRAINS['snow']]).astype(np.uint8); walkable[vegetation>0]=0
    return Landscape(prompt,seed,BIOMES[params[0]],terrain,regions,vegetation,rgb,obj,walkable,(obj>0).astype(np.uint8),params,objects)

sample=generate_landscape('tropical coast beach forest portal',424242)
sample.terrain_params,sample.objects,int(sample.vegetation.sum())
""")

set_cell(4, """def show_landscape(w):
    fig,ax=plt.subplots(1,6,figsize=(20,4))
    items=[(w.rgb,'RGB'),(w.terrain,'Terrain'),(REGION_PALETTE[w.regions],'Regions'),(w.vegetation,'Vegetation'),(w.object_map,'Landmarks'),(w.interaction,'Interaction')]
    for a,(data,title) in zip(ax,items): a.imshow(data,interpolation='nearest'); a.set_title(title); a.axis('off')
    plt.tight_layout(); plt.show()
show_landscape(sample)
""")

set_cell(5, """## 2. Getrennte Terrain-, Placement-, Presence- und Attribute-Pfade

Der Placement Encoder klassifiziert pro Slot eine Terrainregion und einen Anchor. Absolute Pixelkoordinaten sind nur noch ein deterministisches Renderergebnis und kein Lernziel.
""")

set_cell(6, """import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

torch.manual_seed(SEED); DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; COORD_CLASSES=SIZE+1
print('Device:',DEVICE)
if DEVICE=='cuda':
    print('GPU:',torch.cuda.get_device_name(0)); torch.set_float32_matmul_precision('high')

def prompt_vector(prompt):
    p=prompt.lower(); v=np.zeros(len(BIOMES)+6,np.float32)
    for i,b in enumerate(BIOMES): v[i]=float(b in p)
    for i,w in enumerate(['coast','beach','forest','rock','portal','dark']): v[len(BIOMES)+i]=float(w in p)
    return v

def condition_vector(prompt,seed): return np.concatenate([prompt_vector(prompt),layout_from_seed(seed)]).astype(np.float32)

def scene_targets(w):
    biome,orientation,shore,width,rock,forest,density=w.terrain_params
    regions=np.zeros(MAX_SLOTS,np.int64); anchors=np.zeros(MAX_SLOTS,np.int64); presence=np.zeros(MAX_SLOTS,np.float32)
    classes=np.zeros(MAX_SLOTS,np.int64); actions=np.zeros(MAX_SLOTS,np.int64); triggers=np.zeros(MAX_SLOTS,np.int64)
    for slot in range(MAX_SLOTS):
        oid=slot+1
        if oid in w.objects:
            m=w.objects[oid]; presence[slot]=1; regions[slot]=m['region_id']; anchors[slot]=m['anchor_id']
            classes[slot]=LANDMARK_CLASSES.index(m['class']); actions[slot]=ACTIONS.index(m['action']); triggers[slot]=TRIGGER_TYPES.index(m['trigger_type'])
    return np.asarray([shore,width,rock,forest,density]),orientation,biome,regions,anchors,presence,classes,actions,triggers

class LandscapeDataset(Dataset):
    def __init__(self,n=14000):
        # Einmal erzeugen statt 14.000 Welten in jeder der 45 Epochen neu zu bauen.
        self.samples=[]
        print(f'Erzeuge {n:,} Trainingslandschaften einmalig ...')
        for i in range(n):
            p=f'{BIOMES[i%4]} coast beach forest rock portal {i}'; seed=i+1000
            targets=scene_targets(generate_landscape(p,seed))
            self.samples.append(tuple(torch.tensor(x) for x in (condition_vector(p,seed),*targets)))
            if (i+1)%2000==0: print(f'  {i+1:,}/{n:,}')
        print('Datensatz bereit.')
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

class LandscapeNet(nn.Module):
    def __init__(self,condition_dim=len(BIOMES)+6+LAYOUT_DIM,hidden=320):
        super().__init__(); self.slots=MAX_SLOTS
        def enc(): return nn.Sequential(nn.Linear(condition_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        def dec(): return nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.terrain_encoder=enc(); self.placement_encoder=enc(); self.presence_encoder=enc(); self.attribute_encoder=enc()
        self.terrain_numeric=nn.Linear(hidden,5*COORD_CLASSES); self.orientation_head=nn.Linear(hidden,4); self.biome_head=nn.Linear(hidden,4)
        self.slot_queries=nn.Embedding(MAX_SLOTS,hidden); self.placement_decoder=dec(); self.attribute_decoder=dec()
        self.region_head=nn.Linear(hidden,4); self.anchor_head=nn.Linear(hidden,16); self.presence_head=nn.Linear(hidden,MAX_SLOTS)
        self.class_head=nn.Linear(hidden,4); self.action_head=nn.Linear(hidden,3); self.trigger_head=nn.Linear(hidden,4)
    def slots_for(self,world,decoder):
        q=self.slot_queries.weight[None].expand(world.shape[0],-1,-1); c=world[:,None,:].expand(-1,self.slots,-1)
        return decoder(torch.cat([c,q],-1))
    def forward(self,x):
        t=self.terrain_encoder(x); p=self.slots_for(self.placement_encoder(x),self.placement_decoder); a=self.slots_for(self.attribute_encoder(x),self.attribute_decoder)
        return (self.terrain_numeric(t).reshape(-1,5,COORD_CLASSES),self.orientation_head(t),self.biome_head(t),self.region_head(p),self.anchor_head(p),
                self.presence_head(self.presence_encoder(x)),self.class_head(a),self.action_head(a),self.trigger_head(a))

model=LandscapeNet().to(DEVICE); sum(p.numel() for p in model.parameters())
""")

set_cell(7, """dataset=LandscapeDataset(14000)
loader=DataLoader(dataset,batch_size=128,shuffle=True,num_workers=2,pin_memory=(DEVICE=='cuda'),persistent_workers=True)
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
coord_values=torch.arange(COORD_CLASSES,dtype=torch.float32,device=DEVICE); ce=nn.CrossEntropyLoss(reduction='none'); bce=nn.BCEWithLogitsLoss(reduction='none')

def ordinal_loss(logits,target,sigma=1.):
    d=coord_values-target[...,None].float(); soft=torch.exp(-.5*(d/sigma)**2); soft/=soft.sum(-1,keepdim=True)
    expected=(logits.softmax(-1)*coord_values).sum(-1)
    return -(soft*logits.log_softmax(-1)).sum(-1)+4*F.smooth_l1_loss(expected,target.float(),reduction='none')/SIZE

def masked_ce(logits,target,presence):
    e=ce(logits.flatten(0,1),target.flatten()).reshape_as(target); return (e*presence).sum()/presence.sum().clamp_min(1)

EPOCHS=45
for epoch in range(EPOCHS):
    model.train(); totals=np.zeros(8); batches=0
    for batch in loader:
        condition,numeric,orient,biome,regions,anchors,pres,cls,act,trig=[x.to(DEVICE,non_blocking=True) for x in batch]
        num_l,orient_l,biome_l,region_l,anchor_l,pres_l,cls_l,act_l,trig_l=model(condition)
        terrain_loss=ordinal_loss(num_l,numeric).mean()+ce(orient_l,orient).mean()+ce(biome_l,biome).mean()
        placement_loss=masked_ce(region_l,regions,pres)+masked_ce(anchor_l,anchors,pres)
        presence_loss=(bce(pres_l,pres)*torch.where(pres>.5,1.,2.)).mean()
        class_loss=masked_ce(cls_l,cls,pres); action_loss=masked_ce(act_l,act,pres); trigger_loss=masked_ce(trig_l,trig,pres)
        loss=terrain_loss+2*placement_loss+presence_loss+class_loss+action_loss+trigger_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step(); totals += [loss.item(),terrain_loss.item(),placement_loss.item(),presence_loss.item(),class_loss.item(),action_loss.item(),trigger_loss.item(),1]; batches+=1
    v=totals/max(1,batches)
    print(f'Epoch {epoch+1:02d}: loss={v[0]:.3f} terrain={v[1]:.3f} placement={v[2]:.3f} presence={v[3]:.3f} class={v[4]:.3f} action={v[5]:.3f} trigger={v[6]:.3f}')
""")

set_cell(8, """## 3. Auswertung über ungesehene Landschaften

0.6.1 misst Region- und Anchor-Accuracy sowie die daraus deterministisch gerenderte absolute Position. Vegetations- und Terrain-Round-trips bleiben pixelgenau durch Seed und Scene-Graph-Parameter.
""")

set_cell(9, """def decode_ordinal(logits): return (logits.softmax(-1)*coord_values).sum(-1).round().clamp(0,SIZE).long()

def predict(model,prompt,seed):
    x=torch.tensor(condition_vector(prompt,seed))[None].to(DEVICE); model.eval()
    with torch.no_grad(): n,o,b,r,a,p,c,ac,tr=model(x)
    return decode_ordinal(n[0]).cpu().numpy(),int(o[0].argmax()),int(b[0].argmax()),r[0].argmax(-1).cpu().numpy(),a[0].argmax(-1).cpu().numpy(),p[0].sigmoid().cpu().numpy(),c[0].argmax(-1).cpu().numpy(),ac[0].argmax(-1).cpu().numpy(),tr[0].argmax(-1).cpu().numpy()

def rasterize_landmarks(seed,terrain,regions,region_ids,anchors,presence,classes):
    obj=np.zeros((SIZE,SIZE),np.int64); boxes=np.zeros((MAX_SLOTS,4),np.int64)
    for slot in range(MAX_SLOTS):
        if presence[slot]<.5: continue
        kind=LANDMARK_CLASSES[int(classes[slot])]; w,h=LANDMARK_SIZES[kind]
        resolved=resolve_anchor(regions,terrain,int(region_ids[slot]),int(anchors[slot]),w,h,seed,slot,obj)
        if resolved is None: continue
        x,y,_=resolved; obj[y:y+h,x:x+w]=slot+1; boxes[slot]=[x,y,w,h]
    return obj,(obj>0).astype(np.uint8),boxes

eval_seeds=[500000+i*7919 for i in range(30)]; metrics={k:[] for k in ['terrain_iou','biome','orientation','params','presence','region','anchor','position','class','action','trigger','interaction']}
prompt='tropical coast beach forest rock portal'
for seed in eval_seeds:
    target=generate_landscape(prompt,seed); nt,ot,bt,rt,ant,prt,ct,actt,trt=scene_targets(target)
    np_,op,bp,rp,anp,prp,cp,ap,trp=predict(model,prompt,seed); pred_params=(bp,op,*map(int,np_)); pred_terrain=render_terrain(pred_params,seed); pred_regions=render_regions(pred_terrain,pred_params,seed)
    pred_obj,pred_int,pred_boxes=rasterize_landmarks(seed,pred_terrain,pred_regions,rp,anp,prp,cp)
    ious=[]
    for cid in range(len(TERRAINS)):
        union=np.logical_or(pred_terrain==cid,target.terrain==cid).sum()
        if union: ious.append(np.logical_and(pred_terrain==cid,target.terrain==cid).sum()/union)
    metrics['terrain_iou'].append(np.mean(ious)); metrics['biome'].append(bp==bt); metrics['orientation'].append(op==ot); metrics['params'].append(np.abs(np_-nt).mean())
    mask=prt>.5; metrics['presence'].append(((prp>=.5)==mask).mean()); metrics['region'].append((rp[mask]==rt[mask]).mean()); metrics['anchor'].append((anp[mask]==ant[mask]).mean())
    target_boxes=np.zeros((MAX_SLOTS,4),np.int64)
    for slot in range(MAX_SLOTS):
        if slot+1 in target.objects: target_boxes[slot]=target.objects[slot+1]['bbox']
    metrics['position'].append(np.abs(pred_boxes[mask,:2]-target_boxes[mask,:2]).mean()); metrics['class'].append((cp[mask]==ct[mask]).mean()); metrics['action'].append((ap[mask]==actt[mask]).mean()); metrics['trigger'].append((trp[mask]==trt[mask]).mean())
    a=target.interaction>0; p=pred_int>0; metrics['interaction'].append(np.logical_and(a,p).sum()/max(1,np.logical_or(a,p).sum()))

print('Terrain — IoU:',round(float(np.mean(metrics['terrain_iou'])),3),'Biome:',round(float(np.mean(metrics['biome'])),3),'Orientation:',round(float(np.mean(metrics['orientation'])),3),'Parameter-MAE:',round(float(np.mean(metrics['params'])),3))
print('Placement — Region:',round(float(np.mean(metrics['region'])),3),'Anchor:',round(float(np.mean(metrics['anchor'])),3),'Position-MAE:',round(float(np.mean(metrics['position'])),3))
print('Slots — Presence:',round(float(np.mean(metrics['presence'])),3),'Klasse:',round(float(np.mean(metrics['class'])),3),'Aktion:',round(float(np.mean(metrics['action'])),3),'Trigger:',round(float(np.mean(metrics['trigger'])),3),'Interaction IoU:',round(float(np.mean(metrics['interaction'])),3))
print('Vegetation im Beispiel:',int(sample.vegetation.sum()),'Bäume; deterministischer Round-trip:',np.array_equal(sample.vegetation,scatter_vegetation(sample.terrain,sample.regions,sample.terrain_params,sample.seed)))
""")

set_cell(10, """## Nächste Experimente

1. Region-, Anchor- und Interaction-Metriken von 0.6.1 auswerten.
2. Vegetationsarten, Dichte und Mindestabstände erweitern.
3. Wege und freie Korridore als eigene Terrainregion ergänzen.
4. In 0.7 den Settlement Layer für Dörfer auf diesen Regionen platzieren.
""")

out=root/'notebooks'/'PixelWorld_0_6_1.ipynb'
out.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print(out)
