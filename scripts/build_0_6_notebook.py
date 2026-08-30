import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
md("""# PixelWorld-0.6 — Meilenstein 3: Landschaften und Terrain

0.6 erzeugt strukturierte Außenwelten. Ein lernendes Modell sagt Biome, Küstenrichtung, Uferlinie, Strandbreite, Felsigkeit und interaktive Object Slots voraus. Ein deterministischer Rasterizer erzeugt daraus Terrain-, Semantic-, Object-, Walkability- und Interaction-Maps.
"""),
code("""# Notebook-Abhängigkeiten installieren.
%pip install -q ipympl

import hashlib, random
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SEED = 42
random.seed(SEED); np.random.seed(SEED)
SIZE = 64
MAX_SLOTS = 8
SLOT_LATENT_DIM = 6
TERRAIN_LATENT_DIM = 5
LAYOUT_DIM = TERRAIN_LATENT_DIM + MAX_SLOTS*SLOT_LATENT_DIM

BIOMES = ['temperate', 'tropical', 'arid', 'tundra']
ORIENTATIONS = ['north', 'east', 'south', 'west']
TERRAINS = {'water': 0, 'sand': 1, 'grass': 2, 'dirt': 3, 'rock': 4, 'snow': 5}
OBJECT_CLASSES = ['tree', 'rock', 'npc', 'portal']
OBJECT_SIZES = {'tree': (5,8), 'rock': (5,4), 'npc': (5,9), 'portal': (6,8)}
ACTIONS = ['LOOK', 'USE', 'SCAN']
TRIGGER_TYPES = ['NONE', 'WORLD', 'STORY', 'SECRET']
PALETTE = np.asarray([(25,85,145),(224,198,125),(58,125,65),(130,92,55),(105,105,110),(225,232,238)], np.uint8)

@dataclass
class Landscape:
    prompt: str
    seed: int
    biome: str
    terrain: np.ndarray
    rgb: np.ndarray
    semantic: np.ndarray
    object_map: np.ndarray
    walkable: np.ndarray
    interaction: np.ndarray
    terrain_params: tuple
    objects: dict

def world_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

def transition_seed(current_seed, slot, trigger_type, story_state=0):
    return world_seed(f'{current_seed}:{slot}:{trigger_type}:{story_state}')

def layout_from_seed(seed):
    return np.random.default_rng(seed).random(LAYOUT_DIM).astype(np.float32)
"""),
md("""## 1. Terrain Scene Graph

Die Terrainform wird durch wenige bedeutungsvolle Parameter beschrieben. Organische Küstenkrümmung entsteht deterministisch aus dem Welt-Seed und muss nicht als 4.096 unabhängige Pixel gelernt werden.
"""),
code("""def terrain_params(prompt, seed):
    p = prompt.lower(); layout = layout_from_seed(seed)
    biome = next((b for b in BIOMES if b in p), BIOMES[int(layout[0]*len(BIOMES)) % len(BIOMES)])
    orientation = int(layout[1]*4) % 4
    shoreline = 22 + int(layout[2]*21)       # Pixel 22..42
    beach_width = 3 + int(layout[3]*9)       # Pixel 3..11
    rockiness = int(layout[4]*6)              # Stufe 0..5
    return BIOMES.index(biome), orientation, shoreline, beach_width, rockiness

def render_terrain(params, seed):
    biome_id, orientation, shoreline, beach_width, rockiness = map(int, params)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    along = xx if orientation in (0,2) else yy
    across = yy if orientation in (0,2) else xx
    if orientation in (2,1):
        across = SIZE - 1 - across
    phase = (seed % 10007) / 10007 * 2*np.pi
    coast = shoreline + 2.3*np.sin(along/7.0 + phase) + 1.1*np.sin(along/3.3 + phase*1.7)
    terrain = np.full((SIZE,SIZE), TERRAINS['grass'], np.int64)
    terrain[across < coast] = TERRAINS['water']
    terrain[(across >= coast) & (across < coast + beach_width)] = TERRAINS['sand']
    land = across >= coast + beach_width
    base = ['grass','grass','dirt','snow'][biome_id]
    terrain[land] = TERRAINS[base]
    rock_field = np.sin(xx*0.37 + phase) + np.cos(yy*0.29 - phase) + np.sin((xx+yy)*0.17)
    terrain[land & (rock_field > 2.35 - rockiness*0.12)] = TERRAINS['rock']
    return terrain

def nearest_land(terrain, x, y, w, h):
    for radius in range(SIZE):
        for dy, dx in ((0,0),(radius,0),(-radius,0),(0,radius),(0,-radius)):
            px, py = int(np.clip(x+dx,0,SIZE-w)), int(np.clip(y+dy,0,SIZE-h))
            patch = terrain[py:py+h,px:px+w]
            if np.all(patch != TERRAINS['water']): return px,py
    return int(x),int(y)

def generate_landscape(prompt, seed=None):
    seed = world_seed(prompt) if seed is None else int(seed)
    layout = layout_from_seed(seed)
    params = terrain_params(prompt, seed)
    terrain = render_terrain(params, seed)
    obj = np.zeros((SIZE,SIZE), np.int64); objects = {}
    anchors = [(7,8),(20,8),(35,8),(49,8),(7,30),(20,30),(35,30),(49,30)]
    for slot in range(MAX_SLOTS):
        values = layout[TERRAIN_LATENT_DIM+slot*SLOT_LATENT_DIM:TERRAIN_LATENT_DIM+(slot+1)*SLOT_LATENT_DIM]
        present = slot == 0 or values[0] > 0.34
        if not present: continue
        class_id = 3 if slot == 0 else int(values[1]*len(OBJECT_CLASSES)) % len(OBJECT_CLASSES)
        kind = OBJECT_CLASSES[class_id]; w,h = OBJECT_SIZES[kind]
        ax,ay = anchors[slot]
        x,y = nearest_land(terrain, ax+int(values[2]*5), ay+int(values[3]*5), w,h)
        # Vollständige deterministische Suche verhindert Überlappungen auch in Randfällen.
        placed = False; columns = SIZE-w+1; rows = SIZE-h+1
        for shift in range(columns*rows):
            px = (x + shift) % columns
            py = (y + shift//columns) % rows
            if np.all(obj[py:py+h,px:px+w] == 0) and np.all(terrain[py:py+h,px:px+w] != TERRAINS['water']):
                x,y = px,py; placed = True; break
        if not placed: continue
        action = ACTIONS[int(values[4]*len(ACTIONS)) % len(ACTIONS)]
        trigger = TRIGGER_TYPES[int(values[5]*len(TRIGGER_TYPES)) % len(TRIGGER_TYPES)]
        oid = slot+1; obj[y:y+h,x:x+w] = oid
        objects[oid] = {'class':kind,'bbox':[x,y,w,h],'action':action,'trigger_type':trigger,
                        'next_seed':transition_seed(seed,slot,trigger)}
    semantic = terrain.copy()
    rgb = PALETTE[terrain]
    walkable = np.isin(terrain,[TERRAINS['sand'],TERRAINS['grass'],TERRAINS['dirt'],TERRAINS['snow']]).astype(np.uint8)
    interaction = (obj>0).astype(np.uint8)
    return Landscape(prompt,seed,BIOMES[params[0]],terrain,rgb,semantic,obj,walkable,interaction,params,objects)

sample = generate_landscape('tropical coast with palms and portal', 424242)
sample.terrain_params, sample.objects
"""),
code("""def show_landscape(w):
    fig,ax=plt.subplots(1,5,figsize=(18,4))
    items=[(w.rgb,'RGB'),(w.terrain,'Terrain'),(w.object_map,'Objects'),(w.walkable,'Walkable'),(w.interaction,'Interaction')]
    for a,(data,title) in zip(ax,items):
        a.imshow(data,interpolation='nearest'); a.set_title(title); a.axis('off')
    plt.tight_layout(); plt.show()
show_landscape(sample)
"""),
md("""## 2. Terrain- und Object-Slot-Modell

Terrain, Geometry, Presence und Attribute besitzen getrennte Pfade. Die organische Detailstruktur wird vom Rasterizer aus den vorhergesagten Terrainparametern und dem unveränderten Welt-Seed erzeugt.
"""),
code("""import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
COORD_CLASSES=SIZE+1

def prompt_vector(prompt):
    p=prompt.lower(); v=np.zeros(len(BIOMES)+6,np.float32)
    for i,b in enumerate(BIOMES): v[i]=float(b in p)
    for i,w in enumerate(['coast','beach','forest','rock','portal','dark']): v[len(BIOMES)+i]=float(w in p)
    return v

def condition_vector(prompt,seed):
    return np.concatenate([prompt_vector(prompt),layout_from_seed(seed)]).astype(np.float32)

def scene_targets(w):
    biome,orientation,shoreline,width,rockiness=w.terrain_params
    positions=np.zeros((MAX_SLOTS,2),np.int64); presence=np.zeros(MAX_SLOTS,np.float32)
    classes=np.zeros(MAX_SLOTS,np.int64); actions=np.zeros(MAX_SLOTS,np.int64); triggers=np.zeros(MAX_SLOTS,np.int64)
    for slot in range(MAX_SLOTS):
        oid=slot+1
        if oid in w.objects:
            meta=w.objects[oid]; positions[slot]=meta['bbox'][:2]; presence[slot]=1
            classes[slot]=OBJECT_CLASSES.index(meta['class']); actions[slot]=ACTIONS.index(meta['action'])
            triggers[slot]=TRIGGER_TYPES.index(meta['trigger_type'])
    return np.asarray([shoreline,width,rockiness]),orientation,biome,positions,presence,classes,actions,triggers

class LandscapeDataset(Dataset):
    def __init__(self,n=12000):
        self.n=n
    def __len__(self): return self.n
    def __getitem__(self,i):
        p=f'{BIOMES[i%4]} coast beach forest rock portal {i}'; seed=i+1000
        w=generate_landscape(p,seed); numeric,orient,biome,pos,pres,cls,act,trig=scene_targets(w)
        return tuple(torch.tensor(x) for x in (condition_vector(p,seed),numeric,orient,biome,pos,pres,cls,act,trig))

class LandscapeNet(nn.Module):
    def __init__(self,condition_dim=len(BIOMES)+6+LAYOUT_DIM,hidden=320):
        super().__init__(); self.slots=MAX_SLOTS
        def encoder(): return nn.Sequential(nn.Linear(condition_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.terrain_encoder=encoder(); self.geometry_encoder=encoder(); self.presence_encoder=encoder(); self.attribute_encoder=encoder()
        self.terrain_numeric=nn.Linear(hidden,3*COORD_CLASSES); self.orientation_head=nn.Linear(hidden,4); self.biome_head=nn.Linear(hidden,4)
        self.slot_queries=nn.Embedding(MAX_SLOTS,hidden)
        self.geometry_decoder=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.attribute_decoder=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.position_head=nn.Linear(hidden,2*COORD_CLASSES); self.presence_head=nn.Linear(hidden,MAX_SLOTS)
        self.class_head=nn.Linear(hidden,4); self.action_head=nn.Linear(hidden,3); self.trigger_head=nn.Linear(hidden,4)
    def slot_decode(self,world,decoder):
        q=self.slot_queries.weight[None].expand(world.shape[0],-1,-1); c=world[:,None,:].expand(-1,self.slots,-1)
        return decoder(torch.cat([c,q],-1))
    def forward(self,x):
        terrain=self.terrain_encoder(x); geo=self.slot_decode(self.geometry_encoder(x),self.geometry_decoder)
        attr=self.slot_decode(self.attribute_encoder(x),self.attribute_decoder)
        return (self.terrain_numeric(terrain).reshape(-1,3,COORD_CLASSES),self.orientation_head(terrain),self.biome_head(terrain),
                self.position_head(geo).reshape(-1,self.slots,2,COORD_CLASSES),self.presence_head(self.presence_encoder(x)),
                self.class_head(attr),self.action_head(attr),self.trigger_head(attr))

model=LandscapeNet().to(DEVICE)
sum(p.numel() for p in model.parameters())
"""),
code("""loader=DataLoader(LandscapeDataset(12000),batch_size=128,shuffle=True)
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
coord_values=torch.arange(COORD_CLASSES,dtype=torch.float32,device=DEVICE)
ce=nn.CrossEntropyLoss(reduction='none'); bce=nn.BCEWithLogitsLoss(reduction='none')

def ordinal_loss(logits,target,sigma=1.0):
    d=coord_values-target[...,None].float(); soft=torch.exp(-.5*(d/sigma)**2); soft/=soft.sum(-1,keepdim=True)
    soft_ce=-(soft*logits.log_softmax(-1)).sum(-1); expected=(logits.softmax(-1)*coord_values).sum(-1)
    return soft_ce+4*F.smooth_l1_loss(expected,target.float(),reduction='none')/SIZE

def masked_ce(logits,target,presence):
    e=ce(logits.flatten(0,1),target.flatten()).reshape_as(target)
    return (e*presence).sum()/presence.sum().clamp_min(1)

EPOCHS=45
for epoch in range(EPOCHS):
    model.train(); totals=np.zeros(8); batches=0
    for batch in loader:
        condition,numeric,orient,biome,pos,pres,cls,act,trig=[x.to(DEVICE) for x in batch]
        num_l,orient_l,biome_l,pos_l,pres_l,cls_l,act_l,trig_l=model(condition)
        terrain_loss=ordinal_loss(num_l,numeric).mean()+ce(orient_l,orient).mean()+ce(biome_l,biome).mean()
        pos_loss=(ordinal_loss(pos_l,pos).mean(-1)*pres).sum()/pres.sum().clamp_min(1)
        presence_loss=(bce(pres_l,pres)*torch.where(pres>.5,1.,2.)).mean()
        class_loss=masked_ce(cls_l,cls,pres); action_loss=masked_ce(act_l,act,pres); trigger_loss=masked_ce(trig_l,trig,pres)
        loss=terrain_loss+4*pos_loss+presence_loss+class_loss+action_loss+trigger_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        totals += [loss.item(),terrain_loss.item(),pos_loss.item(),presence_loss.item(),class_loss.item(),action_loss.item(),trigger_loss.item(),1]; batches+=1
    v=totals/max(1,batches)
    print(f'Epoch {epoch+1:02d}: loss={v[0]:.3f} terrain={v[1]:.3f} pos={v[2]:.3f} presence={v[3]:.3f} class={v[4]:.3f} action={v[5]:.3f} trigger={v[6]:.3f}')
"""),
md("""## 3. Auswertung über ungesehene Landschaften

Neben den Object-Slot-Metriken misst 0.6 Terrain IoU, Biome Accuracy, Küstenrichtungs-Accuracy sowie MAE für Uferlinie, Strandbreite und Felsigkeit.
"""),
code("""def decode_ordinal(logits):
    return (logits.softmax(-1)*coord_values).sum(-1).round().clamp(0,SIZE).long()

def predict(model,prompt,seed):
    x=torch.tensor(condition_vector(prompt,seed))[None].to(DEVICE); model.eval()
    with torch.no_grad(): out=model(x)
    numeric,orient,biome,pos,pres,cls,act,trig=out
    return (decode_ordinal(numeric[0]).cpu().numpy(),int(orient[0].argmax()),int(biome[0].argmax()),
            decode_ordinal(pos[0]).cpu().numpy(),pres[0].sigmoid().cpu().numpy(),cls[0].argmax(-1).cpu().numpy(),
            act[0].argmax(-1).cpu().numpy(),trig[0].argmax(-1).cpu().numpy())

def rasterize_prediction(seed,numeric,orientation,biome,positions,presence,classes):
    params=(biome,orientation,int(numeric[0]),int(numeric[1]),int(numeric[2])); terrain=render_terrain(params,seed)
    obj=np.zeros((SIZE,SIZE),np.int64)
    for slot in range(MAX_SLOTS):
        if presence[slot]<.5: continue
        kind=OBJECT_CLASSES[int(classes[slot])]; w,h=OBJECT_SIZES[kind]; x,y=positions[slot]
        x=int(np.clip(x,0,SIZE-w)); y=int(np.clip(y,0,SIZE-h)); obj[y:y+h,x:x+w]=slot+1
    return terrain,obj,(obj>0).astype(np.uint8)

eval_seeds=[500000+i*7919 for i in range(30)]; terrain_ious=[]; biome_acc=[]; orientation_acc=[]; terrain_mae=[]
presence_acc=[]; position_mae=[]; class_acc=[]; action_acc=[]; trigger_acc=[]; interaction_iou=[]
prompt='tropical coast beach forest rock portal'
for seed in eval_seeds:
    target=generate_landscape(prompt,seed); nt,ot,bt,pt,prt,ct,at,tt=scene_targets(target)
    np_,op,bp,pp,prp,cp,ap,tp=predict(model,prompt,seed); pred_terrain,pred_obj,pred_int=rasterize_prediction(seed,np_,op,bp,pp,prp,cp)
    terrain_ious.append(np.mean([np.logical_and(pred_terrain==cid,target.terrain==cid).sum()/max(1,np.logical_or(pred_terrain==cid,target.terrain==cid).sum()) for cid in np.unique(target.terrain)]))
    biome_acc.append(bp==bt); orientation_acc.append(op==ot); terrain_mae.append(np.abs(np_-nt).mean())
    mask=prt>.5; presence_acc.append(((prp>=.5)==mask).mean()); position_mae.append(np.abs(pp[mask]-pt[mask]).mean())
    class_acc.append((cp[mask]==ct[mask]).mean()); action_acc.append((ap[mask]==at[mask]).mean()); trigger_acc.append((tp[mask]==tt[mask]).mean())
    a=target.interaction>0; p=pred_int>0; interaction_iou.append(np.logical_and(a,p).sum()/max(1,np.logical_or(a,p).sum()))

print('Terrain — Mean IoU:',round(float(np.mean(terrain_ious)),3),'Biome Accuracy:',round(float(np.mean(biome_acc)),3),'Orientation Accuracy:',round(float(np.mean(orientation_acc)),3),'Parameter-MAE:',round(float(np.mean(terrain_mae)),3))
print('Slots — Presence:',round(float(np.mean(presence_acc)),3),'Position-MAE:',round(float(np.mean(position_mae)),3),'Klasse:',round(float(np.mean(class_acc)),3),'Aktion:',round(float(np.mean(action_acc)),3),'Trigger:',round(float(np.mean(trigger_acc)),3))
print('Interaction IoU:',round(float(np.mean(interaction_iou)),3))
checks=[transition_seed(s,slot,t,state) for s in eval_seeds for slot in range(MAX_SLOTS) for t in TRIGGER_TYPES for state in (0,1)]
print('Deterministische Weltübergänge:',checks==[transition_seed(s,slot,t,state) for s in eval_seeds for slot in range(MAX_SLOTS) for t in TRIGGER_TYPES for state in (0,1)],'geprüft:',len(checks))
"""),
md("""## Nächste Experimente

1. Terrainparameter und Object Slots von 0.6 auswerten.
2. In 0.6.1 Wälder und Vegetation als deterministischen Scatter-Layer ergänzen.
3. Flüsse, Seen und mehrere Küstenabschnitte unterstützen.
4. Terrainübergänge durch gelernte, aber regelbeschränkte Felder erweitern.
5. Innen- und Außenwelten in einem gemeinsamen Scene Graph verbinden.
""")]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / 'notebooks' / 'PixelWorld_0_6.ipynb'
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(out)
