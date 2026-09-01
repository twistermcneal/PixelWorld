"""Dependency-free browser export for a compiled adventure."""

from __future__ import annotations

import json
from pathlib import Path


INDEX_HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PixelWorld Adventure 0.6.3</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <main>
    <header><h1>PixelWorld Adventure</h1><button id="debug-toggle">Walkboxen anzeigen</button></header>
    <section id="game-shell">
      <canvas id="scene" width="1024" height="576" aria-label="Spielbarer Adventure-Raum"></canvas>
      <div id="status" role="status">Klicke auf einen Gegenstand oder auf den Boden.</div>
      <div id="actions" aria-label="Aktionen"></div>
      <div id="inventory" aria-label="Inventar"></div>
    </section>
    <p class="notice">Eigene deterministische Pixel-Platzhalter – keine externen Assets.</p>
  </main>
  <script src="runtime.js"></script>
</body>
</html>
"""


STYLES_CSS = """:root{color-scheme:dark;font-family:ui-monospace,Consolas,monospace;background:#050914;color:#dffcff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 20%,#18304a,#050914 70%);min-height:100vh}
main{width:min(1100px,100%);margin:auto;padding:16px}header{display:flex;align-items:center;justify-content:space-between;gap:12px}
h1{font-size:clamp(18px,3vw,30px);color:#28e7ff;text-shadow:3px 3px #57205f}button{font:inherit;color:#07111f;background:#ffd33d;border:0;border-bottom:4px solid #a86d0b;padding:8px 12px;cursor:pointer}
#game-shell{border:4px solid #335a72;background:#07111f;box-shadow:0 0 30px #28e7ff55}canvas{display:block;width:100%;height:auto;image-rendering:pixelated;cursor:crosshair}
#status{min-height:48px;padding:12px;color:#fff;background:#101d31;border-top:3px solid #335a72}
#actions,#inventory{display:flex;flex-wrap:wrap;gap:8px;padding:10px;border-top:2px solid #223d55;min-height:54px}
#inventory{background:#0a1424}#inventory::before{content:'INVENTAR';color:#ff3b81;padding:9px}.item{background:#9cff57}.notice{color:#8ca8b8;font-size:12px}
"""


RUNTIME_TEMPLATE = r'''"use strict";
// Generic 0.6.3 interpreter. All room-specific entities and rules are compiled data below.
const GAME = __GAME_DATA__;
const canvas=document.querySelector("#scene"),ctx=canvas.getContext("2d"),statusEl=document.querySelector("#status"),actionsEl=document.querySelector("#actions"),inventoryEl=document.querySelector("#inventory");
const scene=GAME.scene_graph,rules=GAME.runtime_rules,S=canvas.width/scene.size[0];
let state=structuredClone(scene.initial_state),selected=null,debug=false;
const entities=new Map(scene.entities.map(e=>[e.id,e]));
function pointIn(p,poly){let inside=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j];if(((a[1]>p[1])!==(b[1]>p[1]))&&(p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]))inside=!inside;}return inside;}
function closest(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],n=dx*dx+dy*dy,t=n?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/n)):0;return[a[0]+t*dx,a[1]+t*dy];}
function project(p){const inside=scene.walkboxes.find(w=>pointIn(p,w.polygon));if(inside)return p;let best=null;for(const w of scene.walkboxes)for(let i=0;i<w.polygon.length;i++){const q=closest(p,w.polygon[i],w.polygon[(i+1)%w.polygon.length]),d=(q[0]-p[0])**2+(q[1]-p[1])**2,c=[d,q[0],q[1],q];if(!best||JSON.stringify(c.slice(0,3))<JSON.stringify(best.slice(0,3)))best=c;}return best[3];}
function walkable(p){return scene.walkboxes.some(w=>pointIn(p,w.polygon))&&!scene.collision_polygons.some(c=>pointIn(p,c.polygon));}
function clear(a,b){const n=Math.max(2,Math.ceil(Math.hypot(b[0]-a[0],b[1]-a[1])*2));for(let i=0;i<=n;i++)if(!walkable([a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n]))return false;return true;}
function boxes(p){return scene.walkboxes.filter(w=>pointIn(p,w.polygon)).map(w=>w.id).sort();}
function route(goal){goal=project(goal);const start=state.player_position;if(clear(start,goal))return[start,goal];const goals=new Set(boxes(goal)),queue=boxes(start).map(id=>({id,pts:[start]})),seen=new Set(queue.map(x=>x.id));while(queue.length){const cur=queue.shift(),last=cur.pts[cur.pts.length-1];if(goals.has(cur.id)&&clear(last,goal))return smooth([...cur.pts,goal]);const edges=scene.navigation_edges.filter(e=>e.from===cur.id||e.to===cur.id).map(e=>({id:e.from===cur.id?e.to:e.from,p:e.point})).sort((a,b)=>a.id.localeCompare(b.id));for(const e of edges)if(!seen.has(e.id)&&clear(last,e.p)){seen.add(e.id);queue.push({id:e.id,pts:[...cur.pts,e.p]});}}return null;}
function smooth(points){const out=[points[0]];let i=0;while(i<points.length-1){let j=points.length-1;while(j>i+1&&!clear(points[i],points[j]))j--;out.push(points[j]);i=j;}return out;}
function animate(path,done){if(!path){say("Dieses Ziel ist nicht erreichbar.");return;}let segment=1;function frame(){if(segment>=path.length){done&&done();return;}const p=state.player_position,q=path[segment],d=Math.hypot(q[0]-p[0],q[1]-p[1]);if(d<.45){state.player_position=[...q];segment++;}else state.player_position=[p[0]+(q[0]-p[0])*.35/d,p[1]+(q[1]-p[1])*.35/d];render();requestAnimationFrame(frame);}frame();}
function read(path){return path.split(".").reduce((v,k)=>v[k],state);}function write(path,value){const parts=path.split("."),key=parts.pop(),target=parts.reduce((v,k)=>v[k],state);target[key]=value;}
function condition(c){if(c.op==="equals")return read(c.path)===c.value;if(c.op==="inventory_contains")return state.inventory.includes(c.value);if(c.op==="inventory_missing")return!state.inventory.includes(c.value);return false;}
function effect(e){if(e.op==="set")write(e.path,e.value);else if(e.op==="inventory_add"&&!state.inventory.includes(e.value))state.inventory.push(e.value);else if(e.op==="inventory_remove")state.inventory=state.inventory.filter(x=>x!==e.value);state.inventory.sort();}
function available(i){const e=entities.get(i.target_id),s=state.objects[i.target_id]||{};return e&&e.visible&&e.enabled&&!s.taken&&i.conditions.every(condition);}
function invoke(interaction){const e=entities.get(interaction.target_id),path=route(e.walk_to_point);animate(path,()=>{interaction.effects.forEach(effect);state.completed=rules.ending_conditions.some(end=>end.conditions.every(condition));say(interaction.text+(state.completed?" Raumziel erreicht!":""));selected=null;render();});}
function say(text){statusEl.textContent=text;}
function polygon(poly,fill,stroke){ctx.beginPath();ctx.moveTo(poly[0][0]*S,poly[0][1]*S);for(const p of poly.slice(1))ctx.lineTo(p[0]*S,p[1]*S);ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();}}
const colors={time_machine:"#28e7ff",control_console:"#ffd33d",chemical_bottle:"#ff3b81",mixing_flask:"#dffcff",time_portal:"#9cff57",robot_arm:"#738fa7",gear:"#b48c5e",mad_scientist:"#f5f0dc"};
function drawEntity(e){const st=state.objects[e.id]||{};if(!e.visible||st.taken)return;const[x,y,w,h]=e.bbox;ctx.fillStyle=colors[e.class]||"#af7ac5";ctx.fillRect(x*S,y*S,w*S,h*S);ctx.fillStyle="#07111f";ctx.fillRect((x+2)*S,(y+2)*S,Math.max(1,w-4)*S,Math.max(1,h-4)*S);ctx.fillStyle=colors[e.class]||"#af7ac5";ctx.fillRect((x+3)*S,(y+3)*S,Math.max(1,w-6)*S,Math.max(1,h-6)*S);if(st.cooled||st.active){ctx.shadowColor="#9cff57";ctx.shadowBlur=24;ctx.strokeStyle="#fff";ctx.lineWidth=3;ctx.strokeRect(x*S,y*S,w*S,h*S);ctx.shadowBlur=0;}if(selected===e.id)polygon(e.hotspot_polygon,null,"#fff");}
function render(){const g=ctx.createLinearGradient(0,0,0,canvas.height);g.addColorStop(0,"#07111f");g.addColorStop(.7,"#18304a");g.addColorStop(1,"#090b12");ctx.fillStyle=g;ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#20364a";ctx.fillRect(0,40*S,canvas.width,32*S);ctx.strokeStyle="#28e7ff33";for(let x=0;x<128;x+=8){ctx.beginPath();ctx.moveTo(x*S,40*S);ctx.lineTo(x*S,72*S);ctx.stroke();}for(let y=40;y<72;y+=8){ctx.beginPath();ctx.moveTo(0,y*S);ctx.lineTo(canvas.width,y*S);ctx.stroke();}scene.entities.slice().sort((a,b)=>a.z_layer-b.z_layer).forEach(drawEntity);const p=state.player_position;ctx.fillStyle="#ffcf8b";ctx.fillRect((p[0]-2)*S,(p[1]-9)*S,4*S,9*S);ctx.fillStyle="#4e9cff";ctx.fillRect((p[0]-3)*S,(p[1]-5)*S,6*S,5*S);scene.occlusion_polygons.forEach(o=>polygon(o.polygon,"#0a1424",null));if(debug){scene.walkboxes.forEach(w=>polygon(w.polygon,"#28e7ff22","#28e7ff"));scene.collision_polygons.forEach(c=>polygon(c.polygon,"#ff3b8144","#ff3b81"));scene.walk_to_points.forEach(w=>{ctx.fillStyle="#ffd33d";ctx.fillRect(w.point[0]*S-3,w.point[1]*S-3,6,6);});}renderUi();}
function renderUi(){actionsEl.replaceChildren();inventoryEl.querySelectorAll("button").forEach(e=>e.remove());for(const id of state.inventory){const b=document.createElement("button");b.className="item";b.textContent=(GAME.adventure.inventory_items.find(x=>x.id===id)||{name:id}).name;inventoryEl.append(b);}if(!selected)return;const e=entities.get(selected);addAction("Ansehen",()=>say(e.description));if(e.hotspot_role==="npc")addAction("Reden",()=>say("Knallbert: Zwei Reagenzien, eine Flasche – dann ab damit in die Maschine!"));for(const i of rules.interactions.filter(x=>x.target_id===selected&&available(x)))addAction(label(i),()=>invoke(i));}
function label(i){if(i.verb==="take")return"Nehmen";if(i.verb==="combine")return"Kombinieren";if(i.verb==="use")return"Benutzen: "+i.item_ids.join(" + ");return i.verb;}
function addAction(label,fn){const b=document.createElement("button");b.textContent=label;b.addEventListener("click",fn);actionsEl.append(b);}
canvas.addEventListener("click",event=>{const box=canvas.getBoundingClientRect(),p=[(event.clientX-box.left)/box.width*scene.size[0],(event.clientY-box.top)/box.height*scene.size[1]],hit=scene.entities.filter(e=>e.visible&&!(state.objects[e.id]||{}).taken&&pointIn(p,e.hotspot_polygon)).sort((a,b)=>b.z_layer-a.z_layer)[0];if(hit){selected=hit.id;say(hit.name+" ausgewählt.");render();}else{selected=null;animate(route(p),()=>say("Ziel erreicht."));}});
document.querySelector("#debug-toggle").addEventListener("click",()=>{debug=!debug;render();});render();
'''


PLACEHOLDER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="128" height="72" shape-rendering="crispEdges"><rect width="128" height="72" fill="#07111f"/><path d="M0 40h128v32H0z" fill="#18304a"/><circle cx="64" cy="34" r="16" fill="#28e7ff" opacity=".65"/><path d="M48 54h32v14H48z" fill="#20364a"/></svg>"""


def export_browser(game: dict, output: str | Path) -> list[str]:
    output = Path(output)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    runtime = RUNTIME_TEMPLATE.replace("__GAME_DATA__", json.dumps(game, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    files = {
        output / "index.html": INDEX_HTML,
        output / "runtime.js": runtime,
        output / "styles.css": STYLES_CSS,
        assets / "lab-placeholder.svg": PLACEHOLDER_SVG,
        assets / "README.txt": "Own deterministic PixelWorld 0.6.3 placeholder assets. No external assets.\n",
    }
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    return [str(path.relative_to(output)).replace("\\", "/") for path in files]

