"""Regenerate reviewed methodological figures from explicitly identified evidence."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets/resultados_finais_monocular"
OUT=[ROOT/"Parte_Escrita"/n/"figuras" for n in ["Monografia_Metodologia_Professor","ModeloTCC_Monografia_CC_Latex","ModeloTCC_Artigo_CC_Latex"]]
provenance={}
def save(fig,name):
 for out in OUT:
  for ext in ["png","svg","pdf"]:fig.savefig(out/(name+"."+ext),dpi=200,bbox_inches="tight")
 plt.close(fig)
def main():
 ids=["20260715_163428","20260721_121851","20260723_125506","20260723_235409"]
 fig,axes=plt.subplots(2,2,figsize=(10,10),layout="constrained")
 imu,iaxes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
 audits=[]
 for id,ax,ix in zip(ids,axes.flat,iaxes.flat):
  file=DATA/"figuras_revisao"/("voo_"+id+".npz")
  with np.load(file) as f:a=f["intervals"]
  d=np.column_stack([a[k] for k in ["delta_x_m","delta_y_m","delta_z_m"]]).astype(float)
  pos=np.vstack([np.zeros(3),np.cumsum(d,axis=0)])
  invalid=(~np.isfinite(a["dt_s"]))|(a["dt_s"]<=0)|(a["dt_s"]>1.)
  # Split at recorded timing anomalies; coordinates keep every finite displacement.
  breaks=np.flatnonzero(invalid)+1;parts=np.split(np.arange(len(pos)),breaks)
  for part in parts:
   if len(part)>1:ax.plot(pos[part,1],pos[part,0],lw=1.5,color="#2563eb")
  for j in np.linspace(0,len(pos)-2,9,dtype=int):
   end=min(j+20,len(pos)-1)
   if invalid[j:end].any():continue
   ax.annotate("",(pos[end,1],pos[end,0]),(pos[j,1],pos[j,0]),arrowprops=dict(arrowstyle="->",color="#163768",lw=1.2))
  ax.scatter(0,0,c="green",s=65,marker="o",label="Início (origem relativa)",zorder=4)
  ax.scatter(pos[-1,1],pos[-1,0],c="red",s=70,marker="X",label="Fim",zorder=5)
  if len(breaks):ax.scatter(pos[breaks,1],pos[breaks,0],s=15,c="orange",label="Quebra temporal",zorder=3)
  ax.set(title=id,xlabel="Leste relativo (m)",ylabel="Norte relativo (m)",aspect="equal")
  ax.grid(alpha=.2);ax.legend(fontsize=7)
  for k,label,c in zip(["delta_roll_speed_rad_s","delta_pitch_speed_rad_s","delta_yaw_speed_rad_s"],["Δω roll","Δω pitch","Δω yaw"],["#2563eb","#e76f00","#00804c"]):
   y=a[k].astype(float).copy();y[invalid]=np.nan
   ix.plot(a["sample_id"],y,label=label,color=c,lw=.65)
  ix.set(title=id,xlabel="Índice de amostra (ordem de registro)",ylabel="Diferença de velocidade angular (rad/s)")
  ix.grid(alpha=.2);ix.legend(fontsize=8)
  audits.append(dict(run=id,samples=len(a),bad_dt_count=int(invalid.sum()),bad_dt_sample_ids=a["sample_id"][invalid].tolist(),
     largest_step_m=float(np.max(np.linalg.norm(d,axis=1))),first_position_relative=[0,0,0],last_position_relative=pos[-1].tolist(),
     displacement_sum_matches_endpoint=bool(np.allclose(pos[-1],d.sum(axis=0))),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 fig.suptitle("Quatro registros históricos — vista superior, todos os deltas acumulados",fontsize=13)
 imu.suptitle("Diferenças entre atualizações; cores constantes nas quatro execuções",fontsize=13)
 save(fig,"metodologia_trajetorias");save(imu,"metodologia_imu")
 provenance["flight_audit"]=audits

 # All 1,234 filtered historical tabular intervals, no concatenated-run lines.
 with np.load(DATA/"figuras_revisao/intervalos_tabulares_40.npz") as z:tabular={k:z[k] for k in z.files}
 assert len(tabular)==40 and sum(map(len,tabular.values()))==1234
 fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
 ax.hist(np.concatenate([a["dt_s"] for a in tabular.values()]),bins=30,color="#2563eb",edgecolor="white")
 ax.set(xlabel="Intervalo RGB válido (s)",ylabel="Número de intervalos",title="40 execuções tabulares: 1.234 intervalos filtrados")
 save(fig,"metodologia_intervalos_visuais")
 selected=next(k for k in reversed(sorted(tabular)) if len(tabular[k])>=10);a=tabular[selected]
 fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,layout="constrained")
 axes[0].plot(np.arange(len(a)),a["flow_mag_p90_px"],color="#2563eb",marker=".",label="P90 do módulo do fluxo")
 axes[1].plot(np.arange(len(a)),a["delta_depth_close_5m_pp"],color="#e76f00",marker=".",label="Δ pixels com profundidade <5 m")
 axes[0].set(ylabel="Fluxo (pixels/intervalo)",title=selected+" — mesmos intervalos nas duas séries")
 axes[1].set(xlabel="Ordem do intervalo válido",ylabel="Variação (p.p.)")
 for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=9)
 save(fig,"metodologia_sincronizacao")
 provenance["tabular"]=dict(runs=40,intervals=1234,synchronization_example=selected,selection="Last chronological run with at least 10 valid intervals")

 # Six input groups drawn from the same real tabular intervals (illustrative, not attribution).
 first=sorted(tabular)[0];a=tabular[first];x=np.arange(len(a))
 fields=[("dt_s","Sincronização: Δt RGB (s)"),("delta_x_m","Movimento: Δx (m)"),
 ("delta_gyro_x_rad_s","IMU: Δ giroscópio x (rad/s)"),("flow_mag_p90_px","Fluxo: módulo P90 (pixels)"),
 ("delta_avoid_brake","Comando: Δ frenagem"),("delta_depth_close_5m_pp","Alvo GT: Δ pixels <5 m (p.p.)")]
 fig,axes=plt.subplots(2,3,figsize=(11,6),layout="constrained")
 for ax,(field,title) in zip(axes.flat,fields):
  values=a[field]
  if field=="delta_depth_close_5m_pp":
   with np.load(DATA/"figuras_revisao/diferenca_imagem_exemplo.npz") as z:values=z["mean_absolute"]
   title="Diferença de imagem: média absoluta"
  ax.plot(x,values,marker=".",lw=1);ax.set(title=title,xlabel="Intervalo válido");ax.grid(alpha=.2)
 fig.suptitle(first+" — seis grupos de atributos observados nos mesmos intervalos")
 save(fig,"metodologia_grupos_sinais")
 with np.load(DATA/"figuras_revisao/cenas_rgb.npz") as z:scenes={k:z[k] for k in z.files}
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
 for ax,(name,img) in zip(axes.flat,scenes.items()):
  ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.set_title(name.replace("rgb_","Quadro "));ax.axis("off")
 fig.suptitle("Baylands: variação visual ao longo do registro de desenvolvimento")
 save(fig,"metodologia_cenas_contexto")
 fig,ax=plt.subplots(figsize=(10,3),layout="constrained");ax.axis("off")
 labels=["Atributos\npadronizados","Linear 64\nReLU\nDropout 0,30","Linear 16\nReLU\nDropout 0,30","Linear 6\nAlvos inversamente\npadronizados"]
 for j,label in enumerate(labels):
  ax.text(.02+j*.25,.5,label,va="center",fontsize=11,bbox=dict(boxstyle="round",fc="#e0ecff"))
  if j<3:ax.annotate("",(.25+j*.25,.5),(.20+j*.25,.5),arrowprops=dict(arrowstyle="->"))
 ax.set_title("MLP tabular; dropout desligado na avaliação; seis variações de proximidade")
 save(fig,"metodologia_mlp")
 # True volumetric octree sketch: every level divides one octant into eight cubes.
 fig=plt.figure(figsize=(10,4),layout="constrained")
 def edges(origin,side):
  from itertools import product
  for axis in range(3):
   for bits in product([0,1],repeat=2):
    a=np.array(origin,dtype=float);j=[v for v in range(3) if v!=axis]
    a[j]+=np.array(bits)*side;b=a.copy();b[axis]+=side
    yield np.array([a,b])
 for level in range(3):
  ax=fig.add_subplot(1,3,level+1,projection="3d")
  segments=list(edges([0,0,0],1))
  for depth in range(level):
   side=1/2**(depth+1)
   from itertools import product
   for index in product([0,1],repeat=3):segments+=list(edges(np.array(index)*side,side))
  ax.add_collection3d(Line3DCollection(segments,colors="#2563eb",linewidths=.8))
  ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel="x",ylabel="y",zlabel="z",title=f"Nível {level}")
  ax.set_box_aspect([1,1,1])
 fig.suptitle("Octree conceitual — um octante é refinado a cada nível")
 save(fig,"metodologia_octree")

 with np.load(DATA/"sample_1.npz") as f:frame={k:f[k] for k in f.files}
 # Full two-branch pipeline with actual RGB as source illustration.
 fig,ax=plt.subplots(figsize=(12,7),layout="constrained");ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
 nodes={"a":(.03,.88,"(a) RGB + IMU + pose"),"b":(.03,.64,"(b) Compensação de rotação\n+ pontos e rastreamento"),
 "c":(.03,.40,"(c) Intervalos e atributos\n+ alvos de proximidade GT"),"d":(.03,.16,"(d) MLP / árvores / ablação\nComparação tabular"),
 "e":(.61,.88,"(e) RGB original → v21\nProfundidade monocular"),"f":(.61,.64,"(f) Retroprojeção + poses\nGrade de ocupação 3D"),
 "g":(.61,.40,"(g) Revisão temporal\n+ saída curva + A*"),"h":(.61,.16,"(h) Avaliação espacial GT\nDecisão: sem liberação para voo")}
 for x,y,label in nodes.values():ax.text(x,y,label,fontsize=11,va="center",bbox=dict(boxstyle="round,pad=.5",fc="#e0ecff",ec="#31629e"))
 for a,b in [("a","b"),("b","c"),("c","d"),("e","f"),("f","g"),("g","h")]:
  x,y,_=nodes[a];xx,yy,_=nodes[b]
  ax.annotate("",(xx+.13,yy+.065),(x+.13,y-.065),arrowprops=dict(arrowstyle="->",color="#193d66"))
 ax.text(.40,.52,"Sem transferência\nda MLP para v21",ha="center",fontsize=9)
 inset=ax.inset_axes([.32,.69,.22,.23]);inset.imshow(cv2.cvtColor(frame["rgb_bgr"],cv2.COLOR_BGR2RGB));inset.axis("off");inset.set_title("RGB real",fontsize=9)
 fig.suptitle("Dois eixos complementares; referências identificadas por função")
 save(fig,"metodologia_fluxograma_completo")
 provenance["generator_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 provenance["command"]="python scripts/generate_methodology_figures.py"

 print(json.dumps(audits,indent=2))
if __name__=="__main__":
 main()
 import subprocess,os
 subprocess.run([sys.executable,str(ROOT/"scripts/reproduce_full_stabilization.py")],check=True,env=dict(os.environ,PYTHONNOUSERSITE="1"))
