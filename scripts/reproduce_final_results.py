from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=json.loads((ROOT/"datasets/resultados_finais_monocular/results.json").read_text())
 r=d
 assert r["frames_integrated"]==77 and r["planning"]["successes"]==12 and abs(r["map"]["false_free_rate"]-4/401)<1e-12
 for folder in [ROOT/"Parte_Escrita/Monografia_TCC"]:
  if not folder.is_dir() or not ((folder/"main.tex").exists() or (folder/"principal.tex").exists()):continue
  out=folder/"figuras";out.mkdir(exist_ok=True)
  f,a=plt.subplots(figsize=(8,4.5),layout="constrained")
  a.bar(["Base","Entrada reta","Revisão temporal","Saída curva"],np.array([7,9,10,12])/17*100)
  a.axhline(80,color="red",ls="--",label="Limiar definido: 80%")
  a.set(ylim=(0,100),ylabel="Pedidos executáveis (%)",title="Desenvolvimento: 77 quadros, 17 pedidos")
  for i,n in enumerate([7,9,10,12]):a.text(i,n/17*100+1,f"{n}/17",ha="center")
  a.legend()
  for ext in ["png","svg","pdf"]:f.savefig(out/("resultados_monocular_final."+ext),dpi=180)
  plt.close(f)
 out=ROOT/"estudos_e_analises/prints";out.mkdir(exist_ok=True)
 f,a=plt.subplots(figsize=(9,3),layout="constrained");a.axis("off")
 a.text(.03,.8,"Estudo monocular encerrado — replay offline",fontsize=17)
 a.text(.03,.48,"70,59% sucesso | 1,00% falso espaço livre",fontsize=18)
 a.text(.03,.16,"77 quadros; 12/17 pedidos; zero colisões discretas nos 12 caminhos.\nDesenvolvimento em Baylands, sem qualificação para voo.",fontsize=11)
 f.savefig(out/"painel_monocular_final.png",dpi=180);plt.close(f)
 print("Final-study counts validated; figures regenerated.")


def illustrations():
 import cv2
 data=ROOT/"datasets/resultados_finais_monocular"
 with np.load(data/"sample_0.npz") as z:a={k:z[k] for k in z.files}
 with np.load(data/"sample_1.npz") as z:b={k:z[k] for k in z.files}
 def publish(fig,name):
  for folder in [ROOT/"Parte_Escrita/Monografia_TCC"]:
   if folder.is_dir() and ((folder/"main.tex").exists() or (folder/"principal.tex").exists()):
    for ext in ["png","svg","pdf"]:fig.savefig(folder/"figuras"/(name+"."+ext),dpi=170)
  plt.close(fig)
 f,ax=plt.subplots(1,3,figsize=(12,4),layout="constrained")
 ax[0].imshow(cv2.cvtColor(a["rgb_bgr"],cv2.COLOR_BGR2RGB));ax[0].set_title("RGB: quadro 1")
 im=ax[1].imshow(a["predicted_depth_m"],vmin=.5,vmax=30,cmap="magma");ax[1].set_title("Profundidade monocular (m)");f.colorbar(im,ax=ax[1])
 d=a["predicted_depth_m"][::12,::12];ys,xs=np.indices(d.shape);k=a["intrinsics"]
 x=(xs*12-k[2])*d/k[0];y=(ys*12-k[3])*d/k[1];ok=np.isfinite(d)&(d>=.5)&(d<=30)
 # Isometric projection of optical 3D coordinates, not an independent reconstruction.
 u=(x-d)*.7071;v=(x+d)*.4082-y*.8165
 ax[2].scatter(u[ok],v[ok],c=d[ok],s=1,cmap="magma",vmin=.5,vmax=30)
 ax[2].scatter([0],[0],marker="^",c="cyan",label="Câmera (origem)")
 ax[2].set(title="Retroprojeção 3D: vista isométrica",xlabel="Projeção horizontal (m)",ylabel="Projeção vertical (m)");ax[2].legend()
 publish(f,"metodologia_rgb_profundidade_3d")
 gray=cv2.cvtColor(a["rgb_bgr"],cv2.COLOR_BGR2GRAY);gray2=cv2.cvtColor(b["rgb_bgr"],cv2.COLOR_BGR2GRAY)
 p=cv2.goodFeaturesToTrack(gray,maxCorners=80,qualityLevel=.02,minDistance=12)
 q,status,_=cv2.calcOpticalFlowPyrLK(gray,gray2,p,None)
 f,ax=plt.subplots(1,2,figsize=(10,4),layout="constrained")
 ax[0].imshow(cv2.cvtColor(a["rgb_bgr"],cv2.COLOR_BGR2RGB));ax[0].set_title("Fluxo entre quadros reais 1 e 2")
 reference=a["reference_depth_m"]
 sampled=[]
 for point,valid in zip(p[:,0],status[:,0]):
  u,v=np.rint(point).astype(int)
  if valid and 0<=v<reference.shape[0] and 0<=u<reference.shape[1]:
   depth=float(reference[v,u])
   if np.isfinite(depth) and depth>0:sampled.append(depth)
 threshold=float(np.median(sampled))
 counts=[0,0]
 for aa,bb,valid in zip(p[:,0],q[:,0],status[:,0]):
  if not valid:continue
  u,v=np.rint(aa).astype(int)
  if not (0<=v<reference.shape[0] and 0<=u<reference.shape[1]):continue
  depth=float(reference[v,u])
  if not np.isfinite(depth) or depth<=0:continue
  group=int(depth>threshold);counts[group]+=1
  color=["#00e5ff","#ffe600"][group]
  ax[0].arrow(*aa,*(bb-aa),color=color,width=.25,head_width=3)
 assert all(counts),counts
 from matplotlib.lines import Line2D
 ax[0].legend(handles=[Line2D([0],[0],color="#00e5ff",label=f"Menor profundidade GT (≤ {threshold:.1f} m)"),
  Line2D([0],[0],color="#ffe600",label=f"Maior profundidade GT (> {threshold:.1f} m)")],loc="lower left",fontsize=7,facecolor="#444444",labelcolor="white")
 ax[1].axis("off")
 ax[1].text(.02,.85,"Quadro t0 → Quadro t1",fontsize=15)
 ax[1].text(.02,.6,"Sinais inerciais e movimento no intervalo\nΔt = t1 − t0\nAtributos → variações de proximidade",fontsize=12)
 ax[1].text(.02,.2,"Cores: grupos relativos de profundidade GT.\nO GT é usado somente nesta ilustração.\nFluxo depende de rotação e translação.\nVetor maior não garante objeto mais próximo.",fontsize=10)
 publish(f,"metodologia_fluxo_intervalo")
 f,ax=plt.subplots(1,2,figsize=(12,4),layout="constrained")
 ax[0].axis("off");ax[0].text(.02,.95,"MLP tabular (não é o estimador denso)",fontsize=13)
 ax[0].text(.02,.75,"Atributos padronizados\n↓ Linear: 64 → ReLU → dropout 0,30\n↓ Linear: 16 → ReLU → dropout 0,30\n↓ Linear: 6 → inversão da padronização\nSeis variações de proximidade\nDropout desativado na avaliação",fontsize=12,va="top")
 ax[1].set_title("Octree conceitual: subdivisão do volume")
 for size in [1,.5,.25]:
  for x in np.arange(0,1.001,size):ax[1].plot([x,x],[0,size],color="navy",lw=1);ax[1].plot([0,size],[x,x],color="navy",lw=1)
 ax[1].set(xlim=(-.1,1.1),ylim=(-.1,1.1),aspect="equal")
 ax[1].text(0,-.25,"Projeção didática de níveis; implementação usa grade fixa.",fontsize=9)
 publish(f,"metodologia_mlp_subdivisao")

if __name__=="__main__":
 main()
 illustrations()
 import subprocess,sys,os
 env=dict(os.environ,PYTHONNOUSERSITE="1")
 subprocess.run([sys.executable,str(ROOT/"scripts/generate_methodology_figures.py")],check=True,env=env)
