"""Full controller stabilization logic on recorded RGB and measured PX4 attitude/gyro."""
from pathlib import Path
import ast,json,math,hashlib,types
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets/resultados_finais_monocular/estabilizacao_completa"
def euler(q):
 w,x,y,z=q
 return math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(np.clip(2*(w*y-z*x),-1,1)),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def controller_methods():
 tree=ast.parse((ROOT/"drone_controller.py").read_text())
 names={"normalizar_angulo_rad","calcular_delta_pan_imu","montar_homografia_compensacao_imu"}
 funcs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(funcs)==3
 namespace=dict(np=np,math=math)
 exec(compile(ast.Module(body=funcs,type_ignores=[]),"controller_stabilization","exec"),namespace)
 return {name:namespace[name] for name in names}
def make_controller(k,parameters):
 state=types.SimpleNamespace(K=k,compensar_pan_yaw=True,compensar_tilt_roll=True,
  stabilization_output_width=640,stabilization_output_height=480,prev_stabilization_stamp_s=None,
  prev_stabilization_yaw=None,**parameters)
 for name,method in controller_methods().items():setattr(state,name,types.MethodType(method,state))
 state.image_timestamp_s=lambda message:message
 return state
def main():
 m=json.loads((DATA/"medicoes.json").read_text());audit=[]
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
 for i,row in enumerate(m["samples"]):
  with np.load(DATA/"rgb_registrado.npz") as z:
   image=z[f"rgb_{i}"];intr=z[f"k_{i}"]
  native=cv2.resize(image,(1280,960),interpolation=cv2.INTER_LINEAR)
  fx,fy,cx,cy=intr*4;k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  state=make_controller(k,m["parameters"])
  state.current_roll,state.current_pitch,state.current_yaw_attitude=euler(row["q_wxyz"])
  state.current_gyro_rad_s=np.array(row["gyro_rad_s"])
  if i:
   previous=m["samples"][i-1];state.prev_stabilization_stamp_s=previous["timestamp_s"]
   state.prev_stabilization_yaw=euler(previous["q_wxyz"])[2]
  h,pan,source=state.montar_homografia_compensacao_imu(row["timestamp_s"])
  original_interval=None if i==0 else row["timestamp_s"]-m["samples"][i-1]["timestamp_s"]
  actual=cv2.warpPerspective(native,h,(640,480),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  state.prev_stabilization_stamp_s=row["sensor_context_timestamp_s"]
  state.prev_stabilization_yaw=euler(row["sensor_context_q_wxyz"])[2]
  mixedh,mixedpan,mixedsource=state.montar_homografia_compensacao_imu(row["timestamp_s"])
  assert mixedsource=="attitude+gyro"
  mixed=cv2.warpPerspective(native,mixedh,(640,480),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  panels=[(image,f"RGB registrado: quadro {i+1}"),
    (actual,f"Reaplicação nos quadros salvos\npan: {source}"),
    (mixed,"Ramo misto: contexto sensorial de 40 ms\natitude + giroscópio medidos; ilustrativo")]
  for ax,(im,title) in zip(axes[i],panels):
   ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(title,fontsize=10);ax.axis("off")
  audit.append(dict(frame=row["frame"],timestamp_s=row["timestamp_s"],saved_rgb_interval_s=original_interval,
    roll_rad=state.current_roll,pitch_rad=state.current_pitch,measured_gyro_z_rad_s=float(state.current_gyro_rad_s[2]),
    saved_sequence_source=source,saved_sequence_pan_rad=pan,mixed_example_source=mixedsource,mixed_example_pan_rad=mixedpan,
    mixed_example_dt_s=row["timestamp_s"]-row["sensor_context_timestamp_s"]))
 fig.suptitle("Algoritmo completo do controlador; o contexto de 40 ms não é um intervalo RGB histórico recuperado",fontsize=12)
 for name in ["Monografia_Metodologia_Professor","ModeloTCC_Monografia_CC_Latex","ModeloTCC_Artigo_CC_Latex"]:
  out=ROOT/"Parte_Escrita"/name/"figuras"
  for ext in ["png","svg","pdf"]:fig.savefig(out/("metodologia_compensacao_rotacao."+ext),dpi=200,bbox_inches="tight")
 plt.close(fig)
 result=dict(controller_source_sha256=hashlib.sha256((ROOT/"drone_controller.py").read_bytes()).hexdigest(),
  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),uses_original_method_bodies=True,
  reconstructs_original_callback_order=False,measurements=m,frames=audit)

 print(json.dumps(audit))
if __name__=="__main__":main()
