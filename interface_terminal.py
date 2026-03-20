import time
import threading
import rclpy

class TerminalInterface(threading.Thread):
    def __init__(self, drone_node):
        super().__init__(daemon=True)
        self.drone_node = drone_node

    def run(self):
        """ Loop independente que fica esperando você digitar sem travar o voo """
        while rclpy.ok():
            if self.drone_node.pronto_para_comando:
                print("\n" + "="*50)
                comando_usuario = input(">>> DIGITE A AÇÃO [pairar, esq, dir, frente, tras, exit]: ").strip().lower()
                
                if comando_usuario == 'esq':
                    self.drone_node.comando_esquerda()
                elif comando_usuario == 'dir':
                    self.drone_node.comando_direita()
                elif comando_usuario == 'frente':
                    self.drone_node.comando_frente()
                elif comando_usuario == 'tras':
                    self.drone_node.comando_tras()
                elif comando_usuario == 'pairar':
                    self.drone_node.comando_pairar()
                elif comando_usuario == 'exit':
                    self.drone_node.comando_exit()
                    break
                else:
                    print(f"[Aviso] Comando '{comando_usuario}' não reconhecido. Digite um comando válido")
            else:
                time.sleep(0.1)