import os
import sys
    
    
class stdrefresh:
    def __init__(self, prescription:list, hand_output:str, medicine_output:str):
        self.prescription = prescription
        self.hand_output = hand_output
        self.medicine_output = medicine_output
        
        super().__init__()
        
    def static_refresh(self, prescription, hand_output, medicine_output):
            # 将处方列表转换为字符串，每个药品名称占一行
            prescription_str = "处方:[ "  +  ",".join(prescription)  +  " ]"
            os.system('cls' if os.name == 'nt' else 'clear')
            # 构建最终的静态刷新字符串
            refresh_string = (f"{prescription_str}" + "\n"
                            f"{hand_output}" + "\n"
                            f"{medicine_output}" + "\n")
            while True:
                sys.stdout.write("\x1b[H")
                # 然后写入新的输出内容
                sys.stdout.write(refresh_string)
                sys.stdout.flush()

if __name__ == "__main__":
    prescription = ["曲前列尼尔注射液", "速碧林(那屈肝素钙注射液)", "依诺肝素钠注射液", "注射用青霉素钠", "注射用头孢唑林钠"]
    hand_output = "0: Bbox [1, 1, 1, 1]"
    medicine_output = "123: Category 一二三"
    test = stdrefresh(prescription,hand_output,medicine_output)
    test.static_refresh(prescription,hand_output,medicine_output)