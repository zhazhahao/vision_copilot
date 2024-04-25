import time
import cv2
from modules.camera_processor import CameraProcessor

if __name__ == "__main__":
    camera = CameraProcessor()
    camera.start()
    i = 0
    a = time.time()
    while(True):
        try:
            bools,mat = camera.achieve_image()
            scale_percent = 50  # 设置缩放比例
            width = int(mat.shape[1] * scale_percent / 200)
            height = int(mat.shape[0] * scale_percent / 200)
            dim = (width, height)
            resized_image = cv2.resize(mat, dim, interpolation=cv2.INTER_AREA)
            if bools:
                # cv2.imwrite("/home/portable-00/VisionCopilot/pharmacy/images/"+str(i)+".jpg",mat)
                # cv2.imshow("win_name",resized_image)
                i = i+1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except KeyboardInterrupt:
            print("KeyBoardInterruption")
            break
        except Exception as e:
            continue
    camera.end_process()
    
