import time
import cv2
from modules.camera_processor import CameraProcessor


if __name__ == "__main__":
    camera = CameraProcessor()
    camera.start()
    a = time.time()
    print(a)
    while(True):
        try:
            bools,mat = camera.achieve_image()
            scale_percent = 50  # 设置缩放比例
            width = int(mat.shape[1] * scale_percent / 400)
            height = int(mat.shape[0] * scale_percent / 400)
            dim = (width, height)
            resized_image = cv2.resize(mat, dim, interpolation=cv2.INTER_AREA)
            if bools:
                cv2.imshow("win_name",resized_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except KeyboardInterrupt:
            print("KeyBoardInterruption")
            break
        except Exception as e:
            continue
    camera.end_process()
    
    
    
    