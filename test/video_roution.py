import cv2
from modules.camera_processor import CameraProcessor


if __name__ == "__main__":
    camera = CameraProcessor()
    camera.start()
    while(True):
        try:
            bools,mat = camera.achieve_image()
            if bools:
                cv2.imshow("hello",mat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            print("KeyBoardInterruption")
            break
        except Exception as e:
            break
    camera.end_process()
    
    
    
    