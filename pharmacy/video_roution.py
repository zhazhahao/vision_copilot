import time
import cv2
from modules.camera_processor import CameraProcessor

if __name__ == "__main__":
    camera = CameraProcessor(doaudio=False)
    camera.start()
    i = 0
    while(True):
        try:
            bools,mat = camera.achieve_image()
            scale_percent = 50  # 设置缩放比例
            width = int(mat.shape[1] * scale_percent / 200)
            height = int(mat.shape[0] * scale_percent / 200)
            dim = (width, height)
            resized_image = cv2.resize(mat, dim, interpolation=cv2.INTER_AREA)
            if bools:
                cv2.imwrite("/home/portable-00/VisionCopilot/pharmacy/images/"+str(i)+".jpg",mat)
                cv2.imshow("win_name",resized_image)
                i = i+1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except KeyboardInterrupt:
            print("KeyBoardInterruption")
            break
        except Exception as e:
            continue
    camera.end_process()
    
    
    
    
    
    
    Counter({'壹丽安注射剂(注射用艾普拉唑钠)': 81, '利奈唑胺葡萄糖注射液': 81, '盐酸昂丹司琼注射液': 74, '奥诺先(注射用右雷佐生)': 71, '灭菌注射用水': 63, '肝素钠注射液': 55, '特比澳(重组人血小板生成素注射液)': 53, '注射用亚叶酸钙': 39, '绿汀诺(注射用谷胱甘肽)': 37, '脂肪乳氨基酸(17)葡萄糖(11%)注': 35, '丽泉(托拉塞米注射液)': 34, '多烯磷脂酰胆碱注射液': 31, '肝素钠封管注射液': 28, '同奥(亚叶酸钙注射液)': 24, '盐酸氨溴索注射液': 23, '注射用艾司奥美拉唑钠': 23, '氯化钠注射液[塑料安瓿]': 21, '氯化钾注射液[塑料安瓿]': 20, '醋酸奥曲肽注射液': 19, '氯化钙注射液': 14, '盐酸精氨酸注射液': 9, '可乐必妥注射剂(左氧氟沙星氯化钠注射液)': 8, '浓氯化钠注射液': 8, '呋塞米注射液': 7, '硫辛酸注射液': 6, '甲强龙(注射用甲泼尼龙琥珀酸钠)': 4, '澳能(卤米松乳膏)': 4, '硫酸阿托品注射液': 2, '注射用奥美拉唑钠': 2, '蔗糖铁注射液': 2, '奥硝唑注射液': 1, '碳酸氢钠注射液': 1, '曲安奈德注射液': 1, '盐酸纳洛酮注射液': 1, '复方曲肽注射液': 1, '注射用甲泼尼龙琥珀酸钠': 1, '甲钴胺注射液': 1, '注射用硝普钠': 1, '垂体后叶注射液': 1})