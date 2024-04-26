from qinglang.data_structure.video.video_toolbox import VideoToolbox
from qinglang.data_structure.image.image_toolbox import ImageToolbox, ImageFlow


itb = ImageToolbox(ImageFlow("/home/portable-00/VisionCopilot/pharmacy/work_dirs/20240426-150401/images"))
itb.to_video()