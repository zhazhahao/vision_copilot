def is_jpeg_header(image_data, header=b'\xFF\xD8\xFF'):
    # 检查图像数据的开头是否与 JPEG 头部匹配
    first_bytes = image_data[:len(header)]
    return first_bytes == header

def is_jpeg_end(image_data, end_bytes=b'\xFF\xD9'):
    # 检查图像数据的最后几个字节是否与 JPEG 结尾标记匹配
    last_bytes = image_data[-len(end_bytes):]
    return last_bytes == end_bytes

def has_jpeg_header(image_data, header=b'\xFF\xD8\xFF'):
    # 检查图像数据的开头是否与 JPEG 头部匹配
    start_pos = image_data.find(header)
    return start_pos != -1, start_pos + len(header)

def has_jpeg_end(image_data, end=b'\xFF\xD9'):
    # 从后往前搜索图像数据，找到是否存在 JPEG 结尾标识符
    end_pos = image_data.rfind(end)
    return end_pos != -1, end_pos + len(end)