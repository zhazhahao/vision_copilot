
import re
from Levenshtein import distance

import Levenshtein



def curr_false(text, data_lists, max_threshold=0.4):
    search_ch_text = re.compile(r'[\u4e00-\u9fff]')
    search_brackets_text = re.compile(r'[()\[\]\u3000\uFF08\uFF09\u3010\u3011]')
    before_keywords = ["集采", "预灌封", "预充式", "预充装","小" ,r"\[N-R\]", r"\[N-N\]", r"\[30R\]", r"\[50R\]", r"笔芯\[Y-R\]", r"\(笔芯\)",r"\(水果味\)", r"笔芯",r"预充",r"预装式",r"预冲式",r"预填充"]
    unit_keywords = ["单位", "万单位", "ml", "mg", "g", "IU", "WU" ,"UG", "mgI","ML","喷","微克","吸","粒","μg","揿","揿×","mg:","g:","喷\*"]
    keyword_pattern = "|".join(before_keywords)
    unit_pattern = "|".join(unit_keywords)
    pattern = rf"^(.+?)\s*(?:{keyword_pattern})?\s*(((\d+(\.\d+)?)%?[a-zA-Z%μ]*\s*)+({unit_pattern})((\d+(\.\d+)?\s*)+({unit_pattern})\s*)*x?\d*\s*)/\s*(\S+)$"
    # search_num_text = re.compile(r'\d')
    if search_ch_text.search(text):
        if text in data_lists:
            return text
        # while searching number，you should do some improvements.
        text = re.sub(r'\d+[^0-9]*$', '', text)
        filtered_data_lists = data_lists.copy()
        match = re.search(pattern, text)
        # print(text)
        if match:
            print("药品名称:", match.group(1))
            print("剂量:", text[match.end(1):])
        if search_brackets_text.search(text):
            filtered_data_lists = [item for item in filtered_data_lists if search_brackets_text.search(item[0])]
        text = text.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
        similarities = [Levenshtein.ratio(text, str2[0]) for str2 in filtered_data_lists]
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        most_similar_drug = filtered_data_lists[max_index][0]
        if max_similarity > max_threshold:
            return most_similar_drug
        
def group_similar_strings(strings, counter):
    groups = []
    for string in strings:
        # Check if the string can be added to any existing group
        added = False
        for group in groups:
            for s in group:
                common_chars = set(string) & set(s)
                if len(common_chars) > 6:
                    group.append(string)
                    added = True
                    break
            if added:
                break
        if not added:
            groups.append([string])

    # Filter out elements with smaller counts based on the given counter
    filtered_groups = []
    for group in groups:
        filtered_group = []
        max_num = max([counter[element]["counts"] for idx,element in enumerate(group)])
        for idx, element in enumerate(group):
            if counter[element]["counts"]/max_num >= 0.2:  # Adjust the threshold as needed
                filtered_group.append(element)
        if filtered_group:
            filtered_groups.append(filtered_group)
    return filtered_groups