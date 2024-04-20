
import re

import Levenshtein



def curr_false(text, data_lists, max_threshold=0.4):
    search_ch_text = re.compile(r'[\u4e00-\u9fff]')
    search_brackets_text = re.compile(r'[()\u3000\uFF08\uFF09\u3010\u3011]')
    # search_num_text = re.compile(r'\d')
    if search_ch_text.search(text):
        if text in data_lists:
            return text
        # while searching number，you should do some improvements.
        text = re.sub(r'\d+[^0-9]*$', '', text)
        filtered_data_lists = data_lists.copy()
        if search_brackets_text.search(text):
            filtered_data_lists = [item for item in filtered_data_lists if search_brackets_text.search(item[0])]
        similarities = [Levenshtein.ratio(text, str2[0]) for str2 in filtered_data_lists]
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        most_similar_drug = filtered_data_lists[max_index][0]
        # if search_num_text.search(text):
        #     print(text,most_similar_drug)
        # print(max_similarity, most_similar_drug, text)
        if max_similarity > max_threshold:
            return most_similar_drug