
import re

import Levenshtein


def curr_false(text,data_lists,max_threshold = 0.4):
    search_ch_text = re.compile(r'[\u4e00-\u9fff]')
    if search_ch_text.search(text):
        if text in data_lists:
            return text
        similarities = [Levenshtein.ratio(text, str2[0]) for str2 in data_lists]
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        most_similar_drug = data_lists[max_index][0]
        # print(max_similarity,most_similar_drug,text)
        if max_similarity > max_threshold:
            return most_similar_drug