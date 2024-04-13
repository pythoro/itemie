# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:05:29 2023

@author: Reuben
"""


WORDCLOUD_MODULE = False
try:
    from wordcloud import WordCloud
    WORDCLOUD_MODULE = True
except ModuleNotFoundError:
    pass


class WordCloud:
    def generate(self, counts):
        wc = WordCloud(background_color="white",
                       max_words=1000,
                       random_state=random_state,
                       **kwargs)
        wc.generate_from_frequencies(counts)
        return wc
    
    def to_svg(self, wc):
        svg_string = wc.to_svg(embed_font=True)
        byts = svg_string.encode()
        encoded = base64.b64encode(byts)
        src = 'data:image/svg+xml;base64,{}'.format(encoded.decode())
        return src