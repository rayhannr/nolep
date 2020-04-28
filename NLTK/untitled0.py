import nltk
nltk.download('words')
nltk.pos_tag('Machine Learning is fucking boring'.split())

sentence = 'Wayne Rooney was only walking through your neighborhood on March 14, 2020'
tags = nltk.pos_tag(sentence.split())
chunk = nltk.ne_chunk(tags).draw()

sent = 'Steve Jobs was a CEO of Apple Corp.'
tag = nltk.pos_tag(sent.split())
nltk.ne_chunk(tag).draw()