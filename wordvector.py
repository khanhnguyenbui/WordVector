# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import io
from itertools import islice
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i=0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = tokens[1:]
        i += 1
        if i==20000:
            break
    return data, n, d

data, n, d = load_vectors('E:\Datasets\cc.en.300.vec\cc.en.300.vec')

def read_each_line(path):
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #fp = open("file")
    for i, line in enumerate(fin):
        if i == 1:
            n, d = map(int, fin.readline().split())
        else:
            tokens = line.rstrip().split(' ')
            
            
    fin.close()
    
read_each_line('E:\Datasets\cc.en.300.vec\cc.en.300.vec')

def read_each_line2(path):
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #fp = open("file")
    lines = list(islice(fin,1000000,1000001))
    print(lines)
    fin.close()
    
read_each_line2('E:\Datasets\cc.en.300.vec\cc.en.300.vec')
    
    

def buildingDictOfAWord(inputWord):
    dictOfAWord = {}
    for word in data:
        difference = 0
        for i in range(0,300):
            a =  (float(data[word][i]) - float(data[inputWord][i]))**2
            difference += a
        dictOfAWord[word] = difference
    return dictOfAWord

dict1 = buildingDictOfAWord('Dog')

dict2 = buildingDictOfAWord('Cat')
dict3 = buildingDictOfAWord('Horse')
dict4 = buildingDictOfAWord('Bird')
dict5 = buildingDictOfAWord('Bear')

# https://www.nytimes.com/2018/10/29/well/family/how-children-learn-to-recognize-faces.html?action=click&module=Discovery&pgtype=Homepage
sentence = "“I’m very, very good at recognizing faces,” said Kang Lee, a professor of applied psychology and human development at the University of Toronto, who studies the development of facial recognition skills in children. If he has met a person once, he said, he will recognize that person two or three years later. “One encounter for me is sufficient, my brain has encoded it.” But young children take years to master this skill. And this is a holiday season when we mess around with faces. For those just developing the ability to recognize faces, Halloween masks, costumes, fake noses, false beards, wigs and elaborate makeup present special challenges. Just as babies can be baffled and delighted by peek-a-boo games, as they learn that people can disappear and come back, they can also be truly confused by the transformation of a masquerade. We’re playing here with complicated neurological systems, with effects on the brain that combine nature and nurture, and, perhaps inevitably, with social and cultural overtones. Research in this field has helped tease out what those skills are that make some of us very good at distinguishing and remembering faces, and make others of us, well, kind of borderline — and I speak here as someone who relies heavily on the nametags which are, thank goodness, required in medical settings, even to confirm I am getting the names right for people I have worked with for a while. Dr. Lee’s ability to recognize and remember, he said, is based in being “excellent at picking up structure.” There are two different kinds of visual information that people use to recognize faces, Dr. Lee said. Young children begin by relying on feature information, like the size of the eyes, the size and shape of the nose, the presence and color of a beard. But as children mature, they learn to use what he called “configural information,” which reflects that underlying structure. Configural information would include the distance between the features, and their relations to the contour of the face. “We have a superior ability to detect minor configural differences,” Dr. Lee said, such as very slight differences in the distance between the eyes. “This ability is amazing. It also allows us to recognize a person we haven’t seen for 20 or 30 years when we go back to a reunion.” Those underlying structural details don’t change very much as people age, which is what makes it possible to recognize someone in a baby picture — at least for those of you who can recognize someone in a baby picture.But small children depend on features, Dr. Lee said, and that’s why they may fail to recognize even a parent who has shaved his beard, or is wearing a very unfamiliar hat. Preschoolers, he said, may have difficulty if someone puts on — or takes off — eyeglasses. Starting in infancy, they are also using configural information, and as children grow and mature, they get better at it, using the underlying structures in recognizing faces, moving beyond a reliance on features. Not till they are around 14 to 16 years old, though, Dr. Lee said, do they reach adult levels of skill. Sarah Gaither, an assistant professor of psychology and neuroscience at Duke University, said that as children grow up in their social contexts, face perception also reflects racial and social experiences. In these studies, first the baby is habituated by being shown a face over and over and over, then offered another face from the same racial or gender group; using devices that track the baby’s eye movements, researchers can see whether the baby perceives the new face as recognizably different. “By the first 3 months, we’ve already seen infants make pretty strong racial and ethnic preferences according to exposure,” she said. Babies get better at distinguishing faces in their own racial groups, less good with other groups, Dr. Gaither said. “Most research argues that this other-race effect comes online between 3 and 6 months of life, but it’s linked to exposure,” she said. Her own research has shown that biracial infants, growing up with exposure to both Asian and white faces, are faster at scanning faces by 3 months than children growing up with only one such group. Other research has suggested that by school age, children’s ability to recognize and remember faces of those in other racial groups is associated with their understanding of race. For children growing up with varied exposure to varied faces, she said, such as transracially adopted children in diverse neighborhoods, “they’re better at recognizing faces within racial out-groups.” Dr. Gaither said this topic can take on political overtones around Halloween, with heightened sensitivities about cultural appropriation. It may raise special questions this year, she said, thanks to the popularity of “Black Panther” and the question of how children can dress as a hero from another group without offending the members of that group by anything that would suggest appropriation or something as overtly offensive as blackface. So thinking about recognizing and altering faces tells us a lot about how we perceive others — and ourselves. On Halloween, Dr. Lee said, a child might fail to recognize a somewhat familiar adult who has on makeup or a false beard, “but adult neighbors will recognize me, they can see through makeup, see through my beard, they know the structure of my face, that’s Kang, that’s not someone else.” Face perception definitely has a genetic component, Dr. Lee said, though exposure is also critical; children who don’t see well during their first two years of life — who have congenital cataracts, for example, which are later removed — may not be able to recover the ability later in life. There is a wide range of normal abilities in recognizing faces, and training is of somewhat limited value, Dr. Lee said (there is also a disorder — prosopagnosia, or “face-blindness — in which people cannot recognize faces, sometimes even of those they know well). So go ahead, take the Cambridge Face Memory Test, and find out how you do. I’m not going to tell you how I did, lest I undermine whatever tenuous authority I may have to write about this subject (I do think I might have done better if there had been some female and nonwhite faces on the test — but maybe I’m just kidding myself). And if you have a small child or cross paths with some on Halloween, remember that in their eyes, the transformation of familiar people by masks and makeup may seem pretty tricky — even if there are treats involved."
sentence = [sentence]

dict1 = {}
for word in ['Horse', 'Cat', 'Bird', 'Bear']:
    dict2 = buildingDictOfAWord(word)
    for element in dict2:
        if element in dict1:
            dict1[element] += dict2[element] 
        else:
            dict1[element] = dict2[element]
    print(word)
    
    

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                         for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return " ".join([feature_names[i]
                         for i in topic.argsort()[:-no_top_words - 1:-1]])

no_features = 1000
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(sentence)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()



no_topics = 1
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
no_top_words = 20


topic = display_topics(nmf, tfidf_feature_names, no_top_words).split()

dict1 = {}
for word in topic:
    try:
        dict2 = buildingDictOfAWord(word)
        for element in dict2:
            if element in dict1:
                dict1[element] += dict2[element] 
            else:
                dict1[element] = dict2[element]
    except:
        print("no word: " + word)
    print(word)


# =============================================================================
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.decomposition import NMF, LatentDirichletAllocation
# 
# def display_topics(model, feature_names, no_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         print("Topic %d:" % (topic_idx))
#         print(" ".join([feature_names[i]
#                         for i in topic.argsort()[:-no_top_words - 1:-1]]))
# 
# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# documents = dataset.data
# 
# no_features = 1000
# 
# # NMF is able to use tf-idf
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
# tfidf = tfidf_vectorizer.fit_transform(documents)
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# 
# # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
# tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
# tf = tf_vectorizer.fit_transform(documents)
# tf_feature_names = tf_vectorizer.get_feature_names()
# 
# no_topics = 20
# 
# # Run NMF
# nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
# 
# # Run LDA
# lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
# 
# no_top_words = 10
# display_topics(nmf, tfidf_feature_names, no_top_words)
# display_topics(lda, tf_feature_names, no_top_words)
# =============================================================================
