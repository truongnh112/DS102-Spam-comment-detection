
import regex as re
from underthesea import word_tokenize

import unidecode


#Bỏ dấu
def remove_accent(text):
    return unidecode.unidecode(text)

"""
    Start section: Chuyển câu văn về kiểu gõ telex khi không bật Unikey
    Ví dụ: thủy = thuyr, tượng = tuwowngj
"""

uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
#bang nguyen am
vowel_board = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']


vowel_to_ids = {}

def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

def convert_unicode(txt):
    dicchar = loaddicchar()
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

#chuan hoa dau cau
def vietnamese_word_punctuate_normalization(word):
    
    if not is_valid_vietnam_word(word):
        return word
 
    chars = list(word)
    dau_cau = 0
    vowel_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = vowel_board[x][0]
        if not qu_or_gi or index != 1:
            vowel_index.append(index)
    if len(vowel_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = vowel_to_ids.get(chars[1])
                chars[1] = vowel_board[x][dau_cau]
            else:
                x, y = vowel_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowel_board[x][dau_cau]
                else:
                    chars[1] = vowel_board[5][dau_cau] if chars[1] == 'i' else vowel_board[9][dau_cau]
            return ''.join(chars)
        return word
 
    for index in vowel_index:
        x, y = vowel_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowel_board[x][dau_cau]
            # for index2 in vowel_index:
            #     if index2 != index:
            #         x, y = vowel_to_ids[chars[index]]
            #         chars[index2] = vowel_board[x][0]
            return ''.join(chars)

    if len(vowel_index) == 2:
        if vowel_index[-1] == len(chars) - 1:
            x, y = vowel_to_ids[chars[vowel_index[0]]]
            chars[vowel_index[0]] = vowel_board[x][dau_cau]
            # x, y = vowel_to_ids[chars[vowel_index[1]]]
            # chars[vowel_index[1]] = vowel_board[x][0]
        else:
            # x, y = vowel_to_ids[chars[vowel_index[0]]]
            # chars[vowel_index[0]] = vowel_board[x][0]
            x, y = vowel_to_ids[chars[vowel_index[1]]]
            chars[vowel_index[1]] = vowel_board[x][dau_cau]
    else:
        # x, y = vowel_to_ids[chars[vowel_index[0]]]
        # chars[vowel_index[0]] = vowel_board[x][0]
        x, y = vowel_to_ids[chars[vowel_index[1]]]
        chars[vowel_index[1]] = vowel_board[x][dau_cau]
        # x, y = vowel_to_ids[chars[vowel_index[2]]]
        # chars[vowel_index[2]] = vowel_board[x][0]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    vowel_index = -1
    for index, char in enumerate(chars):
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x != -1:
            if vowel_index == -1:
                vowel_index = index
            else:
                if index - vowel_index != 1:
                    return False
                vowel_index = index
    return True


def vietnamese_punctuation_normalization(sentence):
    """
        Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
        :param sentence:
        :return:
        """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = vietnamese_word_punctuate_normalization(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)
 
def read_stopwords():
    f = open("stopwords.txt", "r")
    res = f.readlines()
    return res
def remove_stopwords(line):
    stopword = read_stopwords()
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)
 

        
 

# def read_data(csv_file_path):
#     f = open(csv_file_path)
#     csv_reader = csv.reader(f, delimiter=',')
#     line_count = 0
#     raw_sentence = []
#     raw_rating = []
#     raw_label = []
#     for row in csv_reader:
#         if line_count == 0:
#             line_count += 1
#             continue
#         raw_rating.append(row[1])
#         raw_sentence.append(row[2])
#         raw_label.append(row[3])
#         line_count += 1
#     f.close()
#     return  raw_rating, raw_sentence, raw_label



def text_preprocess(document):
    document = convert_unicode(document)
    document = vietnamese_punctuation_normalization(document)
    document = word_tokenize(document, format="text")
    document = document.lower()
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
    document = re.sub(r'\s+', ' ', document).strip()
    document = remove_stopwords(document)
#    document = remove_accent(document)
    return document


#raw_document, _ , _ = read_data("./data/CSV/main_data.csv")
#print(text_preprocess(raw_document[30]))