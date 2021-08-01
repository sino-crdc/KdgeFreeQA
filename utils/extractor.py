# -*- coding: utf-8 -*-
import itertools
from utils.triple import Triple
from utils.triple import Relation
from utils.triple import Entity
from ltp import LTP

'pyltp is a suck'
# from pyltp import Segmentor
# from pyltp import Postagger
# from pyltp import NamedEntityRecognizer
# from pyltp import Parser
# from pyltp import SementicRoleLabeller
# from pyltp import SentenceSplitter


class Extractor():

    def __init__(self):
        '''
        Seq2Graph : sequence to graph based on dependency syntactic parsing

        heavily borrowed from zhopenie : https://github.com/tim5go/zhopenie

        but we have modified it : replace pyltp by ltp, some modifications on details.

        this is used to convert data of textual form into graph form, such as law, background span, question, or add answer.

        example : '星展集团是亚洲最大的金融服务集团之一, 拥有约3千5百亿美元资产和超过280间分行, 业务遍及18个市场。' -->
        ['e1: 星展集团, e2: 亚洲最大的金融服务集团之一, r: 是']
        '''
        self.__clause_list = []
        self.__subclause_dict = {}
        self.__triple_list = []
        # self.__segmentor = Segmentor()
        # self.__postagger = Postagger()
        # self.__recognizer = NamedEntityRecognizer()
        # self.__parser = Parser()
        # self.__labeller = SementicRoleLabeller()
        self.ltp = LTP('small')
        self.__words_full_list = []
        self.__netags_full_list = []

    @property
    def clause_list(self):
        return self.__clause_list

    @property
    def triple_list(self):
        return self.__triple_list

    def split(self, words, postags):
        start = 0
        for j, w in enumerate(words):
            if w == ',' or w == '，' or w == '。':
                clause = Clause(start, j - 1)
                self.__clause_list.append(clause)
                start = j + 1

        for clause in self.__clause_list:
            clause.split(postags)
            for subclause in clause.sub_clause_list:
                self.add_inverted_idx(subclause)

    def add_inverted_idx(self, subclause):
        for i in range(subclause.start_idx, subclause.end_idx):
            self.__subclause_dict[i] = subclause

    # def load(self):
    #     self.__segmentor.load('ltp_data/cws.model')
    #     self.__postagger.load('ltp_data/pos.model')
    #     self.__recognizer.load('ltp_data/ner.model')
    #     self.__parser.load('ltp_data/parser.model')
    #     self.__labeller.load('ltp_data/srl')
    #
    # def release(self):
    #     self.__segmentor.release()
    #     self.__postagger.release()
    #     self.__recognizer.release()
    #     self.__parser.release()
    #     self.__labeller.release()

    def clear(self):
        self.__triple_list = []
        self.__words_full_list = []
        self.__netags_full_list = []

    def resolve_conference(self, entity):

        try:
            e_str = entity.get_content_as_str()
        except Exception:
            return '?'
        ref = e_str
        if e_str == '他' or e_str == '她':
            for i in range(entity.loc, -1, -1):
                if self.__netags_full_list[i].lower().endswith('nh'):
                    ref = self.__words_full_list[i]
                    break
        return ref

    def resolve_all_conference(self):
        '''
         pronoun substitution
         '''
        for t in self.triple_list:
            e_str = self.resolve_conference(t.entity_1)
            try:
                t.entity_1.content = e_str.split()
            except Exception:
                pass

    def chunk_str(self, data):
        '''
        generate graph.

        :param data: string list ['xxx'] or pure string 'xxx'

        '''
        # sents = SentenceSplitter.split(data)
        sents = []
        if data is list:
            sents = self.ltp.sent_split(data)
        else:
            sents = self.ltp.sent_split([data])
        offset = 0
        for sent in sents:
            try:
                # words = self.__segmentor.segment(sent)
                # postags = self.__postagger.postag(words)
                # netags = self.__recognizer.recognize(words, postags)
                # arcs = self.__parser.parse(words, postags)
                # roles = self.__labeller.label(words, postags, netags, arcs)
                sentences, hidden = self.ltp.seg([sent])
                words = sentences[0]
                postags = self.ltp.pos(hidden)[0]
                netags = self.ltp.ner(hidden)[0]
                arcs = self.ltp.dep(hidden)[0]
                roles = self.ltp.srl(hidden)[0]
                self.chunk_sent(list(words), list(postags), list(arcs), offset)
                offset += len(list(words))
                self.__words_full_list.extend(list(words))
                self.__netags_full_list.extend(list(netags))
            except Exception as e:
                print(str(e))
                pass

    def chunk_sent(self, words, postags, arcs, offset):
        root = [j + 1 for j in [i for i, x in enumerate(arcs) if x[2] == 'HED']]
        if len(root) > 1:
            raise Exception('More than 1 HEAD arc is detected!')
        root = root[0]
        relations = [j + 1 for j in [i for i, x in enumerate(arcs) if x[1] == root and x[2] == 'COO']]
        relations.insert(0, root)

        prev_e1 = None
        e1 = None
        for rel in relations:

            left_arc = [j + 1 for j in [i for i, x in enumerate(arcs) if x[1] == rel and x[2] == 'SBV']]

            if len(left_arc) > 1:
                pass
                # raise Exception('More than 1 left arc is detected!')
            elif len(left_arc) == 0:
                e1 = prev_e1
            elif len(left_arc) == 1:
                left_arc = left_arc[0]
                leftmost = find_farthest_att(arcs, left_arc)
                e1 = Entity(1, [words[i-1] for i in range(leftmost , left_arc + 1)], offset + leftmost)

            prev_e1 = e1

            right_arc = [j + 1 for j in [i for i, x in enumerate(arcs) if x[1] == rel and x[2] == 'VOB']]

            e2_list = []
            if not right_arc:
                e2 = Entity(2, None)
                e2_list.append(e2)
            else:
                right_ext = find_farthest_vob(arcs, right_arc[0])

                items = [j + 1 for j in [i for i, x in enumerate(arcs) if x[1] == right_ext and x[2] == 'COO']]
                items = right_arc + items

                count = 0
                for item in items:
                    leftmost = find_farthest_att(arcs, item)

                    e2 = None

                    if count == 0:
                        e2 = Entity(2, [words[i-1] for i in range(leftmost, right_ext + 1)], offset + leftmost)
                    else:
                        # p1 = range(leftmost, right_arc[0])
                        p1 = range(leftmost, item)
                        p2 = range(item, find_farthest_vob(arcs, item) + 1)
                        e2 = Entity(2, [words[i-1] for i in itertools.chain(p1, p2)])

                    e2_list.append(e2)
                    r = Relation(words[rel-1])
                    t = Triple(e1, e2, r)
                    self.__triple_list.append(t)
                    count += 1


def find_farthest_att(arcs, loc):
    if loc == 22:
        print('ok')
    att = [j + 1 for j in [i for i, x in enumerate(arcs) if x[1] == loc and (x[2] == 'ATT' or x[2] == 'SBV' or x[2] == 'ADV')]] # todo : add more dep type
    if not att:
        return loc
    else:
        return find_farthest_att(arcs, min(att))


def find_farthest_vob(arcs, loc):
    vob = [j + 1 for j in [i for i, x in enumerate(arcs) if x[1] == loc and (x[2] == 'VOB')]]
    if not vob:
        return loc
    else:
        return find_farthest_vob(arcs, max(vob))


class Clause(object):

    def __init__(self, start=0, end=0):
        self.start_idx = start
        self.end_idx = end
        self.__sub_clause_list = []

    @property
    def sub_clause_list(self):
        return self.__sub_clause_list

    def __str__(self):
        return '{} {}'.format(self.start_idx, self.end_idx)

    def split(self, postags):
        start = self.start_idx
        for k, pos in enumerate(postags):
            if k in range(self.start_idx, self.end_idx + 1):
                if pos == 'c':
                    subclause = SubClause(start, k - 1)
                    self.__sub_clause_list.append(subclause)
                    start = k + 1


class SubClause():

    def __init__(self, start=0, end=0):
        self.start_idx = start
        self.end_idx = end


if __name__ == '__main__':
    # test = '星展集团是亚洲最大的金融服务集团之一, 拥有约3千5百亿美元资产和超过280间分行, 业务遍及18个市场。'
    # test = '女子看见岛村绷着脸不说话, 默默地站起身来出去。' # todo : extend function find_farthest_vob, add a few more syntax in addition to VOB
    # test = '李某在逃亡的第五天还曾教唆一个15岁的男少年抢劫他人财产1200元；帮助他人运输毒品30克，获得运输费150元。'
    test = '在公共场合，故意以焚烧、毁损、涂划、玷污、践踏等方式侮辱中华人民共和国国旗、国徽的，处三年以下有期徒刑、拘役、管制或者剥夺政治权利。'
    extractor = Extractor()
    extractor.chunk_str(test)
    extractor.resolve_all_conference()
    print("Triple: ")
    print('\n'.join(str(p) for p in extractor.triple_list))
