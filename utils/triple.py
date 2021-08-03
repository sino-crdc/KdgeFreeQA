# -*- coding: utf-8 -*-
class Entity():

    def __init__(self, type=None, content=[], loc=0):
        '''
        entity

        :param type: pos type
        :param content: entity content
        :param loc: the number of the words in the sentence
        '''
        self.type = type
        self.content = content
        self.loc = loc

    def get_content_as_str(self):
        '''
        convert entity to string

        :return: string
        '''
        return ''.join(self.content)

    def __str__(self):
        return 'type:{}, content:{}'.format(self.type, self.content)


class Relation():

    def __init__(self, content=[]):
        '''
        the relation between two entities

        :param content: relation content
        '''
        self.content = content

    def get_content_as_str(self):
        '''
        convert relation to string

        :return: string
        '''
        return ''.join(self.content)

    def __str__(self):
        return 'content:{}'.format(self.content)

class Triple():

    def __init__(self, entity_1=Entity(), entity_2=Entity(), relation=Relation()):
        '''
        triple unit in a graph

        :param entity_1: e1
        :param entity_2: e2
        :param relation: r, relation between e1 and e2
        '''
        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.relation = relation

    def __str__(self):
        try:
            return 'e1:{}, e2:{}, r:{}'.format(''.join(self.entity_1.content), ''.join(self.entity_2.content), self.relation.content)
        except:
            return 'Error occurred in toString() method!'
