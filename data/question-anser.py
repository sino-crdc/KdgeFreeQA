# -*- coding:utf-8 -*-
import json
dict1a={}
dict1b={}
dict2a={}
dict2b={}
dict3a={}
dict3b={}
dict4a={}
dict4b={}
with open(r'C:\pythonProject1\KdgeFreeQA\dataset\final_all_data\first_stage\train.json','r',encoding='utf-8-sig') as f:
    with open(r'C:\pythonProject1\KdgeFreeQA\dataset\final_all_data\first_stage\question-answer-train.json','w',encoding='utf-8')as f1:
    # info_dict=json.load(f)文件过大会报错
        while True:
            line=f.readline()
            if not line:
                break
            info_dict=json.loads(line)
            dict1a['question1']="".join(info_dict["meta"]["criminals"])+'犯了什么罪？'
            dict1b['answer1']="".join(info_dict["meta"]["accusation"])+'。'
            dict2a['question2'] = "".join(info_dict["meta"]["criminals"])+'会不会判死刑？'
            if info_dict["meta"]["term_of_imprisonment"]["death_penalty"]==False:
                dict2b['answer2'] ="".join(info_dict["meta"]["criminals"])+'不会判死刑。'
            else:
                dict2b['answer2']="".join(info_dict["meta"]["criminals"])+'会判死刑。'
            dict3a['question3']=''.join(info_dict["meta"]["criminals"])+'的刑期是多少？'
            dict3b['answer3']=str(info_dict["meta"]["term_of_imprisonment"]["imprisonment"])+'月。'
            dict4a['question4']="".join(info_dict["meta"]["criminals"])+'应当处以多少罚金？'
            dict4b['answer4']=str(info_dict["meta"]["punish_of_money"])+'元人民币。'
            f1.writelines(json.dumps([dict1a,dict1b,dict2a,dict2b,dict3a,dict3b,dict4a,dict4b],ensure_ascii=False)+'\n')
            #f1.write('\n')
    






