import random
from flask import Flask, request, jsonify
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import List, Dict, Type
import json
from time import sleep, time
from zhipuai import ZhipuAI
from json_tool import try_parse_ast_to_json, try_parse_json_object
import logging

# 创建Flask应用
app = Flask(__name__)
client = ZhipuAI(api_key="d0ec437e4b38610fe6a811eff802da77.pcTnm7mFo2Ue30Lm") 

# 定义转换函数
def dataclass_from_dict(klass: Type, data: Dict) -> any:
    if hasattr(klass, "__annotations__"):  # 检查是否为dataclass
        fieldtypes = klass.__annotations__
        return klass(**{f: dataclass_from_dict(fieldtypes[f], data[f]) for f in data})
    elif isinstance(data, (list, tuple)):
        return [dataclass_from_dict(klass.__args__[0], d) for d in data]  # 处理列表中的元素
    else:
        return data  # 基本类型直接返回

# def dataclass_to_dict(instance) -> Dict:
#     if hasattr(instance, "to_dict"):
#         return instance.to_dict()
#     elif isinstance(instance, list):
#         return [dataclass_to_dict(item) for item in instance]
#     elif isinstance(instance, dict):
#         return {k: dataclass_to_dict(v) for k, v in instance.items()}
#     else:
#         try:
#             return asdict(instance)
#         except TypeError:
#             return instance


def dataclass_to_dict(instance) -> Dict:
    if hasattr(instance, "to_dict"):
        return instance.to_dict()
    elif isinstance(instance, list):
        return [dataclass_to_dict(item) for item in instance]
    elif isinstance(instance, dict):
        return {k: dataclass_to_dict(v) for k, v in instance.items()}
    elif is_dataclass(instance):
        return {f.name: dataclass_to_dict(getattr(instance, f.name)) for f in fields(instance)}
    else:
        return instance

@dataclass
class Dimension:
    # 维度名称
    dimension_name: str = ""
    # 一级指标
    first_level_index: str = ""
    # 二级指标
    second_level_index: str = ""
    # 核心字段召回
    core_field_recall: str = ""

@dataclass
class StudentAnswer:
    # 学生姓名
    stu_name: str = ""
    # 学生学号
    stu_id: int = ""
    # 学生答案
    stu_answer: str = ""
    # 老师评分，小数类型
    teacher_score: float = 0.0
    # 老师评分理由
    teacher_score_reason: str = ""
    # ai评分
    ai_score: float = 0.0
    # ai评分理由
    ai_score_reason: str = ""
    # ai评分标签，str列表
    ai_score_tags: List[str] = field(default_factory=list)
    # 学生答案命中得分要点的个数
    hit_view_count: int = 0
    # 学生答案命中得分要点列表
    hit_view_list: List[str] = field(default_factory=list)
    # 学生答案命中得分要点的符合度列表
    stu_answer_score_key_points_match_list: List[float] = field(default_factory=list)
    # 学生答案疑似AI生成可疑度
    stu_answer_ai_suspicious: float = 0.0
    # 学生答案疑似AI生成可疑理由
    stu_answer_ai_suspicious_reason: str = ""
    # 学生答案疑似抄袭可疑度
    stu_answer_plagiarism_suspicious: float = 0.0
    # 学生答案疑似抄袭可疑理由
    stu_answer_plagiarism_suspicious_reason: str = ""
    # 学生答案主旨词
    stu_characteristics: str = ""
    # ai阅卷状态
    ai_status: bool = False

@dataclass
class Question:
    # 老师姓名
    teacher_name: str = ""
    # 考试编号
    exam_id: str = ""
    # 考试名称
    exam_name: str = ""
    # 考试时间
    exam_time: str = ""
    # 考试科目
    exam_subject: str = ""
    # 考题内容
    question_content: str = ""
    # ai prompt
    ai_prompt: str = ""
    # 标准答案，用于和用户答案进行对比
    standard_answer: str = ""
    # ai答案
    ai_answer: str = ""
    # 题目难度分析
    question_difficulty: str  = ""
    # 得分要点列表
    score_key_points: List[str] = field(default_factory=list)
    # 考试维度列表, 类型为Dimension
    exam_dimension_list: List[Dimension] = field(default_factory=list)
    # 学生答案列表，类型为StudentAnswer
    stu_answer_list: List[StudentAnswer] = field(default_factory=list)
    # 考题得分等级人数,A\B\C\D\E
    score_level_count: List[int] = field(default_factory=list)
    # ai标签人数,{"完美试卷": 1,"高分试卷": 1,"疑似AI":1,"雷同试卷":1,"疑似抄袭":1},
    ai_tag_count: List[int] = field(default_factory=list)
    # 主旨词列表
    main_word_list: List[str] = field(default_factory=list)
    # 主旨词分布统计
    main_word_distribution_count: List[int] = field(default_factory=list)

@dataclass
class Test:
    # 考试名称
    name: str = ""
    # 考题，字典类型为str:Question
    questions: Dict[int, Question] = field(default_factory=dict)

# __question_content=""
# __standard_answer=""
# system_prompt_give_dimension=""
# user_prompt_give_dimension=""
# __score_key_points=""
# __stu_answer=""
# __dimsnsions=""
# __core_field_recalls=""

# 定义Test类型的实例，其中questions为空字典
test = Test(name="Midterm Exam", questions={})

# 定义一个函数，参数为questions的key和json格式的字符串，将json字符串转换为Question类型，并添加到test的questions字典中
def add_question(key: int, json_str: str):
    json_str,json_dict=try_parse_json_object(json_str)
    test.questions[key] = dataclass_from_dict(Question, json_dict)

# 假设Test实例已经创建，并且add_question函数也已定义
test = Test(name="Midterm Exam", questions={})

def GLM4_FUNCTION(system_prompt: str, user_prompt: str):
    assert(system_prompt!="")
    assert(user_prompt!="")
    try:
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=chat_history,
            stream=True
        )
        # 获取模型的回答
        model_response = ""
        # 打印模型回答
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                chunk_content = chunk.choices[0].delta.content
                model_response += chunk_content
        # print(model_response)
        return model_response
    except Exception as e:
        print(f"Error in GLM4_FUNCTION: {e}")
        return ""

app = Flask(__name__)
# 自定义JSON序列化
class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        json_str = super().encode(obj)
        return json_str.replace(r'\\n', '\n')

app.json_encoder = CustomJSONEncoder

@app.route('/add_question', methods=['POST'])
def add_question_route():
    json_str = request.json
    key=json_str.get('id')
    key=int(key)
    question=json_str.get('question')
    # question to str
    question = json.dumps(question)
    try:
        add_question(key, question)
        return jsonify({"success": True, "message": "Question added successfully."}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400


@app.route('/update_question_content_standard_answer', methods=['POST'])
def update_question_content_standard_answer_route():
    id = request.form['id']
    question_content = request.form['question_content']
    standard_answer = request.form['standard_answer']
    id=int(id)
    # 创建Question类实例,只设置question_content和standard_answer，其他属性设为默认值
    

    question = Question(question_content=question_content, standard_answer=standard_answer)    
    # 如果key不存在test.questions的key中，则创建键值对
    if id not in test.questions.keys():
        test.questions[id] = question
        print("create:",test.questions[id])
    test.questions[id].question_content = question_content
    test.questions[id].standard_answer = standard_answer
    return jsonify({"success": True, "message": "Question added successfully."}), 200

@app.route('/get_question', methods=['GET'])
def get_question():
    id = request.args.get('id')
    # id转为int
    id = int(id)
    # 使用dataclass_to_dict函数转换Question实例为字典
    question_dict = dataclass_to_dict(test.questions[id])
    # 转换为JSON字符串
    question_json = json.dumps(question_dict, indent=4, ensure_ascii=False)
    print(question_json)
    return jsonify(question_dict), 200

def gen_score_key_points(id: int ,question_content: str, standard_answer: str):
    system_prompt_give_dimension=f"""
##【任务要求】
根据【考题内容】和【标准答案】，生成考题的得分点

##【字段定义】：
试卷和题目请严格按照如下格式仅输出JSON，不要输出python代码，不要返回多余信息，JSON中有多个字段用顿号【、】区隔：
### JSON字段：
{{
    "score_key_points": ["得分点内容"]
}}
"""
    user_prompt_give_dimension=f"""
##【考题内容】
{question_content}

##【标准答案】
{standard_answer}    
"""
    json_str=GLM4_FUNCTION(system_prompt=system_prompt_give_dimension, user_prompt=user_prompt_give_dimension)
    json_str,json_dict=try_parse_json_object(json_str)
    # json_data=json.loads(json_str)
    test.questions[id].score_key_points= json_dict['score_key_points']

@app.route('/give_dimension', methods=['GET'])
def give_dimension_route():

    id = request.args.get('id')
    # id转为int
    id = int(id)
    __standard_answer=test.questions[id].standard_answer
    __question_content=test.questions[id].question_content
    system_prompt_give_dimension=f"""
##【任务要求】
根据我提供的【题目】和【参考答案】，给出对应的(维度,一级指标,二级指标,核心字段召回)JSON列表，列表长度不能大于6。
1. 维度（dimension_name）： 维度是指评价或测试的某个方面或领域。它是评价内容的分类方式，用于确定评价的方向和重点。例如，在一份学生的综合评价中，可能包含“知识掌握”、“技能应用”和“情感态度”等维度。
2. 一级指标（first_level_index）： 一级指标是维度的进一步细分，它具体描述了评价或测试的某个方面需要考虑的主要因素。一级指标通常是评价体系中的主要评判点，例如在“知识掌握”维度下，一级指标可能是“基础知识掌握”、“专业知识掌握”等。
3. 二级指标（second_level_index）： 二级指标是对一级指标的进一步细化，它描述了如何具体评价一级指标。二级指标通常是可量化的具体评价点，例如在“基础知识掌握”一级指标下，二级指标可能是“记忆准确度”、“理解深度”等。
4. 核心字段召回（core_field_recall）： 核心字段召回指的是在评价过程中需要特别关注和记录的关键信息或数据点。这些字段是评价结果的关键组成部分，它们直接关联到评价对象在该指标上的表现。例如，如果评价学生的“记忆准确度”，核心字段召回可能是学生在记忆测试中的正确率。

##【字段定义】：
试卷和题目请严格按照如下格式仅输出JSON，不要输出python代码，不要返回多余信息，JSON中有多个字段用顿号【、】区隔：
### JSON字段：
{{
"exam_dimension_list":[
    {{
        "dimension_name": "【任务要求】1. 维度名称",
        "first_level_index": "【任务要求】2. 一级指标",
        "second_level_index": "【任务要求】3. 二级指标",
        "core_field_recall": "【任务要求】4. 核心字段召回"
    }},
    ...
]
}}

## 注意事项：
1. 基于给出的内容，专业和严谨的回答问题。不允许添加任何编造成分。
"""
    user_prompt_give_dimension=f"""
【题目】：{__question_content}
【参考答案】：{__standard_answer}
"""

    json_str=GLM4_FUNCTION(system_prompt_give_dimension, user_prompt_give_dimension)
    print(system_prompt_give_dimension)
    print(user_prompt_give_dimension)
    # 解析JSON字符串
    json_str,json_dict=try_parse_json_object(json_str)

    # json_data_dict = json.loads(json_str)
    # 转换JSON数据中的每个项为Dimension对象
    new_exam_dimensions = [dataclass_from_dict(Dimension,item) for item in json_dict['exam_dimension_list']]
    # 更新test_instance中question[1]的exam_dimension_list
    test.questions[id].exam_dimension_list = new_exam_dimensions
    question_dict = dataclass_to_dict(test.questions[id])
    return jsonify(question_dict), 200

@app.route('/get_ai_prompt', methods=['get'])
def get_ai_prompt_route():
    # global __question_content,__standard_answer,__score_key_points,__stu_answer,__dimsnsions,__core_field_recalls,system_prompt_give_dimension,user_prompt_give_dimension
    __question_content=""
    __standard_answer=""
    __score_key_points=""
    __dimsnsions=""
    __core_field_recalls=""

    id = request.args.get('id')
    # id转为int
    id = int(id)
    question_dict = dataclass_to_dict(test.questions[id])

    # 如果key不存在test.questions的key中，则创建键值对
    if id not in test.questions.keys():
        test.questions[id] = dataclass_from_dict(Question, question_dict)
    # 获取维度元素列表
    exam_dimension_list = dataclass_from_dict(Question, question_dict).exam_dimension_list
    #打印维度字符串列表
    for dimension in exam_dimension_list:
        __dimsnsions += f"({dimension.dimension_name}, {dimension.first_level_index}, {dimension.second_level_index})"
    # 获取核心字段列表
    unique_core_fields = set()
    for dim in exam_dimension_list:
        if dim.core_field_recall:
            unique_core_fields.add(dim.core_field_recall)
    # 将结果组合成一个字符串
    __core_field_recalls = ",".join(unique_core_fields)
    __core_field_recalls="{"+__core_field_recalls+"}"

    __question_content=dataclass_from_dict(Question, question_dict).question_content
    __standard_answer=dataclass_from_dict(Question, question_dict).standard_answer
    
    gen_score_key_points(id,__question_content,__standard_answer)
    score_key_points=test.questions[id].score_key_points
    score_key_points_string = ", ".join(score_key_points)
    score_key_points_string = "{"+score_key_points_string+"}"
    __score_key_points=score_key_points_string

    system_prompt_give_dimension=f"""
## 角色：你是一个专业的课程老师 ，现在需要你批改一套的试卷，需要按照以下【任务要求】执行。

## 【评分规则】：
1. 总分为100分。
2. 学生答案需要围绕【维度和指标】内容以及【参考答案】展开，必须包含的核心字段有：{__core_field_recalls}，越贴近得分越高。
3. 参考答案中的关键名字不能写错，写错需要扣分。

## 维度和指标：元素格式为(维度,一级指标,二级指标）
{__dimsnsions}

## 考题内容：
{__question_content}

## 参考答案：
{__standard_answer}

## 得分要点列表：
{__score_key_points}

## 【任务要求】：
1. ai_score: AI评分。根据【评分规则】评分，最高得分不能超过100分，最低分为0分。评分的依据在【ai_score_reason】项中给出。
2. ai_score_reason: AI评分依据。每道题目的评分原因的内容不能超过100字。
3. ai_score_tags: AI评分标签列表。分别是："完美试卷"、"高分试卷"、"疑似AI"。其中"高分试卷"的给出依据是得分【ai_score】在90分以上，"疑似AI"的给出依据是学生答案疑似AI生成可疑度【stu_answer_ai_suspicious】大于80%，"完美试卷"的给出依据是【ai_score】在90分以上且学生答案疑似AI生成可疑度【stu_answer_ai_suspicious】小于10%。
4. ai_answer: AI答案。AI答案不超过300字，AI答案需要根据【考题内容】和【参考答案】给出。
5. hit_view_list: 学生答案命中得分要点列表。学生答案的要点与符合【得分要点列表】的交集。元素的个数等于【hit_view_count】
6. stu_answer_score_key_points_match_list: 学生答案命中得分要点的符合度列表。【hit_view_list】中每个要点的符合度，每个元素的类型为百分数，取值越大表示学生答案与得分要点的匹配程度越高。元素的个数等于【hit_view_count】
7. hit_view_count: 学生答案命中得分要点的个数。【hit_view_list】中元素的个数。
8. stu_answer_ai_suspicious: 学生答案疑似AI生成可疑度。表示学生答案疑似AI生成的概率，类型为百分数。疑似AI答案的原因在【stu_answer_ai_suspicious_reason】项中给出。
9. stu_answer_ai_suspicious_reason: 学生答案疑似AI的原因。不超过200字。
10. stu_characteristics: 学生答案主旨词。

##【字段定义】：
试卷和题目请严格按照如下格式仅输出JSON，不要输出python代码，不要返回多余信息，JSON中有多个字段用顿号【、】区隔：
### JSON字段：
{{
    "ai_score": "【任务要求】1. ai_score" ,
    "ai_score_reason": "【任务要求】2. ai_score_reason",
    "ai_score_tags": [
        "【任务要求】3. ai_score_tags，例如: 完美试卷",
    ],
    "ai_answer": "【任务要求】4. ai_answer",
    "hit_view_list": [
        "【任务要求】5. hit_view_list[0]",
        "【任务要求】5. hit_view_list[1]",
    ],
    "stu_answer_score_key_points_match_list": [
        "【任务要求】6. stu_answer_score_key_points_match_list[0]",
        "【任务要求】6. stu_answer_score_key_points_match_list[1]",
    ],
    "hit_view_count": "【任务要求】7. hit_view_count",
    "stu_answer_ai_suspicious": "【任务要求】8. stu_answer_ai_suspicious",
    "stu_answer_ai_suspicious_reason":"【任务要求】9. stu_answer_ai_suspicious_reason",
    "stu_characteristics":"【任务要求】10. stu_characteristics"
}}

## 注意事项：
1. 基于给出的内容，专业和严谨的回答问题。不允许在答案中添加任何编造成分。
"""

    user_prompt_give_dimension=""""""
    
    test.questions[id].ai_prompt=system_prompt_give_dimension
    return jsonify({"prompt":system_prompt_give_dimension}), 200

def create_student_answer(id: int, student_answer: str, stu_id: int, stu_name: str) -> StudentAnswer:
    question:Question
    question=test.questions[id]
    # 创建StudentAnswer类的对象
    student_answer = StudentAnswer(stu_answer = student_answer, stu_id=stu_id, stu_name=stu_name)
    question.stu_answer_list.append(student_answer)
    return student_answer

@app.route('/update_question_student_answer', methods=['POST'])
def update_question_student_answer_route():
    id = request.form['id']
    id=int(id)
    student_answer = request.form['student_answer']
    stu_id = request.form['stu_id']
    stu_id=int(stu_id)
    stu_name = request.form['stu_name']
    student_answer_instance=create_student_answer(id, student_answer, stu_id, stu_name)
    return jsonify({"success": True, "message": "student_answer added successfully."}), 200

def generate_random_number():
    """
    根据给定的最小时间戳生成一个不超过10位的随机数。
    """
    # 确保最小时间戳是整数
    min_timestamp = int(time())

    # 生成一个基于时间戳的随机种子
    random.seed(min_timestamp)

    # 生成一个不超过10位的随机数
    return random.randint(1, 10**10 - 1)

def generate_full_random_name():
    """
    根据当前时间戳生成随机的完整中文名字或英文名字。
    """
    current_timestamp = int(time())
    random.seed(current_timestamp)

    # 中文字符集
    chinese_chars = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刘宋李张赵钱孙李周吴郑王"
    chinese_last_names = list(chinese_chars)
    chinese_first_names = "伟刚勇毅俊峰强军平保东文辉力明永健世广志义兴良海山仁波宁贵福生龙元全国胜学祥才发武新利清飞彬富顺信子杰涛昌成康星光天达安岩中茂进林有坚和彪博诚先敬震振壮会思群豪心邦承乐绍功松善厚庆磊民友裕河哲江超浩亮政谦亨奇固之轮翰朗伯宏言若鸣朋斌梁栋维启克伦翔旭鹏泽晨辰士以建家致树炎德行时泰盛雄琛钧冠策腾楠榕风航弘"
    # 英文名字
    english_first_names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles"]
    english_last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

    # 随机选择生成中文名或英文名
    if random.choice([True, False]):
        # 生成中文名字
        chinese_last_name = random.choice(chinese_last_names)
        chinese_first_name = ''.join(random.sample(chinese_first_names, 2))
        return chinese_last_name + chinese_first_name
    else:
        # 生成英文名字
        english_first_name = random.choice(english_first_names)
        english_last_name = random.choice(english_last_names)
        return english_first_name + " " + english_last_name

@app.route('/set_ai_autogenerate_answer', methods=['POST'])
def set_AI_autogenerate_answer_route() -> StudentAnswer:
    id = request.form['id']
    id=int(id)
    ai_mock_stu_num=request.form['ai_mock_stu_num']
    ai_mock_stu_num=int(ai_mock_stu_num)
    questions=test.questions[id];
    # 使用ai_mock_stu_num迭代
    for i in range(ai_mock_stu_num):    
        ai_mock_answer=GLM4_FUNCTION("帮我回答这个题目，答案字数不超过200字", questions.question_content)
        # ai_mock_stu_id=GLM4_FUNCTION("当我说开始的时候，帮我生成一个随机数，长度不超过10位", "开始")
        ai_mock_stu_id=generate_random_number()
        # ai_mock_stu_name=GLM4_FUNCTION("当我说开始的时候，帮我生成一个中文名，名字不超过3个字，寓意要好", "开始")
        ai_mock_stu_name=generate_full_random_name()
        create_student_answer(id, ai_mock_answer, ai_mock_stu_id, ai_mock_stu_name)
    return jsonify({"success": True, "message": "set_AI_autogenerate_route added successfully."}), 200

@app.route('/start_ai_grading', methods=['POST'])
def start_ai_grading_route() -> StudentAnswer:
    id = request.form['id']
    id=int(id)
    question:Question
    question=test.questions[id];
    # 使用ai_mock_stu_num迭代
    for stu_answer in question.stu_answer_list:
        # print(f"AI Grading Response for Student ID {stu_answer.stu_id}")
        # print("question.ai_prompt=", question.ai_prompt)
        # print("stu_answer.stu_answer", stu_answer.stu_answer)
        ai_grading_json_str=GLM4_FUNCTION(question.ai_prompt, stu_answer.stu_answer)
        # print("ai_grading_json_str=", ai_grading_json_str)
        json_str,ai_grading_json=try_parse_json_object(ai_grading_json_str)        
        
        ai_score=ai_grading_json['ai_score']
        ai_score_reason=ai_grading_json['ai_score_reason']
        ai_score_tags=ai_grading_json['ai_score_tags']
        ai_answer=ai_grading_json['ai_answer']
        hit_view_list=ai_grading_json['hit_view_list']
        stu_answer_score_key_points_match_list=ai_grading_json['stu_answer_score_key_points_match_list']
        hit_view_count=ai_grading_json['hit_view_count']
        stu_answer_ai_suspicious=ai_grading_json['stu_answer_ai_suspicious']
        stu_answer_ai_suspicious_reason=ai_grading_json['stu_answer_ai_suspicious_reason']
        stu_characteristics=ai_grading_json['stu_characteristics']
        
        stu_answer.ai_score=ai_score
        stu_answer.ai_score_reason=ai_score_reason
        stu_answer.ai_score_tags=ai_score_tags
        question.ai_answer=ai_answer
        stu_answer.hit_view_list=hit_view_list
        stu_answer.stu_answer_score_key_points_match_list=stu_answer_score_key_points_match_list
        stu_answer.hit_view_count=hit_view_count
        stu_answer.stu_answer_ai_suspicious=stu_answer_ai_suspicious
        stu_answer.stu_answer_ai_suspicious_reason=stu_answer_ai_suspicious_reason
        stu_answer.stu_characteristics=stu_characteristics
        stu_answer.ai_status=True

    return jsonify({"success": True, "message": "start_ai_grading successfully."}), 200


if __name__ == '__main__':
    app.run(debug=True, host='10.2.8.9', port=8080)
