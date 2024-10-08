from flask import Flask, request, jsonify
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Type
import json
from time import sleep
from zhipuai import ZhipuAI

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

def dataclass_to_dict(instance) -> Dict:
    if hasattr(instance, "to_dict"):
        return instance.to_dict()
    elif isinstance(instance, list):
        return [dataclass_to_dict(item) for item in instance]
    elif isinstance(instance, dict):
        return {k: dataclass_to_dict(v) for k, v in instance.items()}
    else:
        try:
            return asdict(instance)
        except TypeError:
            return instance

@dataclass
class Dimension:
    # 维度名称
    dimension_name: str
    # 一级指标
    first_level_index: str
    # 二级指标
    second_level_index: str
    # 核心字段召回
    core_field_recall: str

@dataclass
class StudentAnswer:
    # 学生姓名
    stu_name: str
    # 学生学号
    stu_id: str
    # 学生答案
    stu_answer: str
    # 老师评分，小数类型
    teacher_score: float
    # 老师评分理由
    teacher_score_reason: str
    # ai评分
    ai_score: float
    # ai评分理由
    ai_score_reason: str
    # ai评分标签，str列表
    ai_score_tags: List[str]
    # 学生答案命中得分要点的个数
    hit_view_count: int
    # 学生答案命中得分要点列表
    hit_view_list: List[str]
    # 学生答案命中得分要点的符合度列表
    stu_answer_score_key_points_match_list: List[float]
    # 学生答案疑似AI生成可疑度
    stu_answer_ai_suspicious: float
    # 学生答案疑似AI生成可疑理由
    stu_answer_ai_suspicious_reason: str
    # 学生答案疑似抄袭可疑度
    stu_answer_plagiarism_suspicious: float
    # 学生答案疑似抄袭可疑理由
    stu_answer_plagiarism_suspicious_reason: str
    # 学生答案主旨词
    stu_characteristics: str

@dataclass
class Question:
    # 老师姓名
    teacher_name: str = "张三"
    # 考试编号
    exam_id: str = "20231201"
    # 考试名称
    exam_name: str = "2023年12月1日考试"
    # 考试时间
    exam_time: str = "2023年12月1日 09:00-11:00"
    # 考试科目
    exam_subject: str = "计算机科学"
    # 考题内容
    question_content: str = "请回答以下问题：什么是人工智能？"
    # ai prompt
    ai_prompt: str = "请根据以下问题，生成答案：什么是人工智能？"
    # 标准答案，用于和用户答案进行对比
    standard_answer: str = "人工智能是一种模拟人类智能的技术，包括机器学习、自然语言处理、计算机视觉等。"
    # ai答案
    ai_answer: str = "人工智能是一种模拟人类智能的技术，包括机器学习、自然语言处理、计算机视觉等。"
    # 题目难度分析
    question_difficulty: str  = "中等"
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
    name: str
    # 考题，字典类型为str:Question
    questions: Dict[int, Question] = field(default_factory=dict)

__question_content=""
__standard_answer=""
system_prompt_give_dimension=""
user_prompt_give_dimension=""
__score_key_points=""
__stu_answer=""
__dimsnsions=""
__core_field_recalls=""

# 定义Test类型的实例，其中questions为空字典
test = Test(name="Midterm Exam", questions={})

# 定义一个函数，参数为questions的key和json格式的字符串，将json字符串转换为Question类型，并添加到test的questions字典中
def add_question(key: int, json_str: str):
    question_json = json.loads(json_str)
    test.questions[key] = dataclass_from_dict(Question, question_json)

# 假设Test实例已经创建，并且add_question函数也已定义
test = Test(name="Midterm Exam", questions={})

def GLM4_FUNCTION(system_prompt: str, user_prompt: str):
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
    print(model_response)
    return model_response

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
    try:
        id = request.args.get('id')
        # id转为int
        id = int(id)
        # 使用dataclass_to_dict函数转换Question实例为字典
        question_dict = dataclass_to_dict(test.questions[id])
        # 转换为JSON字符串
        question_json = json.dumps(question_dict, indent=4, ensure_ascii=False)
        print(question_json)
        return jsonify(question_dict), 200
    except Exception as e:
        # 如果指定的id不存在，返回404错误
        return jsonify({"error": "Question not found"}), 404

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
    json_data=json.loads(json_str)
    test.questions[id].score_key_points= json_data['score_key_points']

@app.route('/give_dimension', methods=['GET'])
def give_dimension_route():
    global __question_content,__standard_answer,system_prompt_give_dimension,user_prompt_give_dimension
    __question_content="""
请简述“功能主义”在现代建筑设计中的原则，并结合具体建筑实例说明其应用
    """
    __standard_answer="""
功能主义的原则：
功能主义是20世纪初期兴起的一种建筑设计理念，它强调建筑的功能和实用性，认为建筑的形式应当服务于其功能。
其主要原则包括：
1.形式追随功能：建筑的设计应以其使用功能为导向，建筑的形状和结构应当反映其用途。
2.简洁性：功能主义提倡简洁的设计，避免不必要的装饰，强调材料和结构的本质。
3.灵活性：建筑设计应考虑到不同使用需求的变化，具备一定的灵活性和适应性。
4.人本设计：关注使用者的需求，强调空间的舒适性和人性化。
实例分析：
一个经典的功能主义建筑实例是巴西利亚的国会大厦，由著名建筑师奥斯卡·尼迈耶设计。
- 形式与功能：主要体现在建筑的两座主要结构——上部的圆顶和下部的矩形体，分别代表着立法和行政的功能。
- 简洁性：国会大厦的外立面采用了光滑的白色混凝土，去除了传统建筑中的繁复装饰，强调了建筑的形式与功能的和谐统一。
- 灵活性：国会大厦的内部空间设计灵活，可以根据不同的会议和活动需求进行调整，满足多种功能的使用。
- 人本设计：建筑的设计不仅关注了功能的实现，也兼顾了使用者的体验，创造了开放、明亮的公共空间，增强了人与建筑的互动。
结论：
功能主义在现代建筑设计中发挥了重要作用，通过强调建筑的功能性和实用性，推动了建筑设计的简洁化和人性化。
"""
    system_prompt_give_dimension=f"""
## 角色：你是一个专业的老师 ，现在需要你根据我提供的题目和参考答案，给出对应的(维度,一级指标,二级指标,核心字段召回)json格式的列表,列表长度不能大于6。

##【字段定义】：
试卷和题目请严格按照如下格式仅输出JSON，不要输出python代码，不要返回多余信息，JSON中有多个字段用顿号【、】区隔：
### JSON字段：
{{
"exam_dimension_list":[
    {{
        "dimension_name": "维度名称",
        "first_level_index": "一级指标",
        "second_level_index": "二级指标",
        "core_field_recall": "核心字段召回"
    }}
    ...
    ]
}}

## 注意事项：
1. 基于给出的内容，专业和严谨的回答问题。不允许添加任何编造成分。
"""

    user_prompt_give_dimension=f"""
## 题目：{__question_content}
## 参考答案：{__standard_answer}
"""

    id = request.args.get('id')
    # id转为int
    id = int(id)
    # 使用dataclass_to_dict函数转换Question实例为字典
    question_dict = dataclass_to_dict(test.questions[id])
    __standard_answer=question_dict['standard_answer']
    __question_content=question_dict['question_content']
    json_str=GLM4_FUNCTION(system_prompt_give_dimension, user_prompt_give_dimension)
    # 解析JSON字符串
    json_data = json.loads(json_str)
    # 转换JSON数据中的每个项为Dimension对象
    new_exam_dimensions = [dataclass_to_dict(item) for item in json_data['exam_dimension_list']]
    # 更新test_instance中question[1]的exam_dimension_list
    test.questions[id].exam_dimension_list = new_exam_dimensions
    question_dict = dataclass_to_dict(test.questions[id])
    return jsonify(question_dict), 200

@app.route('/get_ai_prompt', methods=['get'])
def get_ai_prompt_route():
    global __question_content,__standard_answer,__score_key_points,__stu_answer,__dimsnsions,__core_field_recalls,system_prompt_give_dimension,user_prompt_give_dimension
    __question_content=""
    __standard_answer=""
    __score_key_points=""
    __dimsnsions=""
    __core_field_recalls=""
    system_prompt_give_dimension=f"""
## 角色：你是一个专业的课程老师 ，现在需要你批改一套的试卷，需要按照以下【任务要求】执行。

## 【评分规则】：
1. 总分为100分。
2. 学生答案需要围绕【维度和指标】内容以及【参考答案】展开，必须包含的核心字段有：{__core_field_recalls}，越贴近得分越高。
3. 参考答案中的关键名字不能写错，写错需要扣分。

## 维度和指标：元素格式为(维度,一级指标,二级指标）
{__dimsnsions}

## 考题内容
{__question_content}

## 参考答案：
{__standard_answer}

## 得分要点列表
{__score_key_points}

## 【任务要求】：
1. ai_score: AI评分。根据【评分规则】评分，最高得分不能超过100分，最低分不小于0分。评分的依据在【ai_score_reason】项中给出。
2. ai_score_reason: AI评分依据。每道题目的评分原因的内容不能超过100字。
3. stu_answer_ai_suspicious: AI答案相似度。表示学生答案疑似AI生成的概率，类型为百分数。疑似AI答案的原因在【stu_answer_ai_suspicious_reason】项中给出。
4. stu_answer_ai_suspicious_reason: 学生答案疑似AI的原因。不超过200字。
5. ai_score_tags: AI评分标签列表。分别是："完美试卷"、"高分试卷"、"疑似AI"。其中"高分试卷"的给出依据是得分【ai_score】在90分以上，"疑似AI"的给出依据是学生答案疑似AI生成可疑度【stu_answer_ai_suspicious】大于80%，"完美试卷"的给出依据是【ai_score】在90分以上且学生答案疑似AI生成可疑度【stu_answer_ai_suspicious】小于10%。
6. ai_answer: AI答案。AI答案不超过300字，AI答案需要根据【考题内容】和【参考答案】给出。
7. hit_view_list: 学生答案命中得分要点列表。学生答案的要点与符合【得分要点列表】的交集。元素的个数等于【hit_view_count】
8. stu_answer_score_key_points_match_list: 学生答案命中得分要点的符合度列表。【hit_view_list】中每个要点的符合度，每个元素的类型为百分数，取值越大表示学生答案与得分要点的匹配程度越高。元素的个数等于【hit_view_count】
9. hit_view_count: 学生答案命中得分要点的个数。【hit_view_list】中元素的个数。
10. stu_answer_ai_suspicious: 学生答案疑似AI生成可疑度。表示学生答案疑似AI生成的概率，类型为百分数。疑似AI答案的原因在【stu_answer_ai_suspicious_reason】项中给出。
11. stu_answer_ai_suspicious_reason: 学生答案疑似AI的原因。不超过200字。
12. stu_characteristics: 学生答案主旨词。

##【字段定义】：
试卷和题目请严格按照如下格式仅输出JSON，不要输出python代码，不要返回多余信息，JSON中有多个字段用顿号【、】区隔：
### JSON字段：
{{
    "ai_score": "【任务要求】1. ai_score" ,
    "ai_score_reason": "【任务要求】2. ai_score_reason",
    "stu_answer_ai_suspicious": "【任务要求】3. stu_answer_ai_suspicious",
    "stu_answer_ai_suspicious_reason": "【任务要求】4. stu_answer_ai_suspicious_reason",
    "ai_score_tags": [
        "【任务要求】5. ai_score_tags，例如: 完美试卷",
    ],
    "ai_answer": "【任务要求】6. ai_answer",
    "hit_view_list": [
        "【任务要求】7. hit_view_list[0]",
        "【任务要求】7. hit_view_list[1]",
    ],
    "stu_answer_score_key_points_match_list": [
        "【任务要求】8. stu_answer_score_key_points_match_list[0]",
        "【任务要求】8. stu_answer_score_key_points_match_list[1]",
    ],
    "hit_view_count": "【任务要求】9. hit_view_count",
    "stu_answer_ai_suspicious": "【任务要求】10. stu_answer_ai_suspicious",
    "stu_answer_ai_suspicious_reason":"【任务要求】11. stu_answer_ai_suspicious_reason",
    "stu_characteristics":"【任务要求】12. stu_characteristics"
}}

## 注意事项：
1. 基于给出的内容，专业和严谨的回答问题。不允许在答案中添加任何编造成分。
"""

    user_prompt_give_dimension=""""""
    
    id = request.args.get('id')
    # id转为int
    id = int(id)
    question_dict = dataclass_to_dict(test.questions[id])

    # 如果key不存在test.questions的key中，则创建键值对
    if id not in test.questions.keys():
        test.questions[id] = question_dict
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
    
    test.questions[id].ai_prompt=system_prompt_give_dimension
    return jsonify({"prompt":system_prompt_give_dimension}), 200

if __name__ == '__main__':
    app.run(debug=True, host='10.2.8.9', port=8080)
