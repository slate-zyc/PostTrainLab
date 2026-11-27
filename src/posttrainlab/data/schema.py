"""
训练数据结构定义，规范化对话文本并检查 SFT 和偏好对。

输入
    Hugging Face 原始记录、数据集标识、上游拆分和缓存行号。
输出
    已校验的 SFTExample 或 PreferenceExample 对象，可导出为项目统一的 JSONL。
"""

import re
import unicodedata              # 处理 Unicode 字符
from typing import Literal

from pydantic import BaseModel, StrictStr, field_validator, model_validator     # 数据校验


# ============================================================
# 文本与对话检查
# ============================================================

# 定义异常字符和占位文本
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")
PLACEHOLDERS = {"n/a", "none", "null", "[removed]", "[deleted]", "[placeholder]"}


# 定义 Message 类
# 继承 BaseModel 时，Message 就自动获得了 Pydantic 提供的数据校验能力。
# BaseModel 即提供数据校验能力的基础类
class Message(BaseModel):
    """一轮会话中的角色和正文。

    输入
        role 为 system、user 或 assistant，content 为文本字符串。
    输出
        Pydantic 实例。正文在实例化时完成换行与首尾空白清理。
    """
    # 数据校验
    role: Literal["system", "user", "assistant"]
    content: StrictStr      # 严格字符串

    # 抛出 ValueError 时，Pydantic 会把它包装进 **ValidationError**，其中会包含具体的字段错误信息，错误消息类似 Value error, empty_message。
    @field_validator("content")  # 默认 mode="after", 也就是先完成字段的基本类型校验，再执行编写的校验函数
    # 指定一个字段，并在数据校验过程中**自动调用**对应的函数，对该字段进行进一步检查或清理。
    # 当创建 Message 对象时，Pydantic 会自动调用，对 content 进行清洗
    # 即校验 Message 的 content 字段，**同时返回清理后的正文**，最终存储在 Message 对象中的就是清理后的文本。
    # @field_validator("content") 默认 mode="after"，即先完成 Pydantic 的类型校验（StrictStr），再调用 clean_text
    @classmethod    # Python 内置的装饰器，把一个普通函数定义成类方法，让它通过类来调用，并自动接收类本身作为第一个参数
    # cls 是类方法（classmethod）的第一个参数，代表“类本身”，就像普通实例方法的第一个参数 self 代表“实例本身”。
    # - 为什么不使用 self?
    # - 因为 Pydantic 正在创建对象，还没有完成对象的构建，所以这里使用类方法来定义字段校验函数，而不是依赖已经创建好的实例。
    def clean_text(cls, text: str) -> str:  # 这里 cls 代表 Message 类本身
        """清理消息正文并筛掉明显无效文本。

        输入
            text 是 Pydantic 已检查为字符串的 content 字段。
        输出
            返回清理后的文本，保留正文里的代码缩进和 Markdown。
            空文本、异常字符和占位内容会触发带过滤原因的 ValueError。
        """
        # 统一换行和首尾空白，保留正文里的缩进、代码块和 Markdown。
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        # 去掉首尾 HTML 标签，re.I 表示忽略大小写
        text = re.sub(r"\A<(?:html|body)>\s*|\s*</(?:html|body)>\Z", "", text, flags=re.I).strip()
        if not text:
            raise ValueError("empty_message")
        # \ufffd 是 Unicode 替换字符，显示出来通常是 �。
        # 它经常出现在文本解码错误时。
        # 这里规定出现三个或更多就判定为异常。
        if CONTROL_CHARS.search(text) or text.count("\ufffd") >= 3:
            raise ValueError("invalid_character")
        # .casefold() 转小写；.isalnum()判断字符是否为字母或数字，没有任何字母或数字被过滤
        if text.casefold() in PLACEHOLDERS or not any(char.isalnum() for char in text):
            raise ValueError("placeholder_or_symbols")
        return text


# ============================================================
# SFT 和 Preference 数据结构
# ============================================================

def validate_dialogue(messages: list[Message], last_role: str) -> None:
    """核对对话角色序列和结束角色。

    输入
        messages 是已校验的 Message 列表。last_role 指定期望的最后角色。
        SFT 对话以 assistant 结束，偏好对的共同上文以 user 结束。
    输出
        校验成功返回 None，角色顺序有误时抛出 ValueError。
    """
    roles = [message.role for message in messages]
    if roles[:1] == ["system"]:
        roles = roles[1:]
    # 去掉可选的 system 后，生成 user、assistant 交替的期望序列。
    # 两个源数据集均未使用 system，都是 user和assistant
    expected = ["user" if index % 2 == 0 else "assistant" for index in range(len(roles))]
    if not roles or roles != expected or roles[-1] != last_role:
        raise ValueError("invalid_message_order")


def normalize_prompt(text: str) -> str:
    """为文本比较生成统一的 Prompt 表示。

    输入
        text 是待比较的原文。
    输出
        返回 Unicode 兼容归一化、大小写折叠及合并空白后的文本，用于精确和近似去重。
    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


# # 下载的 HuggingFaceH4/ultrachat_200k 原始数据的其中一条  data/raw/HuggingFaceH4--ultrachat_200k/v1-test_sft-n600-s42-b512.jsonl
# {"prompt": "- The queue size should be dynamically adjusted as elements are added or removed.",
#   "prompt_id": "d089d9db3769166616cbc6b5cce9f465b466fa6b93231c696381e435e5ea0030",
#   "messages": [{"content": "- The queue size should be dynamically adjusted as elements are added or removed.",
#     "role": "user"},
#     {"content": "This is a good practice as it helps in efficient utilization of resources and prevents overflow or underflow situations. When elements are added to the queue, the size of the queue should be increased dynamically to accommodate the new elements. Similarly, when elements are removed from the queue, the size of the queue should be reduced dynamically to free up the memory resources. This ensures that the queue always has sufficient space to hold the elements being added, without wasting any unnecessary memory. Additionally, dynamically adjusting the queue size also helps in improving the performance of the queue operations, such as enqueue and dequeue, as the queue size is always optimized based on the current number of elements in the queue.",
#       "role": "assistant"},
#     {"content": "Could you please add some real-life examples of when dynamically adjusting queue size would be useful?",
#       "role": "user"},
#     {"content": "Certainly! Here are some real-life examples where dynamically adjusting queue sizes would be useful:\n\n1. Server request processing: In a server, requests are received continuously, and they are processed by the server sequentially. In order to avoid overloading the server, it would be useful to implement a queue that dynamically adjusts its size based on the number of requests being processed. If the queue becomes too long, the server can slow down its processing or stop accepting new requests until the queue has reduced its size. 2. Resource allocation: In a computer system, resources such as memory or CPU time are allocated to running programs. If the resources are limited, a queue can be implemented to dynamically adjust its size to accommodate the programs waiting in the queue. This ensures that the resources are allocated based on the needs of the running programs, optimizing the utilization of the resources. 3. Customer support: In a customer support system, customers may submit support tickets continuously. By having a queue that dynamically adjusts its size based on the number of tickets in the queue, the support team can manage their workload effectively. They can also prioritize urgent tickets over less urgent ones, ensuring that the most important issues are addressed first. Overall, implementing a queue that dynamically adjusts its size is a useful technique in situations where resources or demands are continuously changing.", "role": "assistant"}, {"content": "Can you give me some more information on how dynamically adjusting queue size affects the efficiency of queue operations?", "role": "user"}, {"content": "Sure, here are some ways in which dynamically adjusting the queue size can improve the efficiency of queue operations:\n\n1. Reduced memory usage: Dynamically adjusting the queue size ensures that no unnecessary memory is allocated. The memory allocated for the queue is proportional to the number of elements currently in the queue, which reduces memory usage and avoids wastage. 2. Faster operations: By having a smaller queue, enqueue and dequeue operations can be performed more quickly. Smaller queues also mean that there's less data to process or search through, resulting in faster run times. 3. Improved resource utilization: Dynamically adjusting the queue size allows for optimal resource utilization. Resource handling is always done based on the current need, which means that it's never wasted or underutilized. 4. Prevents underflow and overflow: Dynamically adjusting the queue size prevents underflow or overflow situations, which can result in errors or data loss. By automatically expanding and contracting the queue size as elements are added and removed, the queue is always optimized and ready to handle new data. 5. Better responsiveness: Dynamic queue sizing can improve the responsiveness of a system by reducing the waiting time for queue operations. If a queue is too large, it may take longer to add or remove an element. By dynamically adjusting the queue size, the wait time and response time of the system can be optimized. Overall, dynamically adjusting the queue size not only optimizes the use of resources, but also has a significant impact on the performance and efficiency of queue operations.", "role": "assistant"}, {"content": "Thanks for the information, it's really helpful! Could you maybe give me some more examples of how dynamically adjusting queue size could be useful in different industries or contexts? That would be great!", "role": "user"}, {"content": "Sure, here are some more examples of how dynamically adjusting queue size can be useful in different industries or contexts:\n\n1. Retail: In a retail environment during peak shopping seasons, more customers may be waiting in line to complete their purchases. A queue management system that adjusts the queue size can help to minimize the wait times for customers and also ensures that the store's cashiers are efficiently utilized.\n\n2. Transportation: In public transportation systems like buses or trains, there can be a variable number of passengers at different times of the day. A dynamic queuing system can be used to balance the number of people boarding the vehicles and prevent overcrowding at specific times.\n\n3. Healthcare: In healthcare settings, there is often variability in the number of patients needing care. A dynamic queuing system can be used to adjust the size of the queue based on the number of available medical professionals. For instance, during a busy day with more patients, the queue can be expanded to accommodate the increased demand.\n\n4. Manufacturing: In a production line, raw materials or products may need to be transferred between different workstations. A dynamic queue system can be used to allocate the resources required for these transfers and adjust the queue size based on the current production needs.\n\n5. Call center: In a call center, there can be a variable number of customer calls coming in at different times of the day. A dynamic queuing system can be used to adjust the queue size based on the volume of calls and ensure that customers are not waiting too long to be assisted.\n\nThese are just a few examples of how dynamic queue sizing can be useful across different industries and contexts. In general, a dynamic queue system can be used anywhere where there is variability in demand, and resources need to be optimally allocated to manage that demand.",
#       "role": "assistant"}]}

# # 处理过的数据 data/interim/sft/test.jsonl 中的某一条，prepare.py 文件实现
# {"id": "HuggingFaceH4/ultrachat_200k:test_sft:388",
# "source": "HuggingFaceH4/ultrachat_200k",
# "language": "en",
#  "messages": [{"role": "user", "content": "What steps has the UN Security Council taken to address the ongoing humanitarian crisis in Syria?"},
#               {"role": "assistant", "content": "1. Adopted resolutions on humanitarian aid: The Security Council has adopted a number of resolutions, including Resolution 2139 and Resolution 2165, which call for the immediate and full implementation of humanitarian aid to Syria.\n\n2. Established a humanitarian task force: In 2014, the Security Council established a humanitarian task force to facilitate the delivery of aid to those in need in Syria. The task force includes representatives from the UN, the Syrian government, and Syrian opposition groups.\n\n3. Called for ceasefires: The Security Council has called for ceasefires in Syria, particularly in areas where humanitarian aid is needed. These calls have not always been heeded, but they demonstrate the Council’s commitment to addressing the humanitarian crisis in Syria.\n\n4. Condemned attacks on civilians: The Security Council has repeatedly condemned attacks on civilians in Syria. In August 2016, for example, the Council issued a statement condemning the bombing of a humanitarian aid convoy near Aleppo.\n\n5. Worked to address the root causes of the crisis: While the Security Council’s primary focus has been on providing humanitarian aid, it has also worked to address the root causes of the crisis. In 2015, for example, the Council adopted Resolution 2254, which called for a political solution to the conflict in Syria."},
#               {"role": "user", "content": "It seems like the UN Security Council has taken some steps to address the humanitarian crisis in Syria, but have these steps actually made a difference on the ground? What tangible results have been achieved so far?"},
#               {"role": "assistant", "content": "While the UN Security Council has taken some steps to address the humanitarian crisis in Syria, there are still significant challenges on the ground. Some tangible results have been achieved, but the overall impact of these efforts has been limited due to ongoing violence, political challenges, and logistical obstacles.\n\nSome of the tangible results of the Security Council's efforts include:\n\n1. Increased access to humanitarian aid: The establishment of a humanitarian task force and the adoption of resolutions calling for the immediate and full implementation of humanitarian aid have led to increased access to aid for some Syrians. However, access to aid remains limited in some areas due to ongoing violence and political challenges.\n\n2. Brief ceasefires: The Security Council has called for ceasefires in Syria, particularly in areas where humanitarian aid is needed. Some brief ceasefires have been achieved, allowing for the delivery of aid to certain areas. However, these ceasefires have not been sustained, and violence has resumed in many parts of Syria.\n\n3. Condemnation of attacks on civilians: The Security Council's condemnation of attacks on civilians has helped raise awareness of the humanitarian crisis in Syria and the need for urgent action. However, these condemnations have not prevented the continued targeting of civilians in the conflict.\n\n4. Political dialogue: The adoption of Resolution 2254 calling for a political solution to the conflict in Syria was an important step towards addressing the root causes of the crisis. However, progress towards a political solution has been slow, and the conflict in Syria continues to escalate.\n\nIn summary, while the UN Security Council's efforts have led to some tangible results, the overall impact of these efforts on the ground has been limited. The ongoing violence, political challenges, and logistical obstacles in Syria continue to hinder efforts to address the humanitarian crisis."},
#               {"role": "user", "content": "It sounds like the UN Security Council's efforts have been mostly unsuccessful in addressing the humanitarian crisis in Syria. What more needs to be done?"},
#               {"role": "assistant", "content": "Yes, while the UN Security Council's efforts have led to some tangible results, such as increased access to humanitarian aid, brief ceasefires, and political dialogue, the overall impact of these efforts has been limited in addressing the humanitarian crisis in Syria. Much more needs to be done to effectively address the crisis. \n\nSome of the key steps that should be taken include:\n\n1. Increase access to humanitarian aid: While some progress has been made in increasing access to aid, there is still a need to expand the delivery of aid to all parts of Syria. This can be achieved through increased funding for aid programs, improved coordination between aid agencies and the Syrian government, and greater support for local aid organizations.\n\n2. End the violence: The ongoing violence in Syria is the root cause of the humanitarian crisis. All parties to the conflict must commit to ending the violence and protecting civilians. This can be achieved through the cessation of hostilities and the implementation of a comprehensive ceasefire.\n\n3. Ensure accountability for human rights abuses: The perpetrators of human rights abuses in Syria, including war crimes and crimes against humanity, must be held accountable for their actions. This can be achieved through international investigations and prosecutions of those responsible for these crimes.\n\n4. Support political dialogue: A political solution to the conflict is essential for addressing the root causes of the humanitarian crisis in Syria. All parties to the conflict must be encouraged to engage in inclusive political dialogue towards a lasting solution to the conflict.\n\nIn summary, addressing the humanitarian crisis in Syria requires a multifaceted approach. The international community, including the UN Security Council and its member states, must take urgent action to address the root causes of the crisis, end the violence, and support the delivery of humanitarian aid to those in need."},
#               {"role": "user", "content": "It's frustrating that the Security Council has taken so many steps but there hasn't been much improvement on the ground. Why can't they do more to help the people of Syria?"},
#               {"role": "assistant", "content": "There are many complex factors that contribute to the Security Council's limited ability to address the humanitarian crisis in Syria, despite the Council's efforts. Some of these factors include:\n\n1. Political divisions among key Security Council members: Political divisions between the United States and Russia, among other key Council members, have hindered efforts to find a comprehensive solution to the conflict in Syria. This has in turn made it more challenging to address the humanitarian crisis.\n\n2. Sovereignty concerns: The Syrian government has shown resistance to accepting outside help, citing concerns about national sovereignty. This has made it difficult for international aid organizations to access some areas of the country and to provide critical aid to the millions of people in need.\n\n3. Logistical challenges: The ongoing violence in Syria, including the targeting of aid convoys and aid workers, has made it difficult for aid organizations to deliver aid to those in need. The physical destruction of infrastructure and the displacement of people to different parts of the country has also made aid delivery more challenging.\n\n4. Limited resources: Funding for aid programs in Syria is limited and there are constraints on the resources available to the international community to address the crisis.\n\nIn summary, there are many complex factors that contribute to the Security Council's limited ability to address the humanitarian crisis in Syria. Addressing these challenges will require increased political will and the allocation of more resources towards addressing the ongoing humanitarian crisis in Syria."}],
#  "metadata": {"source_split": "test_sft", "source_row": 388, "source_prompt_id": "e301de3ec9e77cbb352c656a6880d1fb1bcd7ceed359c936a75406200a862fa5",
#               "tokens": {"prompt": 121, "answer": 1256, "total": 1377}, "task_category": "other"}}

# SFTExample 接收的数据为未处理过的原始数据，返回希望的结构
class SFTExample(BaseModel):
    """统一的 SFT 样本结构。

    输入
        id、来源、语言、完整多轮 messages 和来源 metadata。
    输出
        已校验的 SFTExample。model_dump() 可转为 JSONL 所需结构。
        model_dump() 是 Pydantic V2 的 BaseModel 提供的方法。
        作用是：将 Pydantic 模型对象转换成 Python 字典，内部的 Message 对象也会递归转换成字典。
    """
    id: str
    source: str
    language: str
    messages: list[Message]     # 表示 messages 是一个列表，列表中每个元素都必须符合 Message 类，传入普通字典时 Pydantic 会自动校验并转为 Message 对象
    metadata: dict

    @model_validator(mode="after")  # messages 内的字段解析完成后再检查角色序列, 即Pydantic 首先会将普通字典转换成 Message 对象，然后再执行 check_dialogue()
    # - field_validator 和 model_validator 的区别
    # - field_validator 的任务是检查一个具体字段；model_validator 的任务是检查整个模型的数据
    def check_dialogue(self):   # self 就是当前创建的 SFTExample 实例
        # 为什么这里不用 cls？
        # 因为两种校验器的工作方式不同：
        # - field_validator 通常 在字段解析过程中处理单个字段，因此使用类方法形式。
        # - model_validator(mode="after") 在模型已经构建完成后执行，因此可以直接获得完整的实例，使用 self。
        # Pydantic V2 的 after 模型校验器要求使用实例方法。
        """检查 SFT 样本是否具有完整的问答轮次。

        输入
            self 是各字段已解析完成的 SFTExample。
        输出
            返回通过校验的实例。对话必须以 assistant 的回答结束。
        """
        validate_dialogue(self.messages, "assistant")
        return self     # Pydantic V2 的 after 模型校验器中的重要要求


# # 下载的 HuggingFaceH4/ultrafeedback_binarized 原始数据的其中一条
# {"prompt": "You will be given a definition of a task first, then some input of the task.\nGiven a scientific question, generate an incorrect answer to the given question. The incorrect answer should be a plausible alternative to the correct answer. The answer should be some other item belonging to the same category as the correct answer but should be incorrect.\n\nWhat type of fertilization do most reptiles have?\nOutput:",
#   "prompt_id": "bd6d080fed4f632b8dca1adbac9701439cd418841c8d67c4819b003982fb5699",
#   "chosen": [{"content": "You will be given a definition of a task first, then some input of the task.\nGiven a scientific question, generate an incorrect answer to the given question. The incorrect answer should be a plausible alternative to the correct answer. The answer should be some other item belonging to the same category as the correct answer but should be incorrect.\n\nWhat type of fertilization do most reptiles have?\nOutput:",
#     "role": "user"},
#     {"content": "External fertilization", "role": "assistant"}],
#   "rejected": [{"content": "You will be given a definition of a task first, then some input of the task.\nGiven a scientific question, generate an incorrect answer to the given question. The incorrect answer should be a plausible alternative to the correct answer. The answer should be some other item belonging to the same category as the correct answer but should be incorrect.\n\nWhat type of fertilization do most reptiles have?\nOutput:",
#     "role": "user"},
#     {"content": "External fertilization", "role": "assistant"}],
#   "messages": [{"content": "You will be given a definition of a task first, then some input of the task.\nGiven a scientific question, generate an incorrect answer to the given question. The incorrect answer should be a plausible alternative to the correct answer. The answer should be some other item belonging to the same category as the correct answer but should be incorrect.\n\nWhat type of fertilization do most reptiles have?\nOutput:",
#     "role": "user"},
#     {"content": "External fertilization", "role": "assistant"}],
#   "score_chosen": 8.5, "score_rejected": 7.5}

# # 处理过的数据 data/interim/preference/test.jsonl 中的某一条，prepare.py 文件实现
# {"id": "HuggingFaceH4/ultrafeedback_binarized:test_prefs:32",
#  "source": "HuggingFaceH4/ultrafeedback_binarized",
#  "prompt": [{"role": "user", "content": "Given the sentence \"A spread of asian food at a bar.\" is it true that \"There are several types of chicken.\"?\n--\nA spread of food could be many things besides several types of chicken.. The answer is: it is not possible to tell.\n----\nPremise: \"A man in a white shirt and plaid shorts sits down.\"\nHypothesis: \"The man is sitting.\"\nIs the hypothesis entailed by the premise?\nOptions:\n- yes\n- it is not possible to tell\n- no\n--\nThe man sits down and the man is sitting is the same thing.. The answer is: yes.\n----\nPremise: \"A bike is on its side in the middle of the road.\"\nBased on this premise, can we conclude that the hypothesis \"And firefighters are talking to a man sitting on the curb.\" is true?\nOptions:\n- yes\n- it is not possible to tell\n- no\n--"}],
#  "chosen": [{"role": "assistant", "content": "The premise only talks about a bike on its side, and there is no mention of firefighters or a man sitting on the curb. The answer is: it is not possible to tell.\n\nConfidence: 95%"}],
#  "rejected": [{"role": "assistant", "content": "Sure, I'll do my best to assist you! Here are my responses to each question:\n\n1. Given the sentence \"A spread of asian food at a bar.\" is it true that \"There are several types of chicken.\"?\n\nConfidence: 0%\n\nThe premise is not sufficient to support the conclusion that there are several types of chicken. The sentence \"A spread of asian food at a bar\" could refer to many different types of food, and it is not possible to infer that chicken is present just because it is an asian food.\n\n2. Premise: \"A man in a white shirt and plaid shorts sits down.\"\n\nHypothesis: \"The man is sitting.\"\n\nIs the hypothesis entailed by the premise?\n\nConfidence: 100%\n\nYes, the hypothesis is entailed by the premise. The premise states that a man in a white shirt and plaid shorts sits down, which implies that the man is sitting.\n\n3. Premise: \"A bike is on its side in the middle of the road.\"\n\nCan we conclude that the hypothesis \"And firefighters are talking to a man sitting on the curb.\" is true?\n\nConfidence: 0%\n\nNo, we cannot conclude that the hypothesis \"And firefighters are talking to a man sitting on the curb\" is true based on the premise \"A bike is on its side in the middle of the road.\" The two premises are unrelated and do not provide enough information to support the conclusion."}],
#  "metadata": {"source_split": "test_prefs", "source_row": 32, "source_prompt_id": "72a7a896df9c82c706d0a217123c3999575a7200713f04eaf883e98ee65a8011",
#               "language": "en", "preference_source": "model_judgment", "score_chosen": 8.0, "score_rejected": 7.5,
#               "tokens": {"prompt": 188, "answer": 296, "total": 484, "chosen": 44, "rejected": 296}, "task_category": "other"}}

# PreferenceExample 接收的数据为未处理过的原始数据，返回希望的结构
class PreferenceExample(BaseModel):
    """统一的偏好对结构，显式保存共同上文。

    输入
        id、来源、共同 Prompt、chosen、rejected 和来源 metadata。
    输出
        已校验的 PreferenceExample，两个回答各包含一条 assistant 消息。
    """
    id: str
    source: str
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]
    metadata: dict

    @model_validator(mode="after")
    def check_pair(self):
        """检查共同 Prompt 和两个候选回答。

        输入
            self 是字段解析完成的 PreferenceExample。
        输出
            返回通过校验的实例。非法角色或归一化后相同的回答会触发 ValueError。
        """
        # 确保共同 Prompt 合法
        validate_dialogue(self.prompt, "user")
        for response in (self.chosen, self.rejected):
            # 必须恰好包含一条消息；这条消息的角色必须是 assistant
            if len(response) != 1 or response[0].role != "assistant":
                raise ValueError("invalid_preference_answer")
        # 检查两个回答是否相同
        if normalize_prompt(self.chosen[0].content) == normalize_prompt(self.rejected[0].content):
            raise ValueError("identical_preference_answers")
        return self


def first_user(example: SFTExample | PreferenceExample) -> str:
    """提取样本中的第一条用户消息。

    输入
        example 是通过结构校验的 SFT 或偏好样本。
    输出
        返回首条 user 消息的正文，用于任务类别统计。
    """
    # SFT 和 Preference 的对话字段名称不同
    messages = example.messages if isinstance(example, SFTExample) else example.prompt
    return next(message.content for message in messages if message.role == "user")


# ============================================================
# 官方数据格式 → 项目数据格式
# ============================================================

def convert_record(row: dict, dataset_id: str, split: str, row_number: int, kind: str):
    """把上游原始样本转成项目的统一训练结构。

    输入
        row 是官方数据中的单条记录，dataset_id、split 和 row_number 用于追溯来源。
        kind 指定 sft 或 preference，决定解析 messages 还是 chosen、rejected。
    输出
        返回对应的 Pydantic 样本。偏好分支的上文不一致时抛出 ValueError。

    偏好样本示例
        chosen   = [共同 Prompt, 较优回答]
        rejected = [共同 Prompt, 较差回答]
        清洗时拆成 prompt、chosen、rejected 三个字段，并保留上游评分。
    """
    metadata = {"source_split": split, "source_row": row_number, "source_prompt_id": row.get("prompt_id")}
    example_id = f"{dataset_id}:{split}:{row_number}"  # 通过上游拆分和缓存行号定位原始样本
    if kind == "sft":
        return SFTExample.model_validate({
            "id": example_id,
            "source": dataset_id,
            "language": "en",
            "messages": row["messages"],
            "metadata": metadata,
        })

    # model_validate() 做了什么？
    # - 检查传入的数据是否符合模型定义，校验失败会抛出 ValidationError
    # - 如果数据合法，就返回一个对应的示例（SFTExample/Message/PreferenceExample）
    # 补充：model_validate() 是 Pydantic V2 的方法，Pydantic V1 中常见的对应方法是 parse_obj()。

    chosen = [Message.model_validate(message) for message in row["chosen"]]
    rejected = [Message.model_validate(message) for message in row["rejected"]]
    # UltraFeedback 两条分支都包含上文。先核对共同 Prompt，再分别取最后一句回答。
    if chosen[:-1] != rejected[:-1]:
        raise ValueError("preference_prompt_mismatch")
    metadata.update({
        "language": "en",
        "preference_source": "model_judgment",
        "score_chosen": row.get("score_chosen"),
        "score_rejected": row.get("score_rejected"),
    })
    return PreferenceExample.model_validate({
        "id": example_id,
        "source": dataset_id,
        "prompt": chosen[:-1],
        "chosen": chosen[-1:],
        "rejected": rejected[-1:],
        "metadata": metadata,
    })
