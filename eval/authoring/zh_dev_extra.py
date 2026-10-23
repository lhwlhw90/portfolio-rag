"""M5：中文评测集新增的 *仅限 dev* 题目（M5 全新编写）。

用途：扩充用来选择和校准可答性判别器的 dev 集。这些题不进入 test。
scripts/build_dev_extra.py 会先检查它们与 test（以及原有 dev）题目既不重复也不近似、
test 文件未被改动，然后写入 eval/datasets/zh.dev_extra.jsonl。

下面每道题都对照 sample_docs/zh 人工核对过：文档里没有题目所问的信息。
大多是"近似题"（实体存在、属性缺失），少数是领域外问题。
"""
Q = [
    ("unanswerable", "丧假一共有几天？", [], []),
    ("unanswerable", "加班费按什么标准计算？", [], []),
    ("unanswerable", "员工离职要提前多久提出申请？", [], []),
    ("unanswerable", "公司年会一般安排在几月份？", [], []),
    ("unanswerable", "住房公积金能提取出来付房租吗？", [], []),
    ("unanswerable", "补充商业医疗保险能给配偶投保吗？", [], []),
    ("unanswerable", "高级体检套餐具体包含哪些检查项目？", [], []),
    ("unanswerable", "出差坐高铁能买一等座吗？", [], []),
    ("unanswerable", "招待客户时的烟酒费用可以报销吗？", [], []),
    ("unanswerable", "研发岗位配的笔记本电脑是什么品牌？", [], []),
    ("unanswerable", "VPN客户端从哪里下载安装？", [], []),
    ("unanswerable", "S7的App能查看每次清扫的面积统计吗？", [], []),
    ("unanswerable", "S7充电座的额定功率是多少瓦？", [], []),
    ("unanswerable", "S7能接入米家App控制吗？", [], []),
    ("unanswerable", "固件v2.0.0更新了哪些内容？", [], []),
    ("unanswerable", "星河大厦一共有几部电梯？", [], []),
    ("unanswerable", "地下车库的充电桩充电要收费吗？", [], []),
    ("unanswerable", "信息安全培训多久必须参加一次？", [], []),
    ("unanswerable", "内部技术分享会在哪个会议室举行？", [], []),
    ("unanswerable", "生产服务器权限到期前系统会提醒吗？", [], []),
    ("unanswerable", "明天杭州会下雨吗？", [], []),
]
