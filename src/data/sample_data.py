"""内置示例数据（CAIL2024多人多罪格式，用于无真实数据时的演示与测试）"""

import json
from typing import List

from .loader import CAILDataset, CAILParser

_CAIL2024_SAMPLES: List[dict] = [
    {
        "Facts": "2023年7月15日14时许，被告人林某A、林某B经预谋后，窜至本市XX区某商场内，由林某B望风掩护，林某A在三楼运动品牌专柜趁店员不备，将三件运动服（总价值人民币4800元）塞入其随身背包内后离开商场。次日11时许，二人再次到该商场二楼化妆品区域，林某A以试用名义支开店员后，趁机窃取高档面霜两瓶（总价值人民币6200元），被商场安保人员当场抓获。",
        "Court View": "公诉机关指控被告人林某A、林某B犯盗窃罪，事实清楚，证据确实充分。被告人林某A到案后如实供述，林某B有坦白情节，可以从轻处罚。赃物已追回，可酌情从轻处罚。辩护人提出林某B系初犯，建议从轻处罚的意见，本院予以采纳。",
        "Defendants": ["林某A", "林某B"],
        "Outcomes": [
            {
                "Name": "林某A",
                "Judgment": [
                    {
                        "Accusation": "盗窃罪",
                        "Standard Accusation": "盗窃罪",
                        "Related Law": ["264"],
                        "Penalty": {
                            "Imprisonment": 14,
                            "Detention": 0,
                            "Surveillance": 0,
                            "Death Penalty": False,
                            "Life Imprisonment": False,
                            "Fine": 3000,
                            "Fine Without Amount": False,
                        },
                    }
                ],
            },
            {
                "Name": "林某B",
                "Judgment": [
                    {
                        "Accusation": "盗窃罪",
                        "Standard Accusation": "盗窃罪",
                        "Related Law": ["264"],
                        "Penalty": {
                            "Imprisonment": 10,
                            "Detention": 0,
                            "Surveillance": 0,
                            "Death Penalty": False,
                            "Life Imprisonment": False,
                            "Fine": 2000,
                            "Fine Without Amount": False,
                        },
                    }
                ],
            },
        ],
    },
    {
        "Facts": "2024年1月20日21时许，被告人赵某与被害人钱某因感情纠纷在本市XX区某酒吧门口发生口角，赵某持啤酒瓶多次击打钱某头部，致钱某倒地。随后赵某上前踢踹钱某腹部。经鉴定，钱某面部软组织挫伤构成轻微伤，右眶内壁骨折构成轻伤二级，左第12肋骨骨折构成轻伤二级。赵某在打人后报警自首，到案后如实供述上述事实。",
        "Court View": "被告人赵某故意伤害他人身体，致一人轻伤二级，其行为已构成故意伤害罪。赵某犯罪后主动投案，如实供述，系自首，可以从轻处罚。赵某已赔偿被害人损失并取得谅解，可从轻处罚。",
        "Defendants": ["赵某"],
        "Outcomes": [
            {
                "Name": "赵某",
                "Judgment": [
                    {
                        "Accusation": "故意伤害罪",
                        "Standard Accusation": "故意伤害罪",
                        "Related Law": ["234"],
                        "Penalty": {
                            "Imprisonment": 8,
                            "Detention": 0,
                            "Surveillance": 0,
                            "Death Penalty": False,
                            "Life Imprisonment": False,
                            "Fine": 0,
                            "Fine Without Amount": False,
                        },
                    }
                ],
            }
        ],
    },
    {
        "Facts": "2023年9月5日，被告人陈某驾驶其所有的小型轿车沿本市XX路由南向北行驶，因超速行驶，车辆失控撞上路边人行道护栏，随后侧翻，致使该车内乘坐人孙某（陈某之友）受重伤（经鉴定为十级伤残）。被告人陈某在事故发生后报警并在现场等待，到案后如实供述。经检测，陈某血液酒精含量为0mg/100ml，排除酒驾。交警部门认定陈某负全部责任。",
        "Court View": "被告人陈某违反交通运输管理法规，发生重大事故，致一人重伤，负事故全部责任，其行为已构成交通肇事罪。陈某系自首，且自愿认罪认罚，可以从轻处罚。陈某积极赔偿被害人损失并取得书面谅解，可从轻处罚并适用缓刑。",
        "Defendants": ["陈某"],
        "Outcomes": [
            {
                "Name": "陈某",
                "Judgment": [
                    {
                        "Accusation": "交通肇事罪",
                        "Standard Accusation": "交通肇事罪",
                        "Related Law": ["133"],
                        "Penalty": {
                            "Imprisonment": 10,
                            "Detention": 0,
                            "Surveillance": 0,
                            "Death Penalty": False,
                            "Life Imprisonment": False,
                            "Fine": 5000,
                            "Fine Without Amount": False,
                        },
                    }
                ],
            }
        ],
    },
    {
        "Facts": "2024年3月12日15时许，被告人吴某醉酒后在本市XX区地铁站内，无故对正在排队购票的被害人周某进行辱骂并扇打其耳光，后又推搡并拳击另一被害人郑某右面部，致使周某面部软组织挫伤（轻微伤），郑某右耳鼓膜穿孔（轻伤二级）。吴某被当场抓获归案。",
        "Court View": "被告人吴某随意殴打他人，情节恶劣，其行为已构成寻衅滋事罪。吴某有前科劣迹，可酌情从重处罚。吴某当庭自愿认罪认罚，可从轻处罚。",
        "Defendants": ["吴某"],
        "Outcomes": [
            {
                "Name": "吴某",
                "Judgment": [
                    {
                        "Accusation": "寻衅滋事罪",
                        "Standard Accusation": "寻衅滋事罪",
                        "Related Law": ["293"],
                        "Penalty": {
                            "Imprisonment": 16,
                            "Detention": 0,
                            "Surveillance": 0,
                            "Death Penalty": False,
                            "Life Imprisonment": False,
                            "Fine": 3000,
                            "Fine Without Amount": False,
                        },
                    }
                ],
            }
        ],
    },
    {
        "Facts": "2024年5月18日，被告人杨某与同伙唐某（另案处理）共谋后，在本市XX区偏僻路段，由杨某驾驶摩托车尾随被害人谭某，唐某下车抢夺谭某挎包（内有现金1200元、手机一部价值3200元），谭某被拽倒后腿部擦伤。得手后二人驾车逃离。三日后，杨某被抓获，唐某在逃。杨某到案后如实供述，协助警方抓获唐某（在逃尚未归案）。",
        "Court View": "被告人杨某以非法占有为目的，伙同他人以暴力、胁迫方法抢劫公私财物，其行为已构成抢劫罪。杨某到案后如实供述且有协助抓获同案犯的立功表现，可以从轻或减轻处罚。辩护人所提相关辩护意见本院予以采纳。",
        "Defendants": ["杨某"],
        "Outcomes": [
            {
                "Name": "杨某",
                "Judgment": [
                    {
                        "Accusation": "抢劫罪",
                        "Standard Accusation": "抢劫罪",
                        "Related Law": ["263"],
                        "Penalty": {
                            "Imprisonment": 36,
                            "Detention": 0,
                            "Surveillance": 0,
                            "Death Penalty": False,
                            "Life Imprisonment": False,
                            "Fine": 5000,
                            "Fine Without Amount": False,
                        },
                    }
                ],
            }
        ],
    },
    {
        "Facts": "2024年2月10日，被告人刘某在本市XX区某小区门口，使用其捡到的一张社保卡冒名刷卡购药（药品价值人民币1800元）。经查，该社保卡系被害人王某于2024年1月25日丢失。刘某到案后如实供述。家属已代为退赔全部赃款。",
        "Court View": "被告人刘某以非法占有为目的，冒用他人信用卡，进行信用卡诈骗活动，数额较大，其行为已构成信用卡诈骗罪。刘某到案后如实供述，积极退赔，可从轻处罚。",
        "Defendants": ["刘某"],
        "Outcomes": [
            {
                "Name": "刘某",
                "Judgment": [
                    {
                        "Accusation": "信用卡诈骗罪",
                        "Standard Accusation": "信用卡诈骗罪",
                        "Related Law": ["196"],
                        "Penalty": {
                            "Imprisonment": 6,
                            "Detention": 0,
                            "Surveillance": 0,
                            "Death Penalty": False,
                            "Life Imprisonment": False,
                            "Fine": 2000,
                            "Fine Without Amount": False,
                        },
                    }
                ],
            }
        ],
    },
]


def get_sample_cases() -> CAILDataset:
    parser = CAILParser()
    cases = []
    for item in _CAIL2024_SAMPLES:
        try:
            cases.append(parser.parse_record(item))
        except Exception as e:
            continue
    return CAILDataset(cases)


def get_sample_records() -> List[dict]:
    return list(_CAIL2024_SAMPLES)