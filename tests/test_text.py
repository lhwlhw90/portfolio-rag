from rag.answerer import split_subquestions
from rag.text import content_tokens, is_cjk, split_sentences, tokenize


def test_english_tokens_stem_and_split_alnum():
    assert tokenize("Passwords IP54") == ["password", "ip54", "ip", "54"]


def test_chinese_bigrams():
    assert tokenize("年假制度") == ["年假", "假制", "制度"]
    assert tokenize("S7的电池") == ["s7", "的电", "电池"]
    assert "什么" not in content_tokens("年假是什么")


def test_chinese_sentence_split_without_spaces():
    text = "整机保修期为2年，电池保修6个月。人为损坏不在保修范围内！是否支持5GHz？不支持。"
    assert split_sentences(text) == ["整机保修期为2年，电池保修6个月。", "人为损坏不在保修范围内！",
                                     "是否支持5GHz？", "不支持。"]


def test_chinese_hard_wrap_joined_without_space():
    text = "当年未休完的年假可以顺延到次年3月31日，逾期作废，入职不满一年的员工按照实际月份\n折算年假天数。"
    assert split_sentences(text) == ["当年未休完的年假可以顺延到次年3月31日，逾期作废，入职不满一年的员工按照实际月份折算年假天数。"]


def test_is_cjk():
    assert is_cjk("年假") and not is_cjk("vacation")


def test_split_subquestions():
    assert split_subquestions("How long is the warranty, and when was the firmware released?") == [
        "How long is the warranty", "when was the firmware released"]
    assert split_subquestions("每周能在家办公几天？设备补贴有多少？") == ["每周能在家办公几天", "设备补贴有多少"]
    assert split_subquestions("What is the minimum password length?") == ["What is the minimum password length"]


def test_heading_like():
    from rag.text import is_heading_like
    assert is_heading_like("星河 S7 智能扫地机器人 产品说明书")
    assert is_heading_like("Section 2. Incident response")
    assert not is_heading_like("整机保修期为2年。")
    assert not is_heading_like("The X2 SDK supports Python and C++ and a ROS 2 driver is available as a package")


def test_list_bullets_stripped():
    assert split_sentences("- First release for the X2 hardware.\n* Second item here.") == [
        "First release for the X2 hardware.", "Second item here."]
