"""Rebuild the deterministic three-city mock and RAG evaluation fixture.

This script is intentionally checked in instead of keeping the generated JSON
and Markdown as an undocumented one-off.  It gives the project a repeatable
fixture-generation command: the same date window, routes, hotel distribution,
weather risks and Gold Chunk IDs can be recreated after a corpus change.

The script is a data migration tool, not production application logic.  It
only writes the explicitly scoped ``data/mock``, ``data/rag_docs`` and
``eval/rag_cases.jsonl`` paths after the caller has made a backup.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = ROOT / "data" / "mock"
RAG_DIR = ROOT / "data" / "rag_docs"
CASES_PATH = ROOT / "eval" / "rag_cases.jsonl"
CHALLENGE_CASES_PATH = ROOT / "eval" / "rag_challenge_cases.jsonl"

CORPUS_VERSION = "v5-agentic-challenge"

START_DATE = date(2026, 10, 1)
WINDOW_DAYS = 7
CITIES = ("杭州", "北京", "上海")
CITY_CODE = {"杭州": "hgh", "北京": "pek", "上海": "sha"}
AIRPORTS = {
    "杭州": ("杭州萧山国际机场", "HGH"),
    "北京": ("北京首都国际机场", "PEK"),
    "上海": ("上海虹桥国际机场", "SHA"),
}


def dates() -> list[date]:
    """Return exactly the seven supported departure/arrival dates."""

    return [START_DATE + timedelta(days=offset) for offset in range(WINDOW_DAYS)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 JSON with stable ordering for reviewable diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_flights() -> dict[str, Any]:
    """Build 6 directed routes x 7 dates x 4 flights = 168 records.

    Every directed pair is materialised independently.  This is important for
    testing an origin that is not fixed to Hangzhou: a return query must find
    the reverse route rather than relying on an accidental one-way fixture.
    """

    times = (("07:20", 160), ("10:10", 155), ("13:40", 170), ("18:30", 165))
    route_base = {
        ("杭州", "北京"): 720,
        ("北京", "杭州"): 680,
        ("杭州", "上海"): 420,
        ("上海", "杭州"): 390,
        ("北京", "上海"): 760,
        ("上海", "北京"): 730,
    }
    items: list[dict[str, Any]] = []

    for depart_day in dates():
        for origin in CITIES:
            for destination in CITIES:
                if origin == destination:
                    continue
                origin_code = CITY_CODE[origin]
                destination_code = CITY_CODE[destination]
                airport_origin = AIRPORTS[origin][0]
                airport_destination = AIRPORTS[destination][0]
                for index, (depart_time, duration) in enumerate(times, start=1):
                    flight_id = (
                        f"flight_{origin_code}_{destination_code}_"
                        f"{depart_day:%Y%m%d}_{index:02d}"
                    )
                    price = route_base[(origin, destination)] + (index - 1) * 55
                    arrive_at = _add_minutes(depart_time, duration)
                    items.append(
                        {
                            "flight_id": flight_id,
                            "flight_no": f"MO{depart_day.day:02d}{index}{origin_code.upper()[:1]}",
                            "airline": ["东航示例", "国航示例", "吉祥示例", "春秋示例"][index - 1],
                            "departure_city": origin,
                            "arrival_city": destination,
                            "departure_airport": airport_origin,
                            "arrival_airport": airport_destination,
                            "depart_date": depart_day.isoformat(),
                            "depart_time": depart_time,
                            "arrive_date": depart_day.isoformat(),
                            "arrive_time": arrive_at,
                            "duration_minutes": duration,
                            "price": price,
                            "available_seats": 9,
                            "cabin": "economy",
                            "baggage": "20kg",
                            "is_direct": True,
                            "refundable": index != 4,
                            "changeable": True,
                            "updated_at": "2026-09-30T08:00:00+08:00",
                        }
                    )

    assert len(items) == 168
    return {
        "provider": "mock_flight_provider",
        "version": 4,
        "scope": {
            "date_start": "2026-10-01",
            "date_end": "2026-10-07",
            "cities": list(CITIES),
            "route_policy": "all_directed_city_pairs",
        },
        "currency": "CNY",
        "items": items,
    }


def _add_minutes(hhmm: str, minutes: int) -> str:
    hour, minute = (int(part) for part in hhmm.split(":"))
    total = hour * 60 + minute + minutes
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def build_hotels() -> dict[str, Any]:
    """Build 11 hotels per city with explicit, queryable preference fields."""

    profiles = [
        ("经济型", 2, 268, 4.2, 0.72, True),
        ("经济型", 2, 318, 4.3, 0.76, True),
        ("中档型", 2, 498, 4.5, 0.84, True),
        ("中档型", 2, 568, 4.6, 0.86, True),
        ("高档型", 2, 828, 4.7, 0.90, True),
        ("高档型", 2, 1080, 4.8, 0.93, True),
        ("地铁便利", 2, 458, 4.5, 0.82, True),
        ("商圈便利", 2, 628, 4.6, 0.78, True),
        ("安静型", 2, 528, 4.6, 0.96, True),
        ("安静型", 2, 688, 4.7, 0.98, True),
        ("明显不足", 1, 238, 3.4, 0.36, False),
    ]
    districts = {
        "杭州": ("拱墅区", "武林商圈", "杭州东站").__iter__(),
        "北京": ("东城区", "朝阳商圈", "北京站").__iter__(),
        "上海": ("静安区", "人民广场商圈", "上海虹桥站").__iter__(),
    }
    items: list[dict[str, Any]] = []

    for city in CITIES:
        district, business_area, station = districts[city]
        city_code = CITY_CODE[city]
        for index, (kind, count, price, rating, quiet, near_subway) in enumerate(profiles, start=1):
            # ``count`` documents the requested room inventory per category;
            # the actual hotel count remains one record for each category row.
            suffix = f"{index:02d}"
            hotel_id = f"hotel_{city_code}_{suffix}"
            name_by_kind = {
                "经济型": f"{city}{district}轻住酒店{suffix}",
                "中档型": f"{city}{business_area}城市精选{suffix}",
                "高档型": f"{city}{business_area}云庭酒店{suffix}",
                "地铁便利": f"{city}{station}换乘酒店{suffix}",
                "商圈便利": f"{city}{business_area}步行酒店{suffix}",
                "安静型": f"{city}{district}静巷酒店{suffix}",
                "明显不足": f"{city}旧站口特价客房{suffix}",
            }
            items.append(
                {
                    "hotel_id": hotel_id,
                    "name": name_by_kind[kind],
                    "city": city,
                    "district": district,
                    "address": f"{city}{district}示例路{index}号，靠近{station}",
                    "price_per_night": price,
                    "rating": rating,
                    "available_rooms": count,
                    "near_subway": near_subway,
                    "distance_to_subway_meters": 220 + index * 35 if near_subway else 1180,
                    "quiet_score": quiet,
                    "cleanliness_score": 0.88 if kind != "明显不足" else 0.61,
                    "hotel_type": kind,
                    "tags": [kind, district, "示例可订", "近" + station],
                    "amenities": ["WiFi", "空调", "行李寄存", "自助洗衣"],
                    "cancel_policy": "入住前24小时可免费取消" if kind != "明显不足" else "不可免费取消",
                    "notes": "公共区域和隔音维护较弱，适合作为价格敏感的备选" if kind == "明显不足" else "房型、价格和可取消条款以预订时为准",
                    "updated_at": "2026-09-30T09:00:00+08:00",
                }
            )

    assert len(items) == 33
    assert all(sum(1 for item in items if item["city"] == city) == 11 for city in CITIES)
    return {
        "provider": "mock_hotel_provider",
        "version": 4,
        "scope": {"date_start": "2026-10-01", "date_end": "2026-10-07", "cities": list(CITIES)},
        "currency": "CNY",
        "items": items,
    }


def build_weather() -> dict[str, Any]:
    """Build one deterministic seven-day weather window for every city."""

    patterns = [
        ("晴", 15, 25, "微风", 5, [], []),
        ("多云", 16, 24, "东风2级", 15, [], []),
        ("小雨", 17, 23, "东南风3级", 65, ["rain"], ["建议携带雨具，室外路线预留遮雨点。"]),
        ("高温晴", 22, 34, "南风2级", 10, ["heat"], ["午间减少暴晒，安排室内休息和补水。"]),
        ("多云", 18, 26, "北风6级", 20, ["wind"], ["高架步道和江边活动注意阵风，收好轻小物品。"]),
        ("强降雨", 19, 25, "东北风5级", 90, ["heavy_rain"], ["优先室内活动，关注积水、雷电和交通延误。"]),
        ("晴间多云", 16, 27, "微风", 10, [], []),
    ]
    items: list[dict[str, Any]] = []
    for city in CITIES:
        for index, day in enumerate(dates()):
            condition, low, high, wind, precipitation, risks, warnings = patterns[index]
            items.append(
                {
                    "weather_id": f"weather_{CITY_CODE[city]}_{day:%Y%m%d}",
                    "city": city,
                    "date": day.isoformat(),
                    "condition": condition,
                    "temperature_low": low,
                    "temperature_high": high,
                    "humidity": 48 + index * 4,
                    "wind": wind,
                    "precipitation_probability": precipitation,
                    "risk_tags": risks,
                    "warnings": warnings,
                    "updated_at": "2026-09-30T06:00:00+08:00",
                }
            )
    assert len(items) == 21
    return {
        "provider": "mock_weather_provider",
        "version": 4,
        "scope": {"date_start": "2026-10-01", "date_end": "2026-10-07", "cities": list(CITIES)},
        "items": items,
    }


CITY_TEXT = {
    "杭州": {
        "guide": [
            ("运河与小河直街", "杭州的城市文化可以从拱宸桥、桥西历史街区和小河直街串成一条水岸主线。上午先看运河博物馆，午后在桥西工艺作坊停留，晚餐选择本地面馆或小馆，不把西湖和运河塞进同一天。"),
            ("西湖低步行路线", "西湖慢游宜把断桥、北山街和曲院风荷拆成短段，地铁站或公交站作为撤退点。想看湖景时优先坐船或沿树荫走一段，遇到人流密集就回到安静茶室，不用追求环湖一周。"),
            ("展览与美术馆", "杭州的展览安排适合采用一主一备：主场馆可选中国美术学院美术馆或良渚博物院，备用点放在同一交通方向的设计展。提前核对周一闭馆、预约时段和身份证入场规则，避免跨区赶场。"),
            ("龙井茶园体验", "龙井村体验的重点是茶园地形、炒茶工艺和一顿简洁的农家饭。不要把多个山路观景点连续排在午后；从梅家坞或龙井村选择一个入口，给返程公交和下坡路留出余量。"),
            ("本地饮食安排", "想吃得丰富，可以将片儿川、葱包桧、定胜糕和杭帮菜分在不同餐段，而不是每顿寻找网红店。武林、湖滨和河坊街的排队长度差异明显，优先选择能线上取号、离地铁近的店。"),
            ("雨天替代方案", "小雨天把湖边长距离步行换成运河博物馆、茶文化展和商场连廊；强降雨时不安排桥面拍照或临水骑行。雨停后先确认地面积水和公交恢复情况，再决定是否恢复户外路线。"),
            ("夜间街区", "杭州夜间可以安排湖滨步行、南宋御街小吃或大运河夜游三选一。夜游核心是控制往返交通和人流，不建议晚饭后再跨城去多个打卡点；返程前保存地铁末班时间。"),
            ("节奏与休息", "三天行程每天保留一个明确主题，上午安排信息量大的场馆，午后留一段咖啡店或酒店休息。携带长者时，把西湖和茶园分别放在不同日期，避免连续爬坡和长距离换乘。"),
        ],
        "hotel": [
            ("位置与到达", "这家杭州酒店靠近武林门一带，步行到地铁站约三百米，去西湖、运河和杭州东站都能通过一次换乘完成。带箱子时应确认地铁出口是否有电梯，网约车下客点不一定就在大堂门口。"),
            ("安静程度", "内侧高楼层房间远离主干道，夜间更适合早睡和第二天赶展览的人。临街房仍可能听到公交和商圈收摊声，轻睡眠住客预订时应备注避开电梯、餐厅和消防门。"),
            ("卫生与维护", "住客反馈显示床品、卫生间和公共走廊维护较稳定，连续住两三晚不会频繁遇到清洁问题。入住后仍应检查排水、空调滤网和窗帘遮光，发现异常及时拍照联系前台。"),
            ("地铁与商圈", "酒店到武林广场、湖滨和运河方向的交通选择多，晚餐后回酒店不必穿过大段无照明道路。商圈便利也意味着周末人流增加，追求安静时不要只看地图上的商业距离。"),
            ("房间设施", "房间提供稳定网络、书桌、热水和行李寄存，适合白天外出、晚上整理照片的短住客。洗衣房和早餐的开放时间可能随日期调整，长住前最好向前台确认而不是默认全天可用。"),
            ("适合人群", "它适合重视地铁、清洁和夜间睡眠的城市游客，也适合需要去杭州东站或武林商圈办公的人。若主要需求是度假景观、泳池或大面积公共空间，应比较城西度假型住宿。"),
            ("可能的不足", "不足是公共休闲空间不大，部分低楼层房间会受到道路声音影响，周末早餐也可能排队。价格与房型差异明显，预订时应同时查看窗向、楼层和取消政策。"),
            ("价格判断", "参考价位属于中档城市酒店，适合把预算更多留给展览、餐饮和夜间活动。实际价格会随国庆窗口和房型变化，不能只凭历史评分判断当天是否值得预订。"),
        ],
        "packing": [
            ("证件与预约", "身份证、航班订单、酒店确认单和博物馆预约码保存电子版与离线截图。杭州地铁和场馆入口可能分别需要不同二维码，出发前把东站、酒店和主要场馆地址写成文字。"),
            ("衣物与鞋", "十月城市游准备速干上衣、薄外套和防滑步行鞋；西湖湖岸早晚温差比商场明显。每天安排一个主线即可，不要因为带了多套衣服就把行程排成连续换装打卡。"),
            ("雨具与防晒", "折叠伞、防晒霜和轻便防水袋适合运河、湖滨和短时阵雨。强降雨日不要把防水装备当成继续骑行的理由，应把相机、预约凭证和备用袜子分区密封。"),
            ("电子设备", "手机、移动电源、充电线和耳机足够应对地铁导航、展览导览与夜间拍照。需要带相机时准备一只防震袋，移动电源按航空规定携带并在出发前确认电量。"),
            ("药品与补水", "常用药按原包装携带，容易晕车的人可准备熟悉的晕车药。湖边和茶园步行时带小水瓶与纸巾，出现头晕、恶心或明显疲劳应先休息，不要为了完成路线硬撑。"),
            ("低步行加项", "低步行行程可带折叠坐垫、轻便雨披和小零食，每半天保留一个可随时取消的节点。长者同行时把药品、证件和水分开装，避免一个背包过重。"),
            ("行李整理", "证件、药品、充电设备和一套换洗衣物放在随身包；雨具和液体分别密封。三到七天城市游一件主行李加日用包通常够用，返程预留给茶叶和伴手礼的空间。"),
            ("出发前复核", "按证件、交通、住宿、预约、天气、药品、充电顺序复核。把杭州东站和酒店的文字地址另存一份，网络不稳定或地铁换乘匆忙时仍能快速交给工作人员。"),
        ],
    },
    "北京": {
        "guide": [
            ("博物馆主线", "北京文化游适合用一个大型博物馆加一个小型专题馆组成主线，例如国家博物馆配合中国美术馆。上午看信息量大的展厅，午后安排同区午餐和休息，不把故宫、国博和多个展馆压缩在一天。"),
            ("胡同与社区", "胡同体验应选择东四、史家或杨梅竹斜街其中一片，先看街巷肌理，再进入书店、社区咖啡和小型展览。胡同不是景点清单，尊重居民生活、控制拍照距离比走遍所有巷口更重要。"),
            ("展览预约", "故宫、国博和热门专题展常有实名预约、安检和分时入场。出发前保存官方预约成功页，确认闭馆日、检票门和证件要求；第三方平台过期信息不能作为到场依据。"),
            ("城市文化", "北京城市文化可以用中轴线、旧书店、戏剧空间和社区餐馆串联。每天保留一个文化核心，其余时间用于阅读展签、吃饭和观察街区，避免把文化体验变成跨区奔波。"),
            ("本地美食", "美食安排可按区域选择豆汁焦圈、炸酱面、涮肉或清真小吃，不必一天吃遍名店。热门店排队会消耗博物馆时段，优先选择能预约或离地铁近的替代店。"),
            ("天气替代", "小雨和高温时把露天长线换成博物馆、商场连廊和剧场；大风或强降雨预警时取消城墙边缘、开放广场和树下停留。天气转好后先重新检查交通与场馆公告。"),
            ("夜间活动", "夜间可在前门、鼓楼或亮马河选择一个区域散步、看演出或吃夜宵。晚间活动要与酒店地铁方向一致，提前确认末班车和网约车上车点，不建议连续跨越多个城区。"),
            ("步行管理", "北京景区和博物馆的入口、安检、展厅之间距离常被低估。每看完一个高密度展区安排十分钟坐下，长者同行优先使用电梯、馆内座椅和短路线。"),
        ],
        "hotel": [
            ("到达动线", "这家北京酒店位于东城区地铁沿线，去国博、东四胡同和前门都方便。带箱子时要确认地铁出口是否有电梯，历史街区的石板路会让最后三百米比地图显示更耗时。"),
            ("夜间睡眠", "内向房和远离电梯的高楼层对早起看展的人更友好，临街房可能听到公交、外卖和胡同夜间聊天。预订时明确提出安静房请求，并把耳塞作为备用。"),
            ("清洁维护", "客房床品、卫生间和公共区域的清洁反馈相对稳定，适合三到七天城市短住。入住后仍要检查热水、空调和窗帘遮光，任何小问题尽早通知前台处理。"),
            ("地铁与文化区", "酒店到东四、王府井和前门的换乘关系清晰，晚饭后回程不需要穿过偏僻道路。靠近文化区也意味着节假日人流较高，安静诉求需要结合房间朝向判断。"),
            ("房间与办公", "房间有书桌、网络、热水和行李寄存，适合白天看展、晚上整理资料。公共健身和餐饮配套较简单，长时间不出门办公的人应提前确认早餐、洗衣和插座数量。"),
            ("适用旅客", "它适合博物馆、胡同和城市文化路线，也适合需要地铁换乘的商务短住。若希望拥有大堂社交、泳池或度假景观，则应选择更大型的综合酒店。"),
            ("潜在不足", "部分房型面积偏紧，周末早餐可能排队，临街低楼层会有交通声；安静型房间通常数量有限。预订前应核对楼层、窗向、早餐和取消规则，不能只看总评分。"),
            ("预算取舍", "中档价格可以把更多预算留给展览、演出和餐饮，适合不把酒店公共设施当作主要活动的人。国庆窗口价格波动明显，建议比较含早、可取消和地铁距离后的总成本。"),
        ],
        "packing": [
            ("证件预约", "身份证、博物馆预约码、酒店订单和返程航班分别保存截图。国博、故宫等场馆检票规则不同，出发前记录检票门、闭馆日和地铁出口，避免在安检口反复查找。"),
            ("胡同步行装备", "胡同街巷路面高低不一，准备防滑鞋、薄外套和小容量水壶。行程以一片街区为单位，别因穿了舒适鞋就把多个相距很远的胡同串成马拉松。"),
            ("温差防护", "十月早晚温差可能明显，薄外套、可叠穿上衣和防风层比厚重单件更灵活。博物馆空调、地铁站台和户外中轴线的体感差异要同时考虑。"),
            ("雨风装备", "折叠伞、防风帽和防水袋应放在随身包外层。遇到大风或强降雨，装备只能帮助短距离移动，不能替代取消露天平台和开放广场的安全决策。"),
            ("电子设备", "手机、充电线、移动电源和耳机足够导航与听讲解。需要拍摄展览时准备静音相机设置和备用存储，不要把闪光灯、三脚架等可能被场馆限制的设备带进核心展厅。"),
            ("常用药品", "按原包装带好处方药和个人过敏信息，久站看展可准备创可贴。高温或人流密集时，饮水和休息比继续排队更重要，同行者应知道集合点和紧急联系人。"),
            ("行李分层", "证件、预约码、药品和一套换洗衣物放在随身包，雨具与液体分开密封。三日文化游主行李不必过大，给书籍、文创和伴手礼留出返程空间。"),
            ("复核顺序", "按交通、场馆预约、酒店、天气、药品和充电顺序复核。把北京站、首都机场和酒店地址保存成文字，网络拥堵时可以直接出示给司机或工作人员。"),
        ],
    },
    "上海": {
        "guide": [
            ("博物馆与展览", "上海适合把人民广场博物馆群、当代艺术展和一处独立书店拆成两天，上午安排预约场馆，下午留给咖啡与街区观察。展馆之间优先选同一地铁方向，减少在高峰时段跨江。"),
            ("里弄街区", "武康路、愚园路或苏州河沿线各自足够走半天，体验重点是里弄建筑、社区店铺和街角展览。不要把多个网红街区作为打卡清单，给居民道路和小店正常经营留出空间。"),
            ("城市文化", "上海城市文化可以从近代建筑、设计展、剧场和咖啡馆切入。每天留一个核心主题，晚上选择一场小型演出或书店活动，比连续排外滩、豫园和多个展馆更有连贯性。"),
            ("美食分区", "小笼、生煎、本帮菜和夜间甜品可以按静安、黄浦、徐汇分开体验。热门店排队时优先找同一商圈的备选，避免为了一个店跨越整座城市而错过预约场次。"),
            ("外滩安排", "外滩和苏州河适合安排在早晚风力较弱的时段，白天用博物馆或商场休息替代长时间暴晒。拍照点不必全部走完，地铁站和室内商场都应作为撤退节点。"),
            ("雨天替代", "小雨可以转入展馆、书店和商场连廊；强降雨时避免外滩临江步道、低洼街巷和临时骑行。雨停后先看积水、地铁延误和场馆公告，再恢复户外路线。"),
            ("夜生活选择", "夜间可从安福路演出、静安夜宵或黄浦江室内观景中选一项。活动结束后尽量沿同一条地铁线回酒店，提前确认末班车和打车上车位置。"),
            ("低密度节奏", "上海地铁换乘看似快捷，但站内步行和出入口距离会累积体力。每天安排一到两个主节点，下午保留一段坐下整理照片或临时调整路线的时间。"),
        ],
        "hotel": [
            ("交通位置", "这家上海酒店靠近静安与人民广场方向的地铁网络，去展馆、里弄和虹桥枢纽都较方便。带行李时要确认出口电梯和雨天遮挡，最后一段步行可能穿过繁忙路口。"),
            ("夜间安静", "高楼层内侧房比临街房更适合睡眠，周末商圈和外卖车辆仍可能带来短时声音。对噪声敏感的人应预订时备注远离电梯、酒吧和主干道，并准备耳塞。"),
            ("清洁和维护", "客房清洁、床品和卫浴维护在近期反馈中较稳定，适合把酒店作为白天行程之间的休息基地。入住后检查空调、热水和窗帘遮光，问题尽早反馈更容易处理。"),
            ("商圈便利", "周边餐饮、便利店和地铁出口密集，晚间临时改变计划也容易找到替代餐厅。便利的另一面是节假日人流与车辆较多，安静程度不能只由商圈距离判断。"),
            ("设施边界", "房间提供网络、书桌、行李寄存和基础洗衣服务，适合城市短住与轻办公。公共泳池、会议厅和大型餐饮不是它的强项，需要这些设施的旅客应看清房型和酒店定位。"),
            ("适合人群", "它适合博物馆、街区、美食和夜间活动为主的旅客，也适合需要往返虹桥枢纽的人。若更重视江景、度假设施或酒店内娱乐，应比较外滩及大型综合酒店。"),
            ("明显不足", "部分低楼层房间受道路声音影响，公共休息区偏小，热门早餐时段可能需要排队。预订时要核对房间朝向、楼层、早餐和取消条款，低价不一定代表综合成本最低。"),
            ("价格与选择", "中档价位能把预算留给展览、餐饮和演出，适合白天大部分时间在外活动的人。国庆期间建议以含早、可取消和地铁距离后的总价比较，不要只看每晚标价。"),
        ],
        "packing": [
            ("订单与证件", "身份证、航班与酒店订单、展馆预约码保存离线截图。上海不同场馆的入口和检票规则差异大，把人民广场、虹桥枢纽和酒店地址另存为文字更稳妥。"),
            ("湿热与温差", "准备速干上衣、薄外套和舒适防滑鞋；地铁、商场和展馆空调可能较强。外滩风大，轻薄防风层比厚重外套更适合在室内外切换。"),
            ("雨具防水", "折叠伞、轻便雨衣、防水袋和备用袜子放在随身包外层。雷雨或积水预警时，应把外滩和低洼街区换成室内路线，不要把装备当成继续暴露的许可。"),
            ("拍摄设备", "手机、移动电源、充电线和耳机可以覆盖导航、导览和夜间拍照。带相机时使用防潮袋并准备静音设置，展馆拍摄规则要以现场提示为准。"),
            ("药品补水", "常用药按原包装携带，随身水瓶和纸巾适合里弄、展馆排队与地铁换乘。若出现头晕、恶心或疲劳，优先找室内座位和饮水点，不要继续追赶下一个展馆。"),
            ("低步行装备", "低步行方案可带轻便坐垫、折叠雨披和小零食，每半天保留一个可取消节点。老人同行时将药品、证件和水分开装，减少一人背包负担。"),
            ("行李收纳", "证件、药品、充电设备和一套替换衣物放在随身包，雨具、液体和可能潮湿的衣物分别密封。三到七天城市游一件主行李加日用包足够。"),
            ("返程复核", "出发前按证件、交通、住宿、预约、天气、药品、充电顺序检查。把酒店到虹桥枢纽的地铁路线和备用打车点保存为截图，避免返程高峰时临时搜索。"),
        ],
    },
}


SAFETY_TEXT = {
    "杭州": {
        "rain": ("杭州小雨下的湖岸与运河安全", "雨天路面湿滑，西湖石板路、桥面和运河亲水平台应缩短停留。鞋底防滑、手机与预约码放入防水袋，遇到能见度下降时优先转入博物馆和商场。"),
        "heat": ("杭州高温城市步行安全", "高温日把茶园、湖岸等户外段放在早晚，午间进入展馆或酒店休息。少量多次补水，出现头晕、恶心或乏力要立即停止行走并寻找空调空间。"),
        "wind": ("杭州大风时临水活动提醒", "北风增强时减少钱塘江边、湖面游船和高架观景平台停留，帽子、雨伞和轻小物品要收好。看到临时封闭提示时不要绕过围挡。"),
        "heavy_rain": ("杭州强降雨交通与积水提醒", "强降雨期间不安排骑行、桥面拍照和低洼街区长距离步行，关注地铁出入口积水与线路延误。行程应保留室内替代点和返程缓冲。"),
    },
    "北京": {
        "rain": ("北京小雨胡同步行提醒", "小雨会让胡同砖石路和台阶变滑，雨具应放在随身包外层。拍照时避开车道和居民门口，行程可切换到同区域博物馆、书店或茶馆。"),
        "heat": ("北京高温文化游提醒", "高温日把中轴线、故宫外围等露天段放在清晨或傍晚，午间安排博物馆、商场和补水。出现头晕、恶心、异常乏力时立即进入有空调的室内。"),
        "wind": ("北京大风开放空间提醒", "大风天气减少城墙、广场、湖边和高处平台停留，注意落物与临时围挡。不要在孤立树下或金属栏杆旁等待，改用地铁站内或有管理的室内空间。"),
        "heavy_rain": ("北京强降雨出行提醒", "强降雨可能造成胡同积水、场馆入口排队变化和道路拥堵，取消露天长线，优先确认国博、故宫等场馆公告。返程至少预留一小时交通缓冲。"),
    },
    "上海": {
        "rain": ("上海小雨里弄与展馆转换", "小雨时里弄石板、斜坡和地铁出入口容易湿滑，雨具、备用袜子和防水袋放在随身包。室外路线缩短后，可转入展馆、书店或商场连廊。"),
        "heat": ("上海高温湿热活动提醒", "高温湿热日把外滩与街区长走安排在早晚，午间进入有空调的展馆或酒店。补水、休息和降低背包重量比完成更多打卡点更重要。"),
        "wind": ("上海江边大风提醒", "大风天气减少外滩、苏州河岸和高层观景平台停留，固定帽子、雨伞与相机背带。遇到临江步道关闭或广播提示时应立即转入室内。"),
        "heavy_rain": ("上海强降雨积水与交通提醒", "强降雨时避开外滩低洼段、地下通道和临时积水路口，先确认地铁出入口及场馆是否正常开放。保留室内备选和虹桥返程缓冲时间。"),
    },
}


# Agentic RAG 专用的精细偏好语料。每个章节对应一个独立
# Information Need，使 Planner 可以为同一 guides 类别拆出多条
# query，Judge 也能按 need 判断是否真正覆盖。
AGENTIC_GUIDE_TEXT = {
    "杭州": [
        ("本地味道与排队备选", "想体验杭州本地味道，可把片儿川与葱包桧放在早午餐，晚餐再选一家杭帮菜小馆。湖滨排队过长时转向武林或拱宸桥同类店，每顿只保留一个必吃项，避免为网红店跨区往返。"),
        ("运河文化深度体验", "文化线可以从中国京杭大运河博物馆开始，步行到桥西历史文化街区看传统工艺，再在小河直街休息。关注运河交通、仓储和手工业的关系，比只在桥边拍照更能理解城市发展。"),
        ("长者同行低步行方案", "长者同行时，西湖只选断桥到北山街的短段，用游船或公交代替环湖步行。每四十分钟设一个茶室、展馆或酒店休息点，茶园与西湖不排在同一天，并提前确认地铁电梯出口。"),
        ("亲子雨天室内备案", "带孩子遇雨时，上午可选运河博物馆的交互展项，午后转到同方向的手工作坊或茶文化展。每个场馆之间只保留一段短途交通，准备换洗袜子和小零食，强降雨时取消湖岸与桥面活动。"),
        ("三十六小时商务短停", "三十六小时商务行程应围绕杭州东站、武林商圈和酒店同一条地铁方向。会议后只保留一项本地体验：就近吃片儿川或在运河边短距离散步，返程前留出九十分钟枢纽缓冲。"),
        ("安静摄影时段", "想拍照又不喜欢拥挤，可在日出后一小时内拍北山街与西湖树影，晚间则选择小河直街外围而不是湖滨核心打卡位。相机使用肩带，不占用居民通道，风雨增强时转入展馆。"),
    ],
    "北京": [
        ("本地味道与排队备选", "北京本地餐食可把炸酱面、涎肉和清真小吃分到不同餐段。前门名店排队过长时，改选东四或护国寺同类老店，不让排队挤占国博、故宫等实名预约时段。"),
        ("中轴线与胡同文化", "中轴线文化不必覆盖全程，可先在正阳门或鼓楼理解城市空间，再进入东四或史家胡同观察社区生活。选一家书店或小型展览作为收尾，避免把居民街巷当成纯拍照布景。"),
        ("长者同行低步行方案", "北京场馆从地铁口到安检口的距离容易被低估。长者同行时，一天只选一个大型场馆，馆内优先电梯与短展线，午后在同区茶馆休息；胡同只走一片，不与故宫长线连排。"),
        ("亲子雨天室内备案", "雨天亲子行程可选一个有交互展项的科技或自然类博物馆，午后转到同方向的绘本书店或剧场。提前保存实名预约码，强降雨时取消广场、城墙和长距离胡同步行。"),
        ("三十六小时商务短停", "北京商务短停应尽量让首都机场、亮马桥商务区和酒店保持单一交通主线。会议后只安排东四短逛或一顿涎肉，不再横跨城区追景点；返程按安检和高峰拥堵留足缓冲。"),
        ("安静摄影时段", "安静摄影可选清晨的史家胡同外围或日落前的亮马河普通步道，避开升旗和网红街区峰值。不用三脚架堵路，不对居民门窗长时拍摄，大风时取消开放高处机位。"),
    ],
    "上海": [
        ("本地味道与排队备选", "上海本地味道可把小笼、生煎和本帮菜分到黄浦、静安和徐汇的不同餐段。人民广场附近排队过长时，在同一商圈选菜品相近的备选，不为一家店错过展馆预约。"),
        ("里弄与近代城市文化", "上海城市文化可先在一处建筑或设计展理解租界与工业变迁，再到愚园路或苏州河沿线观察里弄、仓库和社区店铺。每天只选一片街区，用书店或剧场补充背景，而不是连续打卡。"),
        ("长者同行低步行方案", "长者同行可以人民广场一处博物馆为核心，出馆后只在同方向餐厅休息；里弄路线控制在一小时内，优先有座椅和地铁撤退点的路段。不把外滩、豫园和武康路排在同一天。"),
        ("亲子雨天室内备案", "雨天带孩子可在人民广场选一个有交互内容的博物馆，午后转入同区书店、儿童剧或商场休息区。备好换洗袜子，强降雨时取消外滩、低洼里弄与骑行，先确认地铁出入口。"),
        ("三十六小时商务短停", "上海三十六小时商务短停可以虹桥枢纽、静安商务区和酒店为主线。会议后留一项小笼晚餐或愚园路短逛，避免临时跨江；返程时把工作日高峰和虹桥安检算入缓冲。"),
        ("安静摄影时段", "想避开人群拍摄，可在工作日清晨选苏州河普通河段，或在日落前拍愚园路建筑细节，不把外滩核心机位当成必选。相机肩带要固定，大风或雨天转入室内展馆。"),
    ],
}


def build_agentic_decoy_sections(city: str) -> list[tuple[str, str]]:
    """构造词面相关但不足以支持具体 need 的干扰文档。"""

    return [
        ("热门一日全收集", f"{city}热门一日游可同时安排美食、文化、拍照和夜景，尽量多去几个热门地点。本条只是主题概述，不包含餐饮区域、场馆、交通或休息点等可执行信息。"),
        ("轻松但紧凑的打卡", f"如果想在{city}玩得轻松，可以从早到晚紧凑打卡多个片区，并在路上灵活休息。文本没有提供低步行距离、座椅、电梯或撤退点，不能作为长者行程依据。"),
        ("雨天什么都能玩", f"{city}雨天仍然有很多选择，亲子、美食、商务和摄影行程都可以继续。本条没有给出室内场馆、强降雨取消条件或儿童照护措施，只能作为宽泛主题提示。"),
        ("任意人群通用攻略", f"这份{city}攻略声称同时适合孩子、长者、商务旅客和摄影爱好者。由于没有区分体力、时间、天气和交通限制，不能直接证明任何一项特定偏好已被覆盖。"),
    ]


def frontmatter(metadata: dict[str, Any]) -> str:
    """Render the small YAML subset used by the document loader."""

    lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, list):
            rendered = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.extend(["---", ""])
    return "\n".join(lines)


def write_rag_documents() -> list[dict[str, Any]]:
    """Replace the active corpus with baseline and Agentic challenge documents."""

    # The caller already made a backup.  Remove only Markdown files and leave
    # the four required category directories in place for a clean migration.
    for markdown in RAG_DIR.rglob("*.md"):
        markdown.unlink()
    for category in ("guides", "hotel_reviews", "safety_notices", "packing_checklists"):
        (RAG_DIR / category).mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    hotel_catalog = {
        item["hotel_id"]: item
        for item in build_hotels()["items"]
    }
    for city in CITIES:
        city_code = CITY_CODE[city]
        for doc_type, sections in (
            ("guide", CITY_TEXT[city]["guide"]),
            ("hotel_reviews", CITY_TEXT[city]["hotel"]),
            ("packing_checklist", CITY_TEXT[city]["packing"]),
        ):
            prefix = {"guide": "guide", "hotel_reviews": "hotel_review", "packing_checklist": "packing"}[doc_type]
            doc_id = f"{prefix}_{city_code}_scope_v1"
            filename = f"{doc_id}.md"
            metadata: dict[str, Any] = {
                "doc_id": doc_id,
                "title": f"{city}{'城市行程与体验' if doc_type == 'guide' else '城市酒店体验' if doc_type == 'hotel_reviews' else '城市旅行行李清单'}",
                "doc_type": doc_type,
                "city": city,
                "scenario": ["scope_window", "distinct_topic"],
                "language": "zh-CN",
                "corpus_version": CORPUS_VERSION,
            }
            if doc_type == "hotel_reviews":
                selected_hotel = hotel_catalog[f"hotel_{city_code}_10"]
                metadata.update({
                    "hotel_id": selected_hotel["hotel_id"],
                    "hotel_name": selected_hotel["name"],
                    "district": selected_hotel["district"],
                })
            body = "\n\n".join(f"## {title}\n{text}" for title, text in sections)
            path = RAG_DIR / {"guide": "guides", "hotel_reviews": "hotel_reviews", "packing_checklist": "packing_checklists"}[doc_type] / filename
            path.write_text(frontmatter(metadata) + body + "\n", encoding="utf-8")
            documents.append({"doc_id": doc_id, "doc_type": doc_type, "city": city, "path": path})

        # 每城一份精细偏好文档和一份近义干扰文档。两者都进入
        # 真实索引：Retriever 需要区分可执行证据与“只是词面像”的文本，
        # Evidence Judge 则用 success criteria 拒绝不充分的干扰 chunk。
        for suffix, title, sections, scenarios in (
            (
                "preferences",
                f"{city}Agentic多偏好行程证据",
                AGENTIC_GUIDE_TEXT[city],
                ["agentic_challenge", "need_specific"],
            ),
            (
                "decoys",
                f"{city}Agentic近义干扰攻略",
                build_agentic_decoy_sections(city),
                ["agentic_challenge", "semantic_decoy"],
            ),
        ):
            doc_id = f"guide_{city_code}_agentic_{suffix}_v1"
            metadata = {
                "doc_id": doc_id,
                "title": title,
                "doc_type": "guide",
                "city": city,
                "scenario": scenarios,
                "activity_types": [
                    "food",
                    "culture_history",
                    "low_walking_intensity",
                    "family_children",
                    "business_short_stay",
                    "photography",
                ],
                "language": "zh-CN",
                "corpus_version": CORPUS_VERSION,
            }
            body = "\n\n".join(
                f"## {section_title}\n{text}"
                for section_title, text in sections
            )
            path = RAG_DIR / "guides" / f"{doc_id}.md"
            path.write_text(frontmatter(metadata) + body + "\n", encoding="utf-8")
            documents.append(
                {"doc_id": doc_id, "doc_type": "guide", "city": city, "path": path}
            )

        # 原语料每城只有 hotel_*_10，当候选排序选中其他酒店时
        # hotel_reviews 经常为空。这里为中档、高档、地铁便利和
        # 安静型各增一家，覆盖更常见的选中结果。
        for hotel_index in (3, 5, 7, 9):
            hotel_id = f"hotel_{city_code}_{hotel_index:02d}"
            hotel = hotel_catalog[hotel_id]
            doc_id = f"hotel_review_{city_code}_{hotel_index:02d}_agentic_v1"
            sections = [
                (
                    "交通与到达",
                    f"{hotel['name']}位于{hotel['district']}，距地铁约"
                    f"{hotel['distance_to_subway_meters']}米。带行李到达时要核对电梯出口和网约车下客点，"
                    "返程高峰不要只按地图的纯乘车时间估算。",
                ),
                (
                    "安静与清洁",
                    f"该酒店安静分约为{hotel['quiet_score']}，清洁度参考分为"
                    f"{hotel['cleanliness_score']}。轻睡旅客仍应备注高楼层、远离电梯和主干道，"
                    "入住后检查床品、排水与空调并及时反馈。",
                ),
                (
                    "设施与限制",
                    f"房间提供{'、'.join(hotel['amenities'])}，取消条款为“{hotel['cancel_policy']}”。"
                    "公共设施、早餐和洗衣时间可能随房型变化，评价文本不能代替当日房态与价格查询。",
                ),
                (
                    "适合人群与取舍",
                    f"这是一家{hotel['hotel_type']}酒店，适合重视城市交通、短住休息和总预算的旅客。"
                    f"参考价每晚{hotel['price_per_night']}元，评分{hotel['rating']}；选择时要把安静、地铁距离、"
                    "可取消性和房间面积一起比较，不能只看单项分数。",
                ),
            ]
            metadata = {
                "doc_id": doc_id,
                "title": f"{hotel['name']}住宿体验",
                "doc_type": "hotel_reviews",
                "city": city,
                "hotel_id": hotel_id,
                "hotel_name": hotel["name"],
                "district": hotel["district"],
                "scenario": ["agentic_challenge", "selected_hotel_review"],
                "language": "zh-CN",
                "corpus_version": CORPUS_VERSION,
            }
            body = "\n\n".join(
                f"## {section_title}\n{text}"
                for section_title, text in sections
            )
            path = RAG_DIR / "hotel_reviews" / f"{doc_id}.md"
            path.write_text(frontmatter(metadata) + body + "\n", encoding="utf-8")
            documents.append(
                {
                    "doc_id": doc_id,
                    "doc_type": "hotel_reviews",
                    "city": city,
                    "hotel_id": hotel_id,
                    "path": path,
                }
            )

        for risk_type, (title, overview) in SAFETY_TEXT[city].items():
            doc_id = f"safety_notice_{city_code}_{risk_type}_scope_v1"
            metadata = {
                "doc_id": doc_id,
                "title": title,
                "doc_type": "safety_notice",
                "city": city,
                "risk_type": risk_type,
                "risk_level": "high" if risk_type == "heavy_rain" else "medium",
                "scenario": ["scope_window", "weather_safety"],
                "language": "zh-CN",
                "corpus_version": CORPUS_VERSION,
            }
            sections = [
                ("风险概览", overview),
                ("出发前检查", f"{city}出发前查看当天预警、降水概率、风力和场馆公告，不能只凭前一天的体感判断。把取消条件写进行程备注，同行者提前知道集合点和联系人。"),
                ("交通与替代", f"{city}遇到{risk_type}风险时，优先选择有管理的室内空间和靠近地铁的路线。每个户外节点都要有一个同方向替代点，交通异常时先保证安全再调整预约。"),
                ("现场处理", f"如果{city}现场天气突然升级，立即停止拍照和追赶行程，进入场馆、商场或酒店等有管理的空间。不要翻越围挡、进入积水区或在开放高处停留。"),
                ("同行者照护", f"长者、儿童和对温度或噪声敏感的人需要更短的户外段。出现头晕、发冷、呼吸不适或行动迟缓时，先休息、补水并联系工作人员，不要让不适者独自返回。"),
                ("复核边界", f"安全提醒只说明{risk_type}风险下的通用决策边界，实际出发前仍要以气象、交通、场馆和酒店的最新公告为准。必要时取消非核心活动，保留返程缓冲。"),
            ]
            body = "\n\n".join(f"## {section}\n{text}" for section, text in sections)
            path = RAG_DIR / "safety_notices" / f"{doc_id}.md"
            path.write_text(frontmatter(metadata) + body + "\n", encoding="utf-8")
            documents.append({"doc_id": doc_id, "doc_type": "safety_notice", "city": city, "risk_type": risk_type, "path": path})

    return documents


def write_cases() -> None:
    """Create 96 positive and 24 negative cases tied to stable chunk IDs."""

    cases: list[dict[str, Any]] = []
    city_code = CITY_CODE
    for city in CITIES:
        for doc_type, section_data, category, prefix in (
            ("guide", CITY_TEXT[city]["guide"], "guides", "guide"),
            ("hotel_reviews", CITY_TEXT[city]["hotel"], "hotel_reviews", "hotel"),
            ("packing_checklist", CITY_TEXT[city]["packing"], "packing_checklists", "packing"),
        ):
            doc_id = f"{prefix if prefix != 'hotel' else 'hotel_review'}_{city_code[city]}_scope_v1"
            for index, (title, text) in enumerate(section_data, start=1):
                chunk_id = f"{doc_id}::{index:02d}::001"
                query = f"{city}{title}应该怎么安排和注意什么"
                metadata_filter: dict[str, Any] = {"city": city, "doc_type": doc_type}
                if doc_type == "hotel_reviews":
                    metadata_filter["hotel_ids"] = [f"hotel_{city_code[city]}_10"]
                cases.append(
                    {
                        "case_id": f"rag_scope_{category}_{city_code[city]}_{index:02d}",
                        "category": category,
                        "semantic_query": query,
                        "keyword_query": f"{city} {title}",
                        "metadata_filter": metadata_filter,
                        "gold_chunk_ids": [chunk_id],
                        "relevance": {chunk_id: 2},
                        "expected_doc_types": [doc_type],
                    }
                )

        # Two cases per risk document keep the dataset balanced without making
        # the CPU cross-encoder benchmark unnecessarily long.
        for risk_type in SAFETY_TEXT[city]:
            doc_id = f"safety_notice_{city_code[city]}_{risk_type}_scope_v1"
            for index in (1, 4):
                chunk_id = f"{doc_id}::{index:02d}::001"
                cases.append(
                    {
                        "case_id": f"rag_scope_safety_{city_code[city]}_{risk_type}_{index:02d}",
                        "category": "safety_notices",
                        "semantic_query": f"{city}{risk_type}天气风险下怎么保护行程安全",
                        "keyword_query": f"{city} {risk_type} 安全 替代路线",
                        "metadata_filter": {"city": city, "doc_type": "safety_notice", "risk_type": [risk_type]},
                        "gold_chunk_ids": [chunk_id],
                        "relevance": {chunk_id: 2},
                        "expected_doc_types": ["safety_notice"],
                    }
                )

    # Explicit negatives verify city boundaries and both hotel_id aliases.
    for index in range(8):
        cases.append({
            "case_id": f"rag_scope_negative_unknown_city_{index:02d}",
            "category": "negative",
            "semantic_query": "不存在城市的旅行攻略",
            "keyword_query": "不存在城市 攻略",
            "metadata_filter": {"city": "不存在城市", "doc_type": "guide"},
            "gold_chunk_ids": [], "relevance": {}, "expected_doc_types": ["guide"], "expected_no_results": True,
        })
        cases.append({
            "case_id": f"rag_scope_negative_unknown_hotel_{index:02d}",
            "category": "negative",
            "semantic_query": "指定酒店不存在的住客评价",
            "keyword_query": "酒店 missing hotel_id",
            "metadata_filter": {"city": CITIES[index % len(CITIES)], "doc_type": "hotel_reviews", "hotel_ids": [f"hotel_missing_{index:03d}"]},
            "gold_chunk_ids": [], "relevance": {}, "expected_doc_types": ["hotel_reviews"], "expected_no_results": True,
        })
        cases.append({
            "case_id": f"rag_scope_negative_mismatched_metadata_{index:02d}",
            "category": "negative",
            "semantic_query": "城市攻略但文档类型不匹配",
            "keyword_query": "城市 guide metadata mismatch",
            "metadata_filter": {"city": CITIES[index % len(CITIES)], "doc_type": "safety_notice", "risk_type": ["not_a_real_risk"]},
            "gold_chunk_ids": [], "relevance": {}, "expected_doc_types": ["safety_notice"], "expected_no_results": True,
        })

    assert len(cases) == 120
    assert sum(bool(case.get("expected_no_results")) for case in cases) == 24
    CASES_PATH.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def write_challenge_cases() -> None:
    """
    生成 42 条 Agentic RAG Challenge Cases。

    baseline 数据集保留“单主题直问”口径；这份独立数据集专门覆盖：
        - 同一 guides 类别中的多个独立 needs；
        - 将 combined query 拆成 need-level query 后的可比较案例；
        - 近义词很多但没有可执行内容的 semantic decoys；
        - 多属性酒店、多风险安全和 metadata 负例。
    """

    cases: list[dict[str, Any]] = []
    split_specs = [
        ("food", 1, "想吃本地特色，排队太久时要有同区备选", "本地美食 排队 同区备选", 1),
        ("culture", 2, "想理解城市文化，不是只在热门点拍照", "城市文化 历史 深度体验", 1),
        ("low_walking", 3, "带长者少走路，需要座下休息和撤退点", "长者 低步行 休息 电梯", 2),
        ("family_rain", 4, "带孩子遇到雨天，需要室内备案和取消条件", "亲子 雨天 室内 取消条件", 3),
        ("business", 5, "只有三十六小时，出差之余留一项本地体验", "36小时 商务短停 枢纽 缓冲", 4),
        ("photography", 6, "想避开人群安静拍照，也要有风雨取消边界", "安静摄影 错峰 风雨取消", 4),
    ]

    for city in CITIES:
        code = CITY_CODE[city]
        preference_doc = f"guide_{code}_agentic_preferences_v1"
        decoy_doc = f"guide_{code}_agentic_decoys_v1"

        def pref_chunk(index: int) -> str:
            return f"{preference_doc}::{index:02d}::001"

        def decoy_chunk(index: int) -> str:
            return f"{decoy_doc}::{index:02d}::001"

        # 1. 三条 combined query：一条任务需覆盖 2~3 个独立 need。
        combined_specs = [
            (
                "parents_food_culture",
                f"带父母去{city}，少走路，想吃当地特色，也想真正了解城市文化",
                "长者 低步行 本地美食 城市文化",
                ["food", "culture", "low_walking"],
                [1, 2, 3],
                [1, 2, 4],
            ),
            (
                "family_rain_photo",
                f"带孩子去{city}，下雨也要有室内安排，天气好时想错峰拍照",
                "亲子 雨天室内 错峰摄影",
                ["family_rain", "photography"],
                [4, 6],
                [3, 4],
            ),
            (
                "business_food",
                f"去{city}出差只有三十六小时，交通不要折腾，但想留一顿本地味道",
                "36小时 商务短停 枢纽 本地美食",
                ["business", "food"],
                [5, 1],
                [1, 4],
            ),
        ]
        for group, raw_message, keywords, need_ids, gold_indexes, decoy_indexes in combined_specs:
            gold = [pref_chunk(index) for index in gold_indexes]
            cases.append(
                {
                    "case_id": f"rag_challenge_{code}_combined_{group}",
                    "category": "guides",
                    "challenge_group": f"{code}_{group}",
                    "query_mode": "combined",
                    "raw_message": raw_message,
                    "need_ids": need_ids,
                    "semantic_query": raw_message,
                    "keyword_query": f"{city} {keywords}",
                    "metadata_filter": {"city": city, "doc_type": "guide"},
                    "gold_chunk_ids": gold,
                    "relevance": {chunk_id: 2 for chunk_id in gold},
                    "decoy_chunk_ids": [decoy_chunk(index) for index in decoy_indexes],
                    "expected_doc_types": ["guide"],
                    "top_k": 6,
                }
            )

        # 2. 六条 split query：与 combined 案例使用同一批 Gold chunks，
        #    可直接比较“一条大 query”与“多个 need-level query”。
        for need_id, index, semantic, keywords, decoy_index in split_specs:
            chunk_id = pref_chunk(index)
            cases.append(
                {
                    "case_id": f"rag_challenge_{code}_split_{need_id}",
                    "category": "guides",
                    "challenge_group": f"{code}_need_{need_id}",
                    "query_mode": "split_need",
                    "raw_message": semantic,
                    "need_ids": [need_id],
                    "semantic_query": f"{city}{semantic}",
                    "keyword_query": f"{city} {keywords}",
                    "metadata_filter": {"city": city, "doc_type": "guide"},
                    "gold_chunk_ids": [chunk_id],
                    "relevance": {chunk_id: 2},
                    "decoy_chunk_ids": [decoy_chunk(decoy_index)],
                    "expected_doc_types": ["guide"],
                    "top_k": 5,
                }
            )

        # 3. 含蓄表达：不直接说“慢节奏/文化/美食”，更接近真实用户输入。
        implicit_gold = [
            pref_chunk(1),
            pref_chunk(2),
            f"guide_{code}_scope_v1::08::001",
        ]
        cases.append(
            {
                "case_id": f"rag_challenge_{code}_implicit_relaxed_local",
                "category": "guides",
                "challenge_group": f"{code}_implicit_relaxed_local",
                "query_mode": "implicit",
                "raw_message": f"去{city}不想把自己当特种兵，想吃点这里真正的味道，也想知道这座城市是怎么变成现在这样的",
                "need_ids": ["slow_pace", "food", "culture"],
                "semantic_query": f"{city}不赶场的本地味道和城市发展体验",
                "keyword_query": f"{city} 不赶场 本地味道 城市发展",
                "metadata_filter": {"city": city, "doc_type": "guide"},
                "gold_chunk_ids": implicit_gold,
                "relevance": {
                    implicit_gold[0]: 2,
                    implicit_gold[1]: 2,
                    implicit_gold[2]: 1,
                },
                "decoy_chunk_ids": [decoy_chunk(1), decoy_chunk(2)],
                "expected_doc_types": ["guide"],
                "top_k": 6,
            }
        )

        # 4. 酒店多属性：metadata 先限定选中酒店，Gold 同时要求交通与睡眠证据。
        for hotel_index in (3, 9):
            hotel_doc = f"hotel_review_{code}_{hotel_index:02d}_agentic_v1"
            gold = [f"{hotel_doc}::01::001", f"{hotel_doc}::02::001"]
            cases.append(
                {
                    "case_id": f"rag_challenge_{code}_hotel_{hotel_index:02d}_transit_quiet",
                    "category": "hotel_reviews",
                    "challenge_group": f"{code}_hotel_{hotel_index:02d}",
                    "query_mode": "multi_attribute",
                    "raw_message": "已经选好这家酒店，我在意拖行李到地铁是否方便，也在意晚上安静和房间干净",
                    "need_ids": ["hotel_transit", "hotel_quiet_clean"],
                    "semantic_query": f"{city}酒店行李到达地铁便利、夜间安静和清洁维护",
                    "keyword_query": f"{city} 酒店 地铁 行李 安静 清洁",
                    "metadata_filter": {
                        "city": city,
                        "doc_type": "hotel_reviews",
                        "hotel_ids": [f"hotel_{code}_{hotel_index:02d}"],
                    },
                    "gold_chunk_ids": gold,
                    "relevance": {chunk_id: 2 for chunk_id in gold},
                    # 其他酒店章节只是不同属性，不是语义干扰，
                    # 因此不纳入 decoy 指标。
                    "decoy_chunk_ids": [],
                    "expected_doc_types": ["hotel_reviews"],
                    "top_k": 4,
                }
            )

        # 5. 同一 safety category 的多风险覆盖。
        safety_gold = [
            f"safety_notice_{code}_rain_scope_v1::03::001",
            f"safety_notice_{code}_rain_scope_v1::04::001",
            f"safety_notice_{code}_wind_scope_v1::03::001",
            f"safety_notice_{code}_wind_scope_v1::04::001",
        ]
        cases.append(
            {
                "case_id": f"rag_challenge_{code}_safety_rain_wind",
                "category": "safety_notices",
                "challenge_group": f"{code}_safety_rain_wind",
                "query_mode": "multi_risk",
                "raw_message": f"{city}有雨又有大风，户外和交通应该怎么调整",
                "need_ids": ["rain_safety", "wind_safety"],
                "semantic_query": f"{city}降雨和大风对户外、交通与撤退的影响",
                "keyword_query": f"{city} 雨天 大风 户外 交通 安全",
                "metadata_filter": {
                    "city": city,
                    "doc_type": "safety_notice",
                    "risk_type": ["rain", "wind"],
                },
                "gold_chunk_ids": safety_gold,
                "relevance": {
                    safety_gold[0]: 2,
                    safety_gold[1]: 1,
                    safety_gold[2]: 2,
                    safety_gold[3]: 1,
                },
                "decoy_chunk_ids": [
                    f"safety_notice_{code}_rain_scope_v1::06::001",
                    f"safety_notice_{code}_wind_scope_v1::06::001",
                ],
                "expected_doc_types": ["safety_notice"],
                "top_k": 5,
            }
        )

        # 6. 无效 hotel_id 即使 query 词面很像也必须返回空结果。
        cases.append(
            {
                "case_id": f"rag_challenge_{code}_negative_unknown_hotel",
                "category": "negative",
                "challenge_group": f"{code}_negative_unknown_hotel",
                "query_mode": "negative",
                "raw_message": "查询一家不在数据集里的酒店，但文本包含安静、清洁和地铁",
                "need_ids": ["hotel_review"],
                "semantic_query": f"{city}安静清洁靠近地铁的酒店评价",
                "keyword_query": f"{city} 安静 清洁 地铁 酒店评价",
                "metadata_filter": {
                    "city": city,
                    "doc_type": "hotel_reviews",
                    "hotel_ids": [f"hotel_{code}_missing_challenge"],
                },
                "gold_chunk_ids": [],
                "relevance": {},
                "decoy_chunk_ids": [],
                "expected_doc_types": ["hotel_reviews"],
                "expected_no_results": True,
                "top_k": 5,
            }
        )

    assert len(cases) == 42
    assert sum(bool(case.get("expected_no_results")) for case in cases) == 3
    CHALLENGE_CASES_PATH.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def main() -> None:
    """Write all scoped fixtures and a small catalog manifest."""

    write_json(MOCK_DIR / "flights_mock.json", build_flights())
    write_json(MOCK_DIR / "hotels_mock.json", build_hotels())
    write_json(MOCK_DIR / "weather_mock.json", build_weather())
    write_json(
        MOCK_DIR / "seed_catalog.json",
        {
            "catalog_version": CORPUS_VERSION,
            "scope": {"date_start": "2026-10-01", "date_end": "2026-10-07", "max_trip_days": 7, "cities": list(CITIES)},
            "flight_count": 168,
            "hotel_count_per_city": 11,
            "weather_count": 21,
            "rag_case_count": 120,
            "rag_challenge_case_count": 42,
            "rag_document_count": 39,
            "rag_chunk_count": 222,
            "rag_corpus_policy": "city/scenario/risk documents are date-independent; inventory is date-scoped",
        },
    )
    write_rag_documents()
    write_cases()
    write_challenge_cases()
    print(
        "Scoped fixture generated: flights=168 hotels=33 weather=21 "
        "rag_docs=39 rag_chunks=222 baseline_cases=120 challenge_cases=42"
    )


if __name__ == "__main__":
    main()
