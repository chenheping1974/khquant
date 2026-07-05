#!/usr/bin/env python3
"""下载所有申万行业分类 → 战略分级"""
import sys, json, requests, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

d = json.load(open(".industry_cache.json"))
code_to_sample = {}
for sym, code in d.items():
    if code not in code_to_sample: code_to_sample[code] = sym

# 加载已有
old = json.load(open(".industry_names.json"))
names = old.get("code_to_name", {})
done = set(int(k) for k in names.keys())
todo = [(c, code_to_sample.get(c)) for c in sorted(set(d.values())) if c not in done]
print(f"待下载: {len(todo)} 个行业 (已有 {len(done)})")

if not todo:
    print("✅ 全部完成!")
    sys.exit(0)

tier1_kw = ['电池','半导体','芯片','集成','光伏','新能源','锂电','储能','氢能',
    '医药制造','生物制品','医疗器械','中药','化学制药','医疗服务',
    '航空航天','航天','航空','军工','船舶','新材料','稀土','纳米','超导',
    '计算机设备','软件开发','IT服务','互联网服务','人工智能','机器人',
    '工业母机','数控','自动化','通信设备','通信服务','5G','量子','卫星',
    '消费电子','光学光电','电子元件','电子信息','印制电路板']
tier2_kw = ['汽车','电力','环保','风电','核电','化学制品','化学原料',
    '专用设备','仪器仪表','高端装备','电网','输配电','特高压','充电桩',
    '金属新材料','磁性材料','有机硅','氟化工','体外诊断','疫苗',
    '工程机械','重工','矿山机械','显示面板','LED','激光','传感器',
    '智慧城市','车联网','物联网','虚拟现实','软件服务','云服务','数据']

for i, (code, sym) in enumerate(todo):
    if not sym: continue
    try:
        pf = "SH" if sym.startswith("6") else "SZ"
        url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/CompanySurveyAjax?code={pf}{sym}"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if r.status_code == 200:
            hy = r.json().get('jbzl', {}).get('sshy', '')
            if hy: names[str(code)] = hy
    except: pass

    if (i+1) % 50 == 0:
        print(f"  [{i+1}/{len(todo)}] {len(names)}个")
        # 增量保存
        strategic = {}
        for c_str, name in names.items():
            if any(kw in name for kw in tier1_kw): strategic[c_str] = 1
            elif any(kw in name for kw in tier2_kw): strategic[c_str] = 2
            else: strategic[c_str] = 3
        result = {'code_to_name': names, 'strategic_tier': strategic}
        json.dump(result, open('.industry_names.json', 'w'), ensure_ascii=False, indent=2)

    time.sleep(0.12)

# 最终保存
strategic = {}
for c_str, name in names.items():
    if any(kw in name for kw in tier1_kw): strategic[c_str] = 1
    elif any(kw in name for kw in tier2_kw): strategic[c_str] = 2
    else: strategic[c_str] = 3

result = {'code_to_name': names, 'strategic_tier': strategic}
json.dump(result, open('.industry_names.json', 'w'), ensure_ascii=False, indent=2)
t1 = sum(1 for v in strategic.values() if v == 1)
t2 = sum(1 for v in strategic.values() if v == 2)
print(f"\n✅ 完成: {len(names)}个行业")
print(f"  Tier1(核心战略): {t1}")
print(f"  Tier2(政策支持): {t2}")
print(f"  Tier3(普通行业): {len(names)-t1-t2}")
