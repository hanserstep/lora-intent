"""生成工单分类数据集。

为什么用模板生成而不是手工标注：

    微调实验的关键是「数据量够 + 划分合理」。手工标 400 条要一整天，
    而模板生成能在十分钟内产出同样规模的数据，且天然可控。

    代价是模板数据有固定的语言模式，模型可能学到「关键词匹配」而不是
    真正的语义理解。所以划分必须**按模板分组**，不能随机切：

        同一模板生成的不同变体，如果分别进了训练集和测试集，
        测试集测的其实是「模型有没有背下这个模板」，准确率会虚高。

    按模板划分后，测试集全是模型没见过的表达方式，
    这个数字才反映真实的泛化能力。

六个类别对应运维工单的常见类型，和 ops-agent 的工具能对上：
    disk / network / cpu / memory / database / other
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "tickets.jsonl"

LABELS = ["disk", "network", "cpu", "memory", "database", "other"]

HOSTS = ["web-01", "web-02", "db-01", "cache-01", "app-node-3", "gateway-2"]
DIRS = ["/var/log", "/data", "/tmp", "/opt/app", "/home/deploy"]
PROCS = ["java", "python3", "nginx", "mysqld", "node", "elasticsearch"]
IPS = ["10.2.3.11", "192.168.7.45", "172.16.0.9"]
SERVICES = ["order-service", "payment-gateway", "user-center", "report-job"]

# 每个类别 9 个模板，前 6 个进训练集，后 3 个进测试集
TEMPLATES = {
    "disk": [
        ("{host} 的 {dir} 目录已经用满，写入操作报 No space left on device", "train"),
        ("服务器 {host} 磁盘使用率 {disk_pct}%，业务日志无法继续写入", "train"),
        ("{dir} 分区剩余空间不足 1G，需要清理历史归档", "train"),
        ("{host} 上 inode 用尽，无法创建新文件，df -i 显示 100%", "train"),
        ("磁盘 IO 长时间 100%，{dir} 上的大文件导致写入阻塞", "train"),
        ("{host} 的数据盘被日志撑满，{svc} 服务开始报错", "train"),
        ("{host} 清理完日志后空间没有释放，怀疑有进程持有已删除的文件", "test"),
        ("{host} 的备份目录每天增长 {size}G，磁盘快撑不住了", "test"),
        ("容器写入层把 {host} 的磁盘占满了，需要限制日志大小", "test"),
    ],
    "network": [
        ("{host} 到 {ip} 的请求全部超时，ping 丢包率 {loss}%", "train"),
        ("域名解析返回 NXDOMAIN，{svc} 无法连接下游依赖", "train"),
        ("{host} 的带宽跑满，出口流量达到上限，请求排队", "train"),
        ("TCP 连接数接近 ulimit 上限，新的连接被拒绝", "train"),
        ("{host} 和 {ip} 之间的延迟从 2ms 涨到 {lat}ms，抖动明显", "train"),
        ("负载均衡健康检查失败，{host} 被摘除，怀疑网络分区", "train"),
        ("跨机房专线出现丢包，{host} 到 {ip} 的同步延迟持续升高", "test"),
        ("{svc} 调用第三方接口偶发 connection reset，重试可以恢复", "test"),
        ("VPN 拨号后内网地址不通，{host} 的路由表缺少对应条目", "test"),
    ],
    "cpu": [
        ("{host} 的 15 分钟负载平均值达到 {load}，CPU 使用率持续 100%", "train"),
        ("进程 {proc} 占满单核，{svc} 接口响应时间从 50ms 涨到 {rt}ms", "train"),
        ("{host} 出现大量 runnable 进程，上下文切换次数异常偏高", "train"),
        ("定时任务集中触发，{host} 的 CPU 出现周期性尖峰，影响在线请求", "train"),
        ("{svc} 的 worker 线程池打满，CPU 满载导致请求超时", "train"),
        ("{host} 上某个进程 CPU 占用曲线呈锯齿状，疑似死循环", "train"),
        ("{host} 的容器被限制在 2 核，实际负载超过限额，频繁被 throttle", "test"),
        ("压测期间 {host} 的 CPU 软中断占比过高，怀疑网卡中断集中在一个核", "test"),
        ("{host} 的 CPU 温度过高触发降频，性能下降一半", "test"),
    ],
    "memory": [
        ("{host} 内存使用率 {mem_pct}%，OOM killer 杀掉了 {proc} 进程", "train"),
        ("{proc} 的堆内存持续增长，Full GC 后没有回落，疑似泄漏", "train"),
        ("{host} 的 swap 使用量激增，系统开始频繁换页，响应变慢", "train"),
        ("{svc} 的缓存没有设置上限，内存占用随请求量线性上涨", "train"),
        ("{host} 可用内存不足，页缓存被大量回收，读性能下降", "train"),
        ("{host} 上多个进程同时申请大页内存，触发内存碎片问题", "train"),
        ("{svc} 的 JVM 元空间溢出，加载的类数量超出预期", "test"),
        ("{svc} 的容器内存 limit 设得太小，应用启动时就被 kill", "test"),
        ("内存泄漏要几天才显现，{svc} restart 后能恢复但会复发", "test"),
    ],
    "database": [
        ("MySQL 主从延迟达到 {lag} 秒，读请求拿到旧数据", "train"),
        ("慢查询数量激增，连接池被占满，新的请求排队等待", "train"),
        ("{svc} 和数据库之间频繁死锁，事务被回滚", "train"),
        ("单表数据量超过 {rows} 万行，未建索引的查询扫描全表", "train"),
        ("数据库连接数达到 max_connections 上限，应用侧报连接失败", "train"),
        ("{host} 的 binlog 增长速度异常，磁盘被日志占满", "train"),
        ("统计信息过期导致 {svc} 的执行计划走错索引，查询从 10ms 变成 3s", "test"),
        ("大事务长时间未提交，锁住了 {svc} 的热点行，影响其他写入", "test"),
        ("{host} 的只读实例复制线程断了，需要重新建立同步", "test"),
    ],
    "other": [
        ("{svc} 的 HTTPS 证书还有 3 天过期，需要更新", "train"),
        ("定时任务没有按时执行，调度器显示上次运行失败", "train"),
        ("告警通知没有发出来，{host} 的钉钉机器人 token 失效", "train"),
        ("{host} 的系统时间出现偏差，比标准时间慢 {drift} 秒", "train"),
        ("配置变更后 {svc} 没有生效，需要确认是否 reload", "train"),
        ("{svc} 的日志级别被误改为 DEBUG，日志量暴涨十倍", "train"),
        ("新版本发布后 {svc} 的灰度节点出现异常，需要回滚", "test"),
        ("{host} 的监控面板数据缺失，采集 agent 掉线了", "test"),
        ("不同环境的配置不一致，{svc} 测试环境正常生产环境报错", "test"),
    ],
}

FILLERS = {
    "host": HOSTS,
    "dir": DIRS,
    "proc": PROCS,
    "ip": IPS,
    "svc": SERVICES,
}

VALUE_RANGES = {
    "disk_pct": range(90, 100),
    "loss": range(20, 100, 5),
    "lat": range(80, 800, 40),
    "load": ["8.2", "12.5", "16.8", "23.4", "31.7"],
    "rt": [200, 500, 1200, 3000, 8000],
    "mem_pct": range(88, 99),
    "lag": range(5, 120, 5),
    "rows": [500, 1200, 3000, 8000],
    "drift": range(30, 600, 30),
    "size": [10, 20, 50, 100],
}

VARIATIONS_PER_TEMPLATE = 6


def fill(template: str, rng: random.Random) -> str:
    out = template
    for key, choices in FILLERS.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", str(rng.choice(choices)), 1)
    for key, choices in VALUE_RANGES.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", str(rng.choice(list(choices))), 1)
    return out


def build(seed: int = 42):
    rng = random.Random(seed)
    rows = []
    tid = 0
    for label, specs in TEMPLATES.items():
        for tpl_idx, (tpl, split) in enumerate(specs):
            # 每个模板生成若干变体，同一个模板的所有变体落在同一个 split，
            # 这就是「按模板分组划分」，避免测试集泄漏
            #
            # 模板内去重：有些模板不含变量（如「JVM 元空间溢出」），
            # 生成 6 次会得到 6 条一模一样的句子。重复样本会让单个模板
            # 主导该类的指标——一条判断错，recall 直接掉 1/3，看着像模型
            # 不行，其实是数据重复放大了噪声。
            seen = set()
            for v in range(VARIATIONS_PER_TEMPLATE):
                text = fill(tpl, rng)
                if text in seen:
                    continue
                seen.add(text)
                rows.append({
                    "id": f"t{tid:04d}",
                    "text": text,
                    "label": label,
                    "template_id": f"{label}-{tpl_idx}",
                    "split": split,
                })
                tid += 1
    rng.shuffle(rows)
    return rows


def main():
    rows = build()
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"总样本 {len(rows)}：train {len(train)} / test {len(test)}")
    print(f"训练集模板数 {len({r['template_id'] for r in train})}，"
          f"测试集模板数 {len({r['template_id'] for r in test})}")
    print("\n各类分布：")
    for lb in LABELS:
        print(f"  {lb:10} train={sum(1 for r in train if r['label']==lb):3} "
              f"test={sum(1 for r in test if r['label']==lb):3}")
    print("\n样例：")
    for r in rows[:3]:
        print(f"  [{r['label']}/{r['split']}] {r['text']}")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
