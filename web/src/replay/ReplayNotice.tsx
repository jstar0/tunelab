import { useEffect, useRef } from "react";
import "./replay.css";

export function ReplayNotice() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const sync = () => document.documentElement.style.setProperty("--replay-h", `${node.offsetHeight}px`);
    sync();
    const observer = new ResizeObserver(sync);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return (
    <div className="replay-notice" ref={ref} role="note">
      <b>回放数据：结果由本仓库真实运行后录制，在线版不能现场运行搜索</b>
      <span>
        可以：浏览已录制的实验、查看每个 trial、比较 run，用默认空间、48 trials 与已录制 seed
        的组合点“创建”来回放一次运行，并下载录制的 report / CSV / 配置。
        不可用：自定义搜索空间、预算或 seed 的现场搜索（未录制会提示）、暂停 / 继续 / 取消。
        想真实运行请在本地执行 <code>tunelab serve</code>。
      </span>
    </div>
  );
}
