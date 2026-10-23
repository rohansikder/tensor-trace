import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { HeatmapChart, LineChart, BarChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
  LegendComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption } from "echarts/core";

echarts.use([
  HeatmapChart,
  LineChart,
  BarChart,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
  LegendComponent,
  CanvasRenderer,
]);

function Chart({
  option,
  className,
  label,
}: {
  option: EChartsCoreOption;
  className: string;
  label: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!host.current) return;
    const chart = echarts.init(host.current, undefined, { renderer: "canvas" });
    chart.setOption(option);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [option]);
  return <div ref={host} className={className} role="img" aria-label={label} />;
}

const compact = (piece: string) =>
  piece
    .replace(/\n/g, "↵")
    .replace(/\t/g, "⇥")
    .replace(/^ /, "·")
    .slice(0, 12) || "∅";

export function AttentionChart({
  matrix,
  pieces,
  selected = -1,
}: {
  matrix: number[][];
  pieces: string[];
  selected?: number;
}) {
  const n = matrix.length;
  const weights: unknown[] = [];
  const masked: unknown[] = [];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      if (x > y)
        masked.push({
          value: [x, y, 0],
          label: { show: n <= 10, color: "#a1a5ac", formatter: "—" },
        });
      else
        weights.push({
          value: [x, y, matrix[y]?.[x] ?? 0],
          itemStyle:
            y === (selected < 0 ? n - 1 : selected)
              ? { borderColor: "#3669dc", borderWidth: 1.5 }
              : undefined,
        });
    }
  }
  const labels = pieces.map(compact);
  const option: EChartsCoreOption = {
    animationDuration: 220,
    textStyle: { fontFamily: "'JetBrains Mono', monospace", fontSize: 11 },
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: (params: unknown) => {
        const item = params as { value: number[]; seriesIndex: number };
        const [x, y, value] = item.value;
        return `Query ${y}: ${escapeHtml(pieces[y] ?? "")}<br/>Key ${x}: ${escapeHtml(pieces[x] ?? "")}<br/>${item.seriesIndex === 0 ? "Masked future position" : `${(value * 100).toFixed(3)}% attention weight`}`;
      },
    },
    grid: { top: 34, left: 66, right: 77, bottom: n > 12 ? 70 : 36 },
    xAxis: {
      type: "category",
      data: labels,
      position: "top",
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: "#37404e",
        fontSize: n > 20 ? 9 : 11,
        rotate: n > 12 ? 45 : 0,
        interval: n > 24 ? Math.ceil(n / 16) - 1 : 0,
      },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: "#37404e",
        fontSize: n > 20 ? 9 : 11,
        interval: n > 24 ? Math.ceil(n / 16) - 1 : 0,
      },
    },
    visualMap: {
      min: 0,
      max: 1,
      seriesIndex: 1,
      calculable: false,
      orient: "vertical",
      right: 7,
      top: "center",
      itemHeight: 150,
      itemWidth: 10,
      text: ["1.0", "0.0"],
      precision: 2,
      textStyle: { color: "#737d8b", fontSize: 10 },
      inRange: { color: ["#edf2fd", "#bed2fc", "#6394ec", "#2c68d7"] },
    },
    series: [
      {
        name: "Masked",
        type: "heatmap",
        data: masked,
        itemStyle: { color: "#e9eae7", borderColor: "#f8f7f3", borderWidth: 1 },
        emphasis: { disabled: true },
      },
      {
        name: "Attention",
        type: "heatmap",
        data: weights,
        label: {
          show: n <= 10,
          fontSize: n <= 6 ? 12 : 10,
          formatter: (params: unknown) =>
            (params as { value: number[] }).value[2].toFixed(2),
        },
        itemStyle: { borderColor: "#f8f7f3", borderWidth: 1 },
        emphasis: { itemStyle: { borderColor: "#204ba3", borderWidth: 2 } },
      },
    ],
  };
  return (
    <Chart
      option={option}
      className="attention-chart"
      label={`Causal attention matrix with ${n} context tokens. Future token positions are masked. Values are available by hovering each cell.`}
    />
  );
}

export function VectorChart({ values }: { values: number[] }) {
  const option: EChartsCoreOption = {
    textStyle: { fontFamily: "'JetBrains Mono', monospace" },
    grid: { left: 44, top: 20, right: 20, bottom: 28 },
    tooltip: { trigger: "axis", confine: true },
    xAxis: {
      type: "category",
      data: values.map((_, i) => i),
      axisLine: { lineStyle: { color: "#d9dcd7" } },
      axisTick: { show: false },
      axisLabel: { color: "#748092", fontSize: 10 },
    },
    yAxis: {
      type: "value",
      splitLine: { lineStyle: { color: "#e6e7e3" } },
      axisLabel: { color: "#748092", fontSize: 10 },
    },
    series: [
      {
        type: "bar",
        data: values.map((value) => ({
          value,
          itemStyle: {
            color: value >= 0 ? "#5a8de5" : "#9cbca4",
            borderRadius: [2, 2, 0, 0],
          },
        })),
        barMaxWidth: 22,
      },
    ],
  };
  return (
    <Chart
      option={option}
      className="vector-chart"
      label={`First ${values.length} actual embedding values for the selected token.`}
    />
  );
}

export interface ComparisonPoint {
  contextTokens: number;
  cacheOnMs: number;
  cacheOffMs: number;
}
export function ComparisonChart({ points }: { points: ComparisonPoint[] }) {
  const option: EChartsCoreOption = {
    animationDuration: 300,
    textStyle: { fontFamily: "'DM Sans Variable', sans-serif" },
    color: ["#3b71d9", "#929a9b"],
    grid: { top: 34, left: 65, right: 22, bottom: 74 },
    tooltip: {
      trigger: "axis",
      confine: true,
      valueFormatter: (value: unknown) =>
        `${Number(value).toFixed(2)} ms / token`,
    },
    legend: {
      bottom: 0,
      left: 20,
      icon: "circle",
      itemWidth: 9,
      itemHeight: 9,
      textStyle: { color: "#647080" },
    },
    xAxis: {
      type: "category",
      data: points.map((p) => p.contextTokens),
      name: "Context length (tokens)",
      nameLocation: "middle",
      nameGap: 30,
      nameTextStyle: { color: "#75808d", fontSize: 12 },
      axisLine: { lineStyle: { color: "#d8ddd8" } },
      axisTick: { show: false },
      axisLabel: { color: "#6c7685" },
    },
    yAxis: {
      type: "value",
      name: "ms / token",
      nameTextStyle: { color: "#75808d", fontSize: 11 },
      axisLabel: { color: "#6c7685" },
      splitLine: { lineStyle: { color: "#e4e6e1", type: "dashed" } },
    },
    series: [
      {
        name: "Cache on",
        type: "line",
        data: points.map((p) => p.cacheOnMs),
        symbolSize: 7,
        lineStyle: { width: 2.5 },
        emphasis: { focus: "series" },
      },
      {
        name: "Cache off",
        type: "line",
        data: points.map((p) => p.cacheOffMs),
        symbolSize: 7,
        lineStyle: { width: 2.5 },
        emphasis: { focus: "series" },
      },
    ],
  };
  return (
    <Chart
      option={option}
      className="comparison-chart"
      label="Measured decode time per token, comparing cached and uncached generation across context lengths."
    />
  );
}

function escapeHtml(text: string) {
  return text.replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ]!,
  );
}
