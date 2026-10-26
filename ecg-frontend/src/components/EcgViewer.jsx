import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-basic-dist-min";

// Standard 12-lead layout: 3 rows x 4 columns (2.5 s each) + a 10 s lead II rhythm strip.
const GRID = [
  ["I", "aVR", "V1", "V4"],
  ["II", "aVL", "V2", "V5"],
  ["III", "aVF", "V3", "V6"],
];
const MAX_SECONDS = 10;
const Y_HALF = 1.5; // mV above/below baseline per row => 3 mV = 6 big squares tall
const MARGIN = { l: 6, r: 6, t: 6, b: 26 };
const MAJOR = "#f1a3a3"; // big square: 200 ms x 0.5 mV
const MINOR = "#f9d6d6"; // small square: 40 ms x 0.1 mV

const median = (arr) => {
  const s = Float64Array.from(arr).sort();
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

const range = (start, stop, step) => {
  const out = [];
  for (let v = start; v <= stop + 1e-9; v += step) out.push(+v.toFixed(4));
  return out;
};

/** Builds the Plotly traces + layout for a given container width. Exported for testing. */
export function buildFigure(ecg, width) {
  const fs = ecg.fs;
  const byName = Object.fromEntries(ecg.leads.map((l) => [l.name, l.values]));
  const total = ecg.duration_s || (ecg.leads[0]?.values.length ?? 0) / fs;
  const T = Math.min(total, MAX_SECONDS);
  const seg = T / 4;
  const rows = 4;

  // Keep squares square: 25 mm/s horizontally, 10 mm/mV vertically.
  const plotW = Math.max(300, width - MARGIN.l - MARGIN.r);
  const bigPx = plotW / (T * 5); // 5 big squares per second
  const rowPx = Y_HALF * 2 * 2 * bigPx; // (3 mV / 0.5 mV) big squares per row
  const height = Math.round(rows * rowPx + MARGIN.t + MARGIN.b);

  const data = [];
  const annotations = [];
  const yKey = (r) => (r === 0 ? "y" : `y${r + 1}`);

  const addSegment = (name, row, t0, t1, label) => {
    const vals = byName[name];
    if (vals) {
      const base = median(vals);
      const i0 = Math.round(t0 * fs);
      const i1 = Math.min(vals.length, Math.round(t1 * fs));
      const x = new Array(Math.max(0, i1 - i0));
      const y = new Array(x.length);
      for (let k = 0; k < x.length; k++) {
        x[k] = (i0 + k) / fs;
        y[k] = vals[i0 + k] - base;
      }
      data.push({
        type: "scatter",
        mode: "lines",
        x,
        y,
        xaxis: "x",
        yaxis: yKey(row),
        line: { color: "#111", width: 1.2 },
        hovertemplate: `${name}: %{y:.2f} mV at %{x:.2f} s<extra></extra>`,
        showlegend: false,
      });
    }
    annotations.push({
      xref: "x",
      yref: yKey(row),
      x: t0 + T * 0.004,
      y: Y_HALF * 0.82,
      text: `<b>${label ?? name}</b>`,
      showarrow: false,
      xanchor: "left",
      font: { size: 12, color: "#222" },
    });
  };

  GRID.forEach((cols, r) => cols.forEach((name, c) => addSegment(name, r, c * seg, (c + 1) * seg)));
  addSegment("II", 3, 0, T, "II (rhythm strip)");

  const xTickVals = range(0, T, 0.2);
  const layout = {
    width,
    height,
    margin: MARGIN,
    paper_bgcolor: "#fff",
    plot_bgcolor: "#fffafa",
    showlegend: false,
    hovermode: "closest",
    annotations,
    xaxis: {
      range: [0, T],
      domain: [0, 1],
      anchor: `y${rows}`,
      tickmode: "array",
      tickvals: xTickVals,
      ticktext: xTickVals.map((v) => (Math.abs(v - Math.round(v)) < 1e-6 ? `${Math.round(v)} s` : "")),
      showgrid: true,
      gridcolor: MAJOR,
      gridwidth: 1,
      minor: { tickmode: "linear", tick0: 0, dtick: 0.04, showgrid: true, gridcolor: MINOR, gridwidth: 0.6 },
      zeroline: false,
      showline: true,
      linecolor: MAJOR,
      ticks: "outside",
      ticklen: 3,
      tickfont: { size: 10, color: "#555" },
      fixedrange: true,
    },
  };

  const yTickVals = range(-Y_HALF, Y_HALF, 0.5);
  for (let r = 0; r < rows; r++) {
    layout[r === 0 ? "yaxis" : `yaxis${r + 1}`] = {
      domain: [1 - (r + 1) / rows, 1 - r / rows],
      anchor: "x",
      range: [-Y_HALF, Y_HALF],
      tickmode: "array",
      tickvals: yTickVals,
      ticktext: yTickVals.map(() => ""),
      showticklabels: false,
      showgrid: true,
      gridcolor: MAJOR,
      gridwidth: 1,
      minor: { tickmode: "linear", tick0: -Y_HALF, dtick: 0.1, showgrid: true, gridcolor: MINOR, gridwidth: 0.6 },
      zeroline: false,
      showline: false,
      fixedrange: true,
    };
  }
  return { data, layout, shownSeconds: T, totalSeconds: total };
}

const CONFIG = {
  displaylogo: false,
  displayModeBar: "hover",
  responsive: false,
  modeBarButtonsToRemove: [
    "select2d",
    "lasso2d",
    "zoom2d",
    "pan2d",
    "zoomIn2d",
    "zoomOut2d",
    "autoScale2d",
    "resetScale2d",
    "hoverClosestCartesian",
    "hoverCompareCartesian",
    "toggleSpikelines",
  ],
  toImageButtonOptions: { format: "png", filename: "ecg", scale: 2 },
};

export default function EcgViewer({ ecg }) {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  const [info, setInfo] = useState(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(Math.floor(el.clientWidth));
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!ecg?.leads?.length || !width || !ref.current) return;
    const { data, layout, shownSeconds, totalSeconds } = buildFigure(ecg, width);
    setInfo({ shownSeconds, totalSeconds });
    Plotly.react(ref.current, data, layout, CONFIG);
  }, [ecg, width]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return (
    <div className="ecg-viewer">
      <div className="ecg-scroll">
        <div ref={ref} className="ecg-plot" />
      </div>
      <p className="ecg-caption">
        25 mm/s · 10 mm/mV · small square = 40 ms × 0.1 mV, large square = 200 ms × 0.5 mV · baseline-centred
        {info && info.totalSeconds > info.shownSeconds
          ? ` · showing first ${info.shownSeconds} s of ${Math.round(info.totalSeconds)} s`
          : ""}
      </p>
    </div>
  );
}
