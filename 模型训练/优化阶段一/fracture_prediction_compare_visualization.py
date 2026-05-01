from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import statistics
import subprocess
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/方案对比/研究内容一"
)

POWERSHELL_SVG_TO_PNG = r"""
& {
Set-StrictMode -Version 2.0
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase
$SvgPath = $env:CODEX_SVG_PATH
$PngPath = $env:CODEX_PNG_PATH

function Parse-Double([string]$Text, [double]$DefaultValue = 0.0) {
    if ([string]::IsNullOrWhiteSpace($Text)) { return $DefaultValue }
    return [double]::Parse($Text, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Convert-Color([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -eq 'none') { return $null }
    $Value = $Value.Trim()
    if ($Value.StartsWith('#')) {
        if ($Value.Length -eq 7) {
            return [System.Windows.Media.Color]::FromRgb(
                [Convert]::ToByte($Value.Substring(1, 2), 16),
                [Convert]::ToByte($Value.Substring(3, 2), 16),
                [Convert]::ToByte($Value.Substring(5, 2), 16)
            )
        }
        if ($Value.Length -eq 9) {
            return [System.Windows.Media.Color]::FromArgb(
                [Convert]::ToByte($Value.Substring(1, 2), 16),
                [Convert]::ToByte($Value.Substring(3, 2), 16),
                [Convert]::ToByte($Value.Substring(5, 2), 16),
                [Convert]::ToByte($Value.Substring(7, 2), 16)
            )
        }
    }
    return [System.Windows.Media.ColorConverter]::ConvertFromString($Value)
}

function New-Brush([string]$Fill, [double]$Opacity = 1.0) {
    $Color = Convert-Color $Fill
    if ($null -eq $Color) { return $null }
    $Brush = New-Object System.Windows.Media.SolidColorBrush($Color)
    $Brush.Opacity = $Opacity
    $Brush.Freeze()
    return $Brush
}

function New-Pen([string]$Stroke, [double]$Width = 1.0, [double]$Opacity = 1.0) {
    $Brush = New-Brush $Stroke $Opacity
    if ($null -eq $Brush) { return $null }
    $Pen = New-Object System.Windows.Media.Pen($Brush, $Width)
    $Pen.Freeze()
    return $Pen
}

[xml]$Svg = Get-Content -LiteralPath $SvgPath -Encoding UTF8 -Raw
$Root = $Svg.DocumentElement
$Width = [int][math]::Ceiling((Parse-Double $Root.GetAttribute('width') 1200))
$Height = [int][math]::Ceiling((Parse-Double $Root.GetAttribute('height') 860))

$Visual = New-Object System.Windows.Media.DrawingVisual
$Context = $Visual.RenderOpen()

foreach ($Node in $Root.ChildNodes) {
    if ($Node.NodeType -ne [System.Xml.XmlNodeType]::Element) { continue }
    $Name = $Node.LocalName

    if ($Name -eq 'rect') {
        $X = Parse-Double $Node.GetAttribute('x')
        $Y = Parse-Double $Node.GetAttribute('y')
        $W = Parse-Double $Node.GetAttribute('width')
        $H = Parse-Double $Node.GetAttribute('height')
        $RX = Parse-Double $Node.GetAttribute('rx')
        $RY = Parse-Double $Node.GetAttribute('ry')
        $Opacity = Parse-Double $Node.GetAttribute('opacity') 1.0
        $Brush = New-Brush $Node.GetAttribute('fill') $Opacity
        $StrokeOpacity = Parse-Double $Node.GetAttribute('stroke-opacity') 1.0
        $Pen = New-Pen $Node.GetAttribute('stroke') (Parse-Double $Node.GetAttribute('stroke-width') 1.0) $StrokeOpacity
        $Rect = New-Object System.Windows.Rect($X, $Y, $W, $H)
        if ($RX -gt 0 -or $RY -gt 0) {
            $Context.DrawRoundedRectangle($Brush, $Pen, $Rect, $RX, $RY)
        } else {
            $Context.DrawRectangle($Brush, $Pen, $Rect)
        }
        continue
    }

    if ($Name -eq 'line') {
        $Pen = New-Pen $Node.GetAttribute('stroke') (Parse-Double $Node.GetAttribute('stroke-width') 1.0)
        if ($null -ne $Pen) {
            $P1 = New-Object System.Windows.Point((Parse-Double $Node.GetAttribute('x1')), (Parse-Double $Node.GetAttribute('y1')))
            $P2 = New-Object System.Windows.Point((Parse-Double $Node.GetAttribute('x2')), (Parse-Double $Node.GetAttribute('y2')))
            $Context.DrawLine($Pen, $P1, $P2)
        }
        continue
    }

    if ($Name -eq 'text') {
        $TextValue = [string]$Node.InnerText
        $FontSize = Parse-Double $Node.GetAttribute('font-size') 12.0
        $Fill = $Node.GetAttribute('fill')
        $Brush = New-Brush $(if ([string]::IsNullOrWhiteSpace($Fill)) { '#111111' } else { $Fill }) 1.0
        if ($null -eq $Brush) { $Brush = [System.Windows.Media.Brushes]::Black }
        $FontWeight = [System.Windows.FontWeights]::Normal
        if ($Node.GetAttribute('font-weight') -eq 'bold') {
            $FontWeight = [System.Windows.FontWeights]::Bold
        }
        $Typeface = New-Object System.Windows.Media.Typeface(
            (New-Object System.Windows.Media.FontFamily('Microsoft YaHei')),
            [System.Windows.FontStyles]::Normal,
            $FontWeight,
            [System.Windows.FontStretches]::Normal
        )
        $Formatted = New-Object System.Windows.Media.FormattedText(
            $TextValue,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Windows.FlowDirection]::LeftToRight,
            $Typeface,
            $FontSize,
            $Brush,
            1.0
        )

        $X = Parse-Double $Node.GetAttribute('x')
        $Y = Parse-Double $Node.GetAttribute('y')
        $Anchor = $Node.GetAttribute('text-anchor')
        if ($Anchor -eq 'middle') {
            $X -= $Formatted.WidthIncludingTrailingWhitespace / 2.0
        } elseif ($Anchor -eq 'end') {
            $X -= $Formatted.WidthIncludingTrailingWhitespace
        }
        $DrawY = $Y - ($FontSize * 0.82)

        $TransformText = $Node.GetAttribute('transform')
        $Pushed = $false
        if ($TransformText -match 'rotate\(([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\)') {
            $Angle = Parse-Double $Matches[1]
            $CX = Parse-Double $Matches[2]
            $CY = Parse-Double $Matches[3]
            $Context.PushTransform((New-Object System.Windows.Media.RotateTransform($Angle, $CX, $CY)))
            $Pushed = $true
        }

        $Point = New-Object System.Windows.Point($X, $DrawY)
        $Context.DrawText($Formatted, $Point)
        if ($Pushed) {
            $Context.Pop()
        }
        continue
    }
}

$Context.Close()
$Bitmap = New-Object System.Windows.Media.Imaging.RenderTargetBitmap(
    $Width, $Height, 96.0, 96.0, [System.Windows.Media.PixelFormats]::Pbgra32
)
$Bitmap.Render($Visual)
$Encoder = New-Object System.Windows.Media.Imaging.PngBitmapEncoder
$Encoder.Frames.Add([System.Windows.Media.Imaging.BitmapFrame]::Create($Bitmap)) | Out-Null

$TargetDir = Split-Path -Parent $PngPath
if (-not [string]::IsNullOrWhiteSpace($TargetDir)) {
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
}

$Stream = [System.IO.File]::Open($PngPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
try {
    $Encoder.Save($Stream)
} finally {
    $Stream.Close()
}
}
"""


@dataclass
class CompareCase:
    case_id: str
    title: str
    well_name: str
    prediction_kind: str
    prediction_label: str
    actual_label: str
    prediction_path: Path
    actual_path: Path
    track_path: Path


@dataclass
class FracturePoint:
    well_name: str
    source_kind: str
    source_label: str
    source_file: str
    record_id: str
    depth_md: float | None
    depth_tvd: float | None
    x: float | None
    y: float | None
    time: float | None
    azimuth: float | None
    dip: float | None


@dataclass
class MatchResult:
    predicted_record_id: str
    actual_record_id: str
    predicted_md: float | None
    actual_md: float | None
    signed_md_error: float | None
    abs_md_error: float | None
    predicted_tvd: float | None
    actual_tvd: float | None
    signed_tvd_error: float | None
    abs_tvd_error: float | None
    predicted_azimuth: float | None
    actual_azimuth: float | None
    azimuth_diff: float | None
    predicted_dip: float | None
    actual_dip: float | None
    dip_diff: float | None


@dataclass
class TrackInterpolator:
    md_axis: list[float]
    tvd_axis: list[float]
    x_axis: list[float]
    y_axis: list[float]
    md_from_tvd_axis_tvd: list[float]
    md_from_tvd_axis_md: list[float]

    def tvd_from_md(self, depth_md: float | None) -> float | None:
        return interpolate_linear(self.md_axis, self.tvd_axis, depth_md)

    def x_from_md(self, depth_md: float | None) -> float | None:
        return interpolate_linear(self.md_axis, self.x_axis, depth_md)

    def y_from_md(self, depth_md: float | None) -> float | None:
        return interpolate_linear(self.md_axis, self.y_axis, depth_md)

    def md_from_tvd(self, depth_tvd: float | None) -> float | None:
        return interpolate_linear(self.md_from_tvd_axis_tvd, self.md_from_tvd_axis_md, depth_tvd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw/optimized fracture predictions against extracted imaging fractures and export CSV/SVG/HTML outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--depth-bin-size", type=float, default=10.0)
    parser.add_argument("--azimuth-bin-size", type=float, default=30.0)
    parser.add_argument("--dip-bin-size", type=float, default=10.0)
    parser.add_argument("--error-bin-count", type=int, default=12)
    parser.add_argument("--skip-html", action="store_true")
    return parser.parse_args()


def build_default_cases() -> list[CompareCase]:
    raw_prediction_dir = Path(
        r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝角度预测/井斜"
    )
    actual_dir = Path(
        r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝提取"
    )
    optimized_che151_path = Path(
        r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/现有常规测井裂缝预测/车151HF/final_fracture_points.csv"
    )
    track_dir = Path(r"/data/shared/project-oil/wx数据/砂砾岩/井斜")

    return [
        CompareCase(
            case_id="raw_vs_actual_che151hf",
            title="车151HF 原始预测 vs 实际裂缝提取",
            well_name="车151HF",
            prediction_kind="raw",
            prediction_label="原始预测",
            actual_label="实际裂缝提取",
            prediction_path=raw_prediction_dir / "车151HF_predicted_angles.csv",
            actual_path=actual_dir / "车151HF_fractures.csv",
            track_path=track_dir / "车151HF.dat",
        ),
        CompareCase(
            case_id="raw_vs_actual_cheye1",
            title="车页1导眼 原始预测 vs 实际裂缝提取",
            well_name="车页1导眼",
            prediction_kind="raw",
            prediction_label="原始预测",
            actual_label="实际裂缝提取",
            prediction_path=raw_prediction_dir / "车页1导眼_predicted_angles.csv",
            actual_path=actual_dir / "车页1导眼_fractures.csv",
            track_path=track_dir / "车页1导眼.dat",
        ),
        CompareCase(
            case_id="optimized_vs_actual_che151hf",
            title="车151HF 第一轮优化结果 vs 实际裂缝提取",
            well_name="车151HF",
            prediction_kind="optimized",
            prediction_label="第一轮优化结果",
            actual_label="实际裂缝提取",
            prediction_path=optimized_che151_path,
            actual_path=actual_dir / "车151HF_fractures.csv",
            track_path=track_dir / "车151HF.dat",
        ),
    ]


def read_text_auto(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to open {path} with supported encodings.") from last_error


def open_text_auto(path: Path):
    return io.StringIO(read_text_auto(path))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open_text_auto(path) as handle:
        return list(csv.DictReader(handle))


def read_track_dat(path: Path) -> TrackInterpolator:
    md_axis: list[float] = []
    tvd_axis: list[float] = []
    x_axis: list[float] = []
    y_axis: list[float] = []

    with open_text_auto(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            md = parse_float(parts[1])
            tvd = parse_float(parts[2])
            x = parse_float(parts[3])
            y = parse_float(parts[4])
            if None in (md, tvd, x, y):
                continue
            md_axis.append(md)
            tvd_axis.append(tvd)
            x_axis.append(x)
            y_axis.append(y)

    if len(md_axis) < 2:
        raise ValueError(f"Track file {path} has insufficient samples.")

    tvd_for_inverse, md_for_inverse = compress_axis_pairs(tvd_axis, md_axis)
    return TrackInterpolator(
        md_axis=md_axis,
        tvd_axis=tvd_axis,
        x_axis=x_axis,
        y_axis=y_axis,
        md_from_tvd_axis_tvd=tvd_for_inverse,
        md_from_tvd_axis_md=md_for_inverse,
    )


def compress_axis_pairs(xs: list[float], ys: list[float], tol: float = 1e-9) -> tuple[list[float], list[float]]:
    result_x: list[float] = []
    result_y: list[float] = []
    for x, y in zip(xs, ys):
        if result_x and abs(x - result_x[-1]) <= tol:
            result_y[-1] = y
            continue
        result_x.append(x)
        result_y.append(y)
    return result_x, result_y


def interpolate_linear(xs: list[float], ys: list[float], x: float | None) -> float | None:
    if x is None or not xs or len(xs) != len(ys):
        return None
    if x < xs[0] or x > xs[-1]:
        return None
    idx = bisect_left(xs, x)
    if idx == 0:
        return ys[0]
    if idx == len(xs):
        return ys[-1]
    x0 = xs[idx - 1]
    x1 = xs[idx]
    y0 = ys[idx - 1]
    y1 = ys[idx]
    if abs(x1 - x0) <= 1e-12:
        return y1
    weight = (x - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if (
        abs(number - (-99999.0)) <= 1e-9
        or abs(number - 99999.0) <= 1e-9
        or abs(number - (-9999.0)) <= 1e-9
        or abs(number - 9999.0) <= 1e-9
        or abs(number - (-999.25)) <= 1e-9
        or abs(number - 999.25) <= 1e-9
    ):
        return None
    return number


def normalize_azimuth(value: float | None) -> float | None:
    if value is None:
        return None
    return value % 360.0


def circular_diff_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)


def load_prediction_points(case: CompareCase, track: TrackInterpolator) -> list[FracturePoint]:
    rows = read_csv_rows(case.prediction_path)
    points: list[FracturePoint] = []
    for idx, row in enumerate(rows, start=1):
        if case.prediction_kind == "raw":
            depth_tvd = parse_float(row.get("TVD"))
            point = FracturePoint(
                well_name=case.well_name,
                source_kind="predicted",
                source_label=case.prediction_label,
                source_file=str(case.prediction_path),
                record_id=f"{case.case_id}_pred_{idx}",
                # Unified comparison axis:
                # treat prediction-file TVD as the direct depth coordinate.
                depth_md=depth_tvd,
                depth_tvd=depth_tvd,
                x=parse_float(row.get("X")),
                y=parse_float(row.get("Y")),
                time=parse_float(row.get("TIME")),
                azimuth=normalize_azimuth(parse_float(row.get("Dip_Azimuth"))),
                dip=parse_float(row.get("Dip_Angle")),
            )
        elif case.prediction_kind == "optimized":
            depth_tvd = parse_float(row.get("TVD"))
            well_name = (row.get("WellName") or case.well_name).strip()
            point = FracturePoint(
                well_name=well_name,
                source_kind="predicted",
                source_label=case.prediction_label,
                source_file=str(case.prediction_path),
                record_id=f"{case.case_id}_pred_{idx}",
                # Unified comparison axis:
                # treat prediction-file TVD as the direct depth coordinate.
                depth_md=depth_tvd,
                depth_tvd=depth_tvd,
                x=parse_float(row.get("X")),
                y=parse_float(row.get("Y")),
                time=parse_float(row.get("TIME")),
                azimuth=normalize_azimuth(parse_float(row.get("PointAzimuth"))),
                dip=parse_float(row.get("PointDip")),
            )
        else:
            raise ValueError(f"Unsupported prediction kind: {case.prediction_kind}")
        points.append(point)
    return points


def load_actual_points(case: CompareCase, track: TrackInterpolator) -> list[FracturePoint]:
    rows = read_csv_rows(case.actual_path)
    points: list[FracturePoint] = []
    for idx, row in enumerate(rows, start=1):
        depth_md = parse_float(row.get("MD"))
        point = FracturePoint(
            well_name=case.well_name,
            source_kind="actual",
            source_label=case.actual_label,
            source_file=str(case.actual_path),
            record_id=f"{case.case_id}_act_{idx}",
            depth_md=depth_md,
            # Unified comparison axis:
            # actual fracture file uses MD column name, but the numeric values are
            # treated as the same TVD-like depth axis for this comparison.
            depth_tvd=depth_md,
            x=None,
            y=None,
            time=None,
            azimuth=normalize_azimuth(parse_float(row.get("Azimuth(0~360)"))),
            dip=parse_float(row.get("Angle(0~90)")),
        )
        points.append(point)
    return points


def point_to_row(point: FracturePoint) -> dict[str, object]:
    return {
        "WellName": point.well_name,
        "SourceKind": point.source_kind,
        "SourceLabel": point.source_label,
        "SourceFile": point.source_file,
        "RecordID": point.record_id,
        "DepthMD": point.depth_md,
        "DepthTVD": point.depth_tvd,
        "X": point.x,
        "Y": point.y,
        "TIME": point.time,
        "Azimuth": point.azimuth,
        "Dip": point.dip,
    }


def evaluate_case(
    case: CompareCase,
    predicted_points: list[FracturePoint],
    actual_points: list[FracturePoint],
    depth_bin_size: float,
    azimuth_bin_size: float,
    dip_bin_size: float,
    error_bin_count: int,
) -> dict[str, object]:
    predicted_valid = [item for item in predicted_points if item.depth_tvd is not None]
    actual_valid = [item for item in actual_points if item.depth_tvd is not None]

    if not predicted_valid or not actual_valid:
        raise ValueError(f"{case.case_id} has no valid predicted/actual depth samples.")

    actual_tvd_values = [item.depth_tvd for item in actual_valid if item.depth_tvd is not None]

    # Only compare prediction points that fall inside the actual fracture interval.
    overlap_min = min(actual_tvd_values)
    overlap_max = max(actual_tvd_values)
    if overlap_max <= overlap_min:
        raise ValueError(f"{case.case_id} has invalid actual fracture interval.")

    predicted_overlap = [item for item in predicted_valid if item.depth_tvd is not None and overlap_min <= item.depth_tvd <= overlap_max]
    actual_overlap = [item for item in actual_valid if item.depth_tvd is not None and overlap_min <= item.depth_tvd <= overlap_max]

    matches: list[MatchResult] = []
    for predicted in predicted_overlap:
        nearest = nearest_point(predicted, actual_overlap)
        if nearest is None:
            continue
        signed_md_error = None
        abs_md_error = None
        if predicted.depth_md is not None and nearest.depth_md is not None:
            signed_md_error = predicted.depth_md - nearest.depth_md
            abs_md_error = abs(signed_md_error)
        signed_tvd_error = None
        abs_tvd_error = None
        if predicted.depth_tvd is not None and nearest.depth_tvd is not None:
            signed_tvd_error = predicted.depth_tvd - nearest.depth_tvd
            abs_tvd_error = abs(signed_tvd_error)
        matches.append(
            MatchResult(
                predicted_record_id=predicted.record_id,
                actual_record_id=nearest.record_id,
                predicted_md=predicted.depth_md,
                actual_md=nearest.depth_md,
                signed_md_error=signed_md_error,
                abs_md_error=abs_md_error,
                predicted_tvd=predicted.depth_tvd,
                actual_tvd=nearest.depth_tvd,
                signed_tvd_error=signed_tvd_error,
                abs_tvd_error=abs_tvd_error,
                predicted_azimuth=predicted.azimuth,
                actual_azimuth=nearest.azimuth,
                azimuth_diff=circular_diff_deg(predicted.azimuth, nearest.azimuth),
                predicted_dip=predicted.dip,
                actual_dip=nearest.dip,
                dip_diff=(abs(predicted.dip - nearest.dip) if predicted.dip is not None and nearest.dip is not None else None),
            )
        )

    overlap_length = overlap_max - overlap_min
    depth_bin_edges = build_regular_edges(overlap_min, overlap_max, depth_bin_size, include_endpoint=True)
    depth_bin_labels = build_interval_labels(depth_bin_edges, integer_format=True)
    predicted_depth_counts = histogram([item.depth_tvd for item in predicted_overlap if item.depth_tvd is not None], depth_bin_edges)
    actual_depth_counts = histogram([item.depth_tvd for item in actual_overlap if item.depth_tvd is not None], depth_bin_edges)

    azimuth_edges = build_regular_edges(0.0, 360.0, azimuth_bin_size, include_endpoint=True)
    azimuth_labels = build_interval_labels(azimuth_edges, suffix="°")
    predicted_azimuth_probs = normalize_histogram(
        histogram([item.azimuth for item in predicted_overlap if item.azimuth is not None], azimuth_edges)
    )
    actual_azimuth_probs = normalize_histogram(
        histogram([item.azimuth for item in actual_overlap if item.azimuth is not None], azimuth_edges)
    )

    dip_edges = build_regular_edges(0.0, 90.0, dip_bin_size, include_endpoint=True)
    dip_labels = build_interval_labels(dip_edges, suffix="°")
    predicted_dip_probs = normalize_histogram(
        histogram([item.dip for item in predicted_overlap if item.dip is not None], dip_edges)
    )
    actual_dip_probs = normalize_histogram(
        histogram([item.dip for item in actual_overlap if item.dip is not None], dip_edges)
    )

    signed_md_errors = [item.signed_md_error for item in matches if item.signed_md_error is not None]
    abs_md_errors = [item.abs_md_error for item in matches if item.abs_md_error is not None]
    signed_tvd_errors = [item.signed_tvd_error for item in matches if item.signed_tvd_error is not None]
    abs_tvd_errors = [item.abs_tvd_error for item in matches if item.abs_tvd_error is not None]
    azimuth_diffs = [item.azimuth_diff for item in matches if item.azimuth_diff is not None]
    dip_diffs = [item.dip_diff for item in matches if item.dip_diff is not None]

    error_edges = build_signed_error_edges(signed_md_errors, error_bin_count)
    error_labels = build_interval_labels(error_edges, decimal_places=1)
    signed_md_error_counts = histogram(signed_md_errors, error_edges)

    pred_azimuth_mean = circular_mean_deg([item.azimuth for item in predicted_overlap if item.azimuth is not None])
    act_azimuth_mean = circular_mean_deg([item.azimuth for item in actual_overlap if item.azimuth is not None])
    pred_dip_mean = mean_or_none([item.dip for item in predicted_overlap if item.dip is not None])
    act_dip_mean = mean_or_none([item.dip for item in actual_overlap if item.dip is not None])

    summary = {
        "CaseID": case.case_id,
        "Title": case.title,
        "WellName": case.well_name,
        "PredictionLabel": case.prediction_label,
        "ActualLabel": case.actual_label,
        "PredictionFile": str(case.prediction_path),
        "ActualFile": str(case.actual_path),
        "TrackFile": str(case.track_path),
        "DepthAxisPolicy": "UnifiedTVD_PredTVD_And_ActualMDTreatedAsTVD",
        "DepthWindowSource": "ActualFractureIntervalOnly",
        "PredictedTotalCount": len(predicted_points),
        "ActualTotalCount": len(actual_points),
        "OverlapTVDMin": round(overlap_min, 6),
        "OverlapTVDMax": round(overlap_max, 6),
        "OverlapLength": round(overlap_length, 6),
        "PredictedOverlapCount": len(predicted_overlap),
        "ActualOverlapCount": len(actual_overlap),
        "PredictedOverlapPer100m": round(safe_div(len(predicted_overlap) * 100.0, overlap_length), 6),
        "ActualOverlapPer100m": round(safe_div(len(actual_overlap) * 100.0, overlap_length), 6),
        "PredToActualOverlapCountRatio": round(safe_div(len(predicted_overlap), len(actual_overlap)), 6),
        "MatchedPredictedCount": len(matches),
        "SignedMDErrorMean": round_or_none(mean_or_none(signed_md_errors), 6),
        "SignedMDErrorMedian": round_or_none(median_or_none(signed_md_errors), 6),
        "AbsMDErrorMean": round_or_none(mean_or_none(abs_md_errors), 6),
        "AbsMDErrorMedian": round_or_none(median_or_none(abs_md_errors), 6),
        "AbsMDErrorP90": round_or_none(quantile(abs_md_errors, 0.90), 6),
        "AbsMDErrorMax": round_or_none(max_or_none(abs_md_errors), 6),
        "SignedTVDErrorMean": round_or_none(mean_or_none(signed_tvd_errors), 6),
        "AbsTVDErrorMedian": round_or_none(median_or_none(abs_tvd_errors), 6),
        "AbsTVDErrorP90": round_or_none(quantile(abs_tvd_errors, 0.90), 6),
        "AzimuthDiffMean": round_or_none(mean_or_none(azimuth_diffs), 6),
        "AzimuthDiffMedian": round_or_none(median_or_none(azimuth_diffs), 6),
        "AzimuthDiffP90": round_or_none(quantile(azimuth_diffs, 0.90), 6),
        "DipDiffMean": round_or_none(mean_or_none(dip_diffs), 6),
        "DipDiffMedian": round_or_none(median_or_none(dip_diffs), 6),
        "DipDiffP90": round_or_none(quantile(dip_diffs, 0.90), 6),
        "PredictedAzimuthCircularMean": round_or_none(pred_azimuth_mean, 6),
        "ActualAzimuthCircularMean": round_or_none(act_azimuth_mean, 6),
        "AzimuthCircularMeanDiff": round_or_none(circular_diff_deg(pred_azimuth_mean, act_azimuth_mean), 6),
        "PredictedDipMean": round_or_none(pred_dip_mean, 6),
        "ActualDipMean": round_or_none(act_dip_mean, 6),
        "DipMeanDiff": round_or_none(abs_diff(pred_dip_mean, act_dip_mean), 6),
        "AzimuthHistTotalVariation": round_or_none(total_variation_distance(predicted_azimuth_probs, actual_azimuth_probs), 6),
        "DipHistTotalVariation": round_or_none(total_variation_distance(predicted_dip_probs, actual_dip_probs), 6),
    }

    return {
        "case": case,
        "summary": summary,
        "predicted_points": predicted_points,
        "actual_points": actual_points,
        "predicted_overlap": predicted_overlap,
        "actual_overlap": actual_overlap,
        "matches": matches,
        "depth_bin_labels": depth_bin_labels,
        "predicted_depth_counts": predicted_depth_counts,
        "actual_depth_counts": actual_depth_counts,
        "error_bin_labels": error_labels,
        "signed_md_error_counts": signed_md_error_counts,
        "azimuth_labels": azimuth_labels,
        "predicted_azimuth_probs": predicted_azimuth_probs,
        "actual_azimuth_probs": actual_azimuth_probs,
        "dip_labels": dip_labels,
        "predicted_dip_probs": predicted_dip_probs,
        "actual_dip_probs": actual_dip_probs,
    }


def nearest_point(predicted: FracturePoint, actual_points: list[FracturePoint]) -> FracturePoint | None:
    if predicted.depth_tvd is None:
        return None
    sorted_points = sorted(
        [item for item in actual_points if item.depth_tvd is not None],
        key=lambda item: item.depth_tvd if item.depth_tvd is not None else -1.0,
    )
    if not sorted_points:
        return None
    tvd_values = [item.depth_tvd for item in sorted_points if item.depth_tvd is not None]
    idx = bisect_left(tvd_values, predicted.depth_tvd)
    candidates: list[FracturePoint] = []
    if idx < len(sorted_points):
        candidates.append(sorted_points[idx])
    if idx > 0:
        candidates.append(sorted_points[idx - 1])
    return min(candidates, key=lambda item: abs((item.depth_tvd or 0.0) - predicted.depth_tvd)) if candidates else None


def build_regular_edges(start: float, end: float, step: float, include_endpoint: bool) -> list[float]:
    if step <= 0:
        raise ValueError("Step must be positive.")
    edges = [start]
    current = start
    while current + step < end:
        current += step
        edges.append(current)
    if include_endpoint or edges[-1] < end:
        edges.append(end)
    if len(edges) < 2:
        edges = [start, end]
    return edges


def build_signed_error_edges(values: list[float], bin_count: int) -> list[float]:
    if bin_count < 2:
        bin_count = 2
    if not values:
        return [-1.0, 1.0]
    max_abs = max(abs(value) for value in values)
    if max_abs <= 1e-9:
        max_abs = 1.0
    step = (2.0 * max_abs) / float(bin_count)
    edges = [-max_abs + step * idx for idx in range(bin_count + 1)]
    edges[0] = -max_abs
    edges[-1] = max_abs
    return edges


def build_interval_labels(
    edges: list[float],
    suffix: str = "",
    decimal_places: int = 0,
    integer_format: bool = False,
) -> list[str]:
    labels: list[str] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if integer_format:
            labels.append(f"{int(round(start))}-{int(round(end))}{suffix}")
        else:
            labels.append(f"{start:.{decimal_places}f}-{end:.{decimal_places}f}{suffix}")
    return labels


def histogram(values: Iterable[float | None], edges: list[float]) -> list[int]:
    counts = [0 for _ in range(len(edges) - 1)]
    if len(edges) < 2:
        return counts
    for value in values:
        if value is None:
            continue
        if value < edges[0] or value > edges[-1]:
            continue
        idx = bisect_left(edges, value) - 1
        if idx < 0:
            idx = 0
        if idx >= len(counts):
            idx = len(counts) - 1
        if value == edges[-1]:
            idx = len(counts) - 1
        counts[idx] += 1
    return counts


def normalize_histogram(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total <= 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def safe_div(a: float, b: float) -> float:
    if abs(b) <= 1e-12:
        return 0.0
    return a / b


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def round_or_none(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def abs_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def total_variation_distance(a_probs: list[float], b_probs: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(a_probs, b_probs))


def circular_mean_deg(values: list[float]) -> float | None:
    if not values:
        return None
    sum_sin = sum(math.sin(math.radians(value)) for value in values)
    sum_cos = sum(math.cos(math.radians(value)) for value in values)
    if abs(sum_sin) <= 1e-12 and abs(sum_cos) <= 1e-12:
        return None
    angle = math.degrees(math.atan2(sum_sin, sum_cos))
    return angle % 360.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            handle.write("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_case_svg(case_result: dict[str, object], output_path: Path) -> None:
    case = case_result["case"]
    summary = case_result["summary"]
    width = 1200
    height = 860
    figure_elements: list[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        svg_text(40, 40, case.title, font_size=24, font_weight="bold"),
        svg_text(
            40,
            70,
            (
                f"可比深度窗 TVD: {summary['OverlapTVDMin']:.2f} - {summary['OverlapTVDMax']:.2f} m; "
                f"预测重叠点数 {summary['PredictedOverlapCount']} / 实际重叠点数 {summary['ActualOverlapCount']}"
            ),
            font_size=14,
            fill="#333333",
        ),
    ]

    panels = [
        (40, 100, 540, 320, "裂缝数量对比（按 TVD 分箱）"),
        (620, 100, 540, 320, "最近实际裂缝 MD 偏差分布"),
        (40, 470, 540, 320, "裂缝倾向分布对比"),
        (620, 470, 540, 320, "裂缝倾角分布对比"),
    ]

    figure_elements.extend(
        draw_grouped_bar_panel(
            *panels[0][:4],
            title=panels[0][4],
            labels=case_result["depth_bin_labels"],
            series_a=case_result["actual_depth_counts"],
            series_b=case_result["predicted_depth_counts"],
            label_a=case.actual_label,
            label_b=case.prediction_label,
            color_a="#4c78a8",
            color_b="#f58518",
            y_label="数量",
            normalize=False,
        )
    )

    error_summary_lines = [
        f"匹配点数: {summary['MatchedPredictedCount']}",
        f"Signed MD Mean: {format_metric(summary['SignedMDErrorMean'])} m",
        f"Abs MD Median: {format_metric(summary['AbsMDErrorMedian'])} m",
        f"Abs MD P90: {format_metric(summary['AbsMDErrorP90'])} m",
    ]
    figure_elements.extend(
        draw_single_bar_panel(
            *panels[1][:4],
            title=panels[1][4],
            labels=case_result["error_bin_labels"],
            counts=case_result["signed_md_error_counts"],
            color="#54a24b",
            y_label="数量",
            note_lines=error_summary_lines,
        )
    )

    azimuth_note_lines = [
        f"TV Distance: {format_metric(summary['AzimuthHistTotalVariation'])}",
        f"Circular Mean Diff: {format_metric(summary['AzimuthCircularMeanDiff'])}°",
    ]
    figure_elements.extend(
        draw_grouped_bar_panel(
            *panels[2][:4],
            title=panels[2][4],
            labels=case_result["azimuth_labels"],
            series_a=case_result["actual_azimuth_probs"],
            series_b=case_result["predicted_azimuth_probs"],
            label_a=case.actual_label,
            label_b=case.prediction_label,
            color_a="#4c78a8",
            color_b="#f58518",
            y_label="占比",
            normalize=True,
            note_lines=azimuth_note_lines,
        )
    )

    dip_note_lines = [
        f"TV Distance: {format_metric(summary['DipHistTotalVariation'])}",
        f"Dip Mean Diff: {format_metric(summary['DipMeanDiff'])}°",
        f"Matched Median Diff: {format_metric(summary['DipDiffMedian'])}°",
    ]
    figure_elements.extend(
        draw_grouped_bar_panel(
            *panels[3][:4],
            title=panels[3][4],
            labels=case_result["dip_labels"],
            series_a=case_result["actual_dip_probs"],
            series_b=case_result["predicted_dip_probs"],
            label_a=case.actual_label,
            label_b=case.prediction_label,
            color_a="#4c78a8",
            color_b="#f58518",
            y_label="占比",
            normalize=True,
            note_lines=dip_note_lines,
        )
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Microsoft YaHei, SimHei, Arial, sans-serif">'
        + "".join(figure_elements)
        + "</svg>"
    )
    ensure_dir(output_path.parent)
    output_path.write_text(svg, encoding="utf-8")


def create_case_svg_unified(case_result: dict[str, object], output_path: Path) -> None:
    case = case_result["case"]
    summary = case_result["summary"]
    width = 1200
    height = 860
    figure_elements: list[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        svg_text(40, 40, case.title, font_size=24, font_weight="bold"),
        svg_text(
            40,
            70,
            (
                f"统一深度轴按 TVD 解释，实际裂缝井段: {summary['OverlapTVDMin']:.2f} - "
                f"{summary['OverlapTVDMax']:.2f} m; 井段内预测点数 "
                f"{summary['PredictedOverlapCount']} / 实际裂缝点数 {summary['ActualOverlapCount']}"
            ),
            font_size=14,
            fill="#333333",
        ),
    ]

    panels = [
        (40, 100, 540, 320, "裂缝数量对比（实际裂缝井段内，按 TVD 分箱）"),
        (620, 100, 540, 320, "最近实际裂缝深度偏差分布"),
        (40, 470, 540, 320, "裂缝倾向分布对比"),
        (620, 470, 540, 320, "裂缝倾角分布对比"),
    ]

    figure_elements.extend(
        draw_grouped_bar_panel(
            *panels[0][:4],
            title=panels[0][4],
            labels=case_result["depth_bin_labels"],
            series_a=case_result["actual_depth_counts"],
            series_b=case_result["predicted_depth_counts"],
            label_a=case.actual_label,
            label_b=case.prediction_label,
            color_a="#4c78a8",
            color_b="#f58518",
            y_label="数量",
            normalize=False,
        )
    )

    error_summary_lines = [
        f"匹配预测点数: {summary['MatchedPredictedCount']}",
        f"Signed Depth Mean: {format_metric(summary['SignedTVDErrorMean'])} m",
        f"Abs Depth Median: {format_metric(summary['AbsTVDErrorMedian'])} m",
        f"Abs Depth P90: {format_metric(summary['AbsTVDErrorP90'])} m",
    ]
    figure_elements.extend(
        draw_single_bar_panel(
            *panels[1][:4],
            title=panels[1][4],
            labels=case_result["error_bin_labels"],
            counts=case_result["signed_md_error_counts"],
            color="#54a24b",
            y_label="数量",
            note_lines=error_summary_lines,
        )
    )

    azimuth_note_lines = [
        f"TV Distance: {format_metric(summary['AzimuthHistTotalVariation'])}",
        f"Circular Mean Diff: {format_metric(summary['AzimuthCircularMeanDiff'])}°",
    ]
    figure_elements.extend(
        draw_grouped_bar_panel(
            *panels[2][:4],
            title=panels[2][4],
            labels=case_result["azimuth_labels"],
            series_a=case_result["actual_azimuth_probs"],
            series_b=case_result["predicted_azimuth_probs"],
            label_a=case.actual_label,
            label_b=case.prediction_label,
            color_a="#4c78a8",
            color_b="#f58518",
            y_label="占比",
            normalize=True,
            note_lines=azimuth_note_lines,
        )
    )

    dip_note_lines = [
        f"TV Distance: {format_metric(summary['DipHistTotalVariation'])}",
        f"Dip Mean Diff: {format_metric(summary['DipMeanDiff'])}°",
        f"Matched Median Diff: {format_metric(summary['DipDiffMedian'])}°",
    ]
    figure_elements.extend(
        draw_grouped_bar_panel(
            *panels[3][:4],
            title=panels[3][4],
            labels=case_result["dip_labels"],
            series_a=case_result["actual_dip_probs"],
            series_b=case_result["predicted_dip_probs"],
            label_a=case.actual_label,
            label_b=case.prediction_label,
            color_a="#4c78a8",
            color_b="#f58518",
            y_label="占比",
            normalize=True,
            note_lines=dip_note_lines,
        )
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Microsoft YaHei, SimHei, Arial, sans-serif">'
        + "".join(figure_elements)
        + "</svg>"
    )
    ensure_dir(output_path.parent)
    output_path.write_text(svg, encoding="utf-8")


def convert_svg_to_png(svg_path: Path, png_path: Path) -> None:
    ensure_dir(png_path.parent)
    command_env = os.environ.copy()
    command_env["CODEX_SVG_PATH"] = str(svg_path)
    command_env["CODEX_PNG_PATH"] = str(png_path)
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            POWERSHELL_SVG_TO_PNG,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=command_env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SVG to PNG conversion failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not png_path.exists():
        raise RuntimeError(f"PNG file was not created: {png_path}")


def draw_grouped_bar_panel(
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    labels: list[str],
    series_a: list[float] | list[int],
    series_b: list[float] | list[int],
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    y_label: str,
    normalize: bool,
    note_lines: list[str] | None = None,
) -> list[str]:
    note_lines = note_lines or []
    margin_left = 60
    margin_right = 20
    margin_top = 45
    margin_bottom = 72
    plot_x = x + margin_left
    plot_y = y + margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_max = max([1.0] + [float(value) for value in series_a] + [float(value) for value in series_b])
    if normalize:
        y_max = max(y_max, 0.1)

    elements = draw_panel_frame(x, y, width, height, title)
    elements.append(svg_text(x + width - 190, y + 26, label_a, font_size=12, fill=color_a))
    elements.append(svg_text(x + width - 95, y + 26, label_b, font_size=12, fill=color_b))
    elements.append(svg_rect(x + width - 208, y + 17, 10, 10, color_a))
    elements.append(svg_rect(x + width - 113, y + 17, 10, 10, color_b))
    elements.extend(draw_axes(plot_x, plot_y, plot_w, plot_h, y_max, y_label, percentage=normalize))

    count = max(len(labels), 1)
    group_w = plot_w / count
    bar_w = min(group_w * 0.34, 24.0)
    label_step = max(1, math.ceil(count / 8))

    for idx in range(count):
        center_x = plot_x + group_w * (idx + 0.5)
        value_a = float(series_a[idx]) if idx < len(series_a) else 0.0
        value_b = float(series_b[idx]) if idx < len(series_b) else 0.0
        elements.append(draw_bar(center_x - bar_w - 2.0, plot_y, plot_h, bar_w, value_a, y_max, color_a))
        elements.append(draw_bar(center_x + 2.0, plot_y, plot_h, bar_w, value_b, y_max, color_b))
        if idx % label_step == 0:
            elements.append(svg_text(center_x, plot_y + plot_h + 18, labels[idx], font_size=10, rotate=45, anchor="start"))

    for note_idx, note_line in enumerate(note_lines):
        elements.append(svg_text(x + 18, y + height - 16 - 16 * (len(note_lines) - note_idx - 1), note_line, font_size=11, fill="#555555"))
    return elements


def draw_single_bar_panel(
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    labels: list[str],
    counts: list[int],
    color: str,
    y_label: str,
    note_lines: list[str] | None = None,
) -> list[str]:
    note_lines = note_lines or []
    margin_left = 60
    margin_right = 20
    margin_top = 45
    margin_bottom = 72
    plot_x = x + margin_left
    plot_y = y + margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_max = max([1] + counts)
    elements = draw_panel_frame(x, y, width, height, title)
    elements.extend(draw_axes(plot_x, plot_y, plot_w, plot_h, float(y_max), y_label, percentage=False))

    count = max(len(labels), 1)
    group_w = plot_w / count
    bar_w = min(group_w * 0.72, 28.0)
    label_step = max(1, math.ceil(count / 8))

    for idx in range(count):
        center_x = plot_x + group_w * (idx + 0.5)
        value = float(counts[idx]) if idx < len(counts) else 0.0
        elements.append(draw_bar(center_x - bar_w / 2.0, plot_y, plot_h, bar_w, value, float(y_max), color))
        if idx % label_step == 0:
            elements.append(svg_text(center_x, plot_y + plot_h + 18, labels[idx], font_size=10, rotate=45, anchor="start"))

    for note_idx, note_line in enumerate(note_lines):
        elements.append(svg_text(x + 18, y + height - 16 - 16 * (len(note_lines) - note_idx - 1), note_line, font_size=11, fill="#555555"))
    return elements


def draw_panel_frame(x: int, y: int, width: int, height: int, title: str) -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" ry="8" fill="#fafafa" stroke="#d0d0d0" />',
        svg_text(x + 16, y + 26, title, font_size=16, font_weight="bold"),
    ]


def draw_axes(
    plot_x: int,
    plot_y: int,
    plot_w: int,
    plot_h: int,
    y_max: float,
    y_label: str,
    percentage: bool,
) -> list[str]:
    elements = [
        svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, "#777777", 1.2),
        svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, "#777777", 1.2),
    ]
    tick_count = 4
    for tick_idx in range(tick_count + 1):
        value = y_max * tick_idx / tick_count
        tick_y = plot_y + plot_h - plot_h * tick_idx / tick_count
        elements.append(svg_line(plot_x, tick_y, plot_x + plot_w, tick_y, "#e2e2e2", 1.0))
        label = f"{value:.0%}" if percentage else (f"{value:.0f}" if y_max >= 10 else f"{value:.2f}")
        elements.append(svg_text(plot_x - 8, tick_y + 4, label, font_size=10, anchor="end", fill="#555555"))
    elements.append(svg_text(plot_x - 46, plot_y + 12, y_label, font_size=11, fill="#555555"))
    return elements


def draw_bar(x: float, plot_y: int, plot_h: int, width: float, value: float, y_max: float, color: str) -> str:
    if y_max <= 0:
        return ""
    height = plot_h * (value / y_max)
    y = plot_y + plot_h - height
    return svg_rect(x, y, width, height, color)


def svg_rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" opacity="0.88" />'


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: float) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )


def svg_text(
    x: float,
    y: float,
    text: str,
    font_size: int = 12,
    fill: str = "#111111",
    font_weight: str = "normal",
    rotate: int | None = None,
    anchor: str = "start",
) -> str:
    safe = html.escape(text)
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{font_size}" fill="{fill}" '
        f'font-weight="{font_weight}" text-anchor="{anchor}"{transform}>{safe}</text>'
    )


def format_metric(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report_html(
    output_path: Path,
    summary_rows: list[dict[str, object]],
    figure_paths: dict[str, Path],
    details_paths: dict[str, dict[str, Path]],
) -> None:
    ensure_dir(output_path.parent)
    headers = [
        "CaseID",
        "Title",
        "PredictedTotalCount",
        "ActualTotalCount",
        "PredictedOverlapCount",
        "ActualOverlapCount",
        "PredictedOverlapPer100m",
        "ActualOverlapPer100m",
        "AbsMDErrorMedian",
        "AbsMDErrorP90",
        "AzimuthHistTotalVariation",
        "DipHistTotalVariation",
    ]

    table_header = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    table_rows = []
    for row in summary_rows:
        cells = "".join(f"<td>{html.escape(format_metric(row.get(header)))}</td>" for header in headers)
        case_id = str(row["CaseID"])
        figure_rel = figure_paths[case_id].name
        predicted_rel = details_paths[case_id]["predicted"].name
        actual_rel = details_paths[case_id]["actual"].name
        match_rel = details_paths[case_id]["matches"].name
        links = (
            f'<a href="figures/{html.escape(figure_rel)}">图</a> | '
            f'<a href="details/{html.escape(predicted_rel)}">预测标准表</a> | '
            f'<a href="details/{html.escape(actual_rel)}">实际标准表</a> | '
            f'<a href="details/{html.escape(match_rel)}">匹配结果</a>'
        )
        table_rows.append(f"<tr>{cells}<td>{links}</td></tr>")

    sections = []
    for row in summary_rows:
        case_id = str(row["CaseID"])
        figure_rel = figure_paths[case_id].name
        sections.append(
            f"""
            <section class="card">
              <h2>{html.escape(str(row['Title']))}</h2>
              <img src="figures/{html.escape(figure_rel)}" alt="{html.escape(str(row['Title']))}" />
            </section>
            """
        )

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>裂缝预测对比报告</title>
  <style>
    body {{
      font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
      margin: 24px;
      color: #222;
      background: #f6f7fb;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    .sub {{
      margin-bottom: 20px;
      color: #555;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: #fff;
      margin-bottom: 28px;
    }}
    th, td {{
      border: 1px solid #d7d7d7;
      padding: 8px 10px;
      font-size: 13px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f0f3f8;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d7d7d7;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #e0e0e0;
    }}
    a {{
      color: #0b63ce;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <h1>裂缝预测对比报告</h1>
  <div class="sub">脚本输出为纯标准库版本，图件格式为 SVG，统计明细为 CSV。</div>
  <table>
    <thead>
      <tr>{table_header}<th>文件</th></tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
  {''.join(sections)}
</body>
</html>"""
    output_path.write_text(html_content, encoding="utf-8")


def write_report_html_unified(
    output_path: Path,
    summary_rows: list[dict[str, object]],
    figure_paths: dict[str, Path],
    details_paths: dict[str, dict[str, Path]],
) -> None:
    ensure_dir(output_path.parent)
    headers = [
        "CaseID",
        "Title",
        "PredictedTotalCount",
        "ActualTotalCount",
        "PredictedOverlapCount",
        "ActualOverlapCount",
        "PredictedOverlapPer100m",
        "ActualOverlapPer100m",
        "AbsTVDErrorMedian",
        "AbsTVDErrorP90",
        "AzimuthHistTotalVariation",
        "DipHistTotalVariation",
    ]

    table_header = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    table_rows = []
    for row in summary_rows:
        cells = "".join(f"<td>{html.escape(format_metric(row.get(header)))}</td>" for header in headers)
        case_id = str(row["CaseID"])
        figure_rel = figure_paths[case_id].name
        predicted_rel = details_paths[case_id]["predicted"].name
        actual_rel = details_paths[case_id]["actual"].name
        match_rel = details_paths[case_id]["matches"].name
        links = (
            f'<a href="figures/{html.escape(figure_rel)}">图</a> | '
            f'<a href="details/{html.escape(predicted_rel)}">预测标准表</a> | '
            f'<a href="details/{html.escape(actual_rel)}">实际标准表</a> | '
            f'<a href="details/{html.escape(match_rel)}">匹配结果</a>'
        )
        table_rows.append(f"<tr>{cells}<td>{links}</td></tr>")

    sections = []
    for row in summary_rows:
        case_id = str(row["CaseID"])
        figure_rel = figure_paths[case_id].name
        sections.append(
            f"""
            <section class="card">
              <h2>{html.escape(str(row['Title']))}</h2>
              <img src="figures/{html.escape(figure_rel)}" alt="{html.escape(str(row['Title']))}" />
            </section>
            """
        )

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>裂缝预测对比报告</title>
  <style>
    body {{
      font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
      margin: 24px;
      color: #222;
      background: #f6f7fb;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    .sub {{
      margin-bottom: 20px;
      color: #555;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: #fff;
      margin-bottom: 28px;
    }}
    th, td {{
      border: 1px solid #d7d7d7;
      padding: 8px 10px;
      font-size: 13px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f0f3f8;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d7d7d7;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #e0e0e0;
    }}
    a {{
      color: #0b63ce;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <h1>裂缝预测对比报告</h1>
  <div class="sub">当前口径：预测文件的 TVD 与实际裂缝文件的 MD 数值统一按 TVD 解释，只统计实际裂缝井段内的预测结果。</div>
  <table>
    <thead>
      <tr>{table_header}<th>文件</th></tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
  {''.join(sections)}
</body>
</html>"""
    output_path.write_text(html_content, encoding="utf-8")


def validate_inputs(cases: list[CompareCase]) -> None:
    missing = []
    for case in cases:
        for path in (case.prediction_path, case.actual_path, case.track_path):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    cases = build_default_cases()
    validate_inputs(cases)

    output_dir = args.output_dir.resolve()
    detail_dir = output_dir / "details"
    figure_dir = output_dir / "figures"
    ensure_dir(detail_dir)
    ensure_dir(figure_dir)

    summary_rows: list[dict[str, object]] = []
    figure_paths: dict[str, Path] = {}
    detail_paths: dict[str, dict[str, Path]] = {}
    manifest_rows: list[dict[str, object]] = []

    for case in cases:
        track = read_track_dat(case.track_path)
        predicted_points = load_prediction_points(case, track)
        actual_points = load_actual_points(case, track)
        case_result = evaluate_case(
            case,
            predicted_points,
            actual_points,
            depth_bin_size=args.depth_bin_size,
            azimuth_bin_size=args.azimuth_bin_size,
            dip_bin_size=args.dip_bin_size,
            error_bin_count=args.error_bin_count,
        )

        summary_rows.append(case_result["summary"])
        case_id = case.case_id

        predicted_csv = detail_dir / f"{case_id}_predicted_standardized.csv"
        actual_csv = detail_dir / f"{case_id}_actual_standardized.csv"
        match_csv = detail_dir / f"{case_id}_nearest_actual_matches.csv"
        figure_svg = figure_dir / f"{case_id}_overview_tmp.svg"
        figure_png = figure_dir / f"{case_id}_overview.png"

        write_csv(predicted_csv, [point_to_row(item) for item in case_result["predicted_points"]])
        write_csv(actual_csv, [point_to_row(item) for item in case_result["actual_points"]])
        write_csv(match_csv, [asdict(item) for item in case_result["matches"]])
        create_case_svg_unified(case_result, figure_svg)
        convert_svg_to_png(figure_svg, figure_png)
        if figure_svg.exists():
            figure_svg.unlink()

        figure_paths[case_id] = figure_png
        detail_paths[case_id] = {
            "predicted": predicted_csv,
            "actual": actual_csv,
            "matches": match_csv,
        }
        manifest_rows.append(
            {
                "CaseID": case_id,
                "Title": case.title,
                "PredictedStandardizedCSV": str(predicted_csv),
                "ActualStandardizedCSV": str(actual_csv),
                "NearestMatchCSV": str(match_csv),
                "FigurePNG": str(figure_png),
            }
        )

    summary_csv = output_dir / "fracture_compare_summary.csv"
    manifest_json = output_dir / "fracture_compare_manifest.json"
    write_csv(summary_csv, summary_rows)
    manifest_json.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_html:
        write_report_html_unified(output_dir / "report.html", summary_rows, figure_paths, detail_paths)

    print(f"Output directory: {output_dir}")
    print(f"Summary CSV: {summary_csv}")
    if not args.skip_html:
        print(f"HTML report: {output_dir / 'report.html'}")
    for row in summary_rows:
        print(
            f"[{row['CaseID']}] overlap={row['PredictedOverlapCount']}/{row['ActualOverlapCount']}, "
            f"abs_depth_median={format_metric(row['AbsTVDErrorMedian'])}, "
            f"azimuth_tv={format_metric(row['AzimuthHistTotalVariation'])}, "
            f"dip_tv={format_metric(row['DipHistTotalVariation'])}"
        )


if __name__ == "__main__":
    main()
