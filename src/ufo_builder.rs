//! Assemble a Google Fonts-compliant UFO from segmented glyph PNGs.
//!
//! Uses img2bez's trace/place split: each crop is traced to a neutral
//! outline, then placed deterministically into the font's coordinate
//! system from specimen-level metrics (uniform scale + per-row baselines
//! derived from the labeled uppercase letters).

use crate::gf_latin_core;
use crate::manifest::{GlyphEntry, Manifest};
use crate::pipeline::PipelineConfig;
use crate::spacing::{self, SpacingDefaults};
use anyhow::{Context, Result};
use img2bez::norad::fontinfo::{NonNegativeIntegerOrFloat, StyleMapStyle};
use img2bez::norad::{Contour, ContourPoint, Font, FontInfo, Glyph, PointType};
use img2bez::{trace_place, PlacementOptions, Sidebearings, TargetBand, TraceOptions};
use plist::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// What the build produced, for the pipeline's report.
pub struct BuildReport {
    /// Glyphs traced from the specimen this run.
    pub traced: usize,
    /// Hand-corrected (green-marked) glyphs preserved from the existing UFO.
    pub green_kept: usize,
    /// Glyphs included without a codepoint (excluded from the cmap).
    pub unlabeled: usize,
    /// Crops that failed to trace: (file, error).
    pub failed: Vec<(String, String)>,
    /// Encoded codepoints present in the cmap (after composition).
    pub encoded: HashSet<u32>,
    /// Auto-composition results + agent worklist.
    pub completion: crate::composites::CompletionReport,
    /// Where the completion worklist JSON was written.
    pub worklist_path: std::path::PathBuf,
}

// ============================================================================
// Specimen-level metrics: uniform scale + per-row baseline
// ============================================================================

/// Metrics derived from the specimen layout for uniform glyph scaling.
struct SpecimenMetrics {
    /// Pixels-to-font-units scale factor (same for all glyphs).
    uniform_scale: f64,
    /// Baseline y-coordinate (in source image pixels, y-down) per row.
    baselines: HashMap<usize, f64>,
}

impl SpecimenMetrics {
    /// Analyze the manifest to compute a uniform scale and per-row baselines.
    fn from_manifest(manifest: &Manifest, cap_height: f64) -> Self {
        // Uppercase letters (A-Z) determine scale and baselines.
        let uppercase: Vec<&GlyphEntry> = manifest
            .glyphs
            .iter()
            .filter(|g| matches!(g.codepoint(), Some('A'..='Z')))
            .collect();

        // Uniform scale: reference uppercase height -> cap_height.
        // Prefer H (flat top and bottom, no overshoot), else the median.
        let reference_height = uppercase
            .iter()
            .find(|g| g.codepoint() == Some('H'))
            .map(|g| g.bbox.h as f64)
            .unwrap_or_else(|| {
                let mut heights: Vec<f64> =
                    uppercase.iter().map(|g| g.bbox.h as f64).collect();
                heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if heights.is_empty() {
                    // No labeled uppercase — median height of all crops.
                    let mut all: Vec<f64> =
                        manifest.glyphs.iter().map(|g| g.bbox.h as f64).collect();
                    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    all.get(all.len() / 2).copied().unwrap_or(200.0)
                } else {
                    heights[heights.len() / 2]
                }
            });
        let uniform_scale = cap_height / reference_height;

        // Per-row baselines: median bottom edge of uppercase letters that
        // sit flat on the baseline (exclude J and Q which may descend).
        let mut row_uc_bottoms: HashMap<usize, Vec<f64>> = HashMap::new();
        for g in &uppercase {
            if matches!(g.codepoint(), Some('J') | Some('Q')) {
                continue;
            }
            row_uc_bottoms
                .entry(g.row)
                .or_default()
                .push((g.bbox.y + g.bbox.h) as f64);
        }

        let mut baselines: HashMap<usize, f64> = HashMap::new();
        for (row, bottoms) in &row_uc_bottoms {
            let mut sorted = bottoms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            baselines.insert(*row, sorted[sorted.len() / 2]);
        }

        // Rows without uppercase: median bottom of non-descender glyphs,
        // else the nearest row that has a baseline.
        let max_row = manifest.glyphs.iter().map(|g| g.row).max().unwrap_or(0);
        for row in 0..=max_row {
            if baselines.contains_key(&row) {
                continue;
            }
            let mut bottoms: Vec<f64> = manifest
                .glyphs
                .iter()
                .filter(|g| g.row == row)
                .filter(|g| {
                    !matches!(
                        g.codepoint(),
                        Some('g' | 'j' | 'p' | 'q' | 'y' | 'Q' | 'J')
                    )
                })
                .map(|g| (g.bbox.y + g.bbox.h) as f64)
                .collect();
            if !bottoms.is_empty() {
                bottoms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                baselines.insert(row, bottoms[bottoms.len() / 2]);
            } else if let Some(&nearest) = baselines
                .keys()
                .min_by_key(|&&r| (r as isize - row as isize).unsigned_abs())
            {
                baselines.insert(row, baselines[&nearest]);
            }
        }

        Self {
            uniform_scale,
            baselines,
        }
    }

    /// Font-unit vertical band `[y_min, y_max]` for a glyph: its sheet bbox
    /// mapped through the row baseline and the uniform scale.
    fn band(&self, entry: &GlyphEntry) -> (f64, f64) {
        let baseline = self
            .baselines
            .get(&entry.row)
            .copied()
            .unwrap_or((entry.bbox.y + entry.bbox.h) as f64);
        let y_max = (baseline - entry.bbox.y as f64) * self.uniform_scale;
        let y_min = (baseline - (entry.bbox.y + entry.bbox.h) as f64) * self.uniform_scale;
        (y_min, y_max)
    }
}

/// Purple marks a machine draft that the designer has not reviewed. The
/// Font.Garden mark colors: purple is an unreviewed draft; red, orange, yellow
/// and green are the designer's grades.
pub(crate) const DRAFT_MARK: &str = "0.54,0.44,0.88,1";

/// The RGB values of the marks that a rebuild can replace: purple (an
/// unreviewed draft) and red (graded as broken).
const REPLACEABLE_MARKS: [[f64; 3]; 5] = [
    [0.54, 0.44, 0.88],
    [0.5, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.29, 0.24],
    [1.0, 0.3, 0.3],
];

/// True if a rebuild must keep the existing glyph: every glyph that is not
/// purple or red. A graded or unmarked glyph can hold the designer's work.
pub(crate) fn keep_on_rebuild(glyph: &Glyph) -> bool {
    let mark = glyph
        .lib
        .get("public.markColor")
        .and_then(|v| v.as_string())
        .unwrap_or("");
    let rgb: Vec<f64> = mark.split(',').filter_map(|v| v.trim().parse().ok()).collect();
    if rgb.len() < 3 {
        return true;
    }
    !REPLACEABLE_MARKS
        .iter()
        .any(|m| m.iter().zip(&rgb).all(|(a, b)| (a - b).abs() < 0.015))
}

/// Build the UFO. If the output UFO already exists, green-marked glyphs are
/// preserved (not overwritten): every glyph that is not purple or red. All
/// newly traced glyphs are marked purple.
pub fn build(
    config: &PipelineConfig,
    manifest: &Manifest,
    glyph_dir: &Path,
    ufo_path: &Path,
) -> Result<BuildReport> {
    // Load existing UFO if present (to preserve green glyphs).
    let existing_font = if ufo_path.exists() {
        Font::load(ufo_path).ok()
    } else {
        None
    };

    let mut font = Font::new();
    let mut report = BuildReport {
        traced: 0,
        green_kept: 0,
        unlabeled: 0,
        failed: Vec::new(),
        encoded: HashSet::new(),
        completion: Default::default(),
        worklist_path: PathBuf::new(),
    };

    add_notdef(&mut font, config);
    add_space(&mut font, config);
    report.encoded.insert(0x0020);
    report.encoded.insert(0x00A0);

    let specimen = SpecimenMetrics::from_manifest(manifest, config.cap_height as f64);
    if config.verbose {
        eprintln!(
            "ufo_builder: uniform scale = {:.3} units/px (cap ref -> {} units)",
            specimen.uniform_scale, config.cap_height
        );
        let mut rows: Vec<_> = specimen.baselines.iter().collect();
        rows.sort_by_key(|&(row, _)| *row);
        for (row, bl) in rows {
            eprintln!("ufo_builder: row {row} baseline at y={bl:.0}px");
        }
    }

    let trace_opts = trace_options(config);
    let spacing_defaults = SpacingDefaults::for_upm(config.upm as f64, config.grid);

    // Font-wide ink bounds (for win metrics).
    let mut font_y_max = f64::MIN;
    let mut font_y_min = f64::MAX;

    let mut glyph_order: Vec<String> =
        vec![".notdef".into(), "space".into(), "uni00A0".into()];

    for entry in &manifest.glyphs {
        let glyph_name = entry
            .glyph_name
            .clone()
            .unwrap_or_else(|| entry.id.replace('_', ""));
        if entry.codepoint().is_none() {
            report.unlabeled += 1;
        }

        // Preserve green (hand-corrected) glyphs from the existing UFO.
        if let Some(ref existing) = existing_font {
            if let Some(existing_glyph) = existing.default_layer().get_glyph(&glyph_name) {
                if keep_on_rebuild(existing_glyph) {
                    font.default_layer_mut().insert_glyph(existing_glyph.clone());
                    glyph_order.push(glyph_name.clone());
                    report.green_kept += 1;
                    if let Some(ch) = entry.codepoint() {
                        report.encoded.insert(ch as u32);
                    }
                    if config.verbose {
                        eprintln!("ufo_builder: keeping reviewed {glyph_name:?}");
                    }
                    continue;
                }
            }
        }

        let png_path = glyph_dir.join(&entry.file);
        match trace_one(
            entry,
            &glyph_name,
            &png_path,
            config,
            &specimen,
            &trace_opts,
            &spacing_defaults,
        ) {
            Ok((glyph, y_min, y_max)) => {
                font_y_max = font_y_max.max(y_max);
                font_y_min = font_y_min.min(y_min);
                if let Some(ch) = entry.codepoint() {
                    report.encoded.insert(ch as u32);
                }
                font.default_layer_mut().insert_glyph(glyph);
                glyph_order.push(glyph_name.clone());
                report.traced += 1;
                if config.verbose {
                    eprintln!("ufo_builder: traced {:?} -> {}", entry.file, glyph_name);
                }
            }
            Err(e) => {
                eprintln!("ufo_builder: warning: skipping {:?}: {e:#}", entry.file);
                report
                    .failed
                    .push((entry.file.display().to_string(), format!("{e:#}")));
            }
        }
    }

    // ------------------------------------------------------------------
    // Auto-composition: anchors + Latin Core composites from what the
    // specimen provided, plus the completion worklist for the rest.
    // ------------------------------------------------------------------
    let composition = crate::composites::run(
        &mut font,
        existing_font.as_ref(),
        config,
        &mut report.encoded,
        &mut glyph_order,
    )?;
    if composition.ink_y_max.is_finite() {
        font_y_max = font_y_max.max(composition.ink_y_max);
    }
    if composition.ink_y_min.is_finite() {
        font_y_min = font_y_min.min(composition.ink_y_min);
    }
    report.completion = composition.report;

    let stem = ufo_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("font");
    report.worklist_path = config
        .worklist
        .clone()
        .unwrap_or_else(|| ufo_path.with_file_name(format!("{stem}-completion.json")));
    if let Some(parent) = report.worklist_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &report.worklist_path,
        serde_json::to_string_pretty(&report.completion)?,
    )
    .with_context(|| format!("Failed to write worklist {:?}", report.worklist_path))?;

    apply_font_info(&mut font.font_info, config, font_y_min, font_y_max);
    apply_lib(&mut font, &glyph_order);

    font.save(ufo_path)
        .with_context(|| format!("Failed to write UFO to {ufo_path:?}"))?;

    Ok(report)
}

fn trace_options(config: &PipelineConfig) -> TraceOptions {
    let mut opts = TraceOptions::for_profile(config.profile);
    if let Some(acc) = config.accuracy {
        opts.fit_accuracy = acc;
    }
    opts.grid = config.grid;
    opts
}

/// Trace one crop and place it: vertical band from the specimen metrics,
/// sidebearings from the spacing-v0 edge classifier. Returns the finished
/// norad glyph plus its font-unit ink y-range.
#[allow(clippy::too_many_arguments)]
fn trace_one(
    entry: &GlyphEntry,
    glyph_name: &str,
    png_path: &Path,
    config: &PipelineConfig,
    specimen: &SpecimenMetrics,
    trace_opts: &TraceOptions,
    spacing_defaults: &SpacingDefaults,
) -> Result<(Glyph, f64, f64)> {
    let bytes = std::fs::read(png_path)
        .with_context(|| format!("Cannot read glyph image {png_path:?}"))?;

    let (y_min, y_max) = specimen.band(entry);
    anyhow::ensure!(y_max > y_min, "empty vertical band for {glyph_name}");

    // First placement pass: ink at lsb 0 so the outline is in a known frame.
    let mut placement = PlacementOptions::ink_fit(
        TargetBand::Custom { y_min, y_max },
        Sidebearings::Explicit { lsb: 0.0, rsb: 0.0 },
    );
    placement.grid = config.grid;
    let mut opts = trace_opts.clone();
    opts.rtl_start = entry
        .codepoint()
        .map(img2bez::is_rtl_codepoint)
        .unwrap_or(false);
    let (mut placed, _report) = trace_place(&bytes, &opts, &placement)
        .map_err(|e| anyhow::anyhow!("img2bez: {e}"))?;

    // Spacing v0: classify the ink edges, shift to the class LSB, and set
    // the advance from the class RSB.
    let (left_class, right_class) = spacing::classify_edges(&placed.outline, config.upm as f64);
    let lsb = spacing_defaults.sidebearing(left_class);
    let rsb = spacing_defaults.sidebearing(right_class);

    // Scan dust: drop a contour with fewer than 3 points, or smaller than
    // 2% of the em in both directions. The smallest part of a glyph, a dot
    // or a comma, is larger.
    let dust = config.upm as f64 * 0.02;
    placed.outline.contours.retain(|contour| {
        let xs = contour.points.iter().map(|p| p.x);
        let ys = contour.points.iter().map(|p| p.y);
        let width = xs.clone().fold(f64::MIN, f64::max) - xs.fold(f64::MAX, f64::min);
        let height = ys.clone().fold(f64::MIN, f64::max) - ys.fold(f64::MAX, f64::min);
        contour.points.len() >= 3 && width.max(height) >= dust
    });
    anyhow::ensure!(!placed.outline.contours.is_empty(), "no ink in {glyph_name}");

    let (ink_x_min, ink_x_max, ink_y_min, ink_y_max) = outline_bounds(&placed.outline);
    // The ink edge of a curve is not on the grid, so snap the shift, and
    // then every point, to the grid.
    let dx = snap(lsb - ink_x_min, config.grid);
    for contour in &mut placed.outline.contours {
        for point in &mut contour.points {
            point.x = snap(point.x + dx, config.grid);
            point.y = snap(point.y, config.grid);
        }
    }
    placed.advance_width = snap(ink_x_max + dx + rsb, config.grid);

    let codepoints: Vec<char> = entry.codepoint().into_iter().collect();
    let mut glyph = img2bez::ufo::to_glyph(glyph_name, &placed, &codepoints)
        .map_err(|e| anyhow::anyhow!("img2bez glif conversion: {e}"))?;

    // Mark as an unreviewed draft, which the next rebuild can replace.
    glyph
        .lib
        .insert("public.markColor".into(), Value::String(DRAFT_MARK.into()));

    Ok((glyph, ink_y_min, ink_y_max))
}

/// Ink bounding box of an outline from its rendered bezier paths.
fn outline_bounds(outline: &img2bez::Outline) -> (f64, f64, f64, f64) {
    use img2bez::kurbo::Shape;
    let mut x0 = f64::MAX;
    let mut y0 = f64::MAX;
    let mut x1 = f64::MIN;
    let mut y1 = f64::MIN;
    for path in outline.to_bezpaths() {
        let b = path.bounding_box();
        x0 = x0.min(b.x0);
        y0 = y0.min(b.y0);
        x1 = x1.max(b.x1);
        y1 = y1.max(b.y1);
    }
    (x0, x1, y0, y1)
}

fn snap(v: f64, grid: i32) -> f64 {
    if grid > 1 {
        (v / grid as f64).round() * grid as f64
    } else {
        v.round()
    }
}

/// Add a .notdef glyph with a standard open-box rectangle.
fn add_notdef(font: &mut Font, config: &PipelineConfig) {
    let upm = config.upm as f64;
    let grid = config.grid.max(1) * 4;
    let w = snap(upm * 0.5, grid);
    let asc = snap(config.ascender as f64, grid);
    let desc = snap(config.descender as f64, grid);
    let stroke = snap(upm * 0.05, grid);

    let mut glyph = Glyph::new(".notdef");
    glyph.width = w;

    // Outer rectangle
    let outer = Contour::new(
        vec![
            cp(stroke, desc + stroke, PointType::Line),
            cp(w - stroke, desc + stroke, PointType::Line),
            cp(w - stroke, asc - stroke, PointType::Line),
            cp(stroke, asc - stroke, PointType::Line),
        ],
        None,
    );
    // Inner rectangle (counter — opposite winding)
    let inner = Contour::new(
        vec![
            cp(stroke * 2.0, desc + stroke * 2.0, PointType::Line),
            cp(stroke * 2.0, asc - stroke * 2.0, PointType::Line),
            cp(w - stroke * 2.0, asc - stroke * 2.0, PointType::Line),
            cp(w - stroke * 2.0, desc + stroke * 2.0, PointType::Line),
        ],
        None,
    );
    glyph.contours.push(outer);
    glyph.contours.push(inner);

    font.default_layer_mut().insert_glyph(glyph);
}

/// Add space and no-break space (U+00A0 is required by fontspector's
/// whitespace_glyphs check; it clones the space width).
fn add_space(font: &mut Font, config: &PipelineConfig) {
    let width = snap(config.upm as f64 / 4.0, config.grid);
    let mut glyph = Glyph::new("space");
    glyph.width = width;
    glyph.codepoints.insert(' ');
    font.default_layer_mut().insert_glyph(glyph);

    let mut nbsp = Glyph::new("uni00A0");
    nbsp.width = width;
    nbsp.codepoints.insert('\u{00A0}');
    font.default_layer_mut().insert_glyph(nbsp);
}

#[inline]
fn cp(x: f64, y: f64, typ: PointType) -> ContourPoint {
    ContourPoint::new(x, y, typ, false, None, None)
}

/// Populate lib.plist keys.
fn apply_lib(font: &mut Font, glyph_order: &[String]) {
    let order: Vec<Value> = glyph_order
        .iter()
        .map(|s| Value::String(s.clone()))
        .collect();
    font.lib
        .insert("public.glyphOrder".into(), Value::Array(order));
}

/// Populate `FontInfo` per docs/gf-compliance-checklist.md.
///
/// Vertical metrics strategy (checklist section 1):
/// - typoDescender = --descender; typoAscender chosen so that
///   (a) typoAscender - capHeight ~= |typoDescender| (caps centered) and
///   (b) typoAscender + |typoDescender| >= 120% of UPM.
/// - hhea == typo, all lineGaps 0, fsSelection bit 7 (USE_TYPO_METRICS).
/// - win = the actual ink extremes of this build (floored at the typo
///   values). Single-master family, so family-wide == this font.
fn apply_font_info(info: &mut FontInfo, config: &PipelineConfig, ink_y_min: f64, ink_y_max: f64) {
    info.family_name = Some(config.family.clone());
    info.style_name = Some(config.style.clone());
    info.style_map_family_name = Some(config.family.clone());
    info.style_map_style_name = Some(StyleMapStyle::Regular);

    let ps_family = config.family.replace(' ', "");
    let ps_style = config.style.replace(' ', "");
    info.postscript_font_name = Some(format!("{ps_family}-{ps_style}"));
    info.postscript_full_name = Some(format!("{} {}", config.family, config.style));

    info.units_per_em = NonNegativeIntegerOrFloat::new(config.upm as f64);
    info.ascender = Some(config.ascender as f64);
    info.descender = Some(config.descender as f64);
    info.x_height = Some(config.x_height as f64);
    info.cap_height = Some(config.cap_height as f64);
    info.italic_angle = Some(0.0);

    // --- Vertical metrics (GF strategy; see doc comment) ---
    let upm = config.upm as i32;
    let typo_desc = config.descender.min(-1);
    let min_sum = (config.upm as f64 * 1.2).ceil() as i32;
    let typo_asc = config
        .ascender
        .max(config.cap_height + typo_desc.abs())
        .max(min_sum - typo_desc.abs())
        .min(upm.max(config.ascender)); // never past the em unless ascender is
    // If the ascender is capped at the em, lower the line descender until the
    // sum reaches the Google Fonts minimum of 1.2 em, on the 8-unit grid.
    let needed = min_sum - typo_asc;
    let typo_desc = typo_desc.min(-((needed + 7) / 8 * 8));
    info.open_type_os2_typo_ascender = Some(typo_asc);
    info.open_type_os2_typo_descender = Some(typo_desc);
    info.open_type_os2_typo_line_gap = Some(0);
    info.open_type_hhea_ascender = Some(typo_asc);
    info.open_type_hhea_descender = Some(typo_desc);
    info.open_type_hhea_line_gap = Some(0);

    // win metrics from the real ink extremes of the build.
    let y_max = if ink_y_max.is_finite() { ink_y_max } else { 0.0 };
    let y_min = if ink_y_min.is_finite() { ink_y_min } else { 0.0 };
    info.open_type_os2_win_ascent = Some((y_max.ceil() as i32).max(typo_asc) as u32);
    info.open_type_os2_win_descent =
        Some(((-y_min).ceil() as i32).max(typo_desc.abs()) as u32);

    // fsSelection bit 7 = USE_TYPO_METRICS. Bits 0/5/6 come from the
    // style map at compile time.
    info.open_type_os2_selection = Some(vec![7]);
    // fsType = 0 (installable embedding).
    info.open_type_os2_type = Some(vec![]);
    info.open_type_os2_weight_class = Some(400);
    info.open_type_os2_width_class = Some(img2bez::norad::fontinfo::Os2WidthClass::Normal);
    info.open_type_head_flags = Some(vec![0, 3]);

    // --- Name table (checklist section 2) ---
    info.copyright = Some(config.copyright.clone());
    info.open_type_name_designer = Some(config.designer.clone());
    info.open_type_name_version = Some("Version 1.000".to_string());
    info.version_major = Some(1);
    info.version_minor = Some(0);

    // Name ID 13/14, exact strings from the GF guide.
    info.open_type_name_license = Some(
        "This Font Software is licensed under the SIL Open Font License, \
         Version 1.1. This license is available with a FAQ at: \
         https://openfontlicense.org"
            .to_string(),
    );
    info.open_type_name_license_url = Some("https://openfontlicense.org".to_string());

    info.open_type_os2_vendor_id = Some("FTGD".to_string());
}

/// GF Latin Core coverage of this build.
pub fn coverage_summary(encoded: &HashSet<u32>) -> String {
    let (covered, total, missing) = gf_latin_core::coverage(encoded);
    if missing.is_empty() {
        format!("GF Latin Core coverage: {covered}/{total} (complete)")
    } else {
        let shown: Vec<&str> = missing.iter().take(12).copied().collect();
        let more = missing.len().saturating_sub(shown.len());
        let tail = if more > 0 {
            format!(" (+{more} more)")
        } else {
            String::new()
        };
        format!(
            "GF Latin Core coverage: {covered}/{total}; missing: {}{tail}",
            shown.join(" ")
        )
    }
}
