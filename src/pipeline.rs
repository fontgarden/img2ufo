//! Pipeline orchestration: segment (img2glyph) -> label -> trace/assemble
//! (img2bez + norad) -> optional repo emit -> compile + fontspector gate.

use crate::manifest::{Labels, Manifest};
use crate::{qa, ufo_builder};
use anyhow::{bail, Context, Result};
use img2bez::Profile;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct PipelineConfig {
    pub input: PathBuf,
    /// Output UFO path.
    pub output: Option<PathBuf>,
    /// Path of the completion worklist. Default: next to the UFO.
    pub worklist: Option<PathBuf>,
    /// If Some, use this directory for glyph PNGs and manifest.json.
    /// If None, use a temporary directory that is cleaned up on exit.
    pub glyph_dir: Option<PathBuf>,
    /// External labels file: glyph id -> unicode.
    pub labels: Option<PathBuf>,
    pub family: String,
    pub style: String,
    pub designer: String,
    /// Upstream repo URL for the copyright string (GF format).
    pub git_url: Option<String>,
    pub upm: u32,
    pub ascender: i32,
    pub descender: i32,
    pub x_height: i32,
    pub cap_height: i32,
    pub profile: Profile,
    /// Curve-fit accuracy override (None = the profile default).
    pub accuracy: Option<f64>,
    pub grid: i32,
    pub min_area: u32,
    pub max_area: u32,
    /// Skip the compile + fontspector gate.
    pub no_qa: bool,
    pub verbose: bool,

    // Derived at startup:
    /// GF copyright string: `Copyright {year} The {family} Project Authors ({git_url})`.
    pub copyright: String,
}

impl PipelineConfig {
    /// The effective upstream URL (explicit flag or derived default).
    pub fn effective_git_url(&self) -> String {
        self.git_url.clone().unwrap_or_else(|| {
            format!(
                "https://github.com/eliheuer/{}",
                self.family.to_lowercase().replace(' ', "-")
            )
        })
    }
}

/// Byte-exact GF copyright string.
pub fn gf_copyright(year: i32, family: &str, git_url: &str) -> String {
    format!("Copyright {year} The {family} Project Authors ({git_url})")
}

/// Current UTC year from the system clock (civil-from-days, Hinnant).
pub fn current_year() -> i32 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (y + if m <= 2 { 1 } else { 0 }) as i32
}

pub fn run(config: PipelineConfig) -> Result<()> {
    // ------------------------------------------------------------------
    // Resolve paths.
    // ------------------------------------------------------------------
    let ps_family = config.family.replace(' ', "");
    let ps_style = config.style.replace(' ', "");
    let font_stem = format!("{ps_family}-{ps_style}");

    let ufo_path: PathBuf = match &config.output {
        Some(out) => out.clone(),
        // Default: the UFO lands next to the input image.
        None => config
            .input
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{font_stem}.ufo")),
    };
    if let Some(parent) = ufo_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // ------------------------------------------------------------------
    // Stage 1: segment (skipped when the glyph dir already has a manifest).
    // ------------------------------------------------------------------
    let _temp_dir_holder;
    let glyph_dir: PathBuf = match &config.glyph_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir)?;
            dir.clone()
        }
        // Auto-detect: a manifest.json beside the image means the image
        // lives in an img2glyph output directory — reuse it.
        None if config
            .input
            .parent()
            .is_some_and(|p| p.join("manifest.json").exists()) =>
        {
            let dir = config.input.parent().unwrap().to_path_buf();
            if config.verbose {
                eprintln!("pipeline: using {dir:?} as glyph dir (manifest.json found)");
            }
            dir
        }
        None => {
            _temp_dir_holder = tempfile::tempdir()?;
            _temp_dir_holder.path().to_path_buf()
        }
    };
    let manifest_path = glyph_dir.join("manifest.json");
    if manifest_path.exists() {
        if config.verbose {
            eprintln!("pipeline: manifest found at {manifest_path:?}, skipping segmentation");
        }
    } else {
        step_segment(
            &config.input,
            &glyph_dir,
            config.min_area,
            config.max_area,
            config.verbose,
        )?;
    }

    // ------------------------------------------------------------------
    // Stage 2: labels.
    // ------------------------------------------------------------------
    let mut manifest = Manifest::load(&manifest_path)
        .with_context(|| format!("Failed to load manifest at {manifest_path:?}"))?;
    // Auto-detect labels.json in the glyph dir when --labels is not given.
    let labels_path = config
        .labels
        .clone()
        .or_else(|| Some(glyph_dir.join("labels.json")).filter(|p| p.exists()));
    if let Some(labels_path) = &labels_path {
        if config.verbose {
            eprintln!("pipeline: labels from {labels_path:?}");
        }
        let labels = Labels::load(labels_path)
            .with_context(|| format!("Failed to load labels at {labels_path:?}"))?;
        manifest.apply_labels(&labels);
    } else {
        manifest.derive_names();
    }
    if config.verbose {
        eprintln!("pipeline: {} glyphs in manifest", manifest.glyphs.len());
    }

    // ------------------------------------------------------------------
    // Stage 3+4: trace + assemble the UFO.
    // ------------------------------------------------------------------
    let report = ufo_builder::build(&config, &manifest, &glyph_dir, &ufo_path)?;

    println!("img2ufo: wrote {}", ufo_path.display());
    println!(
        "img2ufo: {} traced, {} kept (green), {} unlabeled (no cmap entry), {} failed",
        report.traced,
        report.green_kept,
        report.unlabeled,
        report.failed.len()
    );
    let comp = &report.completion;
    println!(
        "img2ufo: GF Latin Core coverage (traced): {}/{}",
        comp.coverage_traced.covered, comp.coverage_traced.total
    );
    println!(
        "img2ufo: composition: +{} composites, +{} derived ({} marks still missing, {} atomic glyphs need drawing)",
        comp.built_composites.len(),
        comp.derived_glyphs.len(),
        comp.missing_marks.len(),
        comp.missing_atomic.len()
    );
    for m in comp.missing_marks.iter().take(5) {
        println!(
            "  missing mark {} would unlock {} glyphs",
            m.name,
            m.unlocks.len()
        );
    }
    println!("img2ufo: {}", ufo_builder::coverage_summary(&report.encoded));
    println!(
        "img2ufo: completion worklist: {}",
        report.worklist_path.display()
    );

    // ------------------------------------------------------------------
    // Stage 5: compile + fontspector gate.
    // ------------------------------------------------------------------
    if config.no_qa {
        if config.verbose {
            eprintln!("pipeline: QA gate skipped (--no-qa)");
        }
        return Ok(());
    }

    let dir = ufo_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let ttf_path = dir.join(format!("{font_stem}.ttf"));
    let qa_report_path = dir.join(format!("{font_stem}-fontspector.json"));

    qa::compile(&ufo_path, &ttf_path, config.verbose)?;
    println!("img2ufo: compiled {}", ttf_path.display());

    let excludes: &[&str] = &[];
    let summary = qa::run_fontspector(&ttf_path, &qa_report_path, excludes, config.verbose)?;
    println!(
        "img2ufo: fontspector (googlefonts): {} FAIL, {} WARN, {} ERROR — report: {}",
        summary.fails,
        summary.warns,
        summary.errors,
        qa_report_path.display()
    );
    for (check, count) in summary.failed_checks.iter().take(10) {
        println!("  FAIL {check} (x{count})");
    }
    if summary.passed() {
        println!("img2ufo: QA gate PASS");
        Ok(())
    } else {
        bail!(
            "QA gate FAIL: {} FAILing fontspector checks (see {})",
            summary.failed_checks.len(),
            qa_report_path.display()
        );
    }
}

/// Run `img2glyph process <input> --output <glyph_dir>` as a subprocess.
fn step_segment(
    input: &Path,
    glyph_dir: &Path,
    min_area: u32,
    max_area: u32,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("pipeline: segmenting {input:?} -> {glyph_dir:?}");
    }
    let status = Command::new("img2glyph")
        .arg("process")
        .arg(input)
        .arg("--output")
        .arg(glyph_dir)
        .arg("--min-area")
        .arg(min_area.to_string())
        .arg("--max-area")
        .arg(max_area.to_string())
        .status()
        .with_context(|| {
            "Failed to run `img2glyph` — is it installed and on PATH?\n\
             Install it with: cargo install --git https://github.com/fontgarden/img2glyph"
        })?;
    if !status.success() {
        bail!("`img2glyph process` exited with status {status}");
    }
    Ok(())
}
