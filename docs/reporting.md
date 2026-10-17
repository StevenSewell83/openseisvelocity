# Project report

`osv report -c config.yaml` (also the last step of `osv run`) aggregates every stage directory
into one page. It works at any time and shows whatever stages exist; missing stages appear as
"not run".

## Outputs

| File | Content |
|---|---|
| `<workdir>/10_report/index.html` | The project report (links relative to `10_report/`). |
| `<workdir>/index.html` | Same page with links relative to the work directory; open this one. |
| `<workdir>/10_report/summary.md` | Compact Markdown summary to paste into notes. |

The page contains: project header (name, config hash, osv and tool versions, run dates), the
pipeline status table (status, run time, last run, fail/warn counts, link to each stage report),
all QC failures and warnings grouped by stage, headline numbers per stage (events and picks,
Vp/Vs, 1-D model table and RMS, grid size, chosen damping and smoothing, variance reduction,
resolved fraction, validation 1-D vs 3-D RMS), one or two key figures per stage, links to
exportable result files, and a collapsible config snapshot.

The report stage is re-run automatically whenever any stage manifest changes (the manifests are
its hashed inputs), so it needs no `--force`.

## Configuration

```yaml
report:
  embed_images: false   # true: embed figures as base64 for a single portable index.html
```

## Customising

`osv.report.project.HEADLINES` (summary-key glob patterns per stage) and `KEY_FIGURES`
(figure-name glob patterns per stage; first figure is the fallback) control what is shown.
