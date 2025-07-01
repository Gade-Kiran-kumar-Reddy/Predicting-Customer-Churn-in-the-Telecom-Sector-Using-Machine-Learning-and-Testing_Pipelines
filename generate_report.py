# import os

# FIG_DIR  = 'reports/figures'
# OUT_PATH = 'reports/figures_report.html'

# # 1) Define captions for each file (use HTML formatting)
# captions = {
#     'churn_distribution.png':        '<strong>Figure 1.</strong> Distribution of customers by churn status, showing ~26% churn rate.',
#     'confusion_matrix.png':          '<strong>Figure 2.</strong> Confusion matrix on hold-out set: True vs. predicted churn labels.',
#     'contract_vs_churn.png':         '<strong>Figure 3.</strong> Churn rate by contract type (Month-to-month, One-year, Two-year).',
#     'feature_correlation_heatmap.png': '<strong>Figure 4.</strong> Heatmap of feature correlations after preprocessing.',
#     'model_comparison.png':          '<strong>Figure 5.</strong> Accuracy comparison across RF, XGB, and Logistic models.',
#     'roc_curve.png':                 '<strong>Figure 6.</strong> ROC curve for the best calibrated model on test data.',
#     # Add more as needed...
# }

# # Make sure the output directory for the HTML exists
# os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
# # Make sure the figures directory exists
# os.makedirs(FIG_DIR, exist_ok=True)

# # Collect all .png filenames
# pngs = sorted(f for f in os.listdir(FIG_DIR) if f.lower().endswith('.png'))

# html_lines = [
#     "<!DOCTYPE html>",
#     "<html>",
#     "<head>",
#     "  <meta charset='utf-8'>",
#     "  <title>Model Figures Report</title>",
#     "  <style>",
#     "    body { font-family: Arial, sans-serif; margin: 40px; }",
#     "    figure { margin-bottom: 40px; }",
#     "    figcaption { font-style: italic; margin-top: 8px; }",
#     "    hr { border: none; border-top: 1px solid #ccc; margin: 40px 0; }",
#     "  </style>",
#     "</head>",
#     "<body>",
#     "  <h1>Model Figures Report</h1>",
# ]

# html_dir = os.path.dirname(OUT_PATH)  # e.g. 'reports'
# for fn in pngs:
#     full_path = os.path.join(FIG_DIR, fn)
#     rel_path  = os.path.relpath(full_path, html_dir)
#     html_lines.append("  <figure>")
#     html_lines.append(f"    <img src=\"{rel_path}\" alt=\"{fn}\" style=\"max-width:900px; width:100%;\" />")
#     caption_html = captions.get(fn, "")
#     if caption_html:
#         html_lines.append(f"    <figcaption>{caption_html}</figcaption>")
#     html_lines.append("  </figure>")
#     html_lines.append("  <hr>")

# html_lines.extend([
#     "</body>",
#     "</html>"
# ])

# # Write out the HTML file
# with open(OUT_PATH, 'w') as f:
#     f.write("\n".join(html_lines))

# print(f"Report written to {OUT_PATH}")



import os
import re

FIG_DIR  = 'reports/figures'
OUT_PATH = 'reports/figures_report.html'

# Captions for each figure
captions = {
    'churn_distribution.png':        'Distribution of customers by churn status, showing ~26% churn rate.',
    'confusion_matrix.png':          'Confusion matrix on hold-out set: True vs. predicted churn labels.',
    'contract_vs_churn.png':         'Churn rate by contract type (Month-to-month, One-year, Two-year).',
    'feature_correlation_heatmap.png':'Heatmap of feature correlations after preprocessing.',
    'model_comparison.png':          'Accuracy comparison across RF, XGB, and Logistic models.',
    'roc_curve.png':                 'ROC curve for the best calibrated model on test data.',
}

# Helper to turn a filename into an HTML-friendly anchor
def make_id(filename):
    return re.sub(r'[^a-z0-9]+', '-', filename.lower().replace('.png','')).strip('-')

# Ensure directories exist
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Collect PNG filenames
pngs = sorted(f for f in os.listdir(FIG_DIR) if f.lower().endswith('.png'))

html = [
    "<!DOCTYPE html>",
    "<html lang='en'>",
    "<head>",
    "  <meta charset='utf-8'>",
    "  <meta name='viewport' content='width=device-width,initial-scale=1'>",
    "  <title>Model Figures Report</title>",
    "  <style>",
    "    body { font-family: 'Segoe UI', Tahoma, sans-serif; color: #333; margin: 0; padding: 0; background: #f9f9f9; }",
    "    .container { max-width: 960px; margin: auto; padding: 20px; background: #fff; }",
    "    h1 { text-align: center; margin-bottom: 1rem; background: #0366d6; color: #fff; padding: 10px; }",
    "    nav ul { list-style: none; padding: 0; text-align: center; margin-bottom: 20px; background: #eaecef; padding: 10px; border-radius: 5px; }",
    "    nav li { display: inline-block; margin: 0 10px; }",
    "    nav a { text-decoration: none; color: #0366d6; font-weight: bold; }",
    "    .figure-grid { display: grid; grid-template-columns: 1fr; gap: 40px; justify-items: center; }",
    "    @media(min-width:600px) { .figure-grid { grid-template-columns: 1fr 1fr; } }",
    "    figure { margin: 0; text-align: center; }",
    "    figure img { width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; }",
    "    figcaption { margin-top: 8px; font-style: italic; color: #555; font-size: 0.9rem; }",
    "    hr { border: none; border-top: 1px solid #eaeaea; margin: 40px 0; }",
    "  </style>",
    "</head>",
    "<body>",
    "  <div class='container'>",
    "    <h1>Model Figures Report</h1>",
    "    <nav><ul>"
]

# Table of contents
for fn in pngs:
    anchor = make_id(fn)
    title = fn.replace('_', ' ').replace('.png','').title()
    html.append(f"      <li><a href='#{anchor}'>{title}</a></li>")

html.append("    </ul></nav>")
html.append("    <div class='figure-grid'>")

# Figures
for fn in pngs:
    anchor = make_id(fn)
    full = os.path.join(FIG_DIR, fn)
    rel = os.path.relpath(full, os.path.dirname(OUT_PATH))
    caption = captions.get(fn, "")
    html.append(f"      <figure id='{anchor}'>")
    html.append(f"        <img src='{rel}' alt='{fn}'>")
    if caption:
        html.append(f"        <figcaption><strong>{caption}</strong></figcaption>")
    html.append("      </figure>")
    html.append("      <hr>")

html.extend([
    "    </div>",
    "  </div>",
    "</body>",
    "</html>"
])

with open(OUT_PATH, 'w') as f:
    f.write("\n".join(html))

print(f"Report written to {OUT_PATH}")
