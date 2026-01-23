#!/usr/bin/env python3
"""
Generate a static HTML site for browsing ISMIP6 file coverage.

The site includes:
- Index page: Models vs Experiments with file counts
- Model pages: Experiments vs Variables for each model
- Experiment pages: Models vs Variables for each experiment
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import yaml
from ismip6_helper import get_file_index


# Color thresholds for file counts
THRESHOLDS = {
    'zero': {'min': 0, 'max': 0, 'color': '#cccccc', 'label': 'No files'},
    'low': {'min': 1, 'max': 27, 'color': '#ffeb3b', 'label': '1-27 files'},
    'high': {'min': 28, 'max': float('inf'), 'color': '#4caf50', 'label': '28+ files'}
}


def gs_to_https(gs_url: str) -> str:
    """Convert gs:// URL to public HTTPS URL."""
    if gs_url.startswith('gs://'):
        # Remove gs:// prefix and convert to https://storage.googleapis.com/
        path = gs_url[5:]  # Remove 'gs://'
        return f'https://storage.googleapis.com/{path}'
    return gs_url


def load_variable_metadata() -> Dict:
    """Load variable metadata from ismip_metadata/variables.yaml."""
    yaml_path = Path(__file__).parent / 'ismip_metadata' / 'variables.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    return {'variables': {}}


def sort_variables_by_type(variables: list, var_metadata: Dict) -> list:
    """Sort variables with 2D variables first, then scalar variables."""
    var_info = var_metadata.get('variables', {})

    # Categorize variables
    var_2d = []
    var_scalar = []
    var_unknown = []

    for var in variables:
        if var in var_info:
            var_type = var_info[var].get('variable_type', 'unknown')
            if var_type == '2D':
                var_2d.append(var)
            elif var_type == 'scalar':
                var_scalar.append(var)
            else:
                var_unknown.append(var)
        else:
            var_unknown.append(var)

    # Return sorted: 2D first, then scalars, then unknowns (all alphabetically within groups)
    return sorted(var_2d) + sorted(var_scalar) + sorted(var_unknown)


def get_color_for_count(count: int) -> str:
    """Return the background color for a given file count."""
    for threshold in THRESHOLDS.values():
        if threshold['min'] <= count <= threshold['max']:
            return threshold['color']
    return '#ffffff'


def get_html_header(title: str, breadcrumb: str = '') -> str:
    """Generate HTML header with CSS styling."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - ISMIP6 Coverage</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 100%;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #1976d2;
            padding-bottom: 10px;
        }}
        .nav {{
            margin-bottom: 20px;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .nav a {{
            display: inline-block;
            padding: 8px 16px;
            background-color: #1976d2;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-right: 10px;
            font-size: 14px;
        }}
        .nav a:hover {{
            background-color: #1565c0;
            text-decoration: none;
        }}
        .nav a.active {{
            background-color: #0d47a1;
        }}
        .breadcrumb {{
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .breadcrumb a {{
            color: #1976d2;
            text-decoration: none;
        }}
        .breadcrumb a:hover {{
            text-decoration: underline;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            font-size: 13px;
        }}
        th {{
            background-color: #1976d2;
            color: white;
            padding: 12px 8px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        th.row-header {{
            background-color: #1976d2;
            position: sticky;
            left: 0;
            z-index: 11;
        }}
        th.rotated {{
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            white-space: nowrap;
            padding: 8px 4px;
            min-width: 30px;
            max-width: 30px;
        }}
        th.rotated a {{
            color: white;
        }}
        td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        td.row-header {{
            background-color: #e3f2fd;
            font-weight: bold;
            text-align: left;
            position: sticky;
            left: 0;
            z-index: 5;
        }}
        tr:hover td {{
            opacity: 0.8;
        }}
        a {{
            color: #1976d2;
            text-decoration: none;
            font-weight: 500;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        a.file-link {{
            display: block;
            width: 100%;
            height: 100%;
            color: inherit;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 5px;
        }}
        .legend-box {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 1px solid #999;
            margin-right: 5px;
            vertical-align: middle;
        }}
        .stats {{
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f5e9;
            border-radius: 4px;
            border-left: 4px solid #4caf50;
        }}
        .stats p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="index.html">Coverage Index</a>
            <a href="viewer/index.html">Interactive Viewer</a>
        </div>
        {breadcrumb}
        <h1>{title}</h1>
"""


def get_html_footer() -> str:
    """Generate HTML footer."""
    return """
    </div>
</body>
</html>
"""


def generate_index_page(df: pd.DataFrame, output_dir: Path):
    """Generate the main index page: Models vs Experiments."""
    print("Generating index page...")

    # Create model identifier
    df['model'] = df['institution'] + '/' + df['model_name']

    # Load variable metadata to classify variables
    var_metadata = load_variable_metadata()
    var_info = var_metadata.get('variables', {})

    # Get list of all 2D variables from the schema
    all_2d_vars = {var for var, info in var_info.items() if info.get('variable_type') == '2D'}

    # Classify each file's variable as 2D or scalar
    def classify_variable(var):
        if var in var_info:
            return var_info[var].get('variable_type', 'unknown')
        return 'unknown'

    df['var_type'] = df['variable'].apply(classify_variable)

    # Get all unique model-experiment combinations
    model_exp_combos = df[['model', 'experiment']].drop_duplicates()

    # For each model-experiment combo, count 2D and scalar files, and check which variables exist
    results = []
    for _, row in model_exp_combos.iterrows():
        model = row['model']
        exp = row['experiment']
        subset = df[(df['model'] == model) & (df['experiment'] == exp)]

        count_2d = len(subset[subset['var_type'] == '2D'])
        count_scalar = len(subset[subset['var_type'] == 'scalar'])

        # Get unique 2D variables present
        vars_2d_present = set(subset[subset['var_type'] == '2D']['variable'].unique())
        has_all_2d = all_2d_vars.issubset(vars_2d_present)

        results.append({
            'model': model,
            'experiment': exp,
            'count_2d': count_2d,
            'count_scalar': count_scalar,
            'has_all_2d': has_all_2d
        })

    results_df = pd.DataFrame(results)

    # Sort rows and columns
    all_models = sorted(df['model'].unique())
    all_experiments = sorted(df['experiment'].unique())

    # Generate HTML
    html = get_html_header("ISMIP6 File Coverage Overview")

    # Add statistics
    total_files = len(df)
    total_models = df['model'].nunique()
    total_experiments = df['experiment'].nunique()
    total_variables = df['variable'].nunique()

    html += f"""
        <div class="stats">
            <p><strong>Total Files:</strong> {total_files:,}</p>
            <p><strong>Models:</strong> {total_models}</p>
            <p><strong>Experiments:</strong> {total_experiments}</p>
            <p><strong>Variables:</strong> {total_variables}</p>
        </div>
    """

    # Add legend
    html += """
        <div class="legend">
            <strong>Legend:</strong><br>
            <div class="legend-item">
                <span class="legend-box" style="background-color: #cccccc;"></span>
                No files
            </div>
            <div class="legend-item">
                <span class="legend-box" style="background-color: #ffeb3b;"></span>
                Incomplete 2D coverage
            </div>
            <div class="legend-item">
                <span class="legend-box" style="background-color: #4caf50;"></span>
                All 2D variables present
            </div>
        </div>
    """

    # Create table
    html += """
        <table>
            <thead>
                <tr>
                    <th class="row-header">Model</th>
    """

    # Column headers (experiments) - rotated for readability
    for exp in all_experiments:
        html += f'<th class="rotated"><a href="experiments/{exp}.html">{exp}</a></th>\n'

    html += """
                </tr>
            </thead>
            <tbody>
    """

    # Table rows
    for model in all_models:
        html += '<tr>\n'
        # Row header (model)
        model_slug = model.replace('/', '_')
        html += f'<td class="row-header"><a href="models/{model_slug}.html">{model}</a></td>\n'

        # Data cells
        for exp in all_experiments:
            # Find the data for this model-experiment combo
            row_data = results_df[(results_df['model'] == model) & (results_df['experiment'] == exp)]

            if len(row_data) > 0:
                count_2d = row_data.iloc[0]['count_2d']
                count_scalar = row_data.iloc[0]['count_scalar']
                has_all_2d = row_data.iloc[0]['has_all_2d']

                # Determine color based on 2D coverage
                if count_2d == 0 and count_scalar == 0:
                    color = '#cccccc'  # Gray - no files
                    display_text = ''
                elif has_all_2d:
                    color = '#4caf50'  # Green - all 2D variables present
                    display_text = f'{count_2d} / {count_scalar}'
                else:
                    color = '#ffeb3b'  # Yellow - incomplete 2D coverage
                    display_text = f'{count_2d} / {count_scalar}'
            else:
                color = '#cccccc'
                display_text = ''

            html += f'<td style="background-color: {color};">{display_text}</td>\n'

        html += '</tr>\n'

    html += """
            </tbody>
        </table>
    """

    html += get_html_footer()

    # Write to file
    output_file = output_dir / 'index.html'
    output_file.write_text(html)
    print(f"  Created: {output_file}")


def generate_model_pages(df: pd.DataFrame, output_dir: Path, all_experiments: list, all_variables: list):
    """Generate detail pages for each model: Experiments vs Variables."""
    print("Generating model pages...")

    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    df['model'] = df['institution'] + '/' + df['model_name']

    # Load variable metadata
    var_metadata = load_variable_metadata()
    var_info = var_metadata.get('variables', {})

    # Sort variables with 2D first, then scalars
    all_variables = sort_variables_by_type(all_variables, var_metadata)

    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]

        # Create a lookup dict for URLs (convert gs:// to https://)
        url_lookup = {}
        for _, row in model_df.iterrows():
            key = (row['experiment'], row['variable'])
            url_lookup[key] = gs_to_https(row['url'])

        # Create pivot table: experiments vs variables (binary presence)
        pivot = model_df.groupby(['experiment', 'variable']).size().reset_index(name='count')
        pivot_table = pivot.pivot(index='experiment', columns='variable', values='count').fillna(0).astype(int)
        pivot_table = (pivot_table > 0).astype(int)  # Convert to binary

        # Ensure all experiments and variables are present
        pivot_table = pivot_table.reindex(index=all_experiments, fill_value=0)
        pivot_table = pivot_table.reindex(columns=all_variables, fill_value=0)

        # Generate HTML
        breadcrumb = '<div class="breadcrumb"><a href="../index.html">Home</a> &gt; ' + model + '</div>'
        html = get_html_header(f"Model: {model}", breadcrumb)

        # Stats
        total_files = len(model_df)
        total_experiments = model_df['experiment'].nunique()
        total_variables = model_df['variable'].nunique()

        html += f"""
            <div class="stats">
                <p><strong>Total Files:</strong> {total_files}</p>
                <p><strong>Experiments:</strong> {total_experiments}</p>
                <p><strong>Variables:</strong> {total_variables}</p>
            </div>
        """

        # Create table
        html += """
            <table>
                <thead>
                    <tr>
                        <th class="row-header">Experiment</th>
        """

        # Column headers (variables) - rotated with hover text and links
        for var in pivot_table.columns:
            # Get description for hover text
            description = ""
            if var in var_info:
                description = var_info[var].get('description', '')
            title_attr = f' title="{description}"' if description else ''
            html += f'<th class="rotated"{title_attr}><a href="../variables/{var}.html">{var}</a></th>\n'

        html += """
                    </tr>
                </thead>
                <tbody>
        """

        # Table rows
        for exp in pivot_table.index:
            html += '<tr>\n'
            # Row header (experiment)
            html += f'<td class="row-header"><a href="../experiments/{exp}.html">{exp}</a></td>\n'

            # Data cells (green if exists, gray if not)
            for var in pivot_table.columns:
                exists = pivot_table.loc[exp, var]
                color = '#4caf50' if exists else '#cccccc'
                symbol = '✓' if exists else ''

                # Add link to file if it exists
                if exists and (exp, var) in url_lookup:
                    url = url_lookup[(exp, var)]
                    html += f'<td style="background-color: {color};"><a href="{url}" target="_blank" class="file-link">{symbol}</a></td>\n'
                else:
                    html += f'<td style="background-color: {color};">{symbol}</td>\n'

            html += '</tr>\n'

        html += """
                </tbody>
            </table>
        """

        html += get_html_footer()

        # Write to file
        model_slug = model.replace('/', '_')
        output_file = models_dir / f'{model_slug}.html'
        output_file.write_text(html)

    print(f"  Created {len(df['model'].unique())} model pages")


def generate_experiment_pages(df: pd.DataFrame, output_dir: Path, all_models: list, all_variables: list):
    """Generate detail pages for each experiment: Models vs Variables."""
    print("Generating experiment pages...")

    experiments_dir = output_dir / 'experiments'
    experiments_dir.mkdir(exist_ok=True)

    df['model'] = df['institution'] + '/' + df['model_name']

    # Load variable metadata
    var_metadata = load_variable_metadata()
    var_info = var_metadata.get('variables', {})

    # Sort variables with 2D first, then scalars
    all_variables = sort_variables_by_type(all_variables, var_metadata)

    for exp in sorted(df['experiment'].unique()):
        exp_df = df[df['experiment'] == exp]

        # Create a lookup dict for URLs (convert gs:// to https://)
        url_lookup = {}
        for _, row in exp_df.iterrows():
            key = (row['model'], row['variable'])
            url_lookup[key] = gs_to_https(row['url'])

        # Create pivot table: models vs variables (binary presence)
        pivot = exp_df.groupby(['model', 'variable']).size().reset_index(name='count')
        pivot_table = pivot.pivot(index='model', columns='variable', values='count').fillna(0).astype(int)
        pivot_table = (pivot_table > 0).astype(int)  # Convert to binary

        # Ensure all models and variables are present
        pivot_table = pivot_table.reindex(index=all_models, fill_value=0)
        pivot_table = pivot_table.reindex(columns=all_variables, fill_value=0)

        # Generate HTML
        breadcrumb = f'<div class="breadcrumb"><a href="../index.html">Home</a> &gt; {exp}</div>'
        html = get_html_header(f"Experiment: {exp}", breadcrumb)

        # Stats
        total_files = len(exp_df)
        total_models = exp_df['model'].nunique()
        total_variables = exp_df['variable'].nunique()

        html += f"""
            <div class="stats">
                <p><strong>Total Files:</strong> {total_files}</p>
                <p><strong>Models:</strong> {total_models}</p>
                <p><strong>Variables:</strong> {total_variables}</p>
            </div>
        """

        # Create table
        html += """
            <table>
                <thead>
                    <tr>
                        <th class="row-header">Model</th>
        """

        # Column headers (variables) - rotated with hover text and links
        for var in pivot_table.columns:
            # Get description for hover text
            description = ""
            if var in var_info:
                description = var_info[var].get('description', '')
            title_attr = f' title="{description}"' if description else ''
            html += f'<th class="rotated"{title_attr}><a href="../variables/{var}.html">{var}</a></th>\n'

        html += """
                    </tr>
                </thead>
                <tbody>
        """

        # Table rows
        for model in pivot_table.index:
            html += '<tr>\n'
            # Row header (model)
            model_slug = model.replace('/', '_')
            html += f'<td class="row-header"><a href="../models/{model_slug}.html">{model}</a></td>\n'

            # Data cells (green if exists, gray if not)
            for var in pivot_table.columns:
                exists = pivot_table.loc[model, var]
                color = '#4caf50' if exists else '#cccccc'
                symbol = '✓' if exists else ''

                # Add link to file if it exists
                if exists and (model, var) in url_lookup:
                    url = url_lookup[(model, var)]
                    html += f'<td style="background-color: {color};"><a href="{url}" target="_blank" class="file-link">{symbol}</a></td>\n'
                else:
                    html += f'<td style="background-color: {color};">{symbol}</td>\n'

            html += '</tr>\n'

        html += """
                </tbody>
            </table>
        """

        html += get_html_footer()

        # Write to file
        output_file = experiments_dir / f'{exp}.html'
        output_file.write_text(html)

    print(f"  Created {len(df['experiment'].unique())} experiment pages")


def generate_variable_pages(df: pd.DataFrame, output_dir: Path, all_models: list, all_experiments: list):
    """Generate detail pages for each variable: Models vs Experiments."""
    print("Generating variable pages...")

    variables_dir = output_dir / 'variables'
    variables_dir.mkdir(exist_ok=True)

    df['model'] = df['institution'] + '/' + df['model_name']

    # Load variable metadata
    var_metadata = load_variable_metadata()
    var_info = var_metadata.get('variables', {})

    for var in sorted(df['variable'].unique()):
        var_df = df[df['variable'] == var]

        # Create a lookup dict for URLs (convert gs:// to https://)
        url_lookup = {}
        for _, row in var_df.iterrows():
            key = (row['model'], row['experiment'])
            url_lookup[key] = gs_to_https(row['url'])

        # Create pivot table: models vs experiments (binary presence)
        pivot = var_df.groupby(['model', 'experiment']).size().reset_index(name='count')
        pivot_table = pivot.pivot(index='model', columns='experiment', values='count').fillna(0).astype(int)
        pivot_table = (pivot_table > 0).astype(int)  # Convert to binary

        # Ensure all models and experiments are present
        pivot_table = pivot_table.reindex(index=all_models, fill_value=0)
        pivot_table = pivot_table.reindex(columns=all_experiments, fill_value=0)

        # Generate HTML
        breadcrumb = f'<div class="breadcrumb"><a href="../index.html">Home</a> &gt; {var}</div>'
        html = get_html_header(f"Variable: {var}", breadcrumb)

        # Display variable metadata from YAML
        if var in var_info:
            metadata = var_info[var]
            html += """
                <div class="stats">
                    <h2 style="margin-top: 0;">Variable Metadata</h2>
            """

            # Display all metadata fields
            if 'description' in metadata:
                html += f'<p><strong>Description:</strong> {metadata["description"]}</p>\n'
            if 'variable_type' in metadata:
                html += f'<p><strong>Type:</strong> {metadata["variable_type"]}</p>\n'
            if 'dimensions' in metadata:
                html += f'<p><strong>Dimensions:</strong> {metadata["dimensions"]}</p>\n'
            if 'temporal_type' in metadata:
                temp_type = metadata["temporal_type"]
                temp_desc = "Snapshot (end of year)" if temp_type == "ST" else "Flux (yearly average)"
                html += f'<p><strong>Temporal Type:</strong> {temp_type} ({temp_desc})</p>\n'
            if 'standard_name' in metadata:
                html += f'<p><strong>Standard Name:</strong> {metadata["standard_name"]}</p>\n'
            if 'standard_name_alias' in metadata:
                html += f'<p><strong>Standard Name Alias:</strong> {metadata["standard_name_alias"]}</p>\n'
            if 'alias' in metadata:
                html += f'<p><strong>Alias:</strong> {metadata["alias"]}</p>\n'
            if 'units' in metadata:
                html += f'<p><strong>Units:</strong> {metadata["units"]}</p>\n'
            if 'comment' in metadata:
                html += f'<p><strong>Comment:</strong> {metadata["comment"]}</p>\n'

            html += """
                </div>
            """

        # Stats
        total_files = len(var_df)
        total_models = var_df['model'].nunique()
        total_experiments = var_df['experiment'].nunique()

        html += f"""
            <div class="stats">
                <p><strong>Total Files:</strong> {total_files}</p>
                <p><strong>Models with this variable:</strong> {total_models}</p>
                <p><strong>Experiments with this variable:</strong> {total_experiments}</p>
            </div>
        """

        # Create table
        html += """
            <table>
                <thead>
                    <tr>
                        <th class="row-header">Model</th>
        """

        # Column headers (experiments) - rotated
        for exp in pivot_table.columns:
            html += f'<th class="rotated"><a href="../experiments/{exp}.html">{exp}</a></th>\n'

        html += """
                    </tr>
                </thead>
                <tbody>
        """

        # Table rows
        for model in pivot_table.index:
            html += '<tr>\n'
            # Row header (model)
            model_slug = model.replace('/', '_')
            html += f'<td class="row-header"><a href="../models/{model_slug}.html">{model}</a></td>\n'

            # Data cells (green if exists, gray if not)
            for exp in pivot_table.columns:
                exists = pivot_table.loc[model, exp]
                color = '#4caf50' if exists else '#cccccc'
                symbol = '✓' if exists else ''

                # Add link to file if it exists
                if exists and (model, exp) in url_lookup:
                    url = url_lookup[(model, exp)]
                    html += f'<td style="background-color: {color};"><a href="{url}" target="_blank" class="file-link">{symbol}</a></td>\n'
                else:
                    html += f'<td style="background-color: {color};">{symbol}</td>\n'

            html += '</tr>\n'

        html += """
                </tbody>
            </table>
        """

        html += get_html_footer()

        # Write to file
        output_file = variables_dir / f'{var}.html'
        output_file.write_text(html)

    print(f"  Created {len(df['variable'].unique())} variable pages")


def generate_site(output_dir: str = 'site'):
    """Generate the complete static HTML site."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Generating HTML site in: {output_path.absolute()}")

    # Load data
    df = get_file_index()

    # Create model identifier
    df['model'] = df['institution'] + '/' + df['model_name']

    # Get sorted lists of all unique values
    all_experiments = sorted(df['experiment'].unique())
    all_variables = sorted(df['variable'].unique())
    all_models = sorted(df['model'].unique())

    print(f"  Total experiments: {len(all_experiments)}")
    print(f"  Total variables: {len(all_variables)}")
    print(f"  Total models: {len(all_models)}")

    # Generate pages
    generate_index_page(df, output_path)
    generate_model_pages(df, output_path, all_experiments, all_variables)
    generate_experiment_pages(df, output_path, all_models, all_variables)
    generate_variable_pages(df, output_path, all_models, all_experiments)

    print(f"\nSite generation complete!")
    print(f"Open {output_path.absolute() / 'index.html'} in a browser")


if __name__ == "__main__":
    generate_site()
