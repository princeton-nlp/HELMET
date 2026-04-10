import os
import re
import json
import glob
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm
from argparse import ArgumentParser

dataset_to_metrics = {
    "json_kv": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",

    "narrativeqa": ["gpt-4-score"],
    "msmarco_rerank_psg": "NDCG@10",

    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",

    "qmsum": "rougeL_recall",
    "multi_lexsum": ["gpt-4-f1"],

    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",

    "infbench_qa": ["rougeL_f1"],
    "infbench_choice": ["exact_match"],
    "infbench_sum": ["gpt-4-f1"],

    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],

    "longbenchv2": "exact_match",
    "mrcr_4": "score",
    "mrcr_8": "score",
    "graphwalk_bfs": 'f1',
    "graphwalk_parent": 'f1',

    "ppl_longmino": "perplexity",
}

dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}
custom_avgs = {
    "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall", "ruler_niah_mv ruler_recall"],
    "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match', 'triviaqa substring_exact_match',],
    "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match', 'nlu exact_match'],
    "Cite": ['alce_asqa str_em', 'alce_asqa citation_rec', 'alce_asqa citation_prec', 'alce_qampari qampari_rec_top5', 'alce_qampari citation_rec', 'alce_qampari citation_prec', ],
    "Re-rank": ['msmarco_rerank_psg NDCG@10', ],

    "LongQA": ['narrativeqa gpt-4-score', 'infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
    "Summ": ['infbench_sum gpt-4-f1', 'multi_lexsum gpt-4-f1', ],
    "RULER": ['ruler_niah_s_1 ruler_recall', 'ruler_niah_s_2 ruler_recall', 'ruler_niah_s_3 ruler_recall', 'ruler_niah_mk_1 ruler_recall', 'ruler_niah_mk_2 ruler_recall', 'ruler_niah_mk_3 ruler_recall', 'ruler_niah_mq ruler_recall', 'ruler_niah_mv ruler_recall', 'ruler_cwe ruler_recall', 'ruler_fwe ruler_recall', 'ruler_vt ruler_recall', 'ruler_qa_1 substring_exact_match', 'ruler_qa_2 substring_exact_match'],
    "Avg": ['Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],

    "Recall Dev": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mv ruler_recall", "ruler_niah_mq ruler_recall",],
    "Syn+ Dev": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mv ruler_recall", "ruler_niah_mq ruler_recall", "ruler_fwe ruler_recall", "mrcr_4 score", "mrcr_8 score", "graphwalk_bfs f1", "graphwalk_parent f1"],

    "RAG Dev": ['nq substring_exact_match', 'popqa substring_exact_match'],
    "ICL Dev": ['clinic150 exact_match', 'nlu exact_match'],
    "Re-rank Dev": ['msmarco_rerank_psg NDCG@10', ],
    "LBv2 Dev": ['longbenchv2 exact_match'],
    "Dev Avg": ['Recall Dev', 'RAG Dev', 'ICL Dev', 'Re-rank Dev', 'MRCR Dev', "LBv2 Dev"],
}

@dataclass
class arguments:
    tag: str = "v1"
    input_max_length: int = 131072
    generation_max_length: int = 100
    generation_min_length: int = 0
    max_test_samples: int = 100
    shots: int = 2
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_chat_template: bool = False
    seed: int = 42
    test_name: str = ""
    dataset: str = "nq"
    output_dir: str = "output"
    popularity_threshold: float = 3

    category: str = "synthetic"

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_path(self):
        tag = self.tag

        if self.dataset.startswith("ppl"):
            path = os.path.join(self.output_dir, f"{self.dataset}_{self.tag}_{self.seed}.json")
        else:
            path = os.path.join(self.output_dir, "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self, tag=tag))

        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        if "alce" in self.dataset:
            return path.replace(".json", ".json.score")

        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None

    def get_averaged_metric(self):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            return None
        with open(path) as f:
            try:
                results = json.load(f)
            except Exception as e:
                print("exception occurred while trying to load", path, e)
                return None

        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}

        s = {m : v * (100 if m == "gpt-4-f1" else 1) * (100/3 if m == "gpt-4-score" else 1) for m, v in s.items()}
        print("found scores:", s)
        return s

    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            try:
                results = json.load(f)
            except Exception as e:
                print("exception occurred while trying to load", path, e)
                return None

        output = []
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d['ctxs'])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)

        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")

def plot_results_html(lf_df, baseline_model_names, prefix,
                      output_dir="/scratch/gpfs/DANQIC/hyen/visuals/helmet_plots"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    os.makedirs(output_dir, exist_ok=True)

    metric_cols = [c for c in lf_df.columns if c not in ('input_max_length', 'model')]

    baseline_df = lf_df[lf_df['model'].isin(baseline_model_names)].copy()
    checkpoint_df = lf_df[~lf_df['model'].isin(baseline_model_names)].copy()

    # Parse checkpoint number and base model name
    checkpoint_df['checkpoint'] = checkpoint_df['model'].str.extract(r'checkpoint-(\d+)')[0].astype(float)
    checkpoint_df = checkpoint_df.dropna(subset=['checkpoint'])
    checkpoint_df['checkpoint'] = checkpoint_df['checkpoint'].astype(int)
    checkpoint_df['model_name'] = checkpoint_df['model'].str.replace(r'_?checkpoint-\d+', '', regex=True)

    input_lengths = sorted(lf_df['input_max_length'].unique())
    all_model_names = sorted(checkpoint_df['model_name'].unique())

    # Color maps
    color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    model_color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(all_model_names)}

    baseline_colors_list = ['gray', 'black', 'lightgray', 'darkgray', 'silver']
    baseline_color_map = {name: baseline_colors_list[i % len(baseline_colors_list)]
                          for i, name in enumerate(sorted(baseline_model_names))}

    n_cols_plot = 3
    n_rows_plot = (len(metric_cols) + n_cols_plot - 1) // n_cols_plot

    fig = make_subplots(rows=n_rows_plot, cols=n_cols_plot, subplot_titles=metric_cols,
                        horizontal_spacing=0.04, vertical_spacing=0.06)

    trace_length_map = []  # tracks which input_length each trace belongs to

    for length in input_lengths:
        cp_data = checkpoint_df[checkpoint_df['input_max_length'] == length]
        bl_data = baseline_df[baseline_df['input_max_length'] == length]

        # Determine x range for baseline lines
        if not cp_data.empty:
            x_min, x_max = cp_data['checkpoint'].min(), cp_data['checkpoint'].max()
        else:
            x_min, x_max = 0, 1

        shown_legend = set()

        for metric_idx, metric in enumerate(metric_cols):
            row = metric_idx // n_cols_plot + 1
            col = metric_idx % n_cols_plot + 1

            # Checkpoint model lines
            for model_name in all_model_names:
                model_data = cp_data[cp_data['model_name'] == model_name].sort_values('checkpoint')
                if model_data.empty or model_data[metric].isna().all():
                    continue

                show = model_name not in shown_legend
                if show:
                    shown_legend.add(model_name)

                fig.add_trace(go.Scatter(
                    x=model_data['checkpoint'],
                    y=model_data[metric],
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=model_color_map[model_name]),
                    legendgroup=model_name,
                    showlegend=show,
                    visible=(length == input_lengths[0]),
                ), row=row, col=col)
                trace_length_map.append(length)

            # Baseline horizontal lines
            for _, bl_row in bl_data.iterrows():
                bl_model = bl_row['model']
                val = bl_row.get(metric)
                if pd.isna(val):
                    continue

                bl_key = f"bl_{bl_model}"
                show = bl_key not in shown_legend
                if show:
                    shown_legend.add(bl_key)

                fig.add_trace(go.Scatter(
                    x=[x_min, x_max],
                    y=[val, val],
                    mode='lines',
                    name=bl_model,
                    line=dict(color=baseline_color_map[bl_model], dash='dash'),
                    legendgroup=bl_key,
                    showlegend=show,
                    visible=(length == input_lengths[0]),
                ), row=row, col=col)
                trace_length_map.append(length)

    # Dropdown buttons to select input_max_length
    buttons = []
    for length in input_lengths:
        visibility = [l == length for l in trace_length_map]
        buttons.append(dict(
            label=f"input_max_length = {length}",
            method="update",
            args=[{"visible": visibility}],
        ))

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.0,
            y=1.35,
            xanchor='left',
            yanchor='bottom',
            type='dropdown',
        )],
        height=500 * n_rows_plot,
        margin=dict(t=300),
        title_text="",
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            traceorder='grouped',
            itemwidth=30,
        ),
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(title_text="Checkpoint", matches='x')

    # Build HTML tables for each input_length
    table_html_parts = []
    for length in input_lengths:
        length_data = lf_df[lf_df['input_max_length'] == length].copy()

        # Build rows: for checkpoint models, parse model_name and checkpoint
        rows = []
        for _, r in length_data.iterrows():
            model_str = r['model']
            cp_match = re.search(r'checkpoint-(\d+)', model_str)
            if cp_match:
                checkpoint = cp_match.group(1)
                model_name = re.sub(r'_?checkpoint-\d+', '', model_str)
            else:
                checkpoint = '-'
                model_name = model_str
            metric_vals = {col: r[col] for col in metric_cols}
            rows.append({'Model': model_name, 'Checkpoint': checkpoint, **metric_vals})

        if not rows:
            table_html_parts.append(f'<div class="result-table" data-length="{length}" style="display:none;"></div>')
            continue

        table_df = pd.DataFrame(rows)
        # Sort: checkpoint models by name then checkpoint number, baselines at top
        table_df['_sort_key'] = table_df['Checkpoint'].apply(lambda x: int(x) if x != '-' else -1)
        table_df = table_df.sort_values(['Model', '_sort_key']).drop(columns=['_sort_key'])

        # Format numeric columns to 1 decimal place
        for col in metric_cols:
            table_df[col] = table_df[col].apply(lambda x: f'{x:.1f}' if pd.notna(x) else '-')

        display = 'block' if length == input_lengths[0] else 'none'
        table_html = table_df.to_html(index=False, classes='metrics-table', border=0)
        table_html_parts.append(
            f'<div class="result-table" data-length="{length}" style="display:{display};">'
            f'<h3>Results for input_max_length = {length}</h3>'
            f'{table_html}</div>'
        )

    tables_combined = '\n'.join(table_html_parts)

    # Build full HTML with plot + table + JS to sync dropdown
    plot_div = fig.to_html(full_html=False, config={'responsive': True}, default_width='100%')

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .metrics-table {{
    border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px;
  }}
  .metrics-table th, .metrics-table td {{
    border: 1px solid #ddd; padding: 6px 10px; text-align: right;
  }}
  .metrics-table th {{ background-color: #f5f5f5; position: sticky; top: 0; }}
  .metrics-table td:first-child, .metrics-table td:nth-child(2),
  .metrics-table th:first-child, .metrics-table th:nth-child(2) {{
    text-align: left;
  }}
  .metrics-table tr:hover {{ background-color: #f0f7ff; }}
  h3 {{ margin-top: 20px; }}
</style>
</head>
<body>
{plot_div}
<hr>
{tables_combined}
<script>
(function() {{
  var lengths = {json.dumps([int(l) for l in input_lengths])};
  var lengthMap = {json.dumps({i: int(l) for i, l in enumerate(input_lengths)})};
  var plotDiv = document.querySelectorAll('.plotly-graph-div')[0];
  if (plotDiv) {{
    plotDiv.on('plotly_restyle', function(eventData) {{
      if (eventData && eventData[0] && eventData[0].visible) {{
        var vis = eventData[0].visible;
        // Find which length is now visible by checking the trace_length_map pattern
        var traceLengths = {json.dumps([int(l) for l in trace_length_map])};
        var selectedLength = null;
        for (var i = 0; i < vis.length; i++) {{
          if (vis[i] === true && i < traceLengths.length) {{
            selectedLength = traceLengths[i];
            break;
          }}
        }}
        if (selectedLength !== null) {{
          document.querySelectorAll('.result-table').forEach(function(el) {{
            el.style.display = (parseInt(el.dataset.length) === selectedLength) ? 'block' : 'none';
          }});
        }}
      }}
    }});
  }}
}})();
</script>
</body>
</html>"""

    prefix_name = "_".join(os.path.basename(p.rstrip('/')) for p in prefix)
    filepath = os.path.join(output_dir, f"{prefix_name}.html")
    with open(filepath, 'w') as f:
        f.write(full_html)
    print(f"Plot saved to {filepath}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prefix", nargs="+", type=str, default=None)
    args = parser.parse_args()

    # comment out the models you don't want to include, or add the new ones
    models_configs = [
        {"model": "gpt-4-0125-preview", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-mini-2024-07-18", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-2024-05-13", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-2024-08-06", "use_chat_template": True, "training_length": 128000},
        {"model": "claude-3-5-sonnet-20240620", "use_chat_template": True, "training_length": 200000},
        {"model": "gemini-1.5-flash-001", "use_chat_template": True, "training_length": 1048576},
        {"model": "gemini-1.5-pro-001", "use_chat_template": True, "training_length": 2097152},

        # llama 2 based models
        {"model": "Llama-2-7B-32K", "use_chat_template": False, "training_length": 32768},
        {"model": "Llama-2-7B-32K-Instruct", "training_length": 32768},
        {"model": "llama-2-7b-80k", "use_chat_template": False, "training_length": 80000},
        {"model": "Yarn-Llama-2-7b-64k", "use_chat_template": False, "training_length": 65536},
        {"model": "Yarn-Llama-2-7b-128k", "use_chat_template": False, "training_length": 131072},

        # llama 3 models
        {"model": "Meta-Llama-3-8B", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Instruct", "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Theta16M", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Instruct-Theta16M", "training_length": 8192},
        {"model": "Meta-Llama-3-70B-Theta16M", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-70B-Instruct-Theta16M", "training_length": 8192},

        {"model": "Llama-3.1-8B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
        {"model": "Llama-3.1-70B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.1-70B-Instruct", "training_length": 131072},
        {"model": "Llama-3.3-70B-Instruct", "training_length": 131072},

        {"model": "Llama-3.2-1B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.2-1B-Instruct", "training_length": 131072},
        {"model": "Llama-3.2-3B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.2-3B-Instruct", "training_length": 131072},

        # mistral models
        {"model": "Mistral-7B-v0.1", "use_chat_template": False, "training_length": 8192},
        {"model": "Mistral-7B-Instruct-v0.1", "training_length": 8192},
        {"model": "Mistral-7B-Instruct-v0.2", "training_length": 32768},
        {"model": "Mistral-7B-v0.3", "use_chat_template": False, "training_length": 32768},
        {"model": "Mistral-7B-Instruct-v0.3", "training_length": 32768},
        {"model": "Ministral-8B-Instruct-2410", "training_length": 131072},

        {"model": "Mistral-Nemo-Base-2407", "use_chat_template": False, "training_length": 128000},
        {"model": "Mistral-Nemo-Instruct-2407", "training_length": 128000},
        {"model": "MegaBeam-Mistral-7B-512k", "training_length": 524288},

        # yi models
        {"model": "Yi-6B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-9B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-34B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-1.5-9B-32K", "use_chat_template": False, "training_length": 32768},

        # phi models
        {"model": "Phi-3-mini-128k-instruct", "training_length": 131072},
        {"model": "Phi-3-small-128k-instruct", "training_length": 131072},
        {"model": "Phi-3-medium-128k-instruct", "training_length": 131072},
        {"model": "Phi-3.5-mini-instruct", "training_length": 131072},

        # qwen models
        {"model": "Qwen2-7B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2-7B-Instruct", "training_length": 32768},
        {"model": "Qwen2-57B-A14B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2-57B-A14B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-1.5B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2.5-1.5B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-3B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2.5-3B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-7B", "use_chat_template": False, "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct", "training_length": 131072},
        {"model": "Qwen2.5-72B-Instruct", "training_length": 131072},

        # prolong
        {"model": "Llama-3-8B-ProLong-512k-Instruct", "training_length": 524288},

        # gemma 2 models
        {"model": "gemma-2-9b", "use_chat_template": False, "training_length": 8192},
        {"model": "gemma-2-9b-it", "training_length": 8192},
        {"model": "gemma-2-9b-it-Theta320K", "training_length": 8192},

        {"model": "gemma-2-27b", "use_chat_template": False, "training_length": 8192},
        {"model": "gemma-2-27b-it", "training_length": 8192},
        {"model": "gemma-2-27b-it-Theta320K", "training_length": 8192},

        # others
        {"model": "c4ai-command-r-v01", "training_length": 131072},
        {"model": "Jamba-v0.1", "use_chat_template": False, "training_length": 262144},
        {"model": "AI21-Jamba-1.5-Mini", "training_length": 262144},
    ]


    models_configs = [
        {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
        {"model": "Llama-3-8B-ProLong-512k-Base", "training_length": 524288},
        {"model": "Llama-3-8B-ProLong-512k-Instruct", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_dolci_bsz128_steps125_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_dolci_bsz128_steps125_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_dolci_bsz128_steps125_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_dolci_bsz128_steps125_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_dolci_bsz64_steps250_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_dolci_bsz64_steps250_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_dolci_bsz64_steps250_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_dolci_bsz64_steps250_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-even_bsz128_steps125_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-even_bsz128_steps125_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-even_bsz128_steps125_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-even_bsz128_steps125_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-even_bsz64_steps250_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-even_bsz64_steps250_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-even_bsz64_steps250_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-even_bsz64_steps250_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-prop_bsz128_steps125_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-prop_bsz128_steps125_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-prop_bsz128_steps125_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-prop_bsz128_steps125_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-prop_bsz64_steps250_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-prop_bsz64_steps250_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-prop_bsz64_steps250_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_nemotron-v2-prop_bsz64_steps250_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_tulu3_bsz128_steps125_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_tulu3_bsz128_steps125_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_tulu3_bsz128_steps125_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_tulu3_bsz128_steps125_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_tulu3_bsz64_steps250_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_tulu3_bsz64_steps250_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_tulu3_bsz64_steps250_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_tulu3_bsz64_steps250_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_ultrachat_bsz128_steps125_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_ultrachat_bsz128_steps125_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_ultrachat_bsz128_steps125_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_ultrachat_bsz128_steps125_lr5e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_ultrachat_bsz64_steps250_lr1e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_ultrachat_bsz64_steps250_lr2e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_ultrachat_bsz64_steps250_lr3e-5_warmup0.05", "training_length": 524288},
        #{"model": "sft_Llama-3-8B-ProLong-512k-Base_sft-64K_ultrachat_bsz64_steps250_lr5e-5_warmup0.05", "training_length": 524288},
    ]

    baseline_model_names = set(m['model'] for m in models_configs)

    prefix = args.prefix
    paths = [p for path in prefix for p in glob.glob(f"{path}*")]
    add_models = sorted(paths)
    models_configs += [{'model': os.path.basename(m)} for m in add_models]

    # set your configs here, only include the ones that you ran
    config_files = [
        "configs/recall.yaml", "configs/recall_short.yaml",
        "configs/rag.yaml", "configs/rag_short.yaml",
        "configs/longqa.yaml", "configs/longqa_short.yaml",
        "configs/summ.yaml", "configs/summ_short.yaml",
        "configs/rerank.yaml", "configs/rerank_short.yaml",
        "configs/icl.yaml", "configs/icl_short.yaml",
        "configs/cite.yaml", "configs/cite_short.yaml",
        "configs/ruler.yaml", "configs/ruler_short.yaml",
    ]

    config_files = [
        #"configs/dev_32k.yaml", "configs/dev_64k.yaml",
        #"configs/ppl_longmino_64k.yaml",
        "configs/dev_syn_32k_v2.yaml",
        "configs/dev_syn_64k_v2.yaml",
        "configs/dev_syn_128k_v2.yaml",
        "configs/dev_syn_256k_v2.yaml",
        "configs/dev_syn_512k_v2.yaml",
        #"configs/dev_128k.yaml", "configs/dev_256k.yaml",
    ]

    dataset_configs = []
    for file in config_files:
        c = yaml.safe_load(open(file))['dataset_options']

        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
        if isinstance(c['use_chat_template'], bool):
            c["use_chat_template"] = ",".join([str(c["use_chat_template"])] * len(c["datasets"].split(",")))

        for d, t, l, g, ct in zip(c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','), c['generation_max_length'].split(','), c['use_chat_template'].split(',')):
            dataset_configs.append({"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]), "input_max_length": int(l), "generation_max_length": int(g), "max_test_samples": c['max_test_samples'], 'use_chat_template': ct, 'shots': c.get('shots', 0)})
    print(dataset_configs)

    failed_paths = []
    df = []
    for model in tqdm(models_configs):
        args = arguments()
        args.tag = "v1" # SET YOUR TAG HERE
        args.output_dir = f"output/{model['model']}"

        for dataset in dataset_configs:
            args.update(dataset)
            args.update(model)

            metric = args.get_averaged_metric()
            dsimple, mnames = args.get_metric_name()

            if metric is None:
                failed_paths.append(args.get_path())
                continue

            for k, m in metric.items():
                df.append({**asdict(args), **model,
                    "metric name": k, "metric": m,
                    "dataset_simple": dsimple + " " + k, "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                })

    all_df = pd.DataFrame(df)
    lf_df = all_df.pivot_table(index=["input_max_length", "model", ], columns="dataset_simple", values="metric", sort=False)
    lf_df = lf_df.reset_index()
    lf_df = lf_df.sort_values("input_max_length", kind="stable").reset_index(drop=True)

    for k, v in custom_avgs.items():
        if all(col in lf_df.columns for col in v):
            lf_df[k] = lf_df[v].mean(axis=1)

    if prefix is not None:
        for p in prefix:
            lf_df['model'] = lf_df['model'].str.replace(os.path.basename(p), '')
    print(lf_df.to_csv(index=False))

    if prefix is not None:
        plot_results_html(lf_df, baseline_model_names, prefix)
