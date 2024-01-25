import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json


def plot_fig2(data_path, output_path):
    colors = ['#fbbb6b', '#ff9c00', '#7cc3e8', '#44b4e0', '#d7d1e5', '#c0b4d6', '#63e863']
    # font = FontProperties(family='Times New Roman')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    index = -1

    fig = plt.figure(figsize=(13, 4))
    fig.patch.set_alpha(0)

    ax = fig.add_subplot()
    ax.patch.set_facecolor("#e4eef6")
    ax.patch.set_alpha(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    index1_plot = np.arange(4) * 3

    with open(data_path, 'r', encoding='utf-8') as f:
        for item in f.readlines():
            index += 1
            items = item.strip('\n').split(';')
            index_plot = index1_plot + 0.35 * index
            name_plit = items[0]
            data = [float(i) for i in items[1].split(',')]
            std = [float(i) for i in items[2].split(',')]

            plt.bar(index_plot, data, width=0.35, yerr=std, error_kw={'ecolor': '0.2', 'capsize': 3}, alpha=0.7,
                    color=colors[index], edgecolor='k', label=name_plit, zorder=10)
            text_index = -1
            for x, y in enumerate(data):
                text_index += 1
                if y > 0:
                    plt.text(index_plot[text_index], y + std[x] + 0.0001, '%s' % round(y, 4), ha='center', va='bottom', size=6, zorder=20)
                else:
                    plt.text(index_plot[text_index], y - std[x] - 0.0008, '%s' % round(y, 4), ha='center', va='bottom', size=6, zorder=20)

    index_x = [1.025, 4.025, 7.025, 10.025]
    plt.xticks(index_x, ['5%', '10%', '15%', '20%'], fontsize=12)
    plt.yticks([0, 0.005, 0.01, 0.015], ['0', '0.005', '0.010', '0.015'], fontsize=12)
    plt.legend(shadow=True, loc='best', handlelength=1.5, fontsize=10)
    plt.xlabel("k%", fontsize=14, style='italic')
    plt.ylabel(chr(916) + " AUC", fontsize=14)
    plt.xlim(-0.5, 11.7)
    plt.ylim(-0.003, 0.02)
    plt.grid(axis='y', color='white', linestyle="--", zorder=0, alpha=0.9)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')


def plot_fig4(data_path, output_path, simple):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    colors = ['#fbbb6b', '#ff9c00', '#44b4e0', 'blue', '#c0b4d6', '#7b6bae', 'purple', '#009900']
    # font = FontProperties(family='Times New Roman')

    # load running time data
    with open(data_path, "r") as f:
        rt_data = json.load(f)

    fig = plt.figure(figsize=(8, 4))
    fig.patch.set_alpha(0)
    ax = fig.add_subplot()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    if simple:
        ax.plot(rt_data['mdi_default'], label='MDI-default', marker='o', color=colors[0], markersize=5, linewidth=2)
        ax.plot(rt_data['mdi_tuned'], label='MDI-tuned', marker='o', color=colors[1], markersize=5, linewidth=2)
        ax.plot(rt_data['shap_default'], label='SHAP-default', marker='o', color=colors[2], markersize=5, linewidth=2)
        ax.plot(rt_data['lte'], label='LTE', marker='o', color=colors[-1], markersize=5, linewidth=2)
    else:
        ax.plot(rt_data['mdi_default'], label='MDI-default', marker='o', color=colors[0], markersize=5, linewidth=2)
        ax.plot(rt_data['mdi_tuned'], label='MDI-tuned', marker='o', color=colors[1], markersize=5, linewidth=2)
        ax.plot(rt_data['shap_default'], label='SHAP-default', marker='o', color=colors[2], markersize=5, linewidth=2)
        ax.plot(rt_data['shap_tuned'], label='SHAP-tuned', marker='o', color=colors[3], markersize=5, linewidth=2)
        ax.plot(rt_data['pi_single'], label='PI-single', marker='o', color=colors[5], markersize=5, linewidth=2)
        ax.plot(rt_data['pi_ensemble'], label='PI-ensemble', marker='o', color=colors[6], markersize=5, linewidth=2)
        ax.plot(rt_data['lte'], label='LTE', marker='o', color=colors[-1], markersize=5, linewidth=2)

    ax.set_xlim(0, 7.1)
    ax.set_xticks(np.arange(0, 8), ['0', '1M', '2M', '3M', '4M', '5M', '6M', '7M'])

    ax.set_yscale('log')
    ax.set_xlabel('Number of samples', fontsize=16)
    ax.set_ylabel('Running time (s)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    legend = ax.legend(shadow=True, loc='lower right', handlelength=1.5, fontsize=9, ncol=4)
    for text in legend.get_texts():
        text.set_weight('bold')
    ax.grid(linestyle="--", alpha=0.5)
    plt.grid(linestyle="--", alpha=0.5)

    plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight')
