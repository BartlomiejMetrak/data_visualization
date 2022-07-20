import matplotlib.pyplot as plt
from matplotlib import cm
from cycler import cycler
import numpy as np

from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable


plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = [14, 8]


def plot_macd(prices, macd, signal, hist):
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

    ax1.plot(prices)
    ax2.plot(macd, color='grey', linewidth=1.5, label='MACD')
    ax2.plot(signal, color='skyblue', linewidth=1.5, label='SIGNAL')

    for i in range(len(prices)):
        if str(hist[i])[0] == '-':
            ax2.bar(prices.index[i], hist[i], color='#ef5350')
        else:
            ax2.bar(prices.index[i], hist[i], color='#26a69a')

    plt.legend(loc='lower right')
    plt.show()


def plot_one_line_date_series(data, time, label, color, chart_title, save_bool=None, save_path=None, scale_max=False, scale_min=False):
    """ funkcja do rysowania wykresu dla notowań danej spółki """
    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots()
    ax.plot(time, data.values, linewidth=2, color=color, label=label)
    ax.grid(False)

    if scale_min is not False:
        ymin = data.min() * scale_min
        ymax = data.max() * scale_max
        ax.set_ylim([ymin, ymax])

    ax.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)

    plt.title(chart_title, y=1.0, pad=-14, fontname="Times New Roman", fontweight="bold")
    plt.legend()
    plt.show()

    if save_bool is True:
        fig.savefig(save_path, transparent=True)  # save the figure to file



def plot_one_bar(data, time, chart_desc_dict, chart_params, save_bool=None, save_path=None, scale_y=1.5,
                 show_image=False, add_title=False):
    """ funkcja do rysowania wykresu dla notowań danej spółki """
    label_1 = chart_desc_dict['label_1']
    color_1 = chart_desc_dict['color_1']
    title = chart_desc_dict['title']

    font_size_small = chart_params['font_size_small']
    font_size_large = chart_params['font_size_large']
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] = font_type

    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.85)

    ymin = 0
    ymax = max(data) * scale_y
    ax.set_ylim([ymin, ymax])
    #ax.yaxis.labelpad = 20

    bars = ax.bar(time, data.values, color=color_1, label=label_1, width=4, align='center')
    ax.tick_params(axis='x', labelrotation=45)
    ax.bar_label(container=bars, padding=10, size=font_size_small)

    ax.grid(False)
    plt.box(False)

    if add_title is False:
        plt.title(title, fontsize=font_size_large)

    set_size(w=chart_width*cm, h=chart_height*cm)
    fig.tight_layout()
    make_axes_area_auto_adjustable(ax)
    plt.legend(fontsize=font_size_small)  # loc='upper left'

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, bbox_inches='tight', transparent=True, dpi=200)  # save the figure to file


def create_two_lines(data_1, data_2, time, label_1, label_2, color_1, color_2):
    """ tworzenie wykresu z dwoma liniami na oddzielnych skalach - do wykresu spółki """
    plt.style.use('fivethirtyeight')

    fig, ax1 = plt.subplots()

    ax1.set_ylabel(label_1, color=color_1)
    ax1.plot(time, data_1, color=color_1, linewidth=2, label=label_1)
    ax1.tick_params(axis='y', labelcolor=color_1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # ax2.set_ylabel(label_2, color=color_2)  # we already handled the x-label with ax1
    ax2.plot(time, data_2, color=color_2, linewidth=2, label=label_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    ax1.grid(False)
    ax2.grid(False)
    ax1.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(loc='upper left')
    plt.show()


def create_two_sub_plots(data_1, data_2, time, chart_desc_dict, chart_params, height_ratio_1=5, height_ratio_2=2,
                         show_image=False, save_bool=False, save_path=None):
    """ tworzenie dwóch wykresów spółki jeden pod drugim """
    label_1 = chart_desc_dict['label_1']
    label_2 = chart_desc_dict['label_2']
    color_1 = chart_desc_dict['color_1']
    color_2 = chart_desc_dict['color_2']
    title = chart_desc_dict['title']
    ax_1_y_label = chart_desc_dict['ax_1_y_label']
    ax_2_y_label = chart_desc_dict['ax_2_y_label']

    font_size_small = chart_params['font_size_small']
    font_size_large = chart_params['font_size_large']
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] = font_type
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [height_ratio_1, height_ratio_2]})
    fig.suptitle(title, fontsize=font_size_large)
    fig.tight_layout(pad=2)

    ax1.plot(time, data_1, color=color_1, linewidth=2, label=label_1)
    ax2.plot(time, data_2, color=color_2, linewidth=1, label=label_2)

    ax1.set_ylabel(ax_1_y_label, size=font_size_small)
    ax2.set_ylabel(ax_2_y_label, size=font_size_small)

    ax1.axes.get_xaxis().set_visible(False)

    ax1.yaxis.tick_right()
    ax2.yaxis.tick_right()

    ax1.grid(False)
    ax2.grid(False)
    plt.box(False)
    ax1.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)

    y_ax2_max = 1.2*data_2.max() if data_2.max() > 5 else 10
    ax2.set_ylim([0, y_ax2_max])

    fig.tight_layout()
    make_axes_area_auto_adjustable(ax1)
    make_axes_area_auto_adjustable(ax2)
    set_size(w=chart_width*cm, h=chart_height*cm)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.legend(fontsize=font_size_small)  # loc='upper left'

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, transparent=True, dpi=200)  # save the figure to file

def create_one_chart_two_lines(data_1, data_2, time, chart_desc_dict, chart_params, show_image=False, save_bool=False, save_path=None):
    """ rysowanie jednego wykresu i dwóch linii """
    label_1 = chart_desc_dict['label_1']
    label_2 = chart_desc_dict['label_2']
    color_1 = chart_desc_dict['color_1']
    color_2 = chart_desc_dict['color_2']
    title = chart_desc_dict['title']
    ax_1_y_label = chart_desc_dict['ax_1_y_label']
    ax_2_y_label = chart_desc_dict['ax_2_y_label']

    font_size_small = chart_params['font_size_small']
    font_size_medium = chart_params['font_size_medium']
    font_size_large = chart_params['font_size_large']
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] = font_type

    fig, ax1 = plt.subplots()

    ax1.set_ylabel(label_1, color=color_1)
    ax1.plot(time, data_1, color=color_1, linewidth=2, label=label_1)
    ax1.tick_params(axis='y', labelcolor=color_1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel(label_2, color=color_2)  # we already handled the x-label with ax1
    ax2.plot(time, data_2, color=color_2, linewidth=1, label=label_2)
    ax2.tick_params(axis='y', labelcolor=color_2)


    ax1.set_ylabel(ax_1_y_label, size=font_size_medium)
    ax2.set_ylabel(ax_2_y_label, size=font_size_medium)

    #ax1.axes.get_xaxis().set_visible(False)

    ax1.yaxis.tick_right()
    ax2.yaxis.tick_left()
    ax1.yaxis.set_label_position("right")
    ax2.yaxis.set_label_position("left")
    y_ax2_max = 1.5*data_2.max()
    ax2.set_ylim([0, y_ax2_max])

    ax1.grid(False)
    ax2.grid(False)
    #plt.box(False)
    ax1.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)

    fig.tight_layout()
    #make_axes_area_auto_adjustable(ax1)
    #make_axes_area_auto_adjustable(ax2)
    set_size(w=chart_width*cm, h=chart_height*cm)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.legend(fontsize=font_size_small)  # loc='upper left'

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, transparent=True, dpi=200)  # save the figure to file

def create_one_chart_multiple_line(column_names, df, time, chart_desc_dict, chart_params, show_image=False, save_bool=False, save_path=None):
    """ rysowanie jednego wykresu i dwóch linii """
    ax_1_y_label = chart_desc_dict['ax_1_y_label']

    font_size_small = chart_params['font_size_small']
    font_size_medium = chart_params['font_size_medium']
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] = font_type

    fig, ax1 = plt.subplots()

    df = df[column_names]

    for col in df.columns:
        ax1.plot(time, df[col], label=col, linewidth=1)

    plt.legend()
    plt.show()

    ax1.set_ylabel(ax_1_y_label, size=font_size_medium)

    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    max_value = df.max().max()  # max po kolumnie i potem max ogółem
    y_ax2_max = 1.5*max_value
    ax1.set_ylim([0, y_ax2_max])

    ax1.grid(False)
    #plt.box(False)
    ax1.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)

    fig.tight_layout()
    #make_axes_area_auto_adjustable(ax1)
    #make_axes_area_auto_adjustable(ax2)
    set_size(w=chart_width*cm, h=chart_height*cm)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.legend(fontsize=font_size_small)  # loc='upper left'

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, transparent=True, dpi=200)  # save the figure to file

def compare_bars_next_to_each(groups, data_1, data_2, chart_desc_dict, chart_params, scale_y=1.5,
                              add_title=True, show_image=False, save_bool=False, save_path=None):
    """ tworzenie wykresu z słupkami obok siebie """
    label_1 = chart_desc_dict['label_1']
    label_2 = chart_desc_dict['label_2']
    color_1 = chart_desc_dict['color_1']
    color_2 = chart_desc_dict['color_2']
    title = chart_desc_dict['title']
    ax_y_label = chart_desc_dict['ax_y_label']

    font_size_small = chart_params['font_size_small']  # 8
    font_size_medium = chart_params['font_size_medium']
    font_size_large = chart_params['font_size_large']  # 12
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    X_axis = np.arange(len(groups))

    plt.rcParams["font.family"] = font_type
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.85)

    ymin = 0  # min(data_1) * 0
    ymax = max(data_1+data_2) * scale_y
    ax.set_ylim([ymin, ymax])
    ax.yaxis.labelpad = 20

    bars_1 = ax.bar(X_axis - 0.2, data_1, width=0.3, label=label_1, color=color_1, align='center')
    bars_2 = ax.bar(X_axis + 0.2, data_2, width=0.3, label=label_2, color=color_2, align='center')
    ax.bar_label(container=bars_1, padding=10, size=font_size_small)
    ax.bar_label(container=bars_2, padding=10, size=font_size_small)
    ax.axes.get_yaxis().set_visible(False)

    plt.xticks(X_axis, groups, rotation='vertical', size=font_size_medium)
    plt.ylabel(ax_y_label, size=font_size_medium)
    ax.grid(False)
    plt.box(False)

    make_axes_area_auto_adjustable(ax)
    set_size(w=chart_width*cm, h=chart_height*cm)

    if add_title is True:
        plt.title(title, fontsize=font_size_large, pad=40)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(loc='upper right', fontsize=font_size_medium)

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, transparent=True, dpi=200)  # save the figure to file


def compare_barh_next_to_each(groups, data_1, data_2, chart_desc_dict, chart_params, labels, scale_y=1.5, show_image=False,
                              show_difference=False, save_bool=False, save_path=None, add_title=False):
    """ wykres słupkowy porównawczy - poziome słupki """
    font_size_small = chart_params['font_size_small']  # 8
    font_size_large = chart_params['font_size_large']  # 12
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] = font_type
    plt.rc('xtick', labelsize=font_size_small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size_small)
    fig, ax = plt.subplots()

    plt.subplots_adjust(top=0.85)

    y_axis = np.arange(len(groups))
    width = 0.3

    xmin = min(data_1) * 0
    xmax = max(data_1) * scale_y
    ax.set_xlim([xmin, xmax])

    bars_1 = ax.barh(y=y_axis+width, height=width, width=data_1, color=chart_desc_dict['color_1'], label=labels)
    bars_2 = ax.barh(y=y_axis, height=width, width=data_2, color=chart_desc_dict['color_2'], label=labels)

    if show_difference is False:  # jeśli nie pokazujemy różnicy to pokazujemy dane na słupkach
        ax.bar_label(container=bars_1, padding=10, size=font_size_small)
        ax.bar_label(container=bars_2, padding=10, size=font_size_small)
    else:  # pokazujemy różnice na końcu słupków
        rects = ax.patches

        label_custom = [round(data_1[i]-data_2[i], 2) for i in range(len(data_1))]

        for rect, label in zip(rects, label_custom):
            print(rect)
            height = rect.get_width()
            ax.text(rect.get_width() + 0.2, rect.get_y() + rect.get_height()/2, label, va="center", color="red", fontsize=font_size_small)

    ax.set(yticks=y_axis + width / 2, yticklabels=groups, ylim=[2 * width - 1, len(groups)])

    ax.grid(False)
    make_axes_area_auto_adjustable(ax)
    set_size(w=chart_width*cm, h=chart_height*cm)

    if add_title is True:
        title = chart_desc_dict['title']
        plt.title(title, fontsize=font_size_large, pad=40)
    plt.box(False)

    label_names = list(labels.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=labels[x]) for x in label_names]

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(handles, label_names, loc='lower right', fontsize=font_size_small)

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, bbox_inches='tight', transparent=True, dpi=200)


def horizontal_bar_chart(pd_series_names, pd_series_data, chart_desc_dict, chart_params, color, label, show_difference=False,
                         scale_x_max=1.5, add_title=True, show_image=False, save_bool=False, save_path=None):
    """ wykres słupkowy - poziome słupki """
    font_size_small = chart_params['font_size_small']  # 8
    font_size_large = chart_params['font_size_large']  # 12
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] = font_type
    plt.rc('xtick', labelsize=font_size_small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size_small)
    fig, ax = plt.subplots()

    plt.subplots_adjust(top=0.85)

    xmin = min(pd_series_data) * 0
    xmax = max(pd_series_data) * scale_x_max
    ax.set_xlim([xmin, xmax])

    bars = ax.barh(y=pd_series_names, height=0.6, width=pd_series_data, color=color, label=label)

    if show_difference is False:  # jeśli nie pokazujemy różnicy to pokazujemy dane na słupkach
        ax.bar_label(container=bars, padding=10, size=font_size_small)
    else:  # pokazujemy różnice na końcu słupków
        rects = ax.patches

        label_custom = [1]*len(pd_series_names)

        for rect, label in zip(rects, label_custom):
            ax.text(rect.get_width() + 0.3, rect.get_y() + rect.get_height()/2, label, va="center", color="red", fontsize=font_size_small)


    ax.grid(False)
    make_axes_area_auto_adjustable(ax)
    set_size(w=chart_width*cm, h=chart_height*cm)

    if add_title is True:
        title = chart_desc_dict['title']
        plt.title(title, fontsize=font_size_large, pad=40)
    plt.box(False)

    label_names = list(label.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=label[x]) for x in label_names]

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(handles, label_names, loc='lower right', fontsize=font_size_small)

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, bbox_inches='tight', transparent=True, dpi=200)

def horizontal_bar_stacked(groups, data_1, data_2, chart_desc_dict, chart_params, labels, scale_y=1.5, show_image=False,
                           save_bool=False, save_path=None, add_title=False):
    """ wykres słupkowy nakładany - poziome słupki """
    font_size_small = chart_params['font_size_small']  # 8
    font_size_large = chart_params['font_size_large']  # 12
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] = font_type
    plt.rc('xtick', labelsize=font_size_small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size_small)
    fig, ax = plt.subplots()

    plt.subplots_adjust(top=0.85)

    y_axis = np.arange(len(groups))
    width = 0.3

    xmin = min(data_1) * 0
    xmax = max(data_1+data_2) * scale_y
    ax.set_xlim([xmin, xmax])

    data_lst = [data_1, data_2]
    color_lst = [chart_desc_dict['color_1'], chart_desc_dict['color_2']]
    left = 0
    for data, color in zip(data_lst, color_lst):
        ax.barh(y=y_axis, height=width, width=data, color=color, left=left)
        left = data

    ax.set(yticks=y_axis + width, yticklabels=groups, ylim=[2 * width, len(groups)])

    ax.grid(False)
    make_axes_area_auto_adjustable(ax)
    set_size(w=chart_width*cm, h=chart_height*cm)

    if add_title is True:
        title = chart_desc_dict['title']
        plt.title(title, fontsize=font_size_large, pad=40)
    plt.box(False)

    label_names = list(labels.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=labels[x]) for x in label_names]

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(handles, label_names, fontsize=font_size_small)

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, bbox_inches='tight', transparent=True, dpi=200)


def create_image_matplotlib(chart_params, save_path):
    """ zapisywanie surowego obrazka dal wykresów których nie ma """
    font_size_large = chart_params['font_size_large']  # 12
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    cm = 1 / 2.54  # centimeters in inches

    plt.rcParams["font.family"] = font_type
    fig, ax = plt.subplots()
    plt.box(False)

    ax.text(0.5, 0.5, 'Brak danych dla wykresu', transform=ax.transAxes, va='center', ha='center',
            backgroundcolor='white', size=font_size_large)

    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    set_size(w=chart_width * cm, h=chart_height * cm)
    fig.savefig(save_path, transparent=True, dpi=200)


def create_multiline_chart(df, date_col, counter_col, name_col, unique_name_list, chart_desc_dict, chart_params,
                           color_palette='Blues_r', add_title=False, show_image=False, save_bool=False, save_path=None):
    """ tworzenie wykresu z wieloma liniami nałozonymi na tą samą skalę """
    ax_y_label = chart_desc_dict['ax_y_label']

    font_size_small = chart_params['font_size_small']  # 8
    font_size_medium = chart_params['font_size_medium']
    font_size_large = chart_params['font_size_large']  # 12
    chart_width = chart_params['chart_width']  # in cm
    chart_height = chart_params['chart_height']  # in cm
    font_type = chart_params['font_type']
    centimeter = 1 / 2.54  # centimeters in inches

    plt.rcParams["font.family"] = font_type
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.85)

    color_map = cm.get_cmap(color_palette, len(unique_name_list))
    plt.gca().set_prop_cycle(cycler('color', color_map(np.linspace(0.7, 0, len(unique_name_list)))))

    for woj in unique_name_list:
        data = df[df[name_col] == woj]
        plt.plot(data[date_col], data[counter_col], label=woj, linewidth=2)  # plotting t, a separately

    #ax.set_ylabel(ax_y_label, size=font_size_medium)
    y_ax2_max = 1.4 * df[counter_col].max()
    ax.set_ylim([0, y_ax2_max])

    ax.grid(False)
    ax.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)
    ax.yaxis.tick_right()

    fig.tight_layout()
    # make_axes_area_auto_adjustable(ax1)
    # make_axes_area_auto_adjustable(ax2)
    set_size(w=chart_width * centimeter, h=chart_height * centimeter)
    plt.subplots_adjust(wspace=0, hspace=0)

    if add_title is True:
        title = chart_desc_dict['title']
        plt.title(title, fontsize=font_size_large, pad=40)

    plt.legend(fontsize=font_size_small)

    if show_image is True:
        plt.show()

    if save_bool is True:
        fig.savefig(save_path, transparent=True, dpi=200)  # save the figure to file


""" funkcje do wykresów """

def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
