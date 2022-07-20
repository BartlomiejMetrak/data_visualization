from matplotlib.legend_handler import HandlerBase
from adjustText import adjust_text

import matplotlib.pyplot as plt
import geopandas


"""" skyrpt do generowania mapy i zapisywania jej jeśli potrzeba """


class plot_points_geolocation:
    """
    Funkcja do umieszczania punktów na mapie i zapisywania obazka
    Podajemy dowolne współrzędne w dataframie i wizualizujemy je na mapie
    Kolumny konieczne:
        > latitude
        > longitude
        > color
        > size
        > label
        > city
    Mozemy też zapisywać obrazek lub skalować kropki itp
    """

    @staticmethod
    def calculate_size(df, scale):  # between x and 2x
        """ sumowanie po label i mieście aby obliczyć wielkość kropek """
        df_sized = df.groupby(['label', 'city']).size()
        df['size_no_scale'] = 1
        for index, row in df_sized.items():
            df.loc[(df['label'] == index[0]) & (df['city'] == index[1]), 'size_no_scale'] = row

        df['size'] = ((df['size_no_scale'] - min(df['size_no_scale'])) / (max(df['size_no_scale']) - min(df['size_no_scale'])) + 1) * scale
        return df

    @staticmethod
    def rescale(df, scale_column, scale):
        """ zmiana skali dla wskazanej kolumny """
        if len(df) > 1:
            df['size'] = ((df[scale_column] - min(df[scale_column])) / (max(df[scale_column]) - min(df[scale_column])) + 1) * scale
        else:
            df['size'] = scale
        return df

    def plot_only_points(self, df_geo, chart_desc_dict, chart_params, map_link, annotate=False, differ_size=False,
                         show_legend=True, size=300, add_title=True, save_path=None, save_bool=False, show_image=False):
        """ umieszczanie punktów na mapie razem z liczbą w środku """
        color_map = chart_desc_dict['color_map']
        font_color = chart_desc_dict['font_color']
        title = chart_desc_dict['title']

        font_size_small = chart_params['font_size_small']
        font_size_large = chart_params['font_size_large']  # 12
        chart_width = chart_params['chart_width']  # in cm
        chart_height = chart_params['chart_height']  # in cm
        cm = 1 / 2.54  # centimeters in inches

        if differ_size is True:
            df_geo = self.rescale(df=df_geo, scale_column='counter', scale=size)  #between x and 2x
        else:
            df_geo['size'] = size

        lsoas = geopandas.read_file(map_link)
        plt.rcParams.update({'axes.facecolor': 'white'})
        lsoas = lsoas.to_crs(epsg=4326)

        fig, ax = plt.subplots()
        plt.style.use('fivethirtyeight')
        plt.subplots_adjust(top=0.85)
        lsoas.plot(ax=ax, legend=True, color=color_map)

        # label_list = df_geo['label'].unique().tolist()

        texts = []
        x_coordinates = []
        y_coordinates = []
        annotated_cities_labels = []

        for location in df_geo.to_dict('records'):
            city = location['city']
            color = location['color']
            label = location['label']
            size = location['size']

            longitude = float(location['longitude'])
            latitude = float(location['latitude'])
            if (longitude, latitude) not in annotated_cities_labels and 12 < longitude < 26 and 47 < latitude < 56:
                ax.scatter(longitude, latitude, color=color, label=label, s=size, alpha=0.8)
                x_coordinates.append(longitude)
                y_coordinates.append(latitude)

                if annotate is True:
                    text = ax.annotate(city, (longitude, latitude+0.15), color=font_color, va='center', ha='center', fontsize=font_size_small)
                    # ax.annotate(location['counter'], (longitude, latitude), color=font_color, va='center', ha='center', fontsize=font_size_small)
                    texts.append(text)
                    annotated_cities_labels.append((longitude, latitude))

        if annotate is True:
            adjust_text(texts, x=x_coordinates, y=y_coordinates, expand_align=(2.5, 2.5), expand_text=(1.7, 1.7), expand_points=(2.5, 2.5))

        if add_title is True:
            plt.title(title, fontsize=font_size_large, pad=40)
        fig.tight_layout(pad=1)

        """ legenda - zmienione - sprawdzić czy działa bo bez testowania """
        if show_legend is True:
            legend_dict = dict(zip(df_geo['label'], df_geo['color']))
            list_color = list(legend_dict.values())
            list_lab = list(legend_dict.keys())
            list_mak = ["o"] * len(list_lab)
            plt.legend(list(zip(list_color, list_mak)), list_lab, handler_map={tuple: MarkerHandler()},
                       loc='upper right', fontsize=font_size_small, frameon=False)
        plt.box(False)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        self.set_size(w=chart_width * cm, h=chart_height * cm)
        if show_image is True:
            plt.show()

        if save_bool is True:
            fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=200)   # save the figure to file

    def plot_points_ebudownictwo(self, df_geo, chart_desc_dict, chart_params, map_link, label_list, annotate=False, differ_size=False,
                                 size=100, add_title=True, save_path=None, save_bool=False, show_image=False):
        """ umieszczanie punktów na mapie dla raportu z projektów inwestycyjnych """
        color_map = chart_desc_dict['color_map']
        font_color = chart_desc_dict['font_color']
        title = chart_desc_dict['title']

        font_size_small = chart_params['font_size_small']
        font_size_medium = chart_params['font_size_medium']
        font_size_large = chart_params['font_size_large']
        chart_width = chart_params['chart_width']  # in cm
        chart_height = chart_params['chart_height']  # in cm
        cm = 1 / 2.54  # centimeters in inches

        if differ_size is True:
            df_geo = self.rescale(df=df_geo, scale_column='counter', scale=size)  #between x and 2x
        else:
            df_geo['size'] = size

        lsoas = geopandas.read_file(map_link)
        plt.rcParams.update({'axes.facecolor': 'white'})
        lsoas = lsoas.to_crs(epsg=4326)

        fig, ax = plt.subplots()
        plt.style.use('fivethirtyeight')
        plt.subplots_adjust(top=0.85)
        lsoas.plot(ax=ax, legend=True, color=color_map)

        texts = []
        x_coordinates = []
        y_coordinates = []
        annotated_cities_labels = []
        props = dict(boxstyle='round', facecolor="#D9D9D9", alpha=0.5)

        for location in df_geo.to_dict('records'):
            company = location['name_search']
            color = location['color']
            label = location['label']
            size = location['size']

            longitude = float(location['longitude'])
            latitude = float(location['latitude'])
            if location['counter'] == 1 and 12 < longitude < 26 and 47 < latitude < 56:
                ax.scatter(longitude, latitude, color=color, label=label, s=size, alpha=0.8, marker="o")
                x_coordinates.append(longitude)
                y_coordinates.append(latitude)
                if annotate is True:
                    text = ax.annotate(company, (longitude, latitude+0.15), color=font_color, va='center', ha='center',
                                       fontsize=font_size_small, bbox=props)
                    texts.append(text)
                    annotated_cities_labels.append((longitude, latitude))
            else:
                ax.scatter(longitude, latitude, color=color, label=label, s=4*size, alpha=0.8, marker="o")
                x_coordinates.append(longitude)
                y_coordinates.append(latitude)
                if annotate is True:
                    ax.annotate(location['counter'], (longitude, latitude), color=font_color, va='center', ha='center', fontsize=font_size_small)

        if annotate is True:
            adjust_text(texts, x=x_coordinates, y=y_coordinates)

        if add_title is True:
            plt.title(title, fontsize=font_size_medium, pad=40)
        fig.tight_layout(pad=1)

        list_color = list(label_list.values())
        list_lab = list(label_list.keys())
        list_mak = ["o"] * len(list_lab)

        plt.legend(list(zip(list_color, list_mak)), list_lab, handler_map={tuple: MarkerHandler()},
                   bbox_to_anchor=(1.3, 0.5), loc='center right', fontsize=font_size_medium, frameon=False)
        plt.box(False)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        self.set_size(w=chart_width * cm, h=chart_height * cm)
        if show_image is True:
            plt.show()

        if save_bool is True:
            fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=200)   # save the figure to file

    def plot_with_numbers(self, df_geo, chart_desc_dict, chart_params, map_link, annotate=False, differ_size=False,
                          size=100, add_title=True, save_path=None, save_bool=False, show_image=False, annotate_exp=True):
        """ umieszczanie punktów na mapie razem z liczbą w środku """
        color_map = chart_desc_dict['color_map']
        font_color = chart_desc_dict['font_color']
        title = chart_desc_dict['title']

        font_size_small = chart_params['font_size_small']
        font_size_large = chart_params['font_size_large']  # 12
        chart_width = chart_params['chart_width']  # in cm
        chart_height = chart_params['chart_height']  # in cm
        cm = 1 / 2.54  # centimeters in inches

        if differ_size is True:
            df_geo = self.rescale(df=df_geo, scale_column='counter', scale=size)  #between x and 2x
        else:
            df_geo['size'] = size

        lsoas = geopandas.read_file(map_link)
        plt.rcParams.update({'axes.facecolor': 'white'})
        lsoas = lsoas.to_crs(epsg=4326)

        fig, ax = plt.subplots()
        plt.style.use('fivethirtyeight')
        plt.subplots_adjust(top=0.85)
        lsoas.plot(ax=ax, legend=True, color=color_map)

        label_list = df_geo['label'].unique().tolist()

        texts = []
        x_coordinates = []
        y_coordinates = []
        annotated_cities_labels = []

        for location in df_geo.to_dict('records'):
            city = location['city']
            color = location['color']
            label = location['label']
            size = location['size']

            longitude = float(location['longitude'])
            latitude = float(location['latitude'])
            if (longitude, latitude) not in annotated_cities_labels and 12 < longitude < 26 and 47 < latitude < 56:
                ax.scatter(longitude, latitude, color=color, label=label, s=size, alpha=0.8)
                x_coordinates.append(longitude)
                y_coordinates.append(latitude)

                if annotate is True:
                    text = ax.annotate(city, (longitude, latitude+0.15), color=font_color, va='center', ha='center', fontsize=font_size_small)
                    ax.annotate(location['counter'], (longitude, latitude), color=font_color, va='center', ha='center', fontsize=font_size_small)
                    texts.append(text)
                    annotated_cities_labels.append((longitude, latitude))

        if annotate is True:
            if annotate_exp is True:
                adjust_text(texts, x=x_coordinates, y=y_coordinates, expand_align=(2.5, 2.5), expand_text=(1.7, 1.7), expand_points=(2.5, 2.5))
            else:
                adjust_text(texts, x=x_coordinates, y=y_coordinates)

        if add_title is True:
            plt.title(title, fontsize=font_size_large, pad=40)
        fig.tight_layout(pad=1)

        plt.legend(label_list, loc='upper right', fontsize=font_size_small)
        plt.box(False)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        self.set_size(w=chart_width * cm, h=chart_height * cm)

        if show_image is True:
            plt.show()
        if save_bool is True:
            fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=200)   # save the figure to file

    def save_only_map(self, chart_desc_dict, chart_params, map_link, label, add_title=True, save_path=None,
                      save_bool=False, show_image=False):
        """ zapisywanie mapy z label - w przypadku gdy nie ma danych """
        color_map = chart_desc_dict['color_map']
        title = chart_desc_dict['title']

        font_size_small = chart_params['font_size_small']
        font_size_large = chart_params['font_size_large']  # 12
        chart_width = chart_params['chart_width']  # in cm
        chart_height = chart_params['chart_height']  # in cm
        cm = 1 / 2.54  # centimeters in inches

        lsoas = geopandas.read_file(map_link)
        plt.rcParams.update({'axes.facecolor': 'white'})
        lsoas = lsoas.to_crs(epsg=4326)

        fig, ax = plt.subplots()
        plt.style.use('fivethirtyeight')
        plt.subplots_adjust(top=0.85)
        lsoas.plot(ax=ax, legend=True, color=color_map)

        if add_title is True:
            plt.title(title, fontsize=font_size_large, pad=40)
        fig.tight_layout(pad=1)

        label_names = list(label.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=label[x]) for x in label_names]
        plt.legend(handles, label_names, loc='upper right', fontsize=font_size_small)
        plt.box(False)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        self.set_size(w=chart_width * cm, h=chart_height * cm)

        if show_image is True:
            plt.show()
        if save_bool is True:
            fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=200)  # save the figure to file

    """ funkcje do wykresów """

    @staticmethod
    def set_size(w, h, ax=None):
        """ w, h: width, height in inches """
        if not ax:
            ax = plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w) / (r - l)
        figh = float(h) / (t - b)
        ax.figure.set_size_inches(figw, figh)


class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup, xdescent, ydescent, width, height, fontsize, trans):
        return [plt.Line2D([width/2], [height/2.], ls="", marker=tup[1], color=tup[0], transform=trans)]
