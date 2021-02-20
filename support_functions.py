import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn import metrics, config_context
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTENC

import plotly.express as px
import missingno
import pickle
from joblib import dump, load


COLOR = '#1f77b4'
RANDOM_STATE = 51
COL_DESCRIPTIONS = {
        'amount_tsh':'Total static head (amount water available to waterpoint)',
        'date_recorded':'The date the row was entered',
        'funder':'Who funded the well',
        'gps_height':'Altitude of the well',
        'installer':'Organization that installed the well',
        'longitude':'GPS coordinate',
        'latitude':'GPS coordinate',
        'wpt_name':'Name of the waterpoint if there is one',
        'num_private':'',
        'basin':'Geographic water basin',
        'subvillage':'Geographic location',
        'region':'Geographic location',
        'region_code':'Geographic location (coded)',
        'district_code':'Geographic location (coded)',
        'lga':'Geographic location',
        'ward':'Geographic location',
        'population':'Population around the well',
        'public_meeting':'True/False',
        'recorded_by':'Group entering this row of data',
        'scheme_management':'Who operates the waterpoint',
        'scheme_name':'Who operates the waterpoint',
        'permit':'If the waterpoint is permitted',
        'construction_year':'Year the waterpoint was constructed',
        'extraction_type':'The kind of extraction the waterpoint uses',
        'extraction_type_group':'The kind of extraction the waterpoint uses',
        'extraction_type_class':'The kind of extraction the waterpoint uses',
        'management':'How the waterpoint is managed',
        'management_group':'How the waterpoint is managed',
        'payment':'What the water costs',
        'payment_type':'What the water costs',
        'water_quality':'The quality of the water',
        'quality_group':'The quality of the water',
        'quantity':'The quantity of water',
        'quantity_group':'The quantity of water',
        'source':'The source of the water',
        'source_type':'The source of the water',
        'source_class':'The source of the water',
        'waterpoint_type':'The kind of waterpoint',
        'waterpoint_type_group':'The kind of waterpoint'
    }


def underline(string, character='-'):
    """
    Return a string of a given character with the length of a given string.
    """
    return character * len(string)
    
    
def headerize(string, character='*', max_len=80):
    """
    Return a given string with a box (of given character) around it.
    """
    if max_len:
        # Create uniform size boxes for headers with centered text.
        if len(string) > max_len-2:
            string = string[:max_len-5] + '...'
            
        total_space = max_len - 2 - len(string)
        left = total_space // 2
        if total_space % 2 == 0:
            right = left
        else:
            right = left + 1
        
        top = character * 80
        mid = f'{character}{" " * left}{string}{" " * right}{character}'
        bot = top
    else:
        # Create modular header boxes depending on the length of the string.
        top = character * (len(f'{string}')+42)
        mid = f'{character}{" " * 20}{string}{" " * 20}{character}'
        bot = top
        
    return f'{top}\n{mid}\n{bot}'


def preliminary_eda(dataframe, 
                    col='',
                    as_cat=False,
                    figsize=(8, 6),
                    color='#1f77b4',
                    title='',
                    mean_median=True,
                    n_top_features=10,
                    normalize=False,
                    dropna=False,
                    interact_with_col='',
                    interact_with_series=None,
                    try_hist=True):
    """
    Show information on a given pandas Series.
    
    Parameters:
    -----------
    dataframe: pandas.DataFrame OR pandas.Series
        The dataframe to analyze.
        If a Series is passed, it will be transformed into a 
        DataFrame and its column will be used.
    col: string (default: '')
        The column name to slice from the dataframe.
    as_cat: bool (default: False)
        If True, a numerical column will be coerced into an 'object'.
    figsize: tup (default: (8, 6))
        (width, height) of the resulting plots.
    color: string (default: '#1f77b4')
        HEX code for color of the bar/box plots.
    title: string (default: '')
        Optional title for the plot. 
        If none is given, the title will automatically set to '{`col`} Analysis'
    mean_median: bool (default: True)
        If True, the distplot will show the mean and median with vertical lines.
    n_top_features: int (default: 10)
        Max number of features to show for the `.value_counts()` function.
    normalize: bool (default: False)
        If True, the value counts will be normalized (% of total) 
        rather than actual counts.
    dropna: bool (default: False)
        If True, NaN values will be filtered out of the value counts.
    interact_with_col: str(default: '')
        Optional column name to plot `col` against.
    interact_with_series: pandas.Series (default: None)
        Optional pandas.Series to plot with `dataframe[col]`. 
        (Alternative to interact_with_col.)
    try_hist: bool (default: True)
        If True, a boxplot will be tried for the interaction plot.
        If the given `dataframe[col]` is not numeric, 
        a scatterplot will be used instead.
    """
    global COL_DESCRIPTIONS
    
    # Set internal variables.
    if type(dataframe) == pd.Series:
        dataframe = pd.DataFrame(dataframe)
        col = dataframe.columns[0]
    series = dataframe[col].copy()
    
    if as_cat:
        series = series.astype('object')
        
    vc = series.value_counts(normalize=normalize, 
                             dropna=dropna).head(n_top_features)
    
    # Check if main feature is numeric.
    numeric = False
    if series.dtype.kind in ['i', 'f']:
        numeric = True
    
    # Header
    if not title:
        title = f'{col} Analysis'
        
    print(headerize(title))
    if col in COL_DESCRIPTIONS:
        col_des = COL_DESCRIPTIONS[col]
        print(col_des)
        print(underline(col_des))
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    if numeric:
        # Distplot.
        sns.distplot(series, ax=ax)
        if mean_median:
            mean = series.mean()
            median = series.median()
            ax.axvline(mean, color='orange', ls=':', 
                       label=f'Mean: {round(mean, 3)}')
            ax.axvline(median, color='blue', ls=':', 
                       label=f'Median: {round(median, 3)}')
            ax.legend()
    else:
        # Bar Chart: top-n value_counts.
        sns.barplot(x=vc, y=vc.index, orient='h', color=color, ax=ax)
        ax.set(xlabel=f'{col} Count')
    ax.set(title=title)
    fig.tight_layout()
    plt.show()
    print()
    
    # Value counts
    vc_title = f'{col} - Value Counts:'
    print(vc_title)
    print(underline(vc_title))
    display(vc)
    print()
    
    # Num unique values
    unique_title = f'{col} - Unique Values'
    print(unique_title)
    print(underline(unique_title))
    print(f'{len(series.unique())} unique values (out of {len(series)}).')
    print()
    
    # NaN
    nan_title = f'{col} - NaN:'
    print(nan_title)
    print(underline(nan_title))
    n = series.isna().sum()
    tot = len(series)
    print(f'{round((n/tot) * 100, 2)}%')
    print(f'({n}/{tot})')
    
    if not interact_with_col and interact_with_series is None:
        return
    
    # Interact with...
    # Set up `plot_df` with two columns.
    if interact_with_col:
        plot_df = dataframe[[col, interact_with_col]].copy()
    elif type(interact_with_series) == pd.Series:
        plot_df = pd.concat([dataframe[col], 
                             interact_with_series], axis=1).copy()
    x, y = plot_df.columns
    
    fig, ax = plt.subplots(figsize=figsize)
    if try_hist and numeric:
        # Distplot for each unique category in the interact_with column.
        for val in plot_df[y].unique():
            sns.distplot(plot_df[plot_df[y] == val][x], label=val)
        ax.legend()
        ax.set(title=f'{y} vs. {x}')
    else:
        # Plot stacked horizontal bar chart.
        # Set up aggregated group_by df.
        plot_df['count'] = 1
        plot_df = plot_df.groupby([x, y]).sum().reset_index()
        x_cols = list(series.value_counts(
            dropna=True).head(n_top_features).index)
        y_vals = sorted(plot_df[y].unique())
        
        r = list(range(len(x_cols), -1, -1))  # y-tick markers.
        colors_lst = ['#1fb426', '#ccc125', '#b41f38']
        
        # Assign percent values for each category.
        vals_dct = {}
        for col in x_cols:
            total = plot_df[plot_df[x] == col]['count'].sum()
            normalized = plot_df[plot_df[x] == col].set_index(y)['count']/total
            vals_dct[col] = normalized
            
        # Plot stacked bars for each x-value.
        for r_val, k in zip(r, vals_dct):
            # Find where each bar-segment starts (leftmost = 0)
            left_vals = [0] + list(np.cumsum([vals_dct[k].get(y_val, 0) 
                                              for y_val in y_vals[:-1]]))
            
            # Plot bars with labels.
            for y_val, left, color in zip(y_vals, left_vals, colors_lst):
                if r_val == r[0]:
                    label = y_val
                else:
                    label = None
                ax.barh([r_val], vals_dct[k].get([y_val], 0), left=left, 
                        color=color, label=label)
        plt.yticks(r, x_cols)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=5)
        ax.axvline(0.5, ls=':', color=COLOR)
        ax.set(title=f'Percent Stacked Barplot\n{x}',
               xlabel='Percent of Entries')
    fig.tight_layout()
    plt.show()


class Stopwatch:
    """
    A simple stopwatch object.
    --------------------------
    
    Parameters:
    -----------
    auto_start: bool (default: True)
        Set whether to start the stopwatch when it's created.
        
    Attributes:
    -----------
    start_time: datetime object
        The recorded time the stopwatch started.
    times: list
        A list of tuples (`label`, `datetime_object`) for each lap recorded.
        (Includes the start and stop times.)
    stop_time: datetime object
        The recorded time the stopwatch stopped.
    """
    def __init__(self, auto_start=True, default_color='#1f77b4'):
        self.start_time = None
        self.times = []
        self.stop_time = None
        self.default_color=default_color
        
        if auto_start:
            self.start()
    
    def start(self, force_reset=True):
        """
        Start the stopwatch.
        --------------------
        
        Parameters:
        -----------
        force_reset: bool (default: True)
            If true, the stopwatch will reset all stored values back to their defaults before starting.
            If false and the timer has times logged, an exception will be raised.
        """
        if self.times and not force_reset:
            raise Exception('Please reset timer using .reset() or set force_reset=True.')
        if force_reset:
            self.reset()
        self.start_time = datetime.now()
        self.times.append(('Start', self.start_time))
        
    def lap(self, label=None):
        """
        Create a timestamp of a lap being completed.
        Appends the list of times in the stopwatch.
        The overall stopwatch continues running.
        --------------------------------------------
        
        Parameters:
        -----------
        label: str
            The label of the lap for identification and plotting purposes.
            If none is passed, the default 'Lap{n}' will be passed.
        """
        if not label:
            label = f'Lap{len(self.times)}'
        self.times.append((label, datetime.now()))
        
    def stop(self):
        """
        Stop the stopwatch.
        This will freeze all times in its times list.
        """
        if not self.start_time:
            print('Timer not started.')
            return
        self.stop_time = datetime.now()
        self.times.append(('Stop', self.stop_time))
        
    def reset(self):
        """
        Restore to factory settings.
        """
        self.__init__(auto_start=False)
        
        
    def elapsed_time_(self):
        """
        Returns the difference between the start_time and the stop_time.
        """
        if self.stop_time and self.start_time:
            elapsed = self.stop_time - self.start_time
            return elapsed
        
    def display_laps(self, 
                     figsize=(8,4), 
                     mark_elapsed_time=True,
                     minutes_elapsed=False,
                     show_stop=True,
                     annotate=True, 
                     verbose=True,
                     vlines=True,
                     styles=['ggplot', 'seaborn-talk']):
        """
        Plots the stored times - start_time, laps, stop_time.
        -----------------------------------------------------
        
        Parameters:
        -----------
        figsize: tup (default: (8, 4))
            Size of the output figure (width, height).
        mark_elapsed_time: bool (default: True)
            Calculate and plot the difference in time from each point to the starting point (0).
            If false, the points will be plotted on their datetime objects.
        minutes_elapsed: bool (default: False)
            If True, elapsed time will be shown as minutes instead of seconds.
        annotate: bool (default: True)
            Label the points with their stored labels.
        verbose: bool (default: True)
            If true, display a dataframe with columns=[label, timestamp, [elapsed_time]].
        vlines: bool (default: True)
            Plot a dotted vertical line on each point in the times list.
        styles: list of strings (or string) (default: ['ggplot', 'seaborn-talk'])
            Desired style of plot. Must be compatible with matplotlib styles.
        """
        if not self.times:
            print('No times to display.')
            return
        
        if show_stop:
            times = self.times
        else:
            times = self.times[:-1]
        
        # Plot one-dimentional scatter for each lap marked.
        with plt.style.context(styles):
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set x-values.
            if mark_elapsed_time:
                x = [(x[1] - self.start_time).total_seconds() 
                     for x in times]
                if minutes_elapsed:
                    elapsed_str = 'Elapsed Time (min)'
                    x = [round(a/60, 2) for a in x]
                    ax.set(xlabel=elapsed_str)
                else:
                    elapsed_str = 'Elapsed Time (sec)'
                    ax.set(xlabel=elapsed_str)
            else:
                x = [a[1] for a in times]
                ax.set(xlabel='Time Recorded')
                
            # Set y-values (constant).
            y = [0 for _ in x]
            
            # Plot, annotate, format.
            ax.scatter(x=x, y=y, color=self.default_color)
            if annotate:
                [plt.annotate(label, 
                              (x[i], y[i]), 
                              xytext=(x[i], y[i]+0.005)) 
                 for i, (label, x_val) in enumerate(times)]
            if vlines:
                if mark_elapsed_time:
                    [ax.axvline(x_val, ls=':', color=self.default_color) for x_val in x]
                else:
                    [ax.axvline(i[1], ls=':', color=self.default_color) for i in times]
            
            # Hide y-ticks.
            ax.set_yticklabels([])
            ax.set_yticks([])
            
            fig.tight_layout()
            
        # Show accompanying df with timestamp details.
        if verbose:
            df = pd.DataFrame(times, columns=['Label', 'Timestamp'])
            if mark_elapsed_time:
                df[elapsed_str] = x
            display(df)
        plt.show()


def show_metrics(model, 
                 X_test, 
                 y_test, 
                 label='', 
                 target_names=None, 
                 normalize='true',
                 return_report=False):
    """
    Print a Classification Report, a Confusion Matrix, and a ROC-Curve.
    -------------------------------------------------------------------
    
    Parameters:
    -----------
    model: pre-fit model (type: scikit-learn)
        The model from which the assessments will be shown.
    X_test: pandas DataFrame
        Test independent variable data.
    y_test: pandas DataFrame
        Test target data.
    label: str (default: '')
        An optional label for the header of the printout.
    target_names: list (default: None)
        If provided, the confusion matrix will show class names 
        rather than encoded classes.
    normalize: string (default: 'true')
        Provide a string (or None) for whether the values in the 
        confusion matrix should be normalized.
        Passing None will display the actual values.
    return_report: bool (default: False)
        If set to True, the classification report dictionary will be returned.
        
    Returns:
    --------
    If return_report is passed as True, a dictionary of the 
    classification report will be returned.
    """
    y_pred = model.predict(X_test)
    if len(np.unique(y_test)) > 2:
        multi_label = True
    else:
        multi_label = False
    
    # Print Classification Report.
    label += ' Classification Report'
    label = label.strip()
    print(headerize(label))
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))
    
    if not multi_label:  # Show Confusion Matrix and ROC-Curve.
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
        metrics.plot_confusion_matrix(model, X_test, y_test,
                                      display_labels=target_names,
                                      normalize=normalize, cmap='Blues', ax=ax1)
        ax1.set(title='Confusion Matrix')
        ax1.grid(False)

        metrics.plot_roc_curve(model, X_test, y_test, ax=ax2)
        ax2.set(title='Receiving Operator Characteristic (ROC)')
        ax2.plot([0,1], [0,1], ls=':', color='blue')
        fig.tight_layout()
    
    elif multi_label:  # Show Confusion Matrix.
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics.plot_confusion_matrix(model, X_test, y_test,
                                      display_labels=target_names,
                                      normalize=normalize, cmap='Blues', ax=ax)
        ax.set(title='Confusion Matrix')
        ax.grid(False)
        if target_names is not None:
            plt.xticks(rotation=15)
        fig.tight_layout()
    
    plt.show()
    if return_report:
        return metrics.classification_report(y_test, y_pred, 
                                             output_dict=True)


def gridsearch_model(model, 
                     model_name, 
                     params, 
                     X_train,
                     X_test,
                     y_train,
                     y_test,
                     target_names=None,
                     scoring_metrics=['accuracy', 'f1', 
                                      'precision', 'recall', 
                                      'roc_auc'],
                    multiclass_default_average='weighted'):
    """
    Fit and execute a GridSearch.
    Print metrics for each optimization.
    Return DataFrame with metrics.
    
    Parameters
    ----------
    model: An uninstantiated model (sklearn)
        
    model_name: str
        String to label the model-type throughout the process.
    params: dict
        Example: {param: [list, of, values]}
        Each key should point to a list of values to check.
    X_train, X_test, y_train, y_test: training and test data.
        X_ data should be pd.DataFrame.
        y_ data should be pd.Series.
    target_names: list (default: None)
        If provided, the confusion matrix will show class names.
        rather than encoded classes.
    scoring_metrics: list of strings (default: ['accuracy', f1', 'precision', 
                                                'recall', 'roc_auc'])
        Metrics to try and optimize for.
    multiclass_default_average: string (default: 'weighted')
        Built-in sklearn average to accompany metric.
        ['micro', 'macro', 'weighted']
        This will only be used in multi-class gridsearches.
    """
    global RANDOM_STATE
    
    # Set up variables.
    try:
        model = model(random_state=RANDOM_STATE)
    except:
        model = model()
    
    # Determine number of classes.
    if len(y_test.unique()) > 2:
        multi_class = True
    else:
        multi_class = False
        multiclass_default_average = 'binary'
    
    # Prepare df data.
    metrics_headers = ['model_label', 'best_params'] + scoring_metrics
    rows = []
    
    # Iterate over metrics to optimize for.
    watch = Stopwatch()
    for scoring in scoring_metrics:
        if multi_class and scoring != 'accuracy':
            scoring = f'{scoring}_{multiclass_default_average}'
        
        # Run GridSearch.
        gridsearch = GridSearchCV(estimator=model, 
                                  param_grid=params,
                                  scoring=scoring,
                                  cv=5)

        gridsearch.fit(X_train, y_train)

        best_model = gridsearch.best_estimator_
        show_metrics(best_model, X_test, y_test, 
                     label=f'{model_name} (optimized for {scoring})',
                     target_names=target_names)
        
        # Append the metrics to the `rows` list.
        y_pred = best_model.predict(X_test)
        
        metrics_lst = []
        if 'accuracy' in scoring_metrics:
            metrics_lst.append(metrics.accuracy_score(y_test, y_pred))
        if 'f1' in scoring_metrics:
            metrics_lst.append(
                metrics.f1_score(y_test, y_pred, average=multiclass_default_average))
        if 'precision' in scoring_metrics:
            metrics_lst.append(
                metrics.precision_score(y_test, y_pred, average=multiclass_default_average))
        if 'recall' in scoring_metrics:
            metrics_lst.append(
                metrics.recall_score(y_test, y_pred, average=multiclass_default_average))
        if 'roc_auc' in scoring_metrics:
            metrics_lst.append(
                metrics.roc_auc_score(y_test, y_pred))
        
        rows.append([f'{model_name}-{scoring}', gridsearch.best_params_] + metrics_lst)

        watch.lap(label=scoring)
    watch.display_laps()

    return pd.DataFrame(rows, columns=metrics_headers)


def plot_comparisons(gridsearch_model_df):
    """
    Takes a dataframe returned from `gridsearch_model()`.
    Plots a grouped bar chart comparing metrics.
    """
    perf_df_t = gridsearch_model_df.drop('best_params', axis=1).T
    perf_df_t.columns = perf_df_t.iloc[0]
    perf_df_t = perf_df_t.iloc[1:]

    ax = perf_df_t.plot.barh(figsize=(10,6))
    fig = plt.gcf()

    ax.set(title='Metrics Comparison', 
           ylabel='Metric', xlabel='Score')
    ax.legend(loc='lower left')
    fig.tight_layout()
    
    plt.show()
    
    
def show_metrics_for_df(gridsearch_df,
                        model, 
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        target_names=None):
    """
    Takes a dataframe returned from `gridsearch_model()`.
    Fits and plots metrics for each of the models in the dataframe.
    
    Parameters
    ----------
    gridsearch_df: pd.DataFrame
        A dataframe returned from `gridsearch_model()`.
        Columns must include ['model_label', 'best_params']
    model: sklearn model class
        An uninstantiated model from sklearn.
    X_train, X_test, y_train, y_test: Test and train data.
        X_ data should be pd.Dataframes.
        y_ data should be pd.Series.
    target_names: list (default: None)
        If provided, the confusion matrix will show class names.
        rather than encoded classes.
    """
    watch = Stopwatch()
    for idx, data in gridsearch_df.iterrows():
        label = data['model_label']
        params = data['best_params']
        
        classifier = model(**params)
        classifier.fit(X_train, y_train)
        
        show_metrics(classifier, 
                     X_test, 
                     y_test, 
                     label=label,
                     target_names=target_names)
        watch.lap(label=label)
    watch.display_laps()
    
    
def plot_on_map(dataframe, 
                lat_col='latitude',
                lon_col='longitude',
                color=None,
                color_lst=['#b41f38', '#1fb426', '#ccc125']):
    """
    Plot a dataframe on a map using Plotly Express.
    
    Returns a plotly figure.
    """
    m_lat = dataframe['latitude'].mean()
    m_lon = dataframe['longitude'].mean()
    
    fig = px.scatter_geo(data_frame=dataframe,
                         lat=lat_col,
                         lon=lon_col,
                         color=color,
                         color_discrete_sequence=color_lst,
                         opacity=0.5,
                         center={'lat': m_lat, 'lon': m_lon},
                         title='Geography of Wells (by well condition)')
    fig.update_geos(fitbounds="locations")

    return fig