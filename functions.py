import os
import io
import pandas as pd
import numpy as np
import seaborn as sns
import folium

BASIC_COLS = ['UNITID', 'OPEID', 'OPEID6', 
              'INSTNM', 'ZIP', 'LATITUDE', 
              'LONGITUDE', 'CONTROL', 'PREDDEG', 'UGDS']

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


def translation_dict(datadict):
    """

    translation_dict  outputs a dictionary satisfying 
    the following conditions:
    - The keys are the column names of colleges that are 
    strings encoded as integers (i.e. columns for which 
    VALUE and LABEL in datadict are non-empty).
    - The values are also dictionaries; each has keys 
    given by VALUE and values LABEL.

    :param datadict: a dataframe like `datadict`.
    :returns: a dictionary of key-value correspondences

    :Example:
    >>> datadict_path = os.path.join('data', 'CollegeScorecardDataDictionary.xlsx')
    >>> datadict = pd.read_excel(datadict_path, sheet_name='data_dictionary')
    >>> d = translation_dict(datadict)
    >>> len(d.keys())
    28
    >>> set(d['PREDDEG'].keys()) == set([0,1,2,3,4])
    True
    >>> 'Not classified' in d['PREDDEG'].values()
    True
    """
    copy = datadict.copy() #working copy of datadict

    translated = {} #empty dictionary to add to
    intCodes = copy.dropna(subset=['VALUE', 'LABEL']) #get only rows with values in both VALUE and LABEL
    keys = intCodes['VARIABLE NAME'].unique() #get all the unique variables

    col = keys[0] #start with the first variable

    def filler(value): #helper function
        nonlocal col #nonlocal, already introduced, progressively change
        if type(value) == float: #if the value is np.NaN aka not a string
            value = col #change it to the previous variable name (since the next variable set will start with a string)
            return value
        else:
            col = value #if it's a string, we start the next variable
            return col

    intCodes['VARIABLE NAME'] = intCodes['VARIABLE NAME'].apply(filler) #apply, replace NaNs with variable names
    intCodes = intCodes[['VARIABLE NAME', 'VALUE', 'LABEL']] #take these three columns

    grouped = intCodes.groupby(['VARIABLE NAME'])[['VALUE', 'LABEL']] #group by variable name, get value and label columns

    names = grouped.apply(lambda x: x.name).astype(str) #get the names as a Series (since we're gonna go through with apply)

    def grouper(group): #helper function
        nonlocal translated #the empty dictionary we already made
        values = grouped.get_group(group).set_index( #get the group for each variable name, set VALUE to index
            'VALUE').T.to_dict('records')[0]  # transpose so index are columns, to_dict with records style (list like [{column -> value}, … , {column -> value}])
        translated[group] = values #make new entry in main dictionary, with value of smaller dict with key of VALUE and value of LABEL
        return values

    names.apply(grouper) #apply helper function to each variable

    return translated


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


def basic_stats(college):
    """

    basic_stats takes in college and returns 
    a Series of the above st
    atistics index by:
    ['num_schools', 'num_satellite', 'num_students', 'avg_univ_size']

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = basic_stats(college)
    >>> out.index[0] == 'num_schools'
    True
    >>> out['num_students'] > 10**6
    True
    """
    stats = pd.Series( #make empty series with the indexes we want
        index=['num_schools', 'num_satellite', 'num_students', 'avg_univ_size'])

    uniqueSchools = college['OPEID'].nunique() #get number of unique schools
    stats['num_schools'] = uniqueSchools

    mainCampuses = college['OPEID6'].nunique() #get number of main colleges
    # count all unique schools, don't count main campuses, the rest are satellite
    stats['num_satellite'] = uniqueSchools - mainCampuses

    totalUndergrads = college['UGDS'].sum() #get all the undergrads
    stats['num_students'] = totalUndergrads

    avgUndergrad = np.mean(college['UGDS']) #get the mean number of undergrads
    stats['avg_univ_size'] = avgUndergrad

    return stats


def plot_school_sizes(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> ax = plot_school_sizes(college)
    >>> ax.get_ylabel()
    'Frequency'
    """
    graphed = college['UGDS'].hist(bins=20) #plot number of undergrads, set bins to 20
    graphed.set_yscale('log') #set y scale to log scale
    graphed.set_ylabel('Frequency') #set y label

    return graphed

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


def num_of_small_schools(college, k):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> nschools = len(college)
    >>> out = num_of_small_schools(college, nschools - 1)
    >>> out == (len(college) - 1)
    True
    >>> import numbers
    >>> isinstance(num_of_small_schools(college, 2), numbers.Integral)
    True
    """
    topOrdered = college['UGDS'].sort_values(ascending=False) # order the colleges from biggest to smallest
    # get the top k colleges, get the total aka bigsum
    topKtotal = topOrdered.head(k).cumsum().iloc[-1]

    bottomOrdered = college['UGDS'].sort_values().cumsum().reset_index()[
        'UGDS']  # order colleges from smallest to biggest
    # where the cumsum is bigger/equal to the bigsum
    totaled = (bottomOrdered >= topKtotal)
    # get how many schools can fit
    selection = totaled.loc[totaled == True].index[0]

    return selection


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


def col_pop_stats(college, col):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = col_pop_stats(college, "PREDDEG")
    >>> (out.columns == ['size', 'sum', 'mean', 'median']).all()
    True
    >>> (out.index == [1,2,3]).all()
    True
    >>> (out > 0).all().all()
    True
    """
    indexVals = college[col].sort_values().unique() #get the unique values, sort first for convenience
    stats = pd.DataFrame( #make empty dataframe with the columns/index we want
        columns=['size', 'sum', 'mean', 'median'], index=indexVals)

    numColleges = college.groupby(col).size() #get the total number of colleges for each group
    stats['size'] = numColleges

    totalUndergrad = college.groupby(col)['UGDS'].sum() #get the total number of undergrads for each group
    stats['sum'] = totalUndergrad

    meanSize = college.groupby(col)['UGDS'].mean() #get the mean number of undergrads for each group
    stats['mean'] = meanSize

    medianSize = college.groupby(col)['UGDS'].median() #get the median number of undergrads for each group
    stats['median'] = medianSize

    stats.index.name = col #set the index name to the col we're looking at for future reference

    return stats


def col_pop_stats_plots(stats, datadict):
    """

    :Example:
    >>> datadict_path = os.path.join('data', 'CollegeScorecardDataDictionary.xlsx')
    >>> datadict = pd.read_excel(datadict_path, sheet_name='data_dictionary')
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = col_pop_stats(college, "PREDDEG")
    >>> ax = col_pop_stats_plots(out, datadict)
    >>> len(ax)
    4
    >>> ax[-1].get_title()
    'median'
    """
    name = stats.index.name #get the col used from the previous function

    originalIndex = stats.index.values.tolist() #get the integer indexes

    translated = translation_dict(datadict) #translate the datadict
    labelKeys = translated[name] #get the values/labels for the variable

    newIndex = [labelKeys[original] for original in originalIndex] #get the string explaination of the integer
    stats[name] = newIndex #add a column with the corresponding names
    stats = stats.set_index(name) #set the new index to the new names

    return stats.plot(subplots=True, kind='bar') #plot


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


def control_preddeg_stats(college, f):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = control_preddeg_stats(college, lambda x:1)
    >>> (out == 1).all().all()
    True
    >>> out.index.name
    'CONTROL'
    >>> out.columns.name
    'PREDDEG'
    """
    grouped = college.groupby(['CONTROL', 'PREDDEG'])['UGDS'].apply(f) #group by the two cols, get undergrad column, apply function
    #maybe try to use a pivot table

    return grouped.unstack() #unstack to single index


def control_preddeg_stats_plot(out, datadict):
    """
    :Example:
    >>> datadict_path = os.path.join('data', 'CollegeScorecardDataDictionary.xlsx')
    >>> datadict = pd.read_excel(datadict_path, sheet_name='data_dictionary')
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = control_preddeg_stats(college, lambda x:1)
    >>> ax = control_preddeg_stats_plot(out, datadict)
    >>> ax.get_children()[0].get_height()
    1
    >>> ax.get_xlabel()
    'CONTROL'
    """
    control = out.index.name #get the control name
    originalIndex = out.index.values.tolist() #get the original index values

    translated = translation_dict(datadict) #translate the datadict
    controlKeys = translated[control] #get the keys for the col we're looking at

    newIndex = [controlKeys[original] for original in originalIndex] #get the corresponding strings
    out[control] = newIndex  # add a column with the corresponding names
    out = out.set_index(control)  # set the new index to the new names

    return out.plot(kind='bar') #plot


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def scatterplot_us(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> ax = scatterplot_us(college)
    >>> ax.get_xlabel()
    'LONGITUDE'
    >>> ax.get_title()
    'Undergraduate Institutions'
    >>>


    """
    lowerStates = ['AL', 'IL', 'WA', 'AZ', 'NM', 'AR', 'CA', 'MN', 'CO', 'CT', 'DE',
                   'DC', 'FL', 'GA', 'ID', 'IN', 'MI', 'IA', 'KS', 'MO', 'KY', 'LA',
                   'ME', 'MD', 'MA', 'MS', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NY', 'NC',
                   'ND', 'OH', 'WV', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                   'UT', 'VT', 'VA', 'WI', 'WY']  # arrray of 48 lower states + DC
    # if in the 48 lower states or DC
    lower = college.loc[college['STABBR'].isin(lowerStates)]
    # we only need these columns to plot
    lower = lower[['LATITUDE', 'LONGITUDE', 'UGDS', 'CONTROL']]
    # make control into object so sns doesn't think it's quantative
    lower['CONTROL'] = lower['CONTROL'].apply(lambda x: str(x) + '_')

    # plot, size to UGDS size, color by control
    plotted = sns.scatterplot(
        data=lower, x='LONGITUDE', y='LATITUDE', hue='CONTROL', size='UGDS')
    plotted.set_title('Undergraduate Institutions')  # set title

    return plotted

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


def plot_folium():
    """

    :Example:
    >>> d = plot_folium()
    >>> isinstance(d, dict)
    True
    >>> 'geo_data' in d.keys()
    True
    >>> isinstance(d.get('key_on'), str)
    True
    """
    fips = pd.read_csv(os.path.join('data', 'state_fips.txt'), sep="|") #open the fips file
    states = os.path.join('data', 'population-2017.csv') #get the population data path
    state_pop = pd.read_csv(states) #get the population data
    state_geo = os.path.join('data', 'gz_2010_us_040_00_5m.json') #geojson path

    college_path = os.path.join('data', 'MERGED2016_17_PP.csv') #college data path
    college = pd.read_csv(college_path) #college data read

    studentsbyState = college.groupby('STABBR')['UGDS'].sum() #get the total students per state
    population = state_pop.groupby('STATE')['POP'].sum() #get the population per state
    ratio = studentsbyState / population #get the ratio of the two
    ratio = ratio.to_frame('RATIO') #make it into a dataframe
    ratio.index.name = 'STUSAB' #make index STUSAB so we can join with fips

    joined = fips.join(ratio, on='STUSAB') #join with fips

    mapped = folium.Map(location=[48, -102], zoom_start=3) #make the folium map
    folium.GeoJson(state_geo).add_to(mapped) #add the state borders/base coloring

    folium.Choropleth( #add choropleth map
        geo_data=state_geo, #over the state borders/bases
        name='choropleth', #it's a choropleth map
        data=joined, #we're using our joined data
        columns=['STATE_NAME', 'RATIO'], #using the state name and ratio
        key_on='feature.properties.NAME', #bind it to the name property of each feature aka each state
        threshold_scale=ratio['RATIO'].quantile([0, .25, .5, .75, 1]).values, #get the standard quartiles
        fill_color='YlGn', #coloring
        fill_opacity=0.7, #opacity
        line_opacity=0.2, #opacity
        legend_name='Ratio' #legend name
    ).add_to(mapped)

    #mapped.save('pct_students_by_state.html') #save as html

    keyDict = { #the same thing but as a dictionary for returning
        'geo_data': state_geo,
        'name': 'choropleth',
        'data': joined,
        'columns': ['STATE_NAME', 'RATIO'],
        'key_on': 'feature.properties.NAME',
        'threshold_scale': ratio['RATIO'].quantile([0, .25, .5, .75, 1]).values,
        'fill_color': 'YlGn',
        'fill_opacity': 0.7,
        'line_opacity': 0.2,
        'legend_name': 'Ratio'
    }

    return keyDict


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


def control_type_by_state(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = control_type_by_state(college)
    >>> len(out)
    49
    >>> np.allclose(out.sum(axis=1), 1)
    True
    """
    lowerStates = ['AL', 'IL', 'WA', 'AZ', 'NM', 'AR', 'CA', 'MN', 'CO', 'CT', 'DE',
                   'DC', 'FL', 'GA', 'ID', 'IN', 'MI', 'IA', 'KS', 'MO', 'KY', 'LA',
                   'ME', 'MD', 'MA', 'MS', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NY', 'NC',
                   'ND', 'OH', 'WV', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                   'UT', 'VT', 'VA', 'WI', 'WY']  # arrray of 48 lower states + DC
    # if in the 48 lower states or DC
    lower = college.loc[college['STABBR'].isin(lowerStates)]
    counts = lower.groupby(['STABBR', 'CONTROL']).size().unstack() #get the controls for each state
    countsProp = counts.apply(lambda x: x / x.sum(), axis=1) #get the distribution conditional on state

    return countsProp


def tvd_states(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = tvd_states(college)
    >>> len(out)
    49
    >>> 'NV' in out.index[:5]
    True
    >>> 'OR' in out.index[-5:]
    True
    """
    unconditional = college['CONTROL'].value_counts(
    ) / college['CONTROL'].count()  # get the unconditional distribution
    # get conditional distribution by state
    conditional = control_type_by_state(college)

    def tvd(row): return sum(abs(row - unconditional)) / 2  # tvd equation

    # apply tvd equation to each row, sort
    return conditional.apply(tvd, axis=1).sort_values(ascending=False)


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def num_subjects(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = num_subjects(college)
    >>> len(out) == len(college)
    True
    >>> out.nunique()
    34
    """
    subjects = [
        col for col in college.columns if 'PCIP' in col]  # get all the PCIP column labels
    subjectsTaken = college[subjects]  # take only those columns
    # count nonzero aka subjects being taken at each college/row
    subjectCount = subjectsTaken.apply(np.count_nonzero, axis=1)

    return subjectCount


def subject_counts(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = subject_counts(college)
    >>> len(out)
    34
    >>> out.loc[0].sum() == 3060
    True
    """
    offerings = np.sort(num_subjects(college).unique()
                        )  # get all unique counts of subject offerings
    # grab the control and UGDS columns only
    controlled = college[['CONTROL', 'UGDS']]
    # add an offerings column so we can group
    controlled['OFFERINGS'] = num_subjects(college)

    grouped = controlled.groupby(['OFFERINGS', 'CONTROL'])['UGDS'].sum(
    ).unstack()  # group by both, get UGDS of all groups, unstack

    return grouped


def create_specialty(college, datadict):
    """

    :Example:
    >>> datadict_path = os.path.join('data', 'CollegeScorecardDataDictionary.xlsx')
    >>> datadict = pd.read_excel(datadict_path, sheet_name='data_dictionary')
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = create_specialty(college, datadict)
    >>> len(out.columns) == len(college.columns) + 1
    True
    >>> 'Psychology' in out['SPECIALTY'].unique()
    True
    """
    copied = college.copy()  # make a deep copy
    # get the number of subjects so we can find the specialty schools
    subjectCount = num_subjects(copied)

    # get all the subject options
    subjects = [col for col in college.columns if 'PCIP' in col]
    # get all the specialty schools
    specialized = subjectCount.loc[subjectCount == 1].index
    specialties = copied.loc[specialized][subjects].apply(
        pd.Series.idxmax, axis=1)  # get their specialty

    # translate it using datadict
    def labeling(
        subject): return datadict.loc[datadict['VARIABLE NAME'] == subject]['LABEL'].iloc[0]
    copied['SPECIALTY'] = specialties.apply(labeling)  # add the new column

    return copied