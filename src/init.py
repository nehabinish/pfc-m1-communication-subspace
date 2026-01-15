"""
Title: Variable Initialization and Visualization Parameters
Author: @nehabinish
Date: 15/11/2023 | 15:32:00
Updated: 16/09/2024
"""

#%% ==                                                               Variable Initialisation                                                                ==

# Task labels for different context functions 
context = ['Predictive', 'Random']

# Regions of interest (brain regions)
roi = ['R1', 'R2']

# Time points for when specific stimuli were presented
stimulus_times = { 'S1': -4.05,
                   'S2': -2.7, 
                   'S3': -1.35, 
                   'Target': 0.0}

# Initialization of color codes for visualization
colors = ['#ff7f0e', '#1f77b4', '#E61A1B', '#0C4B9D',
          '#458B74', '#00008B', '#483D8B',
          '#EEC900', '#808000', '#CD6889']

# Mapping of brain regions to specific colors for visual representation
roi_colours = {'R1': '#FB8500',
                'R2': '#219EBC'}

# Mapping of context functions to specific colors for visual representation
context_colours = {'Predictive': '#76cd6f',
                   'Random': '#a50930'}

